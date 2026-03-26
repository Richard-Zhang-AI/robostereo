# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Run this command to interactively debug:
PYTHONPATH=. python cosmos_predict2/_src/predict2/action/datasets/dataset_local.py

Adapted from:
https://github.com/bytedance/IRASim/blob/main/dataset/dataset_3D.py
"""

import json
import os
import random
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import imageio
import numpy as np
import torch
from decord import VideoReader, cpu
from einops import rearrange
from safetensors import safe_open
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm

from cosmos_predict2._src.imaginaire.flags import INTERNAL
from cosmos_predict2._src.imaginaire.utils.dataset_utils import Resize_Preprocess, ToTensorVideo, euler2rotm, rotm2euler


class Dataset_3D(Dataset):
    def __init__(
        self,
        train_annotation_path,
        val_annotation_path,
        test_annotation_path,
        video_path,
        fps_downsample_ratio,
        num_action_per_chunk,
        cam_ids,
        accumulate_action,
        video_size,
        val_start_frame_interval,
        debug=False,
        normalize=False,
        pre_encode=False,
        do_evaluate=False,
        load_t5_embeddings=False,
        load_action=True,
        mode="train",
        state_key="state",
        gripper_key="continuous_gripper_state",
        gripper_rescale_factor=1.0,
        is_rollout=None,
        video_source="rgb",  # "rgb" for rgb.mp4, "geometry" for geometry.safetensors
        geometry_normalize=True,
        geometry_min=-0.8,
        geometry_max=1.5,
    ):
        """Dataset class for loading 3D robot action-conditional data.

        This dataset loads robot trajectories consisting of RGB video frames, robot states (arm positions and gripper states),
        and computes relative actions between consecutive frames.

        Args:
            train_annotation_path (str): Path to training annotation files
            val_annotation_path (str): Path to validation annotation files
            test_annotation_path (str): Path to test annotation files
            video_path (str): Base path to video files
            fps_downsample_ratio (int): Interval between sampled frames in a sequence
            num_action_per_chunk (int): Number of frames to load per sequence
            cam_ids (list): List of camera IDs to sample from
            accumulate_action (bool): Whether to accumulate actions relative to first frame
            video_size (list): Target size [H,W] for video frames
            val_start_frame_interval (int): Frame sampling interval for validation/test
            debug (bool, optional): If True, only loads subset of data. Defaults to False.
            normalize (bool, optional): Whether to normalize video frames. Defaults to False.
            pre_encode (bool, optional): Whether to pre-encode video frames. Defaults to False.
            do_evaluate (bool, optional): Whether in evaluation mode. Defaults to False.
            load_t5_embeddings (bool, optional): Whether to load T5 embeddings. Defaults to False.
            load_action (bool, optional): Whether to load actions. Defaults to True.
            mode (str, optional): Dataset mode - 'train', 'val' or 'test'. Defaults to 'train'.
            video_source (str, optional): Video source type - 'rgb' for rgb.mp4 or 'geometry' for geometry.safetensors. Defaults to 'rgb'.
            geometry_normalize (bool, optional): Whether to apply geometry min-max normalization. Defaults to True.
            geometry_min (float, optional): Min XYZ value for geometry min-max normalization. Defaults to -0.8.
            geometry_max (float, optional): Max XYZ value for geometry min-max normalization. Defaults to 1.5.

        The dataset loads robot trajectories and computes:
        - Video frames from specified camera views (RGB or geometry XYZ)
        - Robot arm states (xyz position + euler angles)
        - Gripper states (binary open/closed)
        - Relative actions between consecutive frames

        Actions are computed as relative transforms between frames:
        - Translation: xyz offset in previous frame's coordinate frame
        - Rotation: euler angles of relative rotation
        - Gripper: binary gripper state

        Returns dict with:
            - video: Video frames tensor [T,C,H,W]
            - action: Action tensor [T-1,7]
            - video_name: Dict with episode/frame metadata
            - latent: Pre-encoded video features if pre_encode=True
        """

        super().__init__()
        if mode == "train":
            self.data_path = train_annotation_path
            self.start_frame_interval = 1
        elif mode == "val":
            self.data_path = val_annotation_path
            self.start_frame_interval = val_start_frame_interval
        elif mode == "test":
            self.data_path = test_annotation_path
            self.start_frame_interval = val_start_frame_interval
        self.video_path = video_path
        self.fps_downsample_ratio = fps_downsample_ratio
        self.mode = mode

        # self.sequence_length = num_frames
        self.sequence_length = 1 + num_action_per_chunk
        self.normalize = normalize
        self.pre_encode = pre_encode
        self.load_t5_embeddings = load_t5_embeddings
        self.load_action = load_action

        self.cam_ids = cam_ids
        self.accumulate_action = accumulate_action
        self.is_rollout = is_rollout
        self.video_source = video_source  # "rgb" or "geometry"
        self.geometry_normalize = geometry_normalize
        self.geometry_min = geometry_min
        self.geometry_max = geometry_max
        print(f"Using video source: {self.video_source}")

        self._debug_dataloader = os.getenv("COSMOS_DATALOADER_DEBUG", "0") == "1"
        self._debug_every = max(1, int(os.getenv("COSMOS_DATALOADER_DEBUG_EVERY", "1")))
        self._debug_first = max(0, int(os.getenv("COSMOS_DATALOADER_DEBUG_FIRST", "1")))
        self._error_log_every = max(1, int(os.getenv("COSMOS_DATASET_ERROR_LOG_EVERY", "50")))
        self._error_max = int(os.getenv("COSMOS_DATASET_ERROR_MAX", "0"))
        self._max_retry = max(1, int(os.getenv("COSMOS_DATASET_MAX_RETRY", "100")))
        self._bad_ann_files: set[str] = set()
        self._bad_ann_files_logged: set[str] = set()

        self.action_dim = 7  # ee xyz (3) + ee euler (3) + gripper(1)
        self.c_act_scaler = [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, gripper_rescale_factor]
        self.c_act_scaler = np.array(self.c_act_scaler, dtype=float)
        self.ann_files = self._init_anns(self.data_path)
        self._filter_rollout()

        self._state_key = state_key
        self._gripper_key = gripper_key

        print(f"{len(self.ann_files)} trajectories in total")
        self.samples = self._init_sequences(self.ann_files)

        self.samples = sorted(self.samples, key=lambda x: (x["ann_file"], x["frame_ids"][0]))
        if debug and not do_evaluate:
            self.samples = self.samples[0:10]
        print(f"{len(self.ann_files)} trajectories in total")
        print(f"{len(self.samples)} samples in total")
        # with open('./samples_16.pkl','wb') as file:
        #     pickle.dump(self.samples,file)
        self.wrong_number = 0
        self.transform = T.Compose([T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)])
        self.training = False
        self.preprocess = T.Compose(
            [
                ToTensorVideo(),
                Resize_Preprocess(tuple(video_size)),  # 288 512
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        self.not_norm_preprocess = T.Compose([ToTensorVideo(), Resize_Preprocess(tuple(video_size))])

    def _dbg(self, msg: str) -> None:
        if not self._debug_dataloader:
            return
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}][Dataset_3D:{self.video_source}] {msg}", flush=True)

    def __str__(self):
        return f"{len(self.ann_files)} samples from {self.data_path}"

    def _init_anns(self, data_dir):
        ann_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".json")]
        return ann_files

    def _init_sequences(self, ann_files):
        samples = []
        with ThreadPoolExecutor(32) as executor:
            future_to_ann_file = {
                executor.submit(self._load_and_process_ann_file, ann_file): ann_file for ann_file in ann_files
            }
            for future in tqdm(as_completed(future_to_ann_file), total=len(ann_files)):
                samples.extend(future.result())
        return samples

    def _filter_rollout(self):
        if self.is_rollout is None:
            return

        print(f"Filtering rollout: {self.is_rollout}")
        ann_files = []
        # Check if any file in self.ann_files has "is_eval" set to True
        for ann_file in self.ann_files:
            with open(ann_file, "r") as f:
                ann_data = json.load(f)
            is_eval = ann_data["episode_metadata"]["is_eval"]
            if self.is_rollout and is_eval:
                ann_files.append(ann_file)
            elif not self.is_rollout and not is_eval:
                ann_files.append(ann_file)

        self.ann_files = ann_files
        print(f"Filtered {len(ann_files)} rollout: {self.is_rollout}")
        return

    def _load_and_process_ann_file(self, ann_file):
        samples = []
        with open(ann_file, "r") as f:
            ann = json.load(f)

        n_frames = self._get_num_frames(ann)
        if n_frames <= 0:
            return samples
        for frame_i in range(0, n_frames, self.start_frame_interval):
            sample = dict()
            sample["ann_file"] = ann_file
            sample["frame_ids"] = []
            curr_frame_i = frame_i
            while True:
                if curr_frame_i > (n_frames - 1):
                    break
                sample["frame_ids"].append(curr_frame_i)
                if len(sample["frame_ids"]) == self.sequence_length:
                    break
                curr_frame_i += self.fps_downsample_ratio
            # make sure there are sequence_length number of frames
            if len(sample["frame_ids"]) == self.sequence_length:
                samples.append(sample)
        return samples

    def __len__(self):
        return len(self.samples)

    def _get_num_frames(self, ann: dict) -> int:
        if self._state_key in ann and len(ann[self._state_key]) > 0:
            return len(ann[self._state_key])
        if "action" in ann and len(ann["action"]) > 0:
            return len(ann["action"]) + 1
        if "actions" in ann and len(ann["actions"]) > 0:
            return len(ann["actions"]) + 1
        return 0

    def _load_video(self, video_path, frame_ids):
        if self._debug_dataloader:
            t0 = time.perf_counter()
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        assert (np.array(frame_ids) < len(vr)).all()
        assert (np.array(frame_ids) >= 0).all()
        vr.seek(0)
        frame_data = vr.get_batch(frame_ids).asnumpy()
        if self._debug_dataloader:
            dt = time.perf_counter() - t0
            self._dbg(f"_load_video done in {dt:.3f}s path={video_path} frames={len(frame_ids)}")
        return frame_data

    def _load_geometry_safetensors(self, video_path, frame_ids):
        """Load geometry data from geometry.safetensors file.

        Args:
            video_path: Path to the video folder containing geometry.safetensors
            frame_ids: List of frame indices to load

        Returns:
            numpy array of shape [len(frame_ids), H, W, 3] with XYZ coordinates
        """
        # Replace rgb.mp4 with geometry.safetensors in the path
        if "rgb.mp4" in video_path:
            geometry_path = video_path.replace("rgb.mp4", "geometry.safetensors")
        else:
            # Fallback: try to find geometry.safetensors in the same directory
            video_dir = os.path.dirname(video_path)
            geometry_path = os.path.join(video_dir, "geometry.safetensors")

        # Load the safetensors file
        if self._debug_dataloader:
            t0 = time.perf_counter()
        with safe_open(geometry_path, framework="pt", device="cpu") as f:
            # The tensor key is "xyz" with shape [T, H, W, 3]
            all_frames = f.get_tensor("xyz")  # Shape: [T, H, W, 3], dtype: float16
        if self._debug_dataloader:
            dt = time.perf_counter() - t0
            self._dbg(f"_load_geometry_safetensors done in {dt:.3f}s path={geometry_path} frames={len(frame_ids)}")

        # Convert to numpy and select the requested frames
        all_frames_np = all_frames.numpy().astype(np.float32)  # Convert float16 to float32

        # Validate frame indices
        n_frames = all_frames_np.shape[0]
        assert (np.array(frame_ids) < n_frames).all(), f"Frame indices {frame_ids} out of range for {n_frames} frames"
        assert (np.array(frame_ids) >= 0).all()

        # Select the requested frames
        frame_data = all_frames_np[frame_ids]  # Shape: [len(frame_ids), H, W, 3]

        return frame_data

    def _get_frames(self, label, frame_ids, cam_id, pre_encode):
        if pre_encode:
            raise NotImplementedError("Pre-encoded videos are not supported for this dataset.")
        else:
            video_path = label["videos"][cam_id]["video_path"]
            video_path = os.path.join(self.video_path, video_path)

            # Load frames based on video_source setting
            if self.video_source == "geometry":
                # Load from geometry.safetensors (XYZ coordinates in float16)
                frames = self._load_geometry_safetensors(video_path, frame_ids)
                if self.geometry_normalize:
                    # Min-max normalize XYZ values to [-1, 1] via [0, 1] for the RGB preprocessing pipeline.
                    frames = np.clip(frames, self.geometry_min, self.geometry_max)
                    denom = self.geometry_max - self.geometry_min
                    frames = (frames.astype(np.float32) - self.geometry_min) / denom
                    frames = np.clip(frames, 0.0, 1.0)
                else:
                    # Assume geometry values are already normalized to [0, 1].
                    frames = np.clip(frames.astype(np.float32), 0.0, 1.0)
                frames = (frames * 255.0).round().astype(np.uint8)
                frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (T, C, H, W)
            elif self.video_source == "rgb":
                # Load from rgb.mp4 (standard video frames)
                frames = self._load_video(video_path, frame_ids)
                frames = frames.astype(np.uint8)
                frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (T, C, H, W)
            else:
                raise ValueError(f"Invalid video_source: {self.video_source}. Must be 'rgb' or 'geometry'.")

            def printvideo(videos, filename):
                t_videos = rearrange(videos, "f c h w -> f h w c")
                t_videos = (
                    ((t_videos / 2.0 + 0.5).clamp(0, 1) * 255).detach().to(dtype=torch.uint8).cpu().contiguous().numpy()
                )
                print(t_videos.shape)
                writer = imageio.get_writer(filename, fps=4)  # fps 是帧率
                for frame in t_videos:
                    writer.append_data(frame)  # 1 4 13 23 # fp16 24 76 456 688

            if self.normalize:
                frames = self.preprocess(frames)
            else:
                frames = self.not_norm_preprocess(frames)
                frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
        return frames

    def _get_obs(self, label, frame_ids, cam_id, pre_encode):
        if cam_id is None:
            temp_cam_id = random.choice(self.cam_ids)
        else:
            temp_cam_id = cam_id
        frames = self._get_frames(label, frame_ids, cam_id=temp_cam_id, pre_encode=pre_encode)
        return frames, temp_cam_id

    def _get_robot_states(self, label, frame_ids):
        all_states = np.array(label[self._state_key])
        all_cont_gripper_states = np.array(label[self._gripper_key])
        states = all_states[frame_ids]
        cont_gripper_states = all_cont_gripper_states[frame_ids]
        arm_states = states[:, :6]
        return arm_states, cont_gripper_states

    def _get_actions(self, arm_states, gripper_states, accumulate_action):
        action = np.zeros((self.sequence_length - 1, self.action_dim))
        if accumulate_action:
            base_xyz = arm_states[0, 0:3]
            base_rpy = arm_states[0, 3:6]
            base_rotm = euler2rotm(base_rpy)
            for k in range(1, self.sequence_length):
                curr_xyz = arm_states[k, 0:3]
                curr_rpy = arm_states[k, 3:6]
                curr_gripper = gripper_states[k]
                curr_rotm = euler2rotm(curr_rpy)
                rel_xyz = np.dot(base_rotm.T, curr_xyz - base_xyz)
                rel_rotm = base_rotm.T @ curr_rotm
                rel_rpy = rotm2euler(rel_rotm)
                action[k - 1, 0:3] = rel_xyz
                action[k - 1, 3:6] = rel_rpy
                action[k - 1, 6] = curr_gripper
                if k % 4 == 0:
                    base_xyz = arm_states[k, 0:3]
                    base_rpy = arm_states[k, 3:6]
                    base_rotm = euler2rotm(base_rpy)
        else:
            for k in range(1, self.sequence_length):
                prev_xyz = arm_states[k - 1, 0:3]
                prev_rpy = arm_states[k - 1, 3:6]
                prev_rotm = euler2rotm(prev_rpy)
                curr_xyz = arm_states[k, 0:3]
                curr_rpy = arm_states[k, 3:6]
                curr_gripper = gripper_states[k]
                curr_rotm = euler2rotm(curr_rpy)
                rel_xyz = np.dot(prev_rotm.T, curr_xyz - prev_xyz)
                rel_rotm = prev_rotm.T @ curr_rotm
                rel_rpy = rotm2euler(rel_rotm)
                action[k - 1, 0:3] = rel_xyz
                action[k - 1, 3:6] = rel_rpy
                action[k - 1, 6] = curr_gripper
        return torch.from_numpy(action)  # (l - 1, act_dim)

    def __getitem__(self, index, cam_id=None, return_video=False):
        if self.mode != "train":
            np.random.seed(index)
            random.seed(index)

        retries = 0
        while True:
            debug_this = self._debug_dataloader and (index < self._debug_first or index % self._debug_every == 0)
            sample = self.samples[index]
            ann_file = sample["ann_file"]
            if ann_file in self._bad_ann_files:
                index = np.random.randint(len(self.samples))
                retries += 1
                if retries >= self._max_retry:
                    raise RuntimeError("Exceeded max retries due to bad samples.")
                continue
            try:
                if debug_this:
                    self._dbg(f"__getitem__ start index={index}")
                frame_ids = sample["frame_ids"]
                if debug_this:
                    self._dbg(f"loading ann_file={ann_file} frames={len(frame_ids)}")
                with open(ann_file, "r") as f:
                    label = json.load(f)
                has_state = (
                    self._state_key in label
                    and self._gripper_key in label
                    and len(label[self._state_key]) > 0
                    and len(label[self._gripper_key]) > 0
                )
                if has_state:
                    arm_states, gripper_states = self._get_robot_states(label, frame_ids)
                    actions = self._get_actions(arm_states, gripper_states, self.accumulate_action)
                    actions *= self.c_act_scaler
                else:
                    if "action" in label and len(label["action"]) > 0:
                        all_actions = np.array(label["action"], dtype=float)
                    elif "actions" in label and len(label["actions"]) > 0:
                        all_actions = np.array(label["actions"], dtype=float)
                    else:
                        raise ValueError("No valid state/gripper or action fields found in annotation.")
                    act_ids = [i for i in frame_ids[:-1] if i < len(all_actions)]
                    if len(act_ids) != self.sequence_length - 1:
                        raise ValueError("Not enough actions to match sequence length.")
                    actions = torch.from_numpy(all_actions[act_ids]) * torch.from_numpy(self.c_act_scaler)

                data = dict()
                if self.load_action:
                    data["action"] = actions.float()

                if self.pre_encode:
                    raise NotImplementedError("Pre-encoded videos are not supported for this dataset.")
                else:
                    if debug_this:
                        t0 = time.perf_counter()
                    video, cam_id = self._get_obs(label, frame_ids, cam_id, pre_encode=False)
                    if debug_this:
                        dt = time.perf_counter() - t0
                        self._dbg(f"_get_obs done in {dt:.3f}s cam_id={cam_id}")
                    video = video.permute(1, 0, 2, 3)  # Rearrange from [T, C, H, W] to [C, T, H, W]
                    data["video"] = video.to(dtype=torch.uint8)

                data["annotation_file"] = ann_file

                # NOTE: __key__ is used to uniquely identify the sample, required for callback functions
                if "episode_id" in label:
                    data["__key__"] = label["episode_id"]
                else:
                    try:
                        data["__key__"] = label["original_path"]
                    except Exception:
                        try:
                            data["__key__"] = label["episode_metadata"]["episode_id"]
                        except Exception:
                            data["__key__"] = label["episode_metadata"]["segment_id"]

                # Just add these to fit the interface
                if self.load_t5_embeddings:
                    t5_embeddings = np.squeeze(np.load(ann_file.replace(".json", ".npy")))
                    data["t5_text_embeddings"] = torch.from_numpy(t5_embeddings)
                else:
                    data["t5_text_embeddings"] = torch.zeros(512, 1024, dtype=torch.bfloat16)
                    data["ai_caption"] = ""
                data["t5_text_mask"] = torch.ones(512, dtype=torch.int64)
                data["fps"] = 4
                data["image_size"] = 256 * torch.ones(4)
                data["num_frames"] = self.sequence_length
                data["padding_mask"] = torch.zeros(1, 256, 256)

                if debug_this:
                    self._dbg(f"__getitem__ done index={index}")
                return data
            except Exception as exc:
                self.wrong_number += 1
                self._bad_ann_files.add(ann_file)
                if ann_file not in self._bad_ann_files_logged:
                    warnings.warn(
                        f"Marking bad sample: {ann_file} ({type(exc).__name__}: {exc})"
                    )
                    self._bad_ann_files_logged.add(ann_file)
                if self._error_max > 0 and self.wrong_number >= self._error_max:
                    raise
                if self._debug_dataloader or (self.wrong_number % self._error_log_every == 0):
                    warnings.warn(
                        f"Invalid data encountered: {ann_file}. Skipped (count={self.wrong_number})."
                    )
                    if self._debug_dataloader:
                        warnings.warn("FULL TRACEBACK:")
                        warnings.warn(traceback.format_exc())
                index = np.random.randint(len(self.samples))
                retries += 1
                if retries >= self._max_retry:
                    raise RuntimeError("Exceeded max retries due to bad samples.")


if INTERNAL:
    """ Run this command to interactively debug:
    PYTHONPATH=. python cosmos_predict2/_src/predict2/action/datasets/dataset_local.py
    """
    if __name__ == "__main__":
        """
        PYTHONPATH=. python cosmos_predict2/_src/predict2/action/datasets/dataset_local.py
        """

        base_path_pi_benchmark_local = "/project/cosmos/user/nvidia-cosmos-raw-data/ur5-data/video-evals-raw-data/datasets/action_dataset/single_chunk/"
        train_annotation_path_pi_benchmark_local = os.path.join(base_path_pi_benchmark_local, "annotation/val")
        val_annotation_path_pi_benchmark_local = os.path.join(base_path_pi_benchmark_local, "annotation/val")
        test_annotation_path_pi_benchmark_local = os.path.join(base_path_pi_benchmark_local, "annotation/test")
        dataset = Dataset_3D(
            train_annotation_path=train_annotation_path_pi_benchmark_local,
            val_annotation_path=val_annotation_path_pi_benchmark_local,
            test_annotation_path=test_annotation_path_pi_benchmark_local,
            video_path=base_path_pi_benchmark_local,
            fps_downsample_ratio=1,
            num_action_per_chunk=1,
            cam_ids=["base_0"],
            accumulate_action=False,
            video_size=[480, 640],
            val_start_frame_interval=1,
            mode="train",
            state_key="ee_pose",
            is_rollout=None,
        )

        indices = [0, 13, 200, -1]
        for idx in indices:
            start_time = time.time()
            print(
                (
                    f"{idx=} "
                    f"{dataset[idx]['video'].sum()=}\n"
                    f"{dataset[idx]['video'].shape=}\n"
                    # f"{dataset[idx]['video_name']=}\n"
                    f"{dataset[idx]['action'].sum()=}\n"
                    "---"
                )
            )
            end_time = time.time()
            print(f"Time taken: {end_time - start_time} seconds")
        from IPython import embed

        embed()
