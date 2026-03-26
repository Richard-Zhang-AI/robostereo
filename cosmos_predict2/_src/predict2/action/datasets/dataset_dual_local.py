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
Dual-tower action-conditioned dataset that returns synchronized RGB and geometry batches.
"""

import json
import os
import random
import time
import traceback
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

from cosmos_predict2._src.predict2.action.datasets.dataset_local import Dataset_3D


class DualDataset_3D(Dataset):
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
        video_source="rgb",
        geometry_normalize=True,
        geometry_min=-0.8,
        geometry_max=1.5,
    ):
        super().__init__()
        self.rgb_dataset = Dataset_3D(
            train_annotation_path=train_annotation_path,
            val_annotation_path=val_annotation_path,
            test_annotation_path=test_annotation_path,
            video_path=video_path,
            fps_downsample_ratio=fps_downsample_ratio,
            num_action_per_chunk=num_action_per_chunk,
            cam_ids=cam_ids,
            accumulate_action=accumulate_action,
            video_size=video_size,
            val_start_frame_interval=val_start_frame_interval,
            debug=debug,
            normalize=normalize,
            pre_encode=pre_encode,
            do_evaluate=do_evaluate,
            load_t5_embeddings=load_t5_embeddings,
            load_action=load_action,
            mode=mode,
            state_key=state_key,
            gripper_key=gripper_key,
            gripper_rescale_factor=gripper_rescale_factor,
            is_rollout=is_rollout,
            video_source="rgb",
            geometry_normalize=geometry_normalize,
            geometry_min=geometry_min,
            geometry_max=geometry_max,
        )
        self.xyz_dataset = Dataset_3D(
            train_annotation_path=train_annotation_path,
            val_annotation_path=val_annotation_path,
            test_annotation_path=test_annotation_path,
            video_path=video_path,
            fps_downsample_ratio=fps_downsample_ratio,
            num_action_per_chunk=num_action_per_chunk,
            cam_ids=cam_ids,
            accumulate_action=accumulate_action,
            video_size=video_size,
            val_start_frame_interval=val_start_frame_interval,
            debug=debug,
            normalize=normalize,
            pre_encode=pre_encode,
            do_evaluate=do_evaluate,
            load_t5_embeddings=load_t5_embeddings,
            load_action=load_action,
            mode=mode,
            state_key=state_key,
            gripper_key=gripper_key,
            gripper_rescale_factor=gripper_rescale_factor,
            is_rollout=is_rollout,
            video_source="geometry",
            geometry_normalize=geometry_normalize,
            geometry_min=geometry_min,
            geometry_max=geometry_max,
        )
        self.samples = self.rgb_dataset.samples
        self.mode = mode
        self.cam_ids = cam_ids
        self.load_action = load_action
        self.load_t5_embeddings = load_t5_embeddings
        self.pre_encode = pre_encode
        self.state_key = state_key
        self.gripper_key = gripper_key
        self._debug_dataloader = os.getenv("COSMOS_DATALOADER_DEBUG", "0") == "1"
        self._debug_every = max(1, int(os.getenv("COSMOS_DATALOADER_DEBUG_EVERY", "1")))
        self._debug_first = max(0, int(os.getenv("COSMOS_DATALOADER_DEBUG_FIRST", "1")))
        self._error_log_every = max(1, int(os.getenv("COSMOS_DATASET_ERROR_LOG_EVERY", "50")))
        self._error_max = int(os.getenv("COSMOS_DATASET_ERROR_MAX", "0"))
        self._max_retry = max(1, int(os.getenv("COSMOS_DATASET_MAX_RETRY", "100")))
        self._bad_ann_files: set[str] = set()
        self._bad_ann_files_logged: set[str] = set()
        self.wrong_number = 0

    def _dbg(self, msg: str) -> None:
        if not self._debug_dataloader:
            return
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}][DualDataset_3D] {msg}", flush=True)

    def __len__(self):
        return len(self.samples)

    def _build_data_dict(self, video, ann_file, label, actions):
        data = dict()
        if self.load_action:
            data["action"] = actions.float()
        data["video"] = video.to(dtype=torch.uint8)
        data["annotation_file"] = ann_file

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

        if self.load_t5_embeddings:
            t5_embeddings = np.squeeze(np.load(ann_file.replace(".json", ".npy")))
            data["t5_text_embeddings"] = torch.from_numpy(t5_embeddings)
        else:
            data["t5_text_embeddings"] = torch.zeros(512, 1024, dtype=torch.bfloat16)
            data["ai_caption"] = ""
        data["t5_text_mask"] = torch.ones(512, dtype=torch.int64)
        data["fps"] = 4
        data["image_size"] = 256 * torch.ones(4)
        data["num_frames"] = self.rgb_dataset.sequence_length
        data["padding_mask"] = torch.zeros(1, 256, 256)
        return data

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
                with open(ann_file, "r") as f:
                    label = json.load(f)

                has_state = (
                    self.rgb_dataset._state_key in label
                    and self.rgb_dataset._gripper_key in label
                    and len(label[self.rgb_dataset._state_key]) > 0
                    and len(label[self.rgb_dataset._gripper_key]) > 0
                )
                if has_state:
                    arm_states, gripper_states = self.rgb_dataset._get_robot_states(label, frame_ids)
                    actions = self.rgb_dataset._get_actions(
                        arm_states, gripper_states, self.rgb_dataset.accumulate_action
                    )
                    actions *= self.rgb_dataset.c_act_scaler
                else:
                    if "action" in label and len(label["action"]) > 0:
                        all_actions = np.array(label["action"], dtype=float)
                    elif "actions" in label and len(label["actions"]) > 0:
                        all_actions = np.array(label["actions"], dtype=float)
                    else:
                        raise ValueError("No valid state/gripper or action fields found in annotation.")
                    act_ids = [i for i in frame_ids[:-1] if i < len(all_actions)]
                    if len(act_ids) != self.rgb_dataset.sequence_length - 1:
                        raise ValueError("Not enough actions to match sequence length.")
                    actions = torch.from_numpy(all_actions[act_ids]) * torch.from_numpy(self.rgb_dataset.c_act_scaler)

                if cam_id is None:
                    cam_id = random.choice(self.cam_ids)

                if self.pre_encode:
                    raise NotImplementedError("Pre-encoded videos are not supported for this dataset.")

                if debug_this:
                    t0 = time.perf_counter()
                rgb_video = self.rgb_dataset._get_frames(label, frame_ids, cam_id=cam_id, pre_encode=False)
                if debug_this:
                    dt = time.perf_counter() - t0
                    self._dbg(f"rgb _get_frames done in {dt:.3f}s cam_id={cam_id}")
                    t0 = time.perf_counter()
                xyz_video = self.xyz_dataset._get_frames(label, frame_ids, cam_id=cam_id, pre_encode=False)
                if debug_this:
                    dt = time.perf_counter() - t0
                    self._dbg(f"xyz _get_frames done in {dt:.3f}s cam_id={cam_id}")

                rgb_video = rgb_video.permute(1, 0, 2, 3)  # [T, C, H, W] -> [C, T, H, W]
                xyz_video = xyz_video.permute(1, 0, 2, 3)

                rgb_data = self._build_data_dict(rgb_video, ann_file, label, actions)
                xyz_data = self._build_data_dict(xyz_video, ann_file, label, actions)

                if debug_this:
                    self._dbg(f"__getitem__ done index={index}")
                return {
                    "rgb": rgb_data,
                    "xyz": xyz_data,
                    "__key__": rgb_data["__key__"],
                }
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
