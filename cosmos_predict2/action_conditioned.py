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
Action-conditioned video generation inference.
"""

import json
import os
import re
from glob import glob
from pathlib import Path

import mediapy
import numpy as np
import torch
import torchvision
from loguru import logger
from safetensors import safe_open
from tqdm import tqdm

from cosmos_predict2._src.imaginaire.utils import distributed
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.action.datasets.dataset_utils import euler2rotm, rotm2euler, rotm2quat
from cosmos_predict2._src.predict2.inference.video2world import (
    Video2WorldInference,
)
from cosmos_predict2.action_conditioned_config import (
    ActionConditionedInferenceArguments,
    ActionConditionedSetupArguments,
)
from cosmos_predict2.config import MODEL_CHECKPOINTS, VIDEO_EXTENSIONS


def _get_robot_states(label, state_key="state", gripper_key="continuous_gripper_state"):
    """
    Extracts the robot arm and gripper states from the label dictionary for the specified frame indices.

    Args:
        label (dict): Dictionary containing robot state information, with keys "state" and "continuous_gripper_state".
        state_key (str): Key for robot state in the dictionary.
        gripper_key (str): Key for gripper state in the dictionary.

    Returns:
        tuple:
            - np.ndarray: Array of arm states for the selected frames, shape (len(frame_ids), state_dim).
            - np.ndarray: Array of gripper states for the selected frames, shape (len(frame_ids),).
    """
    all_states = np.array(label[state_key])
    all_cont_gripper_states = np.array(label[gripper_key])

    return all_states, all_cont_gripper_states


def _get_actions(arm_states, gripper_states, sequence_length, use_quat=False):
    """
    Compute the relative actions between consecutive robot states.

    Args:
        arm_states (np.ndarray): Array of arm states with shape (sequence_length, 6), where each state contains
            [x, y, z, roll, pitch, yaw] or similar.
        gripper_states (np.ndarray): Array of gripper states with shape (sequence_length,).
        sequence_length (int): Number of states in the sequence.
        use_quat (bool): If True, represent rotation as quaternion; otherwise, use Euler angles.

    Returns:
        np.ndarray: Array of actions with shape (sequence_length - 1, 7), where each action contains
            [relative_xyz (3), relative_rotation (3), gripper_state (1)].
    """
    if use_quat:
        action = np.zeros((sequence_length - 1, 8))
    else:
        action = np.zeros((sequence_length - 1, 7))

    for k in range(1, sequence_length):
        prev_xyz = arm_states[k - 1, 0:3]
        prev_rpy = arm_states[k - 1, 3:6]
        prev_rotm = euler2rotm(prev_rpy)
        curr_xyz = arm_states[k, 0:3]
        curr_rpy = arm_states[k, 3:6]
        curr_gripper = gripper_states[k]
        curr_rotm = euler2rotm(curr_rpy)
        rel_xyz = np.dot(prev_rotm.T, curr_xyz - prev_xyz)
        rel_rotm = prev_rotm.T @ curr_rotm

        if use_quat:
            rel_rot = rotm2quat(rel_rotm)
            action[k - 1, 0:3] = rel_xyz
            action[k - 1, 3:7] = rel_rot
            action[k - 1, 7] = curr_gripper
        else:
            rel_rot = rotm2euler(rel_rotm)
            action[k - 1, 0:3] = rel_xyz
            action[k - 1, 3:6] = rel_rot
            action[k - 1, 6] = curr_gripper
    return action  # (l - 1, act_dim)


def get_action_sequence_from_states(
    data,
    fps_downsample_ratio=1,
    use_quat=False,
    state_key="state",
    gripper_scale=1.0,
    gripper_key="continuous_gripper_state",
    action_scaler=20.0,
):
    """
    Get the action sequence from the states.
    """
    if state_key not in data or len(data[state_key]) == 0:
        if "action" in data and len(data["action"]) > 0:
            actions = np.array(data["action"])
        elif "actions" in data and len(data["actions"]) > 0:
            actions = np.array(data["actions"])
        else:
            raise ValueError(f"Missing '{state_key}' and action fields in input JSON.")
        actions = actions[::fps_downsample_ratio]
        if actions.shape[-1] == 7:
            actions = actions * np.array(
                [action_scaler, action_scaler, action_scaler, action_scaler, action_scaler, action_scaler, gripper_scale]
            )
        elif actions.shape[-1] == 8:
            scale = np.array([action_scaler, action_scaler, action_scaler, 1.0, 1.0, 1.0, 1.0, gripper_scale])
            actions = actions * scale
        return actions

    arm_states, cont_gripper_states = _get_robot_states(data, state_key, gripper_key)
    actions = _get_actions(
        arm_states[::fps_downsample_ratio],
        cont_gripper_states[::fps_downsample_ratio],
        len(data[state_key][::fps_downsample_ratio]),
        use_quat=use_quat,
    )
    actions *= np.array(
        [action_scaler, action_scaler, action_scaler, action_scaler, action_scaler, action_scaler, gripper_scale]
    )

    return actions


def load_default_action_fn():
    """
    Default action loading function that processes robot states into actions.
    """

    def _resolve_geometry_path(video_path: str) -> str:
        if video_path.endswith("rgb.mp4"):
            geometry_path = video_path.replace("rgb.mp4", "geometry.safetensors")
        elif video_path.endswith("geometry.safetensors"):
            geometry_path = video_path
        else:
            geometry_path = os.path.join(os.path.dirname(video_path), "geometry.safetensors")
        if not os.path.exists(geometry_path):
            if "/videos/train/" in geometry_path:
                alt_path = geometry_path.replace("/videos/train/", "/videos/test/")
                if os.path.exists(alt_path):
                    logger.warning(f"geometry path fallback: {geometry_path} -> {alt_path}")
                    geometry_path = alt_path
            elif "/videos/test/" in geometry_path:
                alt_path = geometry_path.replace("/videos/test/", "/videos/train/")
                if os.path.exists(alt_path):
                    logger.warning(f"geometry path fallback: {geometry_path} -> {alt_path}")
                    geometry_path = alt_path
        if not os.path.exists(geometry_path):
            raise FileNotFoundError(f"geometry.safetensors not found for {video_path}")
        return geometry_path

    def _load_geometry_min_max_from_train_config() -> tuple[float | None, float | None]:
        try:
            from cosmos_predict2._src.predict2.action.configs.action_conditioned import config as train_config

            return float(train_config.Config.geometry_min), float(train_config.Config.geometry_max)
        except Exception as exc:
            logger.warning(f"Failed to read geometry min/max from training config: {exc}")

        config_path = (
            Path(__file__).resolve().parent
            / "_src/predict2/action/configs/action_conditioned/config.py"
        )
        try:
            text = config_path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning(f"Failed to read geometry min/max from {config_path}: {exc}")
            return None, None

        float_pattern = r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
        min_match = re.search(rf"geometry_min\s*:\s*float\s*=\s*{float_pattern}", text)
        max_match = re.search(rf"geometry_max\s*:\s*float\s*=\s*{float_pattern}", text)
        if min_match and max_match:
            return float(min_match.group(1)), float(max_match.group(1))
        logger.warning(f"geometry_min/max not found in {config_path}")
        return None, None

    def _resolve_geometry_min_max(args: ActionConditionedInferenceArguments) -> tuple[float | None, float | None]:
        if not args.geometry_normalize:
            return None, None
        cfg_min, cfg_max = _load_geometry_min_max_from_train_config()
        geometry_min = args.geometry_min if args.geometry_min is not None else cfg_min
        geometry_max = args.geometry_max if args.geometry_max is not None else cfg_max
        if geometry_min is None or geometry_max is None:
            raise ValueError("geometry_min/max must be set or available in training config.")
        if geometry_max <= geometry_min:
            raise ValueError(f"geometry_max must be > geometry_min, got {geometry_min}..{geometry_max}")
        return geometry_min, geometry_max

    def _load_geometry_first_frame(
        video_path: str,
        frame_idx: int,
        geometry_min: float | None,
        geometry_max: float | None,
        geometry_normalize: bool,
    ) -> np.ndarray:
        geometry_path = _resolve_geometry_path(video_path)
        with safe_open(geometry_path, framework="pt", device="cpu") as f:
            all_frames = f.get_tensor("xyz")  # Shape: [T, H, W, 3]
        if frame_idx < 0 or frame_idx >= all_frames.shape[0]:
            raise IndexError(f"frame_idx {frame_idx} out of range for {geometry_path} with {all_frames.shape[0]} frames")
        frame = all_frames[frame_idx].cpu().numpy().astype(np.float32)
        if geometry_normalize:
            if geometry_min is None or geometry_max is None:
                raise ValueError("geometry_min/max must be set when geometry_normalize is enabled.")
            frame = np.clip(frame, geometry_min, geometry_max)
            denom = geometry_max - geometry_min
            frame = (frame - geometry_min) / denom
        else:
            frame = np.clip(frame, 0.0, 1.0)
        frame = np.clip(frame, 0.0, 1.0)
        frame = (frame * 255.0).round().astype(np.uint8)
        return frame

    def load_fn(
        json_data: dict,
        video_path: str,
        args: ActionConditionedInferenceArguments,
    ) -> dict:
        """
        Load action data from JSON and prepare it for inference.

        Args:
            json_data: JSON data containing robot states
            video_path: Path to the video file
            args: Inference arguments

        Returns:
            Dictionary containing actions, video data, and metadata
        """
        # Get action sequence from states
        actions = get_action_sequence_from_states(
            json_data,
            fps_downsample_ratio=args.fps_downsample_ratio,
            state_key=args.state_key,
            gripper_scale=args.gripper_scale,
            gripper_key=args.gripper_key,
            action_scaler=args.action_scaler,
            use_quat=args.use_quat,
        )

        geometry_min = None
        geometry_max = None
        if args.visual_condition_source == "geometry":
            geometry_min, geometry_max = _resolve_geometry_min_max(args)
            video_array = None
            img_array = _load_geometry_first_frame(
                video_path,
                args.start_frame_idx,
                geometry_min,
                geometry_max,
                args.geometry_normalize,
            )
        elif args.visual_condition_source == "rgb":
            # Load video
            video_array = mediapy.read_video(video_path)
            img_array = video_array[args.start_frame_idx]
        else:
            raise ValueError(
                f"visual_condition_source must be 'rgb' or 'geometry', got {args.visual_condition_source}"
            )

        # Resize if specified
        if args.resolution != "none":
            try:
                h, w = map(int, args.resolution.split(","))
                img_array = mediapy.resize_image(img_array, (h, w))
            except Exception as e:
                logger.warning(f"Failed to resize image to {args.resolution}: {e}")

        return {
            "actions": actions,
            "initial_frame": img_array,
            "video_array": video_array,
            "video_path": video_path,
            "geometry_min": geometry_min if args.visual_condition_source == "geometry" else None,
            "geometry_max": geometry_max if args.visual_condition_source == "geometry" else None,
        }

    return load_fn


def _save_xyz_outputs(
    xyz_maps: np.ndarray,
    save_root: str,
    base_name: str,
    fps: int,
    geometry_min: float | None = None,
    geometry_max: float | None = None,
) -> None:
    xyz_mp4_path = str(Path(save_root, f"{base_name}_xyz.mp4").resolve())
    xyz_npy_path = str(Path(save_root, f"{base_name}_xyz.npy").resolve())
    np.save(xyz_npy_path, xyz_maps)
    if geometry_min is not None and geometry_max is not None:
        denom = geometry_max - geometry_min
        xyz_video = (xyz_maps - geometry_min) / denom
    else:
        xyz_video = xyz_maps
    xyz_video = np.clip(xyz_video, 0.0, 1.0)
    xyz_video = (xyz_video * 255.0).round().astype(np.uint8)
    mediapy.write_video(xyz_mp4_path, xyz_video, fps=fps)
    logger.info(f"Saved video to {xyz_mp4_path}")
    logger.info(f"Saved npy to {xyz_npy_path}")


def load_callable(name: str):
    """Load a callable function from a module path string."""
    from importlib import import_module

    idx = name.rfind(".")
    assert idx > 0, "expected <module_name>.<identifier>"
    module_name = name[0:idx]
    fn_name = name[idx + 1 :]

    module = import_module(module_name)
    fn = getattr(module, fn_name)
    return fn


def get_video_id(img_path: str):
    """Extract video ID from image path by removing directory and extension."""
    return img_path.split("/")[-1].split(".")[0]


def inference(
    setup_args: ActionConditionedSetupArguments,
    inference_args: ActionConditionedInferenceArguments,
):
    """Run action-conditioned video generation inference using resolved setup and per-run arguments."""
    torch.enable_grad(False)  # Disable gradient calculations for inference

    # Validate num_latent_conditional_frames at the very beginning
    if inference_args.num_latent_conditional_frames not in [0, 1, 2]:
        raise ValueError(
            f"num_latent_conditional_frames must be 0, 1 or 2, but got {inference_args.num_latent_conditional_frames}"
        )

    if inference_args.visual_condition_source == "geometry" and inference_args.num_latent_conditional_frames != 1:
        raise ValueError("geometry visual condition currently supports num_latent_conditional_frames=1 only")

    # Determine supported extensions based on num_latent_conditional_frames
    if inference_args.num_latent_conditional_frames > 1:
        # Check if input folder contains any videos
        has_videos = False
        for file_name in os.listdir(inference_args.input_root):
            file_ext = os.path.splitext(file_name)[1].lower()
            if file_ext in VIDEO_EXTENSIONS:
                has_videos = True
                break

        if not has_videos:
            raise ValueError(
                f"num_latent_conditional_frames={inference_args.num_latent_conditional_frames} > 1 requires video inputs, "
                f"but no videos found in {inference_args.input_root}. Found extensions: "
                f"{set(os.path.splitext(f)[1].lower() for f in os.listdir(inference_args.input_root) if os.path.splitext(f)[1])}"
            )

        logger.info(f"Using video-only mode with {inference_args.num_latent_conditional_frames} conditional frames")
    elif inference_args.num_latent_conditional_frames == 1:
        logger.info(f"Using image+video mode with {inference_args.num_latent_conditional_frames} conditional frame")

    # Get checkpoint and experiment from setup args
    checkpoint = MODEL_CHECKPOINTS[setup_args.model_key]
    experiment = setup_args.experiment or checkpoint.experiment
    # pyrefly: ignore  # missing-attribute
    checkpoint_path = setup_args.checkpoint_path or checkpoint.s3.uri

    # Ensure experiment is not None
    if experiment is None:
        raise ValueError("Experiment name must be provided either in setup args or checkpoint metadata")

    # Initialize the inference handler with context parallel support
    video2world_cli = Video2WorldInference(
        experiment_name=experiment,
        ckpt_path=checkpoint_path,
        s3_credential_path="",
        # pyrefly: ignore  # bad-argument-type
        context_parallel_size=setup_args.context_parallel_size,
        config_file=setup_args.config_file,
    )

    mem_bytes = torch.cuda.memory_allocated(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"GPU memory usage after model load: {mem_bytes / (1024**3):.2f} GB")

    # Load action loading function
    action_load_fn = load_callable(inference_args.action_load_fn)

    # Get input video and annotation paths
    input_video_path = inference_args.input_root
    input_json_path = inference_args.input_root / inference_args.input_json_sub_folder
    input_json_list = glob(str(input_json_path / "*.json"))
    log.info(
        f"Action-conditioned inference input_root={input_video_path}, "
        f"input_json_path={input_json_path}, json_count={len(input_json_list)}"
    )
    if not input_json_list:
        log.warning("No annotation JSON files found; exiting without generation.")
        return

    # Only process files on rank 0 if using distributed processing
    rank0 = True
    # pyrefly: ignore  # unsupported-operation
    if setup_args.context_parallel_size > 1:
        rank0 = distributed.get_rank() == 0

    # Ensure save directory exists
    inference_args.save_root.mkdir(parents=True, exist_ok=True)

    # Process each file in the input directory
    def _resolve_existing_video_path(video_path: str) -> str:
        if os.path.exists(video_path):
            return video_path
        if "/videos/train/" in video_path:
            alt_path = video_path.replace("/videos/train/", "/videos/test/")
            if os.path.exists(alt_path):
                logger.warning(f"video path fallback: {video_path} -> {alt_path}")
                return alt_path
        elif "/videos/test/" in video_path:
            alt_path = video_path.replace("/videos/test/", "/videos/train/")
            if os.path.exists(alt_path):
                logger.warning(f"video path fallback: {video_path} -> {alt_path}")
                return alt_path
        return video_path

    for annotation_path in input_json_list[inference_args.start : inference_args.end]:
        with open(annotation_path, "r") as f:
            json_data = json.load(f)

        # Convert camera_id to integer if it's a string and can be converted to an integer
        camera_id = (
            int(inference_args.camera_id)
            if isinstance(inference_args.camera_id, str) and inference_args.camera_id.isdigit()
            else inference_args.camera_id
        )

        if isinstance(json_data["videos"][camera_id], dict):
            video_path = str(input_video_path / json_data["videos"][camera_id]["video_path"])
        else:
            video_path = str(input_video_path / json_data["videos"][camera_id])
        video_path = _resolve_existing_video_path(video_path)

        # Load action data using the configured function
        action_data = action_load_fn()(json_data, video_path, inference_args)
        actions = action_data["actions"]
        img_array = action_data["initial_frame"]
        geometry_min = action_data.get("geometry_min")
        geometry_max = action_data.get("geometry_max")

        img_name = annotation_path.split("/")[-1].split(".")[0]

        frames = [img_array]
        chunk_video = []
        chunk_xyz = []

        video_name = str((inference_args.save_root / f"{img_name.replace('.jpg', '.mp4')}").resolve())
        chunk_video_name = str((inference_args.save_root / f"{img_name}_chunk.mp4").resolve())
        logger.info(f"Configured output root: {inference_args.save_root.resolve()}")

        chunk_indices = list(range(inference_args.start_frame_idx, len(actions), inference_args.chunk_size))
        if inference_args.single_chunk:
            chunk_indices = chunk_indices[:1]

        with tqdm(
            total=len(chunk_indices),
            desc=f"{img_name} chunks",
            unit="chunk",
            disable=not rank0,
            dynamic_ncols=True,
        ) as chunk_pbar:
            for i in chunk_indices:
                # Handle incomplete chunks
                actions_chunk = actions[i : i + inference_args.chunk_size]
                if actions_chunk.shape[0] != inference_args.chunk_size:
                    pad_len = inference_args.chunk_size - actions_chunk.shape[0]
                    if pad_len > 0:
                        action_shape = list(actions.shape[1:])
                        pad_shape = [pad_len] + action_shape
                        pad_actions = np.zeros(pad_shape, dtype=actions.dtype)
                        actions_chunk = np.concatenate([actions_chunk, pad_actions], axis=0)

                # Convert img_array to tensor and prepare video input
                # pyrefly: ignore  # implicit-import
                img_tensor = torchvision.transforms.functional.to_tensor(img_array).unsqueeze(0)
                num_video_frames = actions_chunk.shape[0] + 1
                vid_input = torch.cat(
                    [img_tensor, torch.zeros_like(img_tensor).repeat(num_video_frames - 1, 1, 1, 1)], dim=0
                )
                vid_input = (vid_input * 255.0).to(torch.uint8)
                vid_input = vid_input.unsqueeze(0).permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)

                # Call generate_vid2world
                video = video2world_cli.generate_vid2world(
                    prompt=inference_args.prompt or "",
                    input_path=vid_input,
                    action=torch.from_numpy(actions_chunk).float()
                    if isinstance(actions_chunk, np.ndarray)
                    else actions_chunk,
                    guidance=inference_args.guidance,
                    num_video_frames=num_video_frames,
                    num_latent_conditional_frames=inference_args.num_latent_conditional_frames,
                    resolution=inference_args.resolution,
                    seed=inference_args.seed + i,
                    negative_prompt=inference_args.negative_prompt,
                )
                # Extract next frame and video from result
                video_normalized = (video - (-1)) / (1 - (-1))
                video_normalized = torch.clamp(video_normalized, 0, 1)
                xyz_chunk = video_normalized[0].permute(1, 2, 3, 0).detach().cpu().numpy().astype(np.float32)
                if inference_args.visual_condition_source == "geometry":
                    if geometry_min is None or geometry_max is None:
                        if inference_args.geometry_normalize:
                            raise ValueError("geometry_min/max missing for geometry visual condition.")
                    else:
                        denom = geometry_max - geometry_min
                        xyz_chunk = xyz_chunk * denom + geometry_min
                        xyz_chunk = np.clip(xyz_chunk, geometry_min, geometry_max)
                video_clamped = (
                    (video_normalized[0] * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
                )
                next_img_array = video_clamped[-1]  # Last frame is the next frame
                frames.append(next_img_array)
                img_array = next_img_array
                chunk_video.append(video_clamped)
                chunk_xyz.append(xyz_chunk)
                chunk_pbar.update(1)

        chunk_list = [chunk_video[0]] + [
            chunk_video[i][: inference_args.chunk_size] for i in range(1, len(chunk_video))
        ]
        chunk_video = np.concatenate(chunk_list, axis=0)
        chunk_xyz_list = [chunk_xyz[0]] + [chunk_xyz[i][: inference_args.chunk_size] for i in range(1, len(chunk_xyz))]
        xyz_full = np.concatenate(chunk_xyz_list, axis=0)
        if inference_args.single_chunk:
            chunk_video_name = str((inference_args.save_root / f"{img_name}_single_chunk.mp4").resolve())
        else:
            chunk_video_name = str((inference_args.save_root / f"{img_name}_chunk.mp4").resolve())

        if rank0:
            mediapy.write_video(chunk_video_name, chunk_video, fps=inference_args.save_fps)
            logger.info(f"Saved video to {chunk_video_name}")
            if inference_args.visual_condition_source == "geometry":
                _save_xyz_outputs(
                    xyz_full,
                    str(inference_args.save_root),
                    img_name,
                    inference_args.save_fps,
                    geometry_min=geometry_min,
                    geometry_max=geometry_max,
                )

    # Synchronize all processes before cleanup
    # pyrefly: ignore  # unsupported-operation
    if setup_args.context_parallel_size > 1:
        torch.distributed.barrier()

    # Clean up distributed resources
    video2world_cli.cleanup()
