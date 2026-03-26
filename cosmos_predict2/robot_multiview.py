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

import os
import re
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms.v2
from einops import rearrange
from loguru import logger
from PIL import Image
from torchvision import transforms


def reshape_features_to_spatial(features, feature_shape):
    """
    Reshape features from [B, N, D] to [B, D, T, H, W] spatial format.

    Args:
        features: List of feature tensors, each with shape [B, N, D]
        feature_shape: Tuple (T, H, W) - the spatial dimensions of the features

    Returns:
        List of reshaped feature tensors with shape [B, D, T, H, W]
    """
    T, H, W = feature_shape
    reshaped_features = []

    for feat in features:
        # feat shape: [B, N, D] where N = T * H * W
        B, N, D = feat.shape
        # Reshape to [B, T, H, W, D] then permute to [B, D, T, H, W]
        feat_spatial = rearrange(feat, 'b (t h w) d -> b d t h w', t=T, h=H, w=W)
        reshaped_features.append(feat_spatial)

    return reshaped_features

from cosmos_predict2._src.imaginaire.modules.camera import Camera
from cosmos_predict2._src.imaginaire.utils import distributed
from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
from cosmos_predict2.config import MODEL_CHECKPOINTS, load_callable
from cosmos_predict2.robot_multiview_config import (
    CameraLoadFn,
    RobotMultiviewInferenceArguments,
    RobotMultiviewSetupArguments,
)


def load_agibot_camera_fn():
    cam_data_list = ["extrinsic_head", "extrinsic_hand_0", "extrinsic_hand_1"]
    intrinsic_data_list = ["intrinsic_head", "intrinsic_hand_0", "intrinsic_hand_1"]

    def load_fn(
        text: str,
        visual: torch.Tensor,
        path: str,
        base_path: str,
        latent_frames: int,
        width: int,
        height: int,
        input_video_res: str,
        patch_spatial: int,
    ):
        result = []

        # Extract task_id from path
        # New format: frames/648544_0 -> task_id = 648544_0
        # Old format: input_images/0 -> task_id = 0
        if "/frames/" in path:
            # New format: extract task_id from frames/{task_id}
            task_id = os.path.basename(path)
        else:
            # Old format: extract input_idx from input_images/{input_idx}
            # pyrefly: ignore  # missing-attribute
            task_id = re.search(r"input_images/(\d+)", path).group(1)

        data = {"text": text, "video": visual, "path": path}
        extrinsics_list = []
        for cam_type in cam_data_list:
            extrinsics_tgt = torch.tensor(
                np.loadtxt(os.path.join(base_path, "cameras", f"{task_id}_{cam_type}.txt"))
            ).to(torch.bfloat16)
            extrinsics_tgt = extrinsics_tgt[:latent_frames]
            extrinsics_tgt = torch.cat(
                (
                    extrinsics_tgt,
                    torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.bfloat16).unsqueeze(0).expand(latent_frames, -1),
                ),
                dim=1,
            ).reshape(-1, 4, 4)
            extrinsics_list.append(extrinsics_tgt)
        extrinsics = torch.cat(extrinsics_list, dim=0)

        intrinsics_list = []
        for intrinsic_type in intrinsic_data_list:
            intrinsics_tgt = torch.tensor(
                np.loadtxt(os.path.join(base_path, "cameras", f"{task_id}_{intrinsic_type}.txt"))
            ).to(torch.bfloat16)
            intrinsics_tgt = intrinsics_tgt[:latent_frames]
            intrinsics_list.append(intrinsics_tgt)
        intrinsics = torch.cat(intrinsics_list, dim=0)

        if input_video_res == "720p":
            scale_w = 1280 / 768
            scale_h = 704 / 432
            intrinsics[:, [0, 2]] *= scale_w
            intrinsics[:, [1, 3]] *= scale_h

        K = Camera.intrinsic_params_to_matrices(intrinsics)
        w2c = Camera.invert_pose(extrinsics[:, :3, :])

        plucker_flat = Camera.get_plucker_rays(w2c, K, (height, width))
        # pyrefly: ignore  # missing-attribute
        plucker_rays = plucker_flat.view(plucker_flat.shape[0], height, width, 6)
        plucker_rays = rearrange(
            plucker_rays,
            "T (H p1) (W p2) C -> T H W (p1 p2 C)",
            p1=patch_spatial,
            p2=patch_spatial,
        )
        data["camera"] = plucker_rays
        # Also include raw extrinsics and intrinsics for the model
        # No need to unsqueeze here - the collate function will add the batch dimension
        data["extrinsics"] = extrinsics  # Shape: (num_views * latent_frames, 4, 4)
        data["intrinsics"] = intrinsics  # Shape: (num_views * latent_frames, 4)
        # Also include image_size for the conditioner (needed to compute Plücker rays)
        data["image_size"] = torch.tensor([height, width], dtype=torch.float32)  # Shape: (2,)
        result.append(data)
        return result

    return load_fn


class TextImageCameraDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path: str,
        args: RobotMultiviewSetupArguments,
        inference_args: list[RobotMultiviewInferenceArguments],
        num_frames: int,
        max_num_frames: int = 93,
        frame_interval: int = 1,
        patch_spatial: int = 16,
        camera_load_fn: CameraLoadFn | None = None,
    ):
        assert camera_load_fn is not None, "not provided function to load camera metadata"
        self.camera_load_fn = camera_load_fn
        self.base_path = base_path
        self.num_output_video = args.num_output_video
        self.data = inference_args

        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.latent_frames = num_frames // 4 + 1
        self.patch_spatial = patch_spatial
        self.input_video_res = args.input_video_res
        if self.input_video_res == "720p":
            self.height, self.width = 704, 1280
        elif self.input_video_res == "480p":
            self.height, self.width = 432, 768
        self.args = args

        # pyrefly: ignore  # implicit-import
        self.frame_process = transforms.v2.Compose(
            [
                # pyrefly: ignore  # implicit-import
                transforms.v2.CenterCrop(size=(self.height, self.width)),
                # pyrefly: ignore  # implicit-import
                transforms.v2.Resize(size=(self.height, self.width), antialias=True),
                # pyrefly: ignore  # implicit-import
                transforms.v2.ToTensor(),
                # pyrefly: ignore  # implicit-import
                transforms.v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        # pyrefly: ignore  # implicit-import
        image = torchvision.transforms.functional.resize(
            image,
            # pyrefly: ignore  # bad-argument-type
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        return image

    def load_images(self, input_name: str, json_file_name: str = None) -> torch.Tensor:
        images_list = []
        images_name = ["head", "hand_0", "hand_1"]

        # Determine the task_id from json_file_name if provided
        # If json_file_name is "648544_0.json", task_id is "648544_0"
        if json_file_name:
            task_id = os.path.splitext(json_file_name)[0]
            frames_dir = os.path.join(self.base_path, "frames", task_id)
        else:
            # Fallback to old format
            frames_dir = os.path.join(self.base_path, "input_images")
            task_id = input_name

        for image_name in images_name:
            if json_file_name:
                # New format: frames/{task_id}/{image_name}_frame_0000.png
                image_path = os.path.join(frames_dir, f"{image_name}_frame_0000.png")
            else:
                # Old format: input_images/{input_name}_{image_name}.png
                image_path = os.path.join(frames_dir, f"{task_id}_{image_name}.png")

            image = Image.open(image_path)  # 加载单张PNG图片
            image = self.crop_and_resize(image)
            image = self.frame_process(image)
            image = image.unsqueeze(0).expand(self.num_frames, -1, -1, -1)  # 关键：将单帧扩展到93帧（复制93次）
            images_list.append(image)
        images = torch.cat(images_list, dim=0)
        images = rearrange(images, "T C H W -> C T H W")
        return images

    # pyrefly: ignore [bad-param-name-override]
    def __getitem__(self, data_id: int):
        inference_args = self.data[data_id]
        input_name = str(inference_args.input_name)
        text = inference_args.prompt

        # Get json_file_name if available
        json_file_name = getattr(inference_args, 'json_file_name', None)

        images = self.load_images(input_name, json_file_name)

        assert text is not None
        # Determine the path for camera loading
        if json_file_name:
            task_id = os.path.splitext(json_file_name)[0]
            camera_path = os.path.join(self.base_path, "frames", task_id)
        else:
            camera_path = os.path.join(self.base_path, "input_images", input_name)

        result = self.camera_load_fn(
            text=text,
            # pyrefly: ignore  # bad-argument-type
            visual=images,
            path=camera_path,
            base_path=self.base_path,
            latent_frames=self.latent_frames,
            # pyrefly: ignore  # unexpected-keyword
            width=self.width,
            # pyrefly: ignore  # unexpected-keyword
            height=self.height,
            # pyrefly: ignore  # unexpected-keyword
            input_video_res=self.input_video_res,
            # pyrefly: ignore  # unexpected-keyword
            patch_spatial=self.patch_spatial,
        )
        # camera_load_fn returns a list of dicts, but we only expect one dict per sample
        # Extract the first (and only) dict and update it
        data = result[0]
        data.update(
            {
                "seed": inference_args.seed,
                "guidance": inference_args.guidance,
                "negative_prompt": inference_args.negative_prompt,
                "input_name": input_name,
            }
        )

        # Add json_file_name for output naming (prefer json_file_name over input_name)
        if json_file_name:
            data["json_file_name"] = json_file_name

        return data

    def __len__(self):
        return len(self.data)


def inference(
    setup_args: RobotMultiviewSetupArguments,
    all_inference_args: list[RobotMultiviewInferenceArguments],
):
    """Run robot multiview inference using resolved setup and per-run arguments."""
    assert len(all_inference_args) > 0

    create_camera_load_fn = load_callable(setup_args.camera_load_create_fn)
    dataset = TextImageCameraDataset(
        # pyrefly: ignore  # bad-argument-type
        base_path=setup_args.base_path,
        args=setup_args,
        inference_args=all_inference_args,
        num_frames=setup_args.num_output_frames,
        camera_load_fn=create_camera_load_fn(),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=setup_args.dataloader_num_workers,
    )

    checkpoint = MODEL_CHECKPOINTS[setup_args.model_key]
    experiment = setup_args.experiment or checkpoint.experiment
    # pyrefly: ignore  # missing-attribute
    checkpoint_path_list = setup_args.checkpoint_path or [checkpoint.s3.uri]
    # Handle the case where checkpoint_path might be a list
    checkpoint_path = checkpoint_path_list[0] if isinstance(checkpoint_path_list, list) else checkpoint_path_list
    # Ensure experiment is a string (handle potential list type)
    if isinstance(experiment, list):
        experiment = experiment[0] if experiment else "unknown_experiment"

    # 【修改】 获取 num_steps
    num_steps = getattr(setup_args, "num_steps", 35)

    # 【新增】 解析特征提取参数
    extract_layer_ids = None
    extract_at_steps = None
    if hasattr(setup_args, "extract_layer_ids") and setup_args.extract_layer_ids:
        extract_layer_ids = [int(x.strip()) for x in setup_args.extract_layer_ids.split(',')]
        logger.info(f"Feature extraction enabled for layers: {extract_layer_ids}")
    if hasattr(setup_args, "extract_at_steps") and setup_args.extract_at_steps:
        step_str = setup_args.extract_at_steps.strip().lower()
        if step_str in ('none', 'all'):
            extract_at_steps = None  # All steps
        elif step_str == 'first':
            extract_at_steps = 'first'
        elif step_str == 'last':
            extract_at_steps = 'last'
        else:
            # Parse as comma-separated indices
            extract_at_steps = [int(x.strip()) for x in step_str.split(',')]
        logger.info(f"Feature extraction at steps: {extract_at_steps}")

    vid2vid_cli = Video2WorldInference(
        # pyrefly: ignore  # bad-argument-type
        experiment_name=experiment,
        ckpt_path=checkpoint_path,
        s3_credential_path="",
        # pyrefly: ignore  # bad-argument-type
        context_parallel_size=setup_args.context_parallel_size,
        config_file=setup_args.config_file,
    )

    mem_bytes = torch.cuda.memory_allocated(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"GPU memory usage after model load: {mem_bytes / (1024**3):.2f} GB")

    # Only process files on rank 0 if using distributed processing
    rank0 = True
    # pyrefly: ignore  # unsupported-operation
    if setup_args.context_parallel_size > 1:
        rank0 = distributed.get_rank() == 0

    # Process each file in the input directory
    for batch_idx, batch in enumerate(dataloader):
        # batch is a dict (collated from default_collate which stacks dict values)
        # Get the actual batch size from the first tensor value
        first_tensor = next(v for v in batch.values() if isinstance(v, torch.Tensor))
        batch_size = first_tensor.size(0)

        for video_idx in range(batch_size):
            # Index the dict at position video_idx to get a dict of indexed tensors
            # 【修改3】 修复 list 类型的解包问题（例如 input_name）
            ex = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    ex[k] = v[video_idx]
                elif isinstance(v, list):
                    # PyTorch DataLoader 会把字符串 batch 成 list，需要解包
                    ex[k] = v[video_idx]
                else:
                    # 其他情况（通常是数字或单个对象，如果被batch了）
                    # 实际上 DataLoader 通常只返回 Tensor 或 list/tuple
                    # 如果 v 既不是 Tensor 也不是 list，直接赋值（虽然不太可能在 default_collate 下发生）
                    ex[k] = v

            tgt_text = ex["text"]
            input_name = ex["input_name"]
            # Use json_file_name for output if available, otherwise use input_name
            json_file_name = ex.get("json_file_name", None)
            if json_file_name:
                # Remove .json extension to get the task ID
                output_task_name = os.path.splitext(json_file_name)[0]
            else:
                output_task_name = input_name
            src_video = ex["video"]
            tgt_camera = ex["camera"]

            # Add batch dimension to video: (C, T, H, W) -> (1, C, T, H, W)
            src_video = src_video.unsqueeze(0)

            # Access extrinsics and intrinsics from the indexed dict
            # After indexing, shape is (num_views * latent_frames, 4, 4) for extrinsics
            # We need to add batch dimension to get (1, num_views * latent_frames, 4, 4)
            tgt_extrinsics = ex["extrinsics"].unsqueeze(0) if "extrinsics" in ex and ex["extrinsics"] is not None else None
            tgt_intrinsics = ex["intrinsics"].unsqueeze(0) if "intrinsics" in ex and ex["intrinsics"] is not None else None
            # image_size needs batch dimension too: (2,) -> (1, 2)
            tgt_image_size = ex["image_size"].unsqueeze(0) if "image_size" in ex and ex["image_size"] is not None else None

            # Debug: Print shapes to understand the data flow
            logger.info(f"DEBUG: src_video shape: {src_video.shape}")
            if tgt_extrinsics is not None:
                logger.info(f"DEBUG: tgt_extrinsics shape: {tgt_extrinsics.shape}, dim: {tgt_extrinsics.dim()}")
            if tgt_intrinsics is not None:
                logger.info(f"DEBUG: tgt_intrinsics shape: {tgt_intrinsics.shape}, dim: {tgt_intrinsics.dim()}")
            if tgt_image_size is not None:
                logger.info(f"DEBUG: tgt_image_size shape: {tgt_image_size.shape}, dim: {tgt_image_size.dim()}")

            video = vid2vid_cli.generate_vid2world(
                prompt=tgt_text,
                input_path=src_video,
                camera=tgt_camera,
                num_input_video=setup_args.num_input_video,
                num_output_video=setup_args.num_output_video,
                num_latent_conditional_frames=setup_args.num_input_frames,
                num_video_frames=setup_args.num_output_frames,
                seed=ex["seed"].item(),
                guidance=ex["guidance"].item(),
                negative_prompt=ex["negative_prompt"],
                num_steps=num_steps,
                extrinsics=tgt_extrinsics,
                intrinsics=tgt_intrinsics,
                image_size=tgt_image_size,
                extract_layer_ids=extract_layer_ids,
                extract_at_steps=extract_at_steps,
            )

            # 【新增】 处理返回值（如果启用了特征提取，返回值是 (video, features, step_info)）
            features = None
            step_info = None
            if extract_layer_ids is not None:
                video, features, step_info = video
                logger.info(f"Extracted features from {len(features)} timesteps")

            if rank0:
                # Use json_file_name (without extension) as output name, fallback to video_batch_idx format
                output_name = f"{output_task_name}_video"
                # Save all outputs directly under output_dir/task_id/ subdirectory
                save_root = Path(setup_args.output_dir) / output_task_name
                save_root.mkdir(parents=True, exist_ok=True)

                output_path = str(save_root / output_name)
                save_img_or_video((1.0 + video[0]) / 2, output_path, fps=30)

                # Get video shape information
                # video[0] shape: (C, T, H, W) where H is the stitched height (3 views stacked vertically)
                C, T, H, W = video[0].shape
                num_views = setup_args.num_output_video  # Number of views (e.g., 3)
                view_height = H // num_views  # Height of each individual view

                # Log frame count for the stitched video
                logger.info(f"Saved stitched video to {output_path}, frames: {T}")

                # Save each individual view video (unstitched)
                # The vertical layout from top to bottom is: head, right_hand, left_hand
                view_mapping = {
                    0: "head",       # First 1/3 of height (top)
                    1: "right_hand", # Middle 1/3 of height
                    2: "left_hand",  # Last 1/3 of height (bottom)
                }
                for view_idx in range(num_views):
                    # Extract the view by slicing the height dimension
                    start_h = view_idx * view_height
                    end_h = (view_idx + 1) * view_height
                    view_video = video[0][:, :, start_h:end_h, :]  # Shape: (C, T, view_height, W)

                    # Save the individual view video
                    view_name = view_mapping[view_idx]
                    view_output_name = f"{output_name}_view_{view_name}"
                    view_output_path = str(save_root / view_output_name)
                    save_img_or_video((1.0 + view_video) / 2, view_output_path, fps=30)
                    logger.info(f"Saved {view_name} view video to {view_output_path}, frames: {T}")

                # 【新增】 保存特征到文件（可选）
                if features is not None:
                    # Reshape features to spatial format [B, D, T, H, W]
                    features_spatial = []
                    for i, (feat_list, info) in enumerate(zip(features, step_info)):
                        if 'feature_shape' in info:
                            feat_spatial = reshape_features_to_spatial(feat_list, info['feature_shape'])
                            features_spatial.append(feat_spatial)
                            logger.info(f"  Reshaped timestep {i} features to spatial format: {feat_spatial[0].shape}")
                        else:
                            logger.warning(f"  No feature_shape info for timestep {i}, skipping reshape")

                    # Save both original and spatial features
                    feature_save_path = str(save_root / f"{output_name}_features.pt")
                    torch.save({
                        'features': features,  # Original: List[List[Tensor]] with shape [B, N, D]
                        'features_spatial': features_spatial,  # Reshaped: List[List[Tensor]] with shape [B, D, T, H, W]
                        'step_info': step_info,
                    }, feature_save_path)
                    logger.info(f"Saved features to {feature_save_path}")

                    # Also save each timestep's features separately for easier access
                    for i, (feat_list, feat_spatial_list, info) in enumerate(zip(features, features_spatial, step_info)):
                        timestep_save_path = str(save_root / f"{output_name}_timestep_{info['step_idx']}.pt")
                        torch.save({
                            'features': feat_list,
                            'features_spatial': feat_spatial_list,
                            'step_info': info,
                        }, timestep_save_path)
                        logger.info(f"  Saved timestep {info['step_idx']} features to {timestep_save_path}")

    # Synchronize all processes before cleanup
    # pyrefly: ignore  # unsupported-operation
    if setup_args.context_parallel_size > 1:
        torch.distributed.barrier()

    # Clean up distributed resources
    vid2vid_cli.cleanup()