#!/usr/bin/env python
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

'''
batch 推理：
  python examples/action_conditioned_batch4.py \
    -i assets/action_conditioned/geometry/bridge/inference_params.json \
    -o outputs/action_conditioned/geometry_batch \
    --experiment ac_reason_embeddings_rectified_flow_2b_256_320 \
    --batch-size 1
'''


"""Action-conditioned Video2World batch inference (batch size = 4)."""

import json
import time
from pathlib import Path
from typing import Annotated

import mediapy
import numpy as np
import pydantic
import torch
import torchvision
import tyro
from loguru import logger

from cosmos_oss.init import cleanup_environment, init_environment, init_output_dir
from cosmos_predict2.action_conditioned import _save_xyz_outputs, load_callable
from cosmos_predict2.action_conditioned_config import (
    ActionConditionedInferenceArguments,
    ActionConditionedInferenceOverrides,
    ActionConditionedSetupArguments,
)
from cosmos_predict2.config import handle_tyro_exception, is_rank0
from cosmos_predict2._src.imaginaire.utils import distributed
from cosmos_predict2._src.predict2.inference.get_t5_emb import get_text_embedding
from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference


class Args(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    input_file: Annotated[Path, tyro.conf.arg(aliases=("-i",))]
    """Path to the inference parameter file"""
    setup: ActionConditionedSetupArguments
    """Setup arguments. These can only be provided via CLI."""
    overrides: ActionConditionedInferenceOverrides
    """Inference parameter overrides. These can either be provided in the input json file or via CLI. CLI overrides will overwrite the values in the input file."""
    batch_size: int = 4
    """Batch size for duplicated inputs."""


def _build_batch(
    model,
    video: torch.Tensor,
    prompt: str,
    negative_prompt: str | None,
    action: np.ndarray,
    num_conditional_frames: int,
    batch_size: int,
    t5_cache: dict[str, torch.Tensor] | None = None,
) -> dict:
    _, _, _, h, w = video.shape
    data_batch = {
        "dataset_name": "video_data",
        "video": video,
        "action": torch.from_numpy(action).to(dtype=torch.bfloat16),
        "fps": torch.randint(16, 32, (batch_size,)).float(),
        "padding_mask": torch.zeros(batch_size, 1, h, w),
        "num_conditional_frames": num_conditional_frames,
    }

    if t5_cache is None:
        t5_cache = {}

    def _get_cached_embed(text: str, is_negative: bool = False) -> torch.Tensor:
        key = f"neg::{text}" if is_negative else text
        if key in t5_cache:
            return t5_cache[key]
        if model.text_encoder is not None:
            emb = model.text_encoder.compute_text_embeddings_online(
                data_batch={"ai_caption": [text], "images": None},
                input_caption_key="ai_caption",
            )
        else:
            emb = get_text_embedding(text)
        t5_cache[key] = emb
        return emb

    t5 = _get_cached_embed(prompt, is_negative=False)
    if t5.shape[0] == 1:
        t5 = t5.repeat(batch_size, 1, 1)
    data_batch["t5_text_embeddings"] = t5
    if negative_prompt is not None:
        neg_t5 = _get_cached_embed(negative_prompt, is_negative=True)
        if neg_t5.shape[0] == 1:
            neg_t5 = neg_t5.repeat(batch_size, 1, 1)
        data_batch["neg_t5_text_embeddings"] = neg_t5

    for k, v in data_batch.items():
        if isinstance(v, torch.Tensor):
            if torch.is_floating_point(v):
                data_batch[k] = v.cuda().to(dtype=torch.bfloat16)
            else:
                data_batch[k] = v.cuda()
    return data_batch


def inference_batch4(
    setup_args: ActionConditionedSetupArguments,
    inference_args: ActionConditionedInferenceArguments,
    batch_size: int,
) -> None:
    torch.enable_grad(False)

    if inference_args.num_latent_conditional_frames not in [0, 1, 2]:
        raise ValueError(
            f"num_latent_conditional_frames must be 0, 1 or 2, but got {inference_args.num_latent_conditional_frames}"
        )
    if inference_args.visual_condition_source == "geometry" and inference_args.num_latent_conditional_frames != 1:
        raise ValueError("geometry visual condition currently supports num_latent_conditional_frames=1 only")

    checkpoint = setup_args.checkpoint_path
    experiment = setup_args.experiment
    from cosmos_predict2.config import MODEL_CHECKPOINTS

    checkpoint_meta = MODEL_CHECKPOINTS[setup_args.model_key]
    if experiment is None:
        experiment = checkpoint_meta.experiment
    if checkpoint is None:
        checkpoint = checkpoint_meta.s3.uri

    if experiment is None:
        raise ValueError("Experiment name must be provided either in setup args or checkpoint metadata")

    video2world_cli = Video2WorldInference(
        experiment_name=experiment,
        ckpt_path=checkpoint,
        s3_credential_path="",
        context_parallel_size=setup_args.context_parallel_size,
        config_file=setup_args.config_file,
    )
    model = video2world_cli.model
    model.eval()

    action_load_fn = load_callable(inference_args.action_load_fn)
    input_video_path = inference_args.input_root
    input_json_path = inference_args.input_root / inference_args.input_json_sub_folder
    input_json_list = list(input_json_path.glob("*.json"))
    logger.info(
        f"Batch-4 inference input_root={input_video_path}, input_json_path={input_json_path}, "
        f"json_count={len(input_json_list)}"
    )
    if not input_json_list:
        logger.warning("No annotation JSON files found; exiting without generation.")
        return

    rank0 = True
    if setup_args.context_parallel_size > 1:
        rank0 = distributed.get_rank() == 0

    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    inference_args.save_root.mkdir(parents=True, exist_ok=True)

    for annotation_path in input_json_list[inference_args.start : inference_args.end]:
        t5_cache: dict[str, torch.Tensor] = {}
        with open(annotation_path, "r") as f:
            json_data = json.load(f)

        camera_id = (
            int(inference_args.camera_id)
            if isinstance(inference_args.camera_id, str) and inference_args.camera_id.isdigit()
            else inference_args.camera_id
        )
        if isinstance(json_data["videos"][camera_id], dict):
            video_path = str(input_video_path / json_data["videos"][camera_id]["video_path"])
        else:
            video_path = str(input_video_path / json_data["videos"][camera_id])

        action_data = action_load_fn()(json_data, video_path, inference_args)
        actions = action_data["actions"]
        img_array = action_data["initial_frame"]
        geometry_min = action_data.get("geometry_min")
        geometry_max = action_data.get("geometry_max")

        img_name = annotation_path.stem
        img_arrays = [img_array for _ in range(batch_size)]
        chunk_video = [[] for _ in range(batch_size)]
        chunk_xyz = [[] for _ in range(batch_size)]

        for i in range(inference_args.start_frame_idx, len(actions), inference_args.chunk_size):
            chunk_start = time.perf_counter()
            actions_chunk = actions[i : i + inference_args.chunk_size]
            if actions_chunk.shape[0] != inference_args.chunk_size:
                pad_len = inference_args.chunk_size - actions_chunk.shape[0]
                if pad_len > 0:
                    action_shape = list(actions.shape[1:])
                    pad_shape = [pad_len] + action_shape
                    pad_actions = np.zeros(pad_shape, dtype=actions.dtype)
                    actions_chunk = np.concatenate([actions_chunk, pad_actions], axis=0)

            num_video_frames = actions_chunk.shape[0] + 1
            img_tensors = torch.stack(
                [torchvision.transforms.functional.to_tensor(img) for img in img_arrays], dim=0
            )
            vid_input = torch.cat(
                [
                    img_tensors.unsqueeze(2),
                    torch.zeros_like(img_tensors).unsqueeze(2).repeat(1, 1, num_video_frames - 1, 1, 1),
                ],
                dim=2,
            )
            vid_input = (vid_input * 255.0).to(torch.uint8)

            action_batch = np.repeat(actions_chunk[None, ...], batch_size, axis=0)
            prep_end = time.perf_counter()
            data_batch = _build_batch(
                model=model,
                video=vid_input,
                prompt=inference_args.prompt or "",
                negative_prompt=inference_args.negative_prompt,
                action=action_batch,
                num_conditional_frames=inference_args.num_latent_conditional_frames,
                batch_size=batch_size,
                t5_cache=t5_cache,
            )

            sample_start = time.perf_counter()
            sample = model.generate_samples_from_batch(
                data_batch,
                n_sample=batch_size,
                guidance=inference_args.guidance,
                seed=inference_args.seed + i,
                is_negative_prompt=inference_args.negative_prompt is not None,
                num_steps=35,
            )
            sample_end = time.perf_counter()

            if isinstance(sample, list):
                logger.info(f"Latent sample list shapes: {[tuple(s.shape) for s in sample]}")
                video = torch.cat([model.decode(s) for s in sample], dim=3)
            else:
                logger.info(f"Latent sample shape: {tuple(sample.shape)}")
                video = model.decode(sample)
            decode_end = time.perf_counter()

            video_normalized = (video - (-1)) / (1 - (-1))
            video_normalized = torch.clamp(video_normalized, 0, 1)

            for b in range(batch_size):
                xyz_chunk = video_normalized[b].permute(1, 2, 3, 0).detach().cpu().numpy().astype(np.float32)
                if inference_args.visual_condition_source == "geometry":
                    if geometry_min is None or geometry_max is None:
                        if inference_args.geometry_normalize:
                            raise ValueError("geometry_min/max missing for geometry visual condition.")
                    else:
                        denom = geometry_max - geometry_min
                        xyz_chunk = xyz_chunk * denom + geometry_min
                        xyz_chunk = np.clip(xyz_chunk, geometry_min, geometry_max)
                video_clamped = (
                    (video_normalized[b] * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
                )
                img_arrays[b] = video_clamped[-1]
                chunk_video[b].append(video_clamped)
                chunk_xyz[b].append(xyz_chunk)

            post_end = time.perf_counter()
            logger.info(
                "Chunk %d timing: prep=%.2fs sample=%.2fs decode=%.2fs post=%.2fs total=%.2fs",
                i,
                prep_end - chunk_start,
                sample_end - sample_start,
                decode_end - sample_end,
                post_end - decode_end,
                post_end - chunk_start,
            )

            if inference_args.single_chunk:
                break

        for b in range(batch_size):
            chunk_list = [chunk_video[b][0]] + [
                chunk_video[b][i][: inference_args.chunk_size] for i in range(1, len(chunk_video[b]))
            ]
            chunk_video_full = np.concatenate(chunk_list, axis=0)
            chunk_xyz_list = [chunk_xyz[b][0]] + [
                chunk_xyz[b][i][: inference_args.chunk_size] for i in range(1, len(chunk_xyz[b]))
            ]
            xyz_full = np.concatenate(chunk_xyz_list, axis=0)

            if inference_args.single_chunk:
                chunk_video_name = str(inference_args.save_root / f"{img_name}_b{b}_single_chunk.mp4")
            else:
                chunk_video_name = str(inference_args.save_root / f"{img_name}_b{b}_chunk.mp4")

            if rank0:
                mediapy.write_video(chunk_video_name, chunk_video_full, fps=inference_args.save_fps)
                logger.info(f"Saved video to {chunk_video_name}")
                print(f"Saved video to: {chunk_video_name}")
                if inference_args.visual_condition_source == "geometry":
                    _save_xyz_outputs(
                        xyz_full,
                        str(inference_args.save_root),
                        f"{img_name}_b{b}",
                        inference_args.save_fps,
                        geometry_min=geometry_min,
                        geometry_max=geometry_max,
                    )

    if setup_args.context_parallel_size > 1:
        torch.distributed.barrier()
    video2world_cli.cleanup()


def main(args: Args) -> None:
    inference_args = ActionConditionedInferenceArguments.from_files([args.input_file], overrides=args.overrides)[0]
    init_output_dir(args.setup.output_dir, profile=args.setup.profile)
    inference_batch4(args.setup, inference_args, args.batch_size)


if __name__ == "__main__":
    init_environment()

    try:
        args = tyro.cli(
            Args,
            description=__doc__,
            console_outputs=is_rank0(),
            config=(tyro.conf.OmitArgPrefixes,),
        )
    except Exception as e:
        handle_tyro_exception(e)
    # pyrefly: ignore  # unbound-name
    main(args)

    cleanup_environment()
