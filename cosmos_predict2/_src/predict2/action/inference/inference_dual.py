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

import argparse
import glob
import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torchvision
from safetensors import safe_open

from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_predict2._src.predict2.inference.get_t5_emb import get_text_embedding
from cosmos_predict2._src.predict2.utils.model_loader import load_model_from_checkpoint


def _progress(msg: str) -> None:
    ts = datetime.now().strftime("%m-%d %H:%M:%S")
    print(f"[{ts}][dual_infer] {msg}", flush=True)


def _save_uint8_video(video_thwc: np.ndarray, save_path: str, fps: int) -> None:
    video_cthw = torch.from_numpy(video_thwc).permute(3, 0, 1, 2).contiguous()
    save_img_or_video(video_cthw, save_path[:-4] if save_path.endswith(".mp4") else save_path, fps=fps)


def _load_geometry_first_frame(
    geometry_path: str, geometry_min: float, geometry_max: float, geometry_normalize: bool
) -> np.ndarray:
    with safe_open(geometry_path, framework="pt", device="cpu") as f:
        xyz = f.get_tensor("xyz").numpy().astype(np.float32)
    first_frame = xyz[0]
    if geometry_normalize:
        first_frame = np.clip(first_frame, geometry_min, geometry_max)
        denom = max(geometry_max - geometry_min, 1e-6)
        first_frame = (first_frame - geometry_min) / denom
        first_frame = np.clip(first_frame, 0.0, 1.0)
    else:
        first_frame = np.clip(first_frame, 0.0, 1.0)
    return (first_frame * 255.0).round().astype(np.uint8)


def _build_video_from_first_frame(first_frame: np.ndarray, num_frames: int, target_hw: tuple[int, int]) -> torch.Tensor:
    img_tensor = torchvision.transforms.functional.to_tensor(first_frame).unsqueeze(0)
    img_tensor = torchvision.transforms.functional.resize(
        img_tensor, [target_hw[0], target_hw[1]], antialias=True
    )
    vid_input = torch.cat([img_tensor, torch.zeros_like(img_tensor).repeat(num_frames - 1, 1, 1, 1)], dim=0)
    vid_input = (vid_input * 255.0).to(torch.uint8)
    return vid_input.unsqueeze(0).permute(0, 2, 1, 3, 4)


def _load_first_rgb_frame(video_path: str) -> np.ndarray:
    video, _, _ = torchvision.io.read_video(video_path, pts_unit="sec")
    if video.numel() == 0:
        raise ValueError(f"No frames found in {video_path}")
    return video[0].numpy().astype(np.uint8)


def _make_batch(
    model,
    video_tensor: torch.Tensor,
    prompt: str,
    negative_prompt: str | None,
    action: np.ndarray,
    num_conditional_frames: int,
) -> dict:
    B, C, T, H, W = video_tensor.shape
    data_batch = {
        "dataset_name": "video_data",
        "video": video_tensor,
        "fps": torch.randint(16, 32, (1,)).float(),
        "padding_mask": torch.zeros(1, 1, H, W),
        "num_conditional_frames": num_conditional_frames,
    }

    if model.text_encoder is not None:
        data_batch["ai_caption"] = [prompt]
        data_batch["t5_text_embeddings"] = model.text_encoder.compute_text_embeddings_online(
            data_batch={"ai_caption": [prompt], "images": None},
            input_caption_key="ai_caption",
        )
        if negative_prompt is not None:
            data_batch["neg_t5_text_embeddings"] = model.text_encoder.compute_text_embeddings_online(
                data_batch={"ai_caption": [negative_prompt], "images": None},
                input_caption_key="ai_caption",
            )
    else:
        data_batch["t5_text_embeddings"] = get_text_embedding(prompt)
        if negative_prompt is not None:
            data_batch["neg_t5_text_embeddings"] = get_text_embedding(negative_prompt)

    num_action_per_chunk = getattr(model.net, "num_action_per_chunk", 12)
    action_dim = model.net.action_embedder_B_D.fc1.in_features // num_action_per_chunk
    action = np.asarray(action, dtype=np.float32)
    if action.ndim == 1:
        action = action.reshape(-1, action_dim)
    if action.shape[-1] != action_dim:
        raise ValueError(f"action_dim mismatch: got {action.shape[-1]} expected {action_dim}")
    if action.shape[0] < num_action_per_chunk:
        pad = np.zeros((num_action_per_chunk - action.shape[0], action_dim), dtype=np.float32)
        action = np.concatenate([action, pad], axis=0)
    elif action.shape[0] > num_action_per_chunk:
        action = action[:num_action_per_chunk]

    data_batch["action"] = torch.from_numpy(action).to(dtype=torch.bfloat16)[None, ...]
    for k, v in data_batch.items():
        if isinstance(v, torch.Tensor):
            if torch.is_floating_point(v):
                data_batch[k] = v.cuda().to(dtype=torch.bfloat16)
            else:
                data_batch[k] = v.cuda()
    return data_batch


def _run_single(
    model,
    config,
    rgb_frame_path: str,
    geometry_path: str,
    action: np.ndarray,
    prompt: str,
    negative_prompt: str | None,
    guidance: int,
    seed: int,
    num_conditional_frames: int,
    save_dir: str,
    save_fps: int,
    save_latents: bool,
) -> None:
    target_hw = (model.net.max_img_h, model.net.max_img_w)
    rgb_frame = _load_first_rgb_frame(rgb_frame_path)
    geometry_frame = _load_geometry_first_frame(
        geometry_path, config.geometry_min, config.geometry_max, config.geometry_normalize
    )

    num_action_per_chunk = getattr(model.rgb_model.net, "num_action_per_chunk", 12)
    rgb_frames = []
    xyz_frames = []
    rgb_latents = []
    xyz_latents = []

    img_rgb = rgb_frame
    img_xyz = geometry_frame
    total_chunks = (action.shape[0] + num_action_per_chunk - 1) // num_action_per_chunk

    for chunk_idx, i in enumerate(range(0, action.shape[0], num_action_per_chunk), start=1):
        end_idx = min(i + num_action_per_chunk, action.shape[0])
        _progress(
            f"[separate][chunk {chunk_idx}/{total_chunks}] start action[{i}:{end_idx}] "
            f"seed={seed + i} save_dir={save_dir}"
        )
        actions_chunk = action[i : i + num_action_per_chunk]
        if actions_chunk.shape[0] != num_action_per_chunk:
            pad_len = num_action_per_chunk - actions_chunk.shape[0]
            pad_actions = np.zeros((pad_len, actions_chunk.shape[1]), dtype=actions_chunk.dtype)
            actions_chunk = np.concatenate([actions_chunk, pad_actions], axis=0)

        num_frames = num_action_per_chunk + 1
        rgb_video = _build_video_from_first_frame(img_rgb, num_frames, target_hw)
        xyz_video = _build_video_from_first_frame(img_xyz, num_frames, target_hw)

        rgb_batch = _make_batch(
            model.rgb_model, rgb_video, prompt, negative_prompt, actions_chunk, num_conditional_frames
        )
        xyz_batch = _make_batch(
            model.xyz_model, xyz_video, prompt, negative_prompt, actions_chunk, num_conditional_frames
        )

        rgb_sample = model.rgb_model.generate_samples_with_latents_from_batch(
            rgb_batch,
            n_sample=1,
            guidance=guidance,
            seed=seed + i,
            is_negative_prompt=negative_prompt is not None,
        )
        xyz_sample = model.xyz_model.generate_samples_with_latents_from_batch(
            xyz_batch,
            n_sample=1,
            guidance=guidance,
            seed=seed + i,
            is_negative_prompt=negative_prompt is not None,
        )

        rgb_video = ((model.rgb_model.decode(rgb_sample) + 1) / 2).clamp(0, 1)
        xyz_video = ((model.xyz_model.decode(xyz_sample) + 1) / 2).clamp(0, 1)

        rgb_chunk = (rgb_video[0].permute(1, 2, 3, 0) * 255).to(torch.uint8).cpu().numpy()
        xyz_chunk = (xyz_video[0].permute(1, 2, 3, 0) * 255).to(torch.uint8).cpu().numpy()

        rgb_frames.append(rgb_chunk)
        xyz_frames.append(xyz_chunk)
        if save_latents:
            rgb_latents.append(rgb_sample.detach().cpu().numpy())
            xyz_latents.append(xyz_sample.detach().cpu().numpy())

        img_rgb = rgb_chunk[-1]
        img_xyz = xyz_chunk[-1]
        _progress(
            f"[separate][chunk {chunk_idx}/{total_chunks}] done "
            f"rgb_frames={rgb_chunk.shape[0]} xyz_frames={xyz_chunk.shape[0]}"
        )

    rgb_full = np.concatenate([rgb_frames[0]] + [c[:num_action_per_chunk] for c in rgb_frames[1:]], axis=0)
    xyz_full = np.concatenate([xyz_frames[0]] + [c[:num_action_per_chunk] for c in xyz_frames[1:]], axis=0)

    os.makedirs(save_dir, exist_ok=True)
    _save_uint8_video(rgb_full, os.path.join(save_dir, "rgb.mp4"), fps=save_fps)
    _save_uint8_video(xyz_full, os.path.join(save_dir, "xyz.mp4"), fps=save_fps)
    if save_latents and rgb_latents:
        np.save(os.path.join(save_dir, "rgb_latents.npy"), np.concatenate(rgb_latents, axis=2))
        np.save(os.path.join(save_dir, "xyz_latents.npy"), np.concatenate(xyz_latents, axis=2))


def _run_single_dual(
    model,
    config,
    rgb_frame_path: str,
    geometry_path: str,
    action: np.ndarray,
    prompt: str,
    negative_prompt: str | None,
    guidance: int,
    seed: int,
    num_conditional_frames: int,
    save_dir: str,
    save_fps: int,
    save_latents: bool,
) -> None:
    target_hw = (model.net.max_img_h, model.net.max_img_w)
    rgb_frame = _load_first_rgb_frame(rgb_frame_path)
    geometry_frame = _load_geometry_first_frame(
        geometry_path, config.geometry_min, config.geometry_max, config.geometry_normalize
    )

    num_action_per_chunk = getattr(model.rgb_model.net, "num_action_per_chunk", 12)
    rgb_frames = []
    xyz_frames = []
    rgb_latents = []
    xyz_latents = []

    img_rgb = rgb_frame
    img_xyz = geometry_frame
    total_chunks = (action.shape[0] + num_action_per_chunk - 1) // num_action_per_chunk

    for chunk_idx, i in enumerate(range(0, action.shape[0], num_action_per_chunk), start=1):
        end_idx = min(i + num_action_per_chunk, action.shape[0])
        _progress(
            f"[dual][chunk {chunk_idx}/{total_chunks}] start action[{i}:{end_idx}] "
            f"seed={seed + i} save_dir={save_dir}"
        )
        actions_chunk = action[i : i + num_action_per_chunk]
        if actions_chunk.shape[0] != num_action_per_chunk:
            pad_len = num_action_per_chunk - actions_chunk.shape[0]
            pad_actions = np.zeros((pad_len, actions_chunk.shape[1]), dtype=actions_chunk.dtype)
            actions_chunk = np.concatenate([actions_chunk, pad_actions], axis=0)

        num_frames = num_action_per_chunk + 1
        rgb_video = _build_video_from_first_frame(img_rgb, num_frames, target_hw)
        xyz_video = _build_video_from_first_frame(img_xyz, num_frames, target_hw)

        rgb_batch = _make_batch(
            model.rgb_model, rgb_video, prompt, negative_prompt, actions_chunk, num_conditional_frames
        )
        xyz_batch = _make_batch(
            model.xyz_model, xyz_video, prompt, negative_prompt, actions_chunk, num_conditional_frames
        )
        dual_batch = {"rgb": rgb_batch, "xyz": xyz_batch}

        rgb_sample, xyz_sample = model.generate_samples_with_latents_from_batch_dual(
            dual_batch,
            n_sample=1,
            guidance=guidance,
            seed=seed + i,
            is_negative_prompt=negative_prompt is not None,
        )

        rgb_video = ((model.rgb_model.decode(rgb_sample) + 1) / 2).clamp(0, 1)
        xyz_video = ((model.xyz_model.decode(xyz_sample) + 1) / 2).clamp(0, 1)

        rgb_chunk = (rgb_video[0].permute(1, 2, 3, 0) * 255).to(torch.uint8).cpu().numpy()
        xyz_chunk = (xyz_video[0].permute(1, 2, 3, 0) * 255).to(torch.uint8).cpu().numpy()

        rgb_frames.append(rgb_chunk)
        xyz_frames.append(xyz_chunk)
        if save_latents:
            rgb_latents.append(rgb_sample.detach().cpu().numpy())
            xyz_latents.append(xyz_sample.detach().cpu().numpy())

        img_rgb = rgb_chunk[-1]
        img_xyz = xyz_chunk[-1]
        _progress(
            f"[dual][chunk {chunk_idx}/{total_chunks}] done "
            f"rgb_frames={rgb_chunk.shape[0]} xyz_frames={xyz_chunk.shape[0]}"
        )

    rgb_full = np.concatenate([rgb_frames[0]] + [c[:num_action_per_chunk] for c in rgb_frames[1:]], axis=0)
    xyz_full = np.concatenate([xyz_frames[0]] + [c[:num_action_per_chunk] for c in xyz_frames[1:]], axis=0)

    os.makedirs(save_dir, exist_ok=True)
    _save_uint8_video(rgb_full, os.path.join(save_dir, "rgb.mp4"), fps=save_fps)
    _save_uint8_video(xyz_full, os.path.join(save_dir, "xyz.mp4"), fps=save_fps)
    if save_latents and rgb_latents:
        np.save(os.path.join(save_dir, "rgb_latents.npy"), np.concatenate(rgb_latents, axis=2))
        np.save(os.path.join(save_dir, "xyz_latents.npy"), np.concatenate(xyz_latents, axis=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-tower action-conditioned inference")
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--load_mode", type=str, default="full", choices=["full", "tower_init"])
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Full-model checkpoint path. Prefer passing a concrete .pt file; directories are resolved to a preferred .pt when possible.",
    )
    parser.add_argument("--rgb_init_checkpoint_path", type=str, default=None)
    parser.add_argument("--xyz_init_checkpoint_path", type=str, default=None)
    parser.add_argument("--sampling_mode", type=str, default="dual", choices=["dual", "separate"])
    parser.add_argument("--num_action_per_chunk_override", type=int, default=None)
    parser.add_argument("--rgb_frame", type=str, help="Path to an RGB frame (image).")
    parser.add_argument("--geometry_safetensors", type=str, help="Path to geometry.safetensors.")
    parser.add_argument("--action_npy", type=str, help="Numpy file of shape [T-1, action_dim].")
    parser.add_argument("--basic_root", type=str, help="Path to assets/action_conditioned/basic.")
    parser.add_argument("--geometry_root", type=str, help="Path to assets/action_conditioned/geometry.")
    parser.add_argument("--split", type=str, default="test", help="Split name under bridge/videos and bridge/annotation.")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--guidance", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_latent_conditional_frames", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="results/dual_infer")
    parser.add_argument("--save_fps", type=int, default=4)
    parser.add_argument("--save_latents", action="store_true")
    args = parser.parse_args()

    repo_root = Path(os.getcwd()).resolve()
    def _resolve_path(path: str) -> str:
        if os.path.exists(path):
            return path
        legacy_root = "/nfs/rczhang/code/cosmos-predict2.5"
        if path.startswith(legacy_root):
            candidate = str(repo_root) + path[len(legacy_root) :]
            if os.path.exists(candidate):
                return candidate
        return path

    def _prefer_pt_checkpoint(path: str) -> str:
        if os.path.isfile(path) and path.endswith(".pt"):
            return path
        search_dir = path
        if path.rstrip("/").endswith("/model"):
            search_dir = os.path.dirname(path)
        if os.path.isdir(search_dir):
            preferred = [
                "model_ema_converted.pt",
                "model_converted.pt",
                "model_ema_bf16.pt",
                "model_bf16.pt",
                "model.pt",
            ]
            for name in preferred:
                candidate = os.path.join(search_dir, name)
                if os.path.isfile(candidate):
                    return candidate
            candidates = sorted(glob.glob(os.path.join(search_dir, "*.pt")))
            if candidates:
                return candidates[0]
        return path

    experiment_opts = []
    if args.num_action_per_chunk_override is not None:
        experiment_opts.append(f"model.config.net.num_action_per_chunk={args.num_action_per_chunk_override}")
    if args.load_mode == "tower_init":
        if not args.rgb_init_checkpoint_path or not args.xyz_init_checkpoint_path:
            raise ValueError("--load_mode tower_init requires both --rgb_init_checkpoint_path and --xyz_init_checkpoint_path")
        rgb_init_path = args.rgb_init_checkpoint_path
        xyz_init_path = args.xyz_init_checkpoint_path
        if not rgb_init_path.startswith("hf://"):
            rgb_init_path = _resolve_path(rgb_init_path)
        if not xyz_init_path.startswith("hf://"):
            xyz_init_path = _resolve_path(xyz_init_path)
        experiment_opts.extend(
            [
                f"model.config.rgb_init_checkpoint_path={rgb_init_path}",
                f"model.config.xyz_init_checkpoint_path={xyz_init_path}",
                "checkpoint.resume_from_checkpoint=false",
            ]
        )
        ckpt_path = None
        skip_load_model = True
    else:
        if not args.ckpt_path:
            raise ValueError("--ckpt_path is required when --load_mode full")
        ckpt_path = _resolve_path(args.ckpt_path)
        ckpt_path = _prefer_pt_checkpoint(ckpt_path)
        if os.path.isdir(ckpt_path) and not ckpt_path.rstrip("/").endswith("/model"):
            candidate = os.path.join(ckpt_path, "model")
            if os.path.isdir(candidate):
                ckpt_path = candidate
        _progress(f"[full] resolved ckpt_path={ckpt_path}")
        skip_load_model = False

    model, config = load_model_from_checkpoint(
        experiment_name=args.experiment,
        s3_checkpoint_dir=ckpt_path,
        config_file="cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py",
        load_ema_to_reg=True,
        experiment_opts=experiment_opts,
        skip_load_model=skip_load_model,
    )
    model.eval()

    basic_root = _resolve_path(args.basic_root) if args.basic_root else None
    geometry_root = _resolve_path(args.geometry_root) if args.geometry_root else None

    if basic_root and geometry_root:
        ann_dir = os.path.join(basic_root, "annotation", args.split)
        json_paths = sorted(glob.glob(os.path.join(ann_dir, "*.json")))
        if not json_paths:
            raise FileNotFoundError(f"No annotation json files found in {ann_dir}")
        total_instances = len(json_paths)
        for inst_idx, json_path in enumerate(json_paths, start=1):
            instance_id = os.path.splitext(os.path.basename(json_path))[0]
            with open(json_path) as f:
                ann = json.load(f)
            action = np.array(ann["action"], dtype=np.float32)
            prompt = args.prompt
            if not prompt:
                texts = ann.get("texts", [])
                prompt = texts[0] if texts else ""
            rgb_frame_path = os.path.join(basic_root, "videos", args.split, instance_id, "rgb.mp4")
            geometry_path = os.path.join(
                geometry_root, "videos", args.split, instance_id, "geometry.safetensors"
            )
            save_dir = os.path.join(args.save_dir, instance_id)
            runner = _run_single_dual if args.sampling_mode == "dual" else _run_single
            _progress(
                f"[instance {inst_idx}/{total_instances}] start id={instance_id} "
                f"actions={action.shape[0]} mode={args.sampling_mode}"
            )
            runner(
                model=model,
                config=config,
                rgb_frame_path=rgb_frame_path,
                geometry_path=geometry_path,
                action=action,
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                guidance=args.guidance,
                seed=args.seed,
                num_conditional_frames=args.num_latent_conditional_frames,
                save_dir=save_dir,
                save_fps=args.save_fps,
                save_latents=args.save_latents,
            )
            _progress(f"[instance {inst_idx}/{total_instances}] done id={instance_id} output={save_dir}")
    else:
        if not (args.rgb_frame and args.geometry_safetensors and args.action_npy):
            raise ValueError("Provide either dataset roots or explicit rgb/geometry/action paths.")
        action = np.load(args.action_npy)
        runner = _run_single_dual if args.sampling_mode == "dual" else _run_single
        _progress(
            f"[single] start actions={action.shape[0]} mode={args.sampling_mode} output={args.save_dir}"
        )
        runner(
            model=model,
            config=config,
            rgb_frame_path=args.rgb_frame,
            geometry_path=args.geometry_safetensors,
            action=action,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            guidance=args.guidance,
            seed=args.seed,
            num_conditional_frames=args.num_latent_conditional_frames,
            save_dir=args.save_dir,
            save_fps=args.save_fps,
            save_latents=args.save_latents,
        )
        _progress(f"[single] done output={args.save_dir}")


if __name__ == "__main__":
    main()
