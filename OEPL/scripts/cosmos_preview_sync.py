#!/usr/bin/env python3
"""
Cosmos 预览 - Docker 端（与 Conda VLA 仿真同步）

在 Docker 中运行：轮询共享目录 → 读取 Conda 保存的帧与动作 → 运行 Cosmos 预览 →
发出继续信号，让 Conda 执行动作并继续下一轮。

用法（Docker 环境，需先启动 Conda 端的 vla_sim_eval_sync.py）:
  python cosmos_preview_sync.py \\
    --sync-dir /path/to/shared/sync \\
    --output-dir /path/to/cosmos_output \\
    --cosmos-experiment ... --cosmos-checkpoint-path ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sync_protocol import (
    docker_wait_for_conda,
    docker_read_and_clear,
    docker_signal_continue,
)
from vla_cosmos_bridge import _adapt_actions_to_cosmos, _load_cosmos_infer


def _pad_actions(actions: np.ndarray, target_len: int) -> np.ndarray:
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    if actions.shape[0] >= target_len:
        return actions[:target_len]
    pad_len = target_len - actions.shape[0]
    pad = np.zeros((pad_len, actions.shape[1]), dtype=actions.dtype)
    return np.concatenate([actions, pad], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cosmos 预览 - Docker 端（与 Conda 同步）")
    parser.add_argument("--sync-dir", required=True, type=Path, help="Conda 与 Docker 共享目录")
    parser.add_argument("--output-dir", type=Path, default=None, help="Cosmos 输出目录（可选）")
    parser.add_argument("--cosmos-root", type=Path, default=Path("/workspace"))
    parser.add_argument("--cosmos-experiment", required=True)
    parser.add_argument("--cosmos-checkpoint-path", default=None)
    parser.add_argument("--cosmos-model-key", default=None)
    parser.add_argument(
        "--cosmos-config-file",
        default="cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py",
    )
    parser.add_argument("--cosmos-action-chunk-size", type=int, default=12)
    parser.add_argument("--cosmos-action-scale", type=float, default=20.0)
    parser.add_argument("--cosmos-gripper-scale", type=float, default=1.0)
    parser.add_argument("--resolution", type=str, default="256,320")
    parser.add_argument("--guidance", type=float, default=0.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument(
        "--negative-prompt-file",
        type=Path,
        default=Path("/workspace/assets/action_conditioned/openvla-coffee/inference_params.json"),
    )
    parser.add_argument("--wait-timeout", type=float, default=3600, help="等待 Conda 就绪超时(秒)")
    parser.add_argument("--seed-base", type=int, default=0)
    args = parser.parse_args()

    args.sync_dir.mkdir(parents=True, exist_ok=True)
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this script.")

    if args.negative_prompt_file and args.negative_prompt_file.exists():
        try:
            neg_data = json.loads(args.negative_prompt_file.read_text(encoding="utf-8"))
            if isinstance(neg_data, dict) and neg_data.get("negative_prompt"):
                args.negative_prompt = neg_data["negative_prompt"]
        except Exception as exc:
            print(f"[WARN] failed to read negative_prompt: {exc}")

    cosmos_infer = _load_cosmos_infer(
        args.cosmos_root,
        args.cosmos_experiment,
        args.cosmos_checkpoint_path,
        args.cosmos_model_key,
        args.cosmos_config_file,
    )

    chunk_count = 0
    print("[Docker] Cosmos 预览服务已启动，等待 Conda 就绪...", flush=True)

    while True:
        if not docker_wait_for_conda(args.sync_dir, timeout_sec=args.wait_timeout):
            print("[Docker] 等待 Conda 超时，退出", flush=True)
            break

        frame, actions_list, meta = docker_read_and_clear(args.sync_dir)
        chunk_idx = meta.get("chunk_idx", chunk_count)
        episode = meta.get("episode", 0)

        # 将动作转为 Cosmos 格式
        actions_np = np.array(actions_list, dtype=np.float32)
        actions_np = _pad_actions(actions_np, args.cosmos_action_chunk_size)
        actions_cosmos = _adapt_actions_to_cosmos(
            actions_np,
            target_chunk_size=args.cosmos_action_chunk_size,
            action_scale=args.cosmos_action_scale,
            gripper_scale=args.cosmos_gripper_scale,
        )

        # 准备 Cosmos 输入
        img_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        num_video_frames = actions_cosmos.shape[0] + 1
        vid_input = torch.cat(
            [
                img_tensor,
                torch.zeros_like(img_tensor).repeat(num_video_frames - 1, 1, 1, 1),
            ],
            dim=0,
        )
        vid_input = (vid_input * 255.0).to(torch.uint8)
        vid_input = vid_input.unsqueeze(0).permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)

        seed = args.seed_base + chunk_idx + episode * 10000
        print(f"[Docker][Ep {episode:02d}][Chunk {chunk_idx:03d}] 运行 Cosmos 预览...", flush=True)

        video = cosmos_infer.generate_vid2world(
            prompt="",
            input_path=vid_input,
            action=torch.from_numpy(actions_cosmos).float(),
            guidance=args.guidance,
            num_video_frames=num_video_frames,
            num_latent_conditional_frames=1,
            resolution=args.resolution,
            seed=seed,
            negative_prompt=args.negative_prompt,
        )

        video_normalized = (video - (-1)) / (1 - (-1))
        video_normalized = torch.clamp(video_normalized, 0, 1)
        video_clamped = (
            (video_normalized[0] * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
        )

        if args.output_dir is not None:
            chunk_dir = args.output_dir / f"ep{episode:02d}_chunk{chunk_idx:03d}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            try:
                import mediapy
                for f_idx, f in enumerate(video_clamped):
                    mediapy.write_image(str(chunk_dir / f"frame_{f_idx:03d}.png"), f)
            except ImportError:
                import imageio.v2 as imageio_v2
                for f_idx, f in enumerate(video_clamped):
                    imageio_v2.imwrite(str(chunk_dir / f"frame_{f_idx:03d}.png"), f)

        docker_signal_continue(args.sync_dir)
        print(f"[Docker][Ep {episode:02d}][Chunk {chunk_idx:03d}] 已发出继续信号", flush=True)
        chunk_count += 1


if __name__ == "__main__":
    main()
