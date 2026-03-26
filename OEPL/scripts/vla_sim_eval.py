#!/usr/bin/env python3
"""
VLA 仿真评估（无 Cosmos）

在 Robosuite 中运行 VLA 策略，仅做仿真评估，不加载 Cosmos。
适用于终端 1：VLA + 仿真环境，与 Cosmos 分离以规避 GPU/库冲突。

用法:
  CUDA_VISIBLE_DEVICES=7 python vla_sim_eval.py \\
    --vla-path /workspace/WMPO/checkpoint_files/SFT_models/coffee \\
    --unnorm-key coffee_d0_300_demos \\
    --task coffee \\
    --env-config /workspace/WMPO/data_files/core_train_configs/bc_rnn_image_ds_coffee_D0_seed_101.json \\
    --reset-states /workspace/WMPO/verl/utils/dataset/coffee_d0_states.pkl \\
    --output-dir /workspace/WMPO/debug/vla_sim_eval_out \\
    --num-chunks 25 --chunk-size 8 --runs 10 --no-video
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import imageio
import numpy as np
import torch
from PIL import Image
import imageio.v2 as imageio_v2

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from vla_sim_rollout import create_env_from_config
from experiments.robot.openvla_utils import get_processor, get_vla, get_vla_action


def _pad_or_trim(actions: np.ndarray, target_len: int) -> np.ndarray:
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    if actions.shape[0] >= target_len:
        return actions[:target_len]
    pad_len = target_len - actions.shape[0]
    pad = np.zeros((pad_len, actions.shape[1]), dtype=actions.dtype)
    return np.concatenate([actions, pad], axis=0)


def _normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    normalized_action = action.copy()
    normalized_action[..., -1] = 2 * (normalized_action[..., -1] - 0.0) / (1.0 - 0.0) - 1
    if binarize:
        normalized_action[..., -1] = np.sign(normalized_action[..., -1])
    return normalized_action


def _invert_gripper_action(action: np.ndarray) -> np.ndarray:
    inverted_action = action.copy()
    inverted_action[..., -1] = inverted_action[..., -1] * -1.0
    return inverted_action


def _ensure_uint8_image(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 0 or arr.size == 0:
        raise ValueError(f"Invalid image array: shape={arr.shape} dtype={arr.dtype}")
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            scale = 255.0 if float(np.nanmax(arr)) <= 1.0 else 1.0
            arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _pick_agent_image(obs: dict, crop_ratio: float = 0.0) -> np.ndarray:
    if "agentview_image" not in obs:
        raise KeyError("Observation does not contain 'agentview_image'")
    img = obs["agentview_image"]
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    if 0 < crop_ratio < 1:
        h, w, _ = img.shape
        new_h = int(h * crop_ratio)
        new_w = int(w * crop_ratio)
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        cropped = img[top : top + new_h, left : left + new_w]
        img = np.array(Image.fromarray(cropped).resize((w, h), Image.BICUBIC))
    return img


def _write_image(path: Path, arr: np.ndarray) -> None:
    arr = _ensure_uint8_image(arr)
    imageio_v2.imwrite(str(path), arr)


def main() -> None:
    parser = argparse.ArgumentParser(description="VLA 仿真评估（无 Cosmos）")
    parser.add_argument("--vla-path", required=True, type=Path)
    parser.add_argument("--unnorm-key", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--env-config", required=True, type=Path)
    parser.add_argument("--reset-states", default=None, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-chunks", type=int, default=25)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-fps", type=int, default=20)
    parser.add_argument(
        "--gripper-postproc",
        choices=("none", "normalize_invert"),
        default="none",
        help="Gripper post-processing to match eval behavior.",
    )
    parser.add_argument("--no-video", action="store_true", help="不保存视频")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument(
        "--vla-platform",
        choices=("libero", "aloha", "bridge"),
        default="libero",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.device}" if args.device is not None else "cuda")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this script.")

    argv_lower = " ".join(sys.argv).lower()
    if args.vla_platform and args.vla_platform not in argv_lower:
        sys.argv.append(args.vla_platform)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.max_steps is None:
        args.max_steps = {"coffee": 256, "stack_three": 320, "three_piece_assembly": 384, "square": 184}.get(
            args.task, 256
        )

    cfg_ns = SimpleNamespace(
        pretrained_checkpoint=str(args.vla_path),
        center_crop=True,
        num_images_in_input=1,
        use_proprio=False,
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        load_in_8bit=False,
        load_in_4bit=False,
        num_open_loop_steps=args.chunk_size,
        unnorm_key=args.unnorm_key,
    )

    vla = get_vla(cfg_ns).to(device).eval()
    processor = get_processor(cfg_ns)
    action_head = None

    env, _ = create_env_from_config(str(args.env_config))
    reset_states = None
    if args.reset_states:
        import pickle

        with open(args.reset_states, "rb") as f:
            reset_states = pickle.load(f)

    task_description = args.task
    successes = 0
    lengths = []

    for ep in range(args.runs):
        ep_dir = args.output_dir / f"episode_{ep:02d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        frame_inputs_dir = ep_dir / "frame_inputs"
        sim_inputs_dir = ep_dir / "sim_inputs"
        vla_dir = ep_dir / "vla_actions"
        sim_frames_dir = ep_dir / "sim_frames"
        for d in (frame_inputs_dir, sim_inputs_dir, vla_dir, sim_frames_dir):
            d.mkdir(parents=True, exist_ok=True)

        if reset_states is not None and len(reset_states) > 0:
            obs = env.reset_to(reset_states[ep % len(reset_states)])
        else:
            obs = env.reset()
        for _ in range(10):
            obs, _, _, _ = env.step(np.zeros(7))

        video_path = ep_dir / f"{args.task}_vla_sim_rollout.mp4"
        video_writer = None
        if not args.no_video:
            video_writer = imageio.get_writer(str(video_path), fps=args.save_fps)

        sim_frame0 = _pick_agent_image(obs, crop_ratio=0.0)
        _write_image(sim_frames_dir / "step_0000.png", sim_frame0)
        if video_writer is not None:
            video_writer.append_data(sim_frame0)

        log_file = ep_dir / f"{args.task}_rollout_log.jsonl"
        log_fp = open(log_file, "w", encoding="utf-8")

        total_steps = 0
        success = False

        for chunk_idx in range(args.num_chunks):
            if total_steps >= args.max_steps:
                break
            print(f"[Episode {ep:02d}][Chunk {chunk_idx:03d}] start (total_steps={total_steps})", flush=True)

            current_sim_frame = _pick_agent_image(obs, crop_ratio=0.0)
            vla_image = current_sim_frame

            _write_image(sim_inputs_dir / f"chunk_{chunk_idx:03d}_input.png", current_sim_frame)
            _write_image(frame_inputs_dir / f"chunk_{chunk_idx:03d}_input.png", vla_image)

            policy_obs = {"full_image": vla_image, "task_description": task_description}
            actions_chunk = get_vla_action(
                cfg_ns, vla, processor, policy_obs, task_description, action_head, None
            )
            raw_actions = np.asarray(actions_chunk)

            if raw_actions.ndim == 1:
                raw_actions = raw_actions.reshape(1, -1)
            if raw_actions.shape[1] != 7:
                raise RuntimeError(f"Expected action dim 7, got {raw_actions.shape}")

            actions_for_sim = _pad_or_trim(raw_actions, args.chunk_size)
            if args.gripper_postproc == "normalize_invert":
                actions_for_sim = _invert_gripper_action(
                    _normalize_gripper_action(actions_for_sim, binarize=True)
                )

            (vla_dir / f"chunk_{chunk_idx:03d}.json").write_text(
                json.dumps(actions_for_sim.astype(float).tolist(), ensure_ascii=True),
                encoding="utf-8",
            )

            for step_idx, action in enumerate(actions_for_sim):
                if total_steps >= args.max_steps:
                    break

                log_fp.write(json.dumps({"chunk": chunk_idx, "step": total_steps, "action": action.tolist()}) + "\n")
                log_fp.flush()

                obs, reward, done, info = env.step(action)
                sim_frame = _pick_agent_image(obs, crop_ratio=0.0)
                _write_image(sim_frames_dir / f"step_{total_steps + 1:04d}.png", sim_frame)
                if video_writer is not None:
                    video_writer.append_data(sim_frame)

                total_steps += 1
                if reward > 0.0 or done:
                    if done:
                        success = True
                    break

            if success or done:
                break

        log_fp.close()
        if video_writer is not None:
            video_writer.close()

        lengths.append(total_steps)
        successes += int(success)

        summary = {
            "task": args.task,
            "episode": ep,
            "total_steps": total_steps,
            "num_chunks_executed": chunk_idx + 1,
            "success": success,
            "video_path": str(video_path) if not args.no_video else "",
            "log_path": str(log_file),
        }
        (ep_dir / f"{args.task}_summary.json").write_text(json.dumps(summary, indent=2))

    overall_summary = {
        "task": args.task,
        "episodes": args.runs,
        "successes": successes,
        "success_rate": successes / max(1, args.runs),
        "avg_len": float(np.mean(lengths)) if lengths else 0.0,
        "std_len": float(np.std(lengths)) if lengths else 0.0,
        "env_config": str(args.env_config),
        "reset_states": str(args.reset_states) if args.reset_states else "",
    }
    (args.output_dir / f"{args.task}_overall_summary.json").write_text(json.dumps(overall_summary, indent=2))

    try:
        if hasattr(env, "close"):
            env.close()
        if hasattr(env, "env") and hasattr(env.env, "close"):
            env.env.close()
    except Exception as exc:
        print(f"[WARN] Error closing environment: {exc}")

    print(f"[VLA Sim Eval] Success rate: {successes}/{args.runs} = {overall_summary['success_rate']:.2%}")


if __name__ == "__main__":
    main()
