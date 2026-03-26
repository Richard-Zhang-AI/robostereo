#!/usr/bin/env python3
"""
VLA + Cosmos gating rollout in Robosuite.

Flow:
  1) Reset simulator, capture current frame.
  2) VLA samples an action chunk (e.g. 8 steps).
  3) Cosmos previews the chunk (e.g. 13 frames) from the same input frame.
  4) Judge the preview; if accepted, execute actions in simulator.
  5) Repeat with the new simulator frame.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import imageio
import mediapy
import numpy as np
import torch
from PIL import Image
import imageio.v2 as imageio_v2
from types import SimpleNamespace

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from vla_sim_rollout import create_env_from_config
from vla_cosmos_bridge import (
    _adapt_actions_to_cosmos,
    _load_cosmos_infer,
)
from experiments.robot.openvla_utils import get_processor, get_vla, get_vla_action


def _resize_for_cosmos(frame: np.ndarray, resolution: str) -> np.ndarray:
    if resolution == "none":
        return frame
    try:
        h, w = map(int, resolution.split(","))
        return np.array(Image.fromarray(frame).resize((w, h), Image.BICUBIC))
    except Exception as exc:
        print(f"[WARN] failed to resize frame to {resolution}: {exc}")
        return frame


def _judge_cosmos_sequence(frames: np.ndarray) -> bool:
    # TODO: fill in with custom acceptance logic.
    return True


def _pad_or_trim(actions: np.ndarray, target_len: int) -> np.ndarray:
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    if actions.shape[0] >= target_len:
        return actions[:target_len]
    pad_len = target_len - actions.shape[0]
    pad = np.zeros((pad_len, actions.shape[1]), dtype=actions.dtype)
    return np.concatenate([actions, pad], axis=0)


def _ensure_uint8_image(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 0 or arr.size == 0:
        raise ValueError(f"Invalid image array: shape={arr.shape} dtype={arr.dtype}")
    if arr.dtype == object:
        try:
            arr = np.array(arr.tolist(), dtype=np.float32)
        except Exception:
            arr = np.array(arr, dtype=np.float32)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            scale = 255.0 if float(np.nanmax(arr)) <= 1.0 else 1.0
            arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _pick_obs_image(obs: dict) -> np.ndarray:
    for key in ("agentview_image", "robot0_eye_in_hand_image"):
        if key not in obs:
            continue
        img = obs[key]
        img = np.asarray(img)
        if img.size == 0:
            continue
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        if img.ndim == 2:
            img = img[:, :, None]
        if img.ndim != 3:
            continue
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
        return _ensure_uint8_image(img)
    keys = list(obs.keys())
    shapes = {k: getattr(np.asarray(obs[k]), "shape", None) for k in keys}
    raise ValueError(f"No valid image found in obs. keys={keys} shapes={shapes}")


def _write_image(path: Path, arr: np.ndarray) -> None:
    arr = _ensure_uint8_image(arr)
    imageio_v2.imwrite(str(path), arr)


def _pick_agentview_image(obs: dict) -> np.ndarray:
    if "agentview_image" not in obs:
        raise KeyError("Observation does not contain 'agentview_image'")
    img = obs["agentview_image"]
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = img[:, :, None]
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    return img


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
        cropped = img[top:top + new_h, left:left + new_w]
        img = np.array(Image.fromarray(cropped).resize((w, h), Image.BICUBIC))
    return img


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


def main() -> None:
    parser = argparse.ArgumentParser(description="VLA+Cosmos gated rollout in simulator")
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
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable writing the simulator rollout video (frames still saved).",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="CUDA device id (e.g. 7). If not set, uses default GPU.",
    )

    parser.add_argument("--resolution", type=str, default="256,320")
    parser.add_argument("--guidance", type=float, default=0.0)
    parser.add_argument("--cosmos-action-chunk-size", type=int, default=12)
    parser.add_argument("--cosmos-action-scale", type=float, default=20.0)
    parser.add_argument("--cosmos-gripper-scale", type=float, default=1.0)
    parser.add_argument("--use-normalized-actions", action="store_true")
    parser.add_argument(
        "--vla-platform",
        choices=("libero", "aloha", "bridge"),
        default="libero",
        help="Force OpenVLA constants selection (controls NUM_ACTIONS_CHUNK).",
    )
    parser.add_argument("--cosmos-root", type=Path, default=Path("/workspace"))
    parser.add_argument("--enable-cosmos", action="store_true", help="Enable Cosmos preview gating.")
    parser.add_argument("--cosmos-experiment", default=None)
    parser.add_argument("--cosmos-checkpoint-path", default=None)
    parser.add_argument("--cosmos-model-key", default=None)
    parser.add_argument(
        "--cosmos-config-file",
        default="cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py",
    )
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument(
        "--negative-prompt-file",
        type=Path,
        default=Path("/workspace/assets/action_conditioned/openvla-coffee/inference_params.json"),
        help="Optional JSON file to load negative_prompt from (overrides --negative-prompt).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.device is not None:
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this script.")

    # Force OpenVLA to pick the desired platform constants based on argv scanning.
    argv_lower = " ".join(sys.argv).lower()
    if args.vla_platform and args.vla_platform not in argv_lower:
        sys.argv.append(args.vla_platform)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.max_steps is None:
        if args.task == "coffee":
            args.max_steps = 256
        elif args.task == "stack_three":
            args.max_steps = 320
        elif args.task == "three_piece_assembly":
            args.max_steps = 384
        elif args.task == "square":
            args.max_steps = 184
        else:
            args.max_steps = 256

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

    if args.negative_prompt_file and args.negative_prompt_file.exists():
        try:
            neg_data = json.loads(args.negative_prompt_file.read_text(encoding="utf-8"))
            if isinstance(neg_data, dict) and neg_data.get("negative_prompt"):
                args.negative_prompt = neg_data["negative_prompt"]
        except Exception as exc:
            print(f"[WARN] failed to read negative_prompt from {args.negative_prompt_file}: {exc}")

    cosmos_infer = None
    if args.enable_cosmos:
        if not args.cosmos_experiment:
            raise ValueError("--cosmos-experiment is required when --enable-cosmos is set.")
        cosmos_infer = _load_cosmos_infer(
            args.cosmos_root,
            args.cosmos_experiment,
            args.cosmos_checkpoint_path,
            args.cosmos_model_key,
            args.cosmos_config_file,
        )

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
        frame_inputs_dir.mkdir(parents=True, exist_ok=True)
        sim_inputs_dir = ep_dir / "sim_inputs"
        sim_inputs_dir.mkdir(parents=True, exist_ok=True)
        vla_dir = ep_dir / "vla_actions"
        vla_dir.mkdir(parents=True, exist_ok=True)
        cosmos_dir = ep_dir / "cosmos_actions"
        cosmos_dir.mkdir(parents=True, exist_ok=True)
        cosmos_frames_dir = ep_dir / "cosmos_frames"
        cosmos_frames_dir.mkdir(parents=True, exist_ok=True)
        sim_frames_dir = ep_dir / "sim_frames"
        sim_frames_dir.mkdir(parents=True, exist_ok=True)

        if reset_states is not None and len(reset_states) > 0:
            obs = env.reset_to(reset_states[ep % len(reset_states)])
        else:
            obs = env.reset()
        for _ in range(10):
            obs, _, _, _ = env.step(np.zeros(7))

        video_path = ep_dir / f"{args.task}_vla_cosmos_sim_rollout.mp4"
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
            print(
                f"[Episode {ep:02d}][Chunk {chunk_idx:03d}] start (total_steps={total_steps})",
                flush=True,
            )

            current_sim_frame = _pick_agent_image(obs, crop_ratio=0.0)
            vla_image = current_sim_frame
            current_frame = vla_image

            sim_input_path = sim_inputs_dir / f"chunk_{chunk_idx:03d}_input.png"
            _write_image(sim_input_path, current_sim_frame)
            _write_image(frame_inputs_dir / f"chunk_{chunk_idx:03d}_input.png", current_frame)

            attempt = 0
            accepted = False
            actions_for_sim = None
            actions_for_cosmos = None
            video_clamped = None

            while not accepted:
                print(f"  [Attempt {attempt:02d}] sampling VLA actions", flush=True)
                if args.enable_cosmos:
                    attempt_seed = args.seed + chunk_idx * 1000 + attempt + ep * 10000
                    torch.manual_seed(attempt_seed)
                    np.random.seed(attempt_seed)

                policy_obs = {"full_image": vla_image, "task_description": task_description}
                actions_chunk = get_vla_action(
                    cfg_ns,
                    vla,
                    processor,
                    policy_obs,
                    task_description,
                    action_head,
                    None,
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
                try:
                    sim_min = float(np.min(actions_for_sim))
                    sim_max = float(np.max(actions_for_sim))
                    sim_mean = float(np.mean(actions_for_sim))
                    grip = actions_for_sim[:, 6]
                    grip_min = float(np.min(grip))
                    grip_max = float(np.max(grip))
                    grip_mean = float(np.mean(grip))
                    print(
                        f"  [Chunk {chunk_idx:03d}] actions_for_sim stats: "
                        f"min={sim_min:.6f} max={sim_max:.6f} mean={sim_mean:.6f} "
                        f"grip(min={grip_min:.6f} max={grip_max:.6f} mean={grip_mean:.6f})",
                        flush=True,
                    )
                except Exception as exc:
                    print(f"  [WARN] failed to log actions_for_sim stats: {exc}", flush=True)
                (vla_dir / f"chunk_{chunk_idx:03d}_attempt_{attempt:02d}.json").write_text(
                    json.dumps(actions_for_sim.astype(float).tolist(), ensure_ascii=True),
                    encoding="utf-8",
                )

                if args.enable_cosmos:
                    actions_for_cosmos = _adapt_actions_to_cosmos(
                        raw_actions,
                        target_chunk_size=args.cosmos_action_chunk_size,
                        action_scale=args.cosmos_action_scale,
                        gripper_scale=args.cosmos_gripper_scale,
                    )
                    (cosmos_dir / f"chunk_{chunk_idx:03d}_attempt_{attempt:02d}.json").write_text(
                        json.dumps(actions_for_cosmos.astype(float).tolist(), ensure_ascii=True),
                        encoding="utf-8",
                    )

                    img_tensor = torch.from_numpy(current_frame).permute(2, 0, 1).float() / 255.0
                    img_tensor = img_tensor.unsqueeze(0)
                    num_video_frames = actions_for_cosmos.shape[0] + 1
                    vid_input = torch.cat(
                        [img_tensor, torch.zeros_like(img_tensor).repeat(num_video_frames - 1, 1, 1, 1)],
                        dim=0,
                    )
                    vid_input = (vid_input * 255.0).to(torch.uint8)
                    vid_input = vid_input.unsqueeze(0).permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)

                    print(f"  [Attempt {attempt:02d}] running Cosmos preview", flush=True)
                    video = cosmos_infer.generate_vid2world(
                        prompt="",
                        input_path=vid_input,
                        action=torch.from_numpy(actions_for_cosmos).float(),
                        guidance=args.guidance,
                        num_video_frames=num_video_frames,
                        num_latent_conditional_frames=1,
                        resolution=args.resolution,
                        seed=attempt_seed,
                        negative_prompt=args.negative_prompt,
                    )

                    video_normalized = (video - (-1)) / (1 - (-1))
                    video_normalized = torch.clamp(video_normalized, 0, 1)
                    video_clamped = (video_normalized[0] * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()

                    chunk_frame_dir = cosmos_frames_dir / f"chunk_{chunk_idx:03d}" / f"attempt_{attempt:02d}"
                    chunk_frame_dir.mkdir(parents=True, exist_ok=True)
                    for f_idx, frame in enumerate(video_clamped):
                        mediapy.write_image(str(chunk_frame_dir / f"frame_{f_idx:03d}.png"), frame)

                    accepted = _judge_cosmos_sequence(video_clamped)
                else:
                    accepted = True
                print(f"  [Attempt {attempt:02d}] judge result: {accepted}", flush=True)
                attempt += 1
                if not args.enable_cosmos:
                    break

            if actions_for_sim is None:
                raise RuntimeError("No valid actions generated for simulator")

            for step_idx, action in enumerate(actions_for_sim):
                if total_steps >= args.max_steps:
                    break

                log_entry = {
                    "chunk": chunk_idx,
                    "step": total_steps,
                    "action": action.tolist(),
                    "attempts": attempt,
                }
                log_fp.write(json.dumps(log_entry) + "\n")
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
        summary_path = ep_dir / f"{args.task}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

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
    with open(args.output_dir / f"{args.task}_overall_summary.json", "w") as f:
        json.dump(overall_summary, f, indent=2)

    try:
        if hasattr(env, "close"):
            env.close()
        if hasattr(env, "env") and hasattr(env.env, "close"):
            env.env.close()
    except Exception as exc:
        print(f"[WARN] Error closing environment: {exc}")


if __name__ == "__main__":
    main()
