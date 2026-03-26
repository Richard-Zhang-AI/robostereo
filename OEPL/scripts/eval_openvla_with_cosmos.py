#!/usr/bin/env python3
"""
Evaluate OpenVLA in robomimic/mimicgen env with Cosmos preview gating.

This is a separate entrypoint to keep dependencies/openvla-oft/eval.py unchanged.
The core rollout logic mirrors eval.py, but adds a Cosmos preview before each
action chunk is executed in the simulator.
"""
import atexit
import argparse
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
os.environ.setdefault("EGL_DEVICE_ID", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Avoid EGL teardown crashes on exit.
atexit.register(lambda: os._exit(0))

import numpy as np
import torch
import mediapy

from experiments.robot.openvla_utils import (
    get_processor,
    get_vla,
    get_vla_action,
)

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.config import config_factory
import mimicgen.envs.robosuite  # noqa: F401

from PIL import Image

from vla_sim_cosmos_rollout import (
    _adapt_actions_to_cosmos,
    _load_cosmos_infer,
)

# Resolve WMPO root for container/local portability.
_ENV_ROOT = os.environ.get("WMPO_ROOT")
if _ENV_ROOT:
    _WMPO_ROOT = Path(_ENV_ROOT).resolve()
elif Path("/workspace/WMPO").exists():
    _WMPO_ROOT = Path("/workspace/WMPO").resolve()
else:
    _WMPO_ROOT = Path(__file__).resolve().parents[1]


# -----------------------------
# Env helpers
# -----------------------------

def _create_env(cfg, env_config_path: str):
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=cfg.train.data)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=cfg.train.data,
        all_obs_keys=cfg.all_obs_keys,
        verbose=False,
    )
    if cfg.experiment.env is not None:
        env_meta["env_name"] = cfg.experiment.env
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=env_meta["env_name"],
        render=False,
        render_offscreen=True,
        use_image_obs=shape_meta["use_images"],
        use_depth_obs=shape_meta["use_depths"],
    )
    return EnvUtils.wrap_env_from_config(env, config=cfg)


def _pick_agent_image(
    obs: Dict[str, Any],
    crop_ratio: float = 0.0,
) -> np.ndarray:
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
    return True


def _quat_xyzw_to_euler(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = quat_xyzw
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = max(-1.0, min(1.0, t2))
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return np.array([roll, pitch, yaw], dtype=np.float32)


def _extract_state(obs: Dict[str, Any]) -> tuple[np.ndarray, float]:
    pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)
    quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32).reshape(-1)
    rpy = _quat_xyzw_to_euler(quat)
    gripper = float(np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)[0])
    state = np.concatenate([pos, rpy], axis=0)
    return state, gripper


def _normalize_gripper_value(value: float, gmin: float, gmax: float, open_ref: float) -> tuple[float, float, float]:
    gmin = min(gmin, value)
    gmax = max(gmax, value)
    if gmax - gmin < 1e-6:
        norm = 0.0
    else:
        norm = (value - gmin) / (gmax - gmin)
        if open_ref > (gmin + gmax) * 0.5:
            norm = 1.0 - norm
    norm = float(np.clip(norm, 0.0, 1.0))
    return norm, gmin, gmax


def _pad_or_trim(actions: np.ndarray, target_len: int) -> np.ndarray:
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    if actions.shape[0] >= target_len:
        return actions[:target_len]
    pad_len = target_len - actions.shape[0]
    pad = np.zeros((pad_len, actions.shape[1]), dtype=actions.dtype)
    return np.concatenate([actions, pad], axis=0)


def _save_cosmos_video(out_path: Path, frames: np.ndarray, fps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(str(out_path), frames, fps=fps)


def _snapshot_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        try:
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            state["torch_cuda"] = None
    return state


def _restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None:
        try:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
        except Exception:
            pass


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="three_piece_assembly")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--env-config", type=str, default=None)
    parser.add_argument("--reset-states", type=str, default=None)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_video", default=True)
    parser.add_argument("--video_dir", type=str, default="./debug/dpo")
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument(
        "--action-log-dir",
        type=str,
        default=str(_WMPO_ROOT / "debug" / "sim_vla_cosmos"),
        help="Directory to save executed simulator actions (per-episode JSONL).",
    )
    parser.add_argument(
        "--print-actions",
        action="store_true",
        help="Print every executed simulator action to stdout.",
    )
    parser.add_argument(
        "--modify-material",
        action="store_true",
        help="Apply material modifications to reset states (disabled by default to match eval.py).",
    )

    # Cosmos args
    parser.add_argument("--cosmos-root", type=Path, default=Path("/workspace"))
    parser.add_argument(
        "--enable-cosmos",
        action="store_true",
        help="Enable Cosmos preview gating (default: disabled).",
    )
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
    parser.add_argument("--resolution", type=str, default="256,320")
    parser.add_argument("--guidance", type=float, default=0.0)
    parser.add_argument("--cosmos-action-chunk-size", type=int, default=12)
    parser.add_argument("--cosmos-action-scale", type=float, default=20.0)
    parser.add_argument("--cosmos-gripper-scale", type=float, default=1.0)
    parser.add_argument("--cosmos-fps", type=int, default=14)
    parser.add_argument(
        "--cosmos-no-isolate-rng",
        action="store_true",
        help="Disable RNG isolation for Cosmos (not recommended).",
    )
    parser.add_argument(
        "--vla-platform",
        choices=("libero", "aloha", "bridge"),
        default="libero",
        help="Force OpenVLA constants selection (controls NUM_ACTIONS_CHUNK).",
    )
    args = parser.parse_args()

    if not args.enable_cosmos:
        # Delegate to the original eval.py to guarantee identical behavior.
        import subprocess
        import sys as _sys
        eval_py = Path(__file__).resolve().parents[1] / "dependencies" / "openvla-oft" / "eval.py"
        strip_flags = {
            "--enable-cosmos",
            "--cosmos-root",
            "--cosmos-experiment",
            "--cosmos-checkpoint-path",
            "--cosmos-model-key",
            "--cosmos-config-file",
            "--negative-prompt",
            "--negative-prompt-file",
            "--resolution",
            "--guidance",
            "--cosmos-action-chunk-size",
            "--cosmos-action-scale",
            "--cosmos-gripper-scale",
            "--cosmos-fps",
            "--cosmos-no-isolate-rng",
            "--vla-platform",
        }
        passthrough = []
        it = iter(_sys.argv[1:])
        for token in it:
            if token in strip_flags:
                # Skip value if it has one.
                if token in {
                    "--cosmos-root",
                    "--cosmos-experiment",
                    "--cosmos-checkpoint-path",
                    "--cosmos-model-key",
                    "--cosmos-config-file",
                    "--negative-prompt",
                    "--negative-prompt-file",
                    "--resolution",
                    "--guidance",
                    "--cosmos-action-chunk-size",
                    "--cosmos-action-scale",
                    "--cosmos-gripper-scale",
                    "--cosmos-fps",
                    "--vla-platform",
                }:
                    try:
                        next(it)
                    except StopIteration:
                        break
                continue
            passthrough.append(token)
        cmd = [_sys.executable, str(eval_py), *passthrough]
        raise SystemExit(subprocess.call(cmd))

    if args.env_config is None:
        args.env_config = str(
            _WMPO_ROOT
            / "data_files"
            / "core_train_configs"
            / f"bc_rnn_image_ds_{args.task}_D0_seed_101.json"
        )
    if args.task == "square":
        args.task_description = "Insert the square into the stick"
    else:
        args.task_description = args.task
        if args.ckpt is None and args.task == "coffee":
            args.ckpt = str(_WMPO_ROOT / "checkpoint_files" / "SFT_models" / "coffee")

    if args.ckpt is None:
        raise ValueError("--ckpt is required for this task in the local environment.")

    args.task_description = args.task
    if args.task == "coffee":
        args.max_steps = 256
    elif args.task == "stack_three":
        args.max_steps = 320
    elif args.task == "three_piece_assembly":
        args.max_steps = 384
    elif args.task == "square":
        args.max_steps = 184

    # Force OpenVLA to pick the desired platform constants based on argv scanning.
    argv_lower = " ".join(os.sys.argv).lower()
    if args.vla_platform and args.vla_platform not in argv_lower:
        os.sys.argv.append(args.vla_platform)

    # --- Load env config (robomimic) ---
    env_config_path = Path(args.env_config)
    ext_cfg = json.load(open(env_config_path, "r"))
    cfg = config_factory(ext_cfg["algo_name"])
    with cfg.values_unlocked():
        cfg.update(ext_cfg)
        dataset_path = Path(cfg.train.data)
        if not dataset_path.is_absolute():
            candidates = [
                (_WMPO_ROOT / dataset_path).resolve(),
                (env_config_path.parent / dataset_path).resolve(),
                (env_config_path.parent.parent / dataset_path).resolve(),
                (Path(__file__).resolve().parents[1] / dataset_path).resolve(),
            ]
            for candidate in candidates:
                if candidate.exists():
                    cfg.train.data = str(candidate)
                    break
    cfg.lock()
    ObsUtils.initialize_obs_utils_with_config(cfg)

    # --- VLA stack ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    from types import SimpleNamespace
    cfg_ns = SimpleNamespace(
        pretrained_checkpoint=args.ckpt,
        center_crop=True,
        num_images_in_input=1,
        use_proprio=False,
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        load_in_8bit=False,
        load_in_4bit=False,
        num_open_loop_steps=args.chunk,
        unnorm_key=f"{args.task}_d0_300_demos",
    )

    print(f"[INFO] Using checkpoint: {args.ckpt}")
    print(f"[INFO] Building VLA from: {cfg_ns.pretrained_checkpoint}")

    vla = get_vla(cfg_ns).to(device).eval()
    processor = get_processor(cfg_ns)
    action_head = None

    # --- Cosmos ---
    cosmos_infer = None
    if args.enable_cosmos:
        if not args.cosmos_experiment:
            raise ValueError("--cosmos-experiment is required when --enable-cosmos is set.")
        # Isolate RNG so Cosmos init does not perturb VLA sampling.
        cosmos_init_rng = None
        if not args.cosmos_no_isolate_rng:
            cosmos_init_rng = _snapshot_rng_state()
            random.seed(args.seed + 999)
            np.random.seed(args.seed + 999)
            torch.manual_seed(args.seed + 999)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed + 999)
        if args.negative_prompt_file and args.negative_prompt_file.exists():
            try:
                neg_data = json.loads(args.negative_prompt_file.read_text(encoding="utf-8"))
                if isinstance(neg_data, dict) and neg_data.get("negative_prompt"):
                    args.negative_prompt = neg_data["negative_prompt"]
            except Exception as exc:
                print(f"[WARN] failed to read negative_prompt from {args.negative_prompt_file}: {exc}")

        cosmos_infer = _load_cosmos_infer(
            args.cosmos_root,
            args.cosmos_experiment,
            args.cosmos_checkpoint_path,
            args.cosmos_model_key,
            args.cosmos_config_file,
        )
        if cosmos_init_rng is not None:
            _restore_rng_state(cosmos_init_rng)

    # --- Rollout episodes ---
    successes = 0
    lengths = []
    os.makedirs(args.video_dir, exist_ok=True)
    if args.log_file is None:
        args.log_file = os.path.join(args.video_dir, f"{args.task}_eval_log.jsonl")

    import pickle
    if args.reset_states is None:
        args.reset_states = str(_WMPO_ROOT / "verl" / "utils" / "dataset" / f"{args.task}_d0_states.pkl")
    with open(args.reset_states, "rb") as f:
        reset_states = pickle.load(f)

    import xml.etree.ElementTree as ET

    def add_material_look(xml_str: str) -> str:
        root = ET.fromstring(xml_str)
        for m in root.findall(".//asset/material"):
            if m.get("name") in ("base_redwood_mat",):
                m.set("rgba", "0.3 0.8 0.3 1")
                m.set("specular", "0.1")
                m.set("shininess", "0.05")
                m.set("texuniform", "true")
                m.set("texrepeat", "2 2")
                m.set("reflectance", "0.1")
        for g in root.findall(".//worldbody//geom"):
            n = g.get("name", "")
            if n.startswith("base_") and not n.endswith("_vis"):
                g.set("rgba", "0 0 0 0")
        return ET.tostring(root, encoding="unicode")

    modified_states = []
    for state in reset_states:
        if args.modify_material:
            state["model"] = add_material_look(state["model"])
        modified_states.append(state)

    env = _create_env(cfg, str(env_config_path))
    log_fp = open(args.log_file, "a", encoding="utf-8")
    action_log_root = Path(args.action_log_dir)
    action_log_root.mkdir(parents=True, exist_ok=True)

    cosmos_root_out = Path(args.video_dir) / "cosmos_previews"
    for ep in range(args.runs):
        # Match eval.py behavior: use reset_states as-is unless --modify-material is set.
        obs = env.reset_to(modified_states[ep % len(modified_states)])
        for _ in range(10):
            obs, reward, done, info = env.step(np.zeros(7))

        frames = []
        ep_len = 0
        success = False

        gripper_open_ref = float(np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).mean())
        gripper_min = gripper_open_ref
        gripper_max = gripper_open_ref

        if args.enable_cosmos:
            ep_cosmos_dir = cosmos_root_out / f"episode_{ep:02d}"
            cosmos_frames_dir = ep_cosmos_dir / "frames"
            cosmos_videos_dir = ep_cosmos_dir / "videos"
            cosmos_frames_dir.mkdir(parents=True, exist_ok=True)
            cosmos_videos_dir.mkdir(parents=True, exist_ok=True)
        ep_action_log = action_log_root / f"episode_{ep:02d}_actions.jsonl"
        ep_action_fp = open(ep_action_log, "w", encoding="utf-8")

        chunk_idx = 0
        while ep_len < args.max_steps:
            full_img = _pick_agent_image(obs)
            policy_obs = {"full_image": full_img, "task_description": args.task_description}

            attempt = 0
            accepted = False
            actions_for_sim = None
            raw_actions_for_log = None
            raw_actions_len = None

            while not accepted:
                if args.enable_cosmos and attempt > 0:
                    attempt_seed = args.seed + ep * 10000 + chunk_idx * 10 + attempt
                    torch.manual_seed(attempt_seed)
                    np.random.seed(attempt_seed)

                actions = get_vla_action(
                    cfg_ns,
                    vla,
                    processor,
                    policy_obs,
                    policy_obs["task_description"],
                    action_head,
                    None,
                )

                raw_actions = np.asarray(actions)
                if raw_actions.ndim == 1:
                    raw_actions = raw_actions.reshape(1, -1)
                if raw_actions.shape[1] != 7:
                    raise RuntimeError(f"Expected action dim 7, got {raw_actions.shape}")

                if args.enable_cosmos:
                    # -------------------------
                    # HARD ISOLATION (critical)
                    # -------------------------
                    raw_actions = raw_actions.astype(np.float32, copy=True)
                    actions_for_sim = _pad_or_trim(raw_actions, args.chunk).astype(np.float32, copy=True)
                    raw_for_cosmos = raw_actions.astype(np.float32, copy=True)
                else:
                    # Match eval.py behavior when Cosmos is disabled.
                    actions_for_sim = _pad_or_trim(raw_actions, args.chunk)

                raw_actions_for_log = raw_actions
                raw_actions_len = int(raw_actions.shape[0])

                if args.enable_cosmos:
                    # Cosmos preview
                    cosmos_input = _resize_for_cosmos(full_img, args.resolution)
                    raw_before = raw_for_cosmos.copy()
                    actions_for_cosmos = _adapt_actions_to_cosmos(
                        raw_for_cosmos,
                        target_chunk_size=args.cosmos_action_chunk_size,
                        action_scale=args.cosmos_action_scale,
                        gripper_scale=args.cosmos_gripper_scale,
                    )
                    if not np.allclose(raw_for_cosmos, raw_before):
                        print(
                            "[WARN] _adapt_actions_to_cosmos mutated its input in-place "
                            "(OK because we passed a copy).",
                            flush=True,
                        )

                    img_tensor = torch.from_numpy(cosmos_input).permute(2, 0, 1).float() / 255.0
                    img_tensor = img_tensor.unsqueeze(0)
                    num_video_frames = actions_for_cosmos.shape[0] + 1
                    vid_input = torch.cat(
                        [img_tensor, torch.zeros_like(img_tensor).repeat(num_video_frames - 1, 1, 1, 1)],
                        dim=0,
                    )
                    vid_input = (vid_input * 255.0).to(torch.uint8)
                    vid_input = vid_input.unsqueeze(0).permute(0, 2, 1, 3, 4)

                    seed = args.seed + ep * 10000 + chunk_idx * 10 + attempt
                    rng_state = None
                    if not args.cosmos_no_isolate_rng:
                        rng_state = _snapshot_rng_state()
                        random.seed(seed)
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(seed)
                    video = cosmos_infer.generate_vid2world(
                        prompt="",
                        input_path=vid_input,
                        action=torch.from_numpy(actions_for_cosmos).float(),
                        guidance=args.guidance,
                        num_video_frames=num_video_frames,
                        num_latent_conditional_frames=1,
                        resolution=args.resolution,
                        seed=seed,
                        negative_prompt=args.negative_prompt,
                    )
                    if rng_state is not None:
                        _restore_rng_state(rng_state)

                    video_normalized = (video - (-1)) / (1 - (-1))
                    video_normalized = torch.clamp(video_normalized, 0, 1)
                    video_clamped = (
                        (video_normalized[0] * 255)
                        .to(torch.uint8)
                        .permute(1, 2, 3, 0)
                        .cpu()
                        .numpy()
                    )

                    chunk_dir = cosmos_frames_dir / f"chunk_{chunk_idx:03d}" / f"attempt_{attempt:02d}"
                    chunk_dir.mkdir(parents=True, exist_ok=True)
                    for f_idx, frame in enumerate(video_clamped):
                        mediapy.write_image(str(chunk_dir / f"frame_{f_idx:03d}.png"), frame)

                    preview_path = cosmos_videos_dir / f"chunk_{chunk_idx:03d}_attempt_{attempt:02d}.mp4"
                    _save_cosmos_video(preview_path, video_clamped, fps=args.cosmos_fps)

                    accepted = _judge_cosmos_sequence(video_clamped)
                else:
                    accepted = True
                attempt += 1

            if actions_for_sim is None:
                raise RuntimeError("No valid actions generated for simulator")
            if raw_actions_for_log is None:
                raise RuntimeError("No raw actions available for logging")

            state_vec, gripper_val = _extract_state(obs)
            eef_pos = np.array(obs.get("robot0_eef_pos"), dtype=np.float32)
            eef_quat = np.array(obs.get("robot0_eef_quat"), dtype=np.float32)
            gripper_norm, gripper_min, gripper_max = _normalize_gripper_value(
                gripper_val, gripper_min, gripper_max, gripper_open_ref
            )
            log_fp.write(
                json.dumps(
                    {
                        "type": "chunk",
                        "episode": ep,
                        "step": ep_len,
                        "state": state_vec.tolist(),
                        "continuous_gripper_state": gripper_norm,
                        "gripper_raw": gripper_val,
                        "eef_pos": eef_pos.tolist(),
                        "eef_quat": eef_quat.tolist(),
                        "actions": raw_actions_for_log.tolist(),
                        "actions_raw_len": raw_actions_len,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
            log_fp.flush()

            for action in actions_for_sim:
                eef_pos_step = np.array(obs.get("robot0_eef_pos"), dtype=np.float32)
                eef_quat_step = np.array(obs.get("robot0_eef_quat"), dtype=np.float32)
                state_step, gripper_raw = _extract_state(obs)
                gripper_norm, gripper_min, gripper_max = _normalize_gripper_value(
                    gripper_raw, gripper_min, gripper_max, gripper_open_ref
                )
                action_list = np.array(action, dtype=np.float32).tolist()
                log_fp.write(
                    json.dumps(
                        {
                            "type": "step",
                            "episode": ep,
                            "step": ep_len,
                            "state": state_step.tolist(),
                            "continuous_gripper_state": gripper_norm,
                            "gripper_raw": gripper_raw,
                            "eef_pos": eef_pos_step.tolist(),
                            "eef_quat": eef_quat_step.tolist(),
                            "action": action_list,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
                log_fp.flush()
                ep_action_fp.write(
                    json.dumps(
                        {
                            "episode": ep,
                            "step": ep_len,
                            "chunk": chunk_idx,
                            "actions_raw_len": raw_actions_len,
                            "action": action_list,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
                ep_action_fp.flush()
                if args.print_actions:
                    print(f"[Exec][Ep {ep:02d}][Step {ep_len:04d}] action={action_list}", flush=True)

                if args.save_video:
                    frames.append(_pick_agent_image(obs))
                obs, reward, done, info = env.step(action)
                if reward > 0.0 or done:
                    if done:
                        assert False, "Done but not success"
                    success = True
                ep_len += 1
                if done or ep_len >= args.max_steps:
                    break

            if done or ep_len >= args.max_steps:
                break
            chunk_idx += 1

        if args.save_video and len(frames) > 0:
            frames.append(_pick_agent_image(obs))
            out_path = os.path.join(args.video_dir, f"{args.task}_rollout_{ep+1:02d}_{success}.mp4")
            mediapy.write_video(out_path, np.stack(frames, axis=0), fps=20)
            print(f"Saved video -> {out_path}")

        lengths.append(ep_len)
        successes += int(success)
        print(f"[Episode {ep+1:02d}] len={ep_len}, success={success}")
        try:
            ep_action_fp.close()
        except Exception:
            pass

    try:
        if hasattr(env, "close"):
            env.close()
        if hasattr(env, "env") and hasattr(env.env, "close"):
            env.env.close()
        log_fp.close()
    except Exception:
        pass

    success_rate = successes / max(1, args.runs)
    print("\n=============================")
    print(f"Episodes: {args.runs}")
    print(f"Successes: {successes}")
    print(f"Success Rate: {success_rate*100:.1f}%")
    print(f"Avg Length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")

    out_json = os.path.join(args.video_dir, f"openvla_{args.task}_eval.json")
    with open(out_json, "w") as f:
        json.dump(
            {
                "episodes": args.runs,
                "successes": successes,
                "success_rate": success_rate,
                "avg_len": float(np.mean(lengths)) if lengths else 0.0,
                "std_len": float(np.std(lengths)) if lengths else 0.0,
                "chunk": args.chunk,
                "max_steps": args.max_steps,
                "ckpt": args.ckpt,
                "env_config": args.env_config,
                "task": args.task,
            },
            f,
            indent=2,
        )
    print(f"Saved summary -> {out_json}")
    env.env.close()


if __name__ == "__main__":
    main()
