#!/usr/bin/env python3
"""
Bridge VLA-RFT's OpenVLA (flow-matching VLA-Adapter) <-> Cosmos world model.

This is the VLA-RFT variant of WMPO's vla_cosmos_bridge.py.
It uses the VLA-Adapter model from VLA-RFT (with flow-matching action heads)
instead of the standard OpenVLA from WMPO.

Loop:
  1) VLA-Adapter predicts an action chunk via flow-matching
  2) Resample/pad actions to Cosmos chunk size (e.g. 12)
  3) Cosmos generates the next video chunk
  4) Feed the last frame back to VLA and repeat

Usage:
    python iepl_cosmos/vla_cosmos_bridge_rft.py \
        --vla-path checkpoints/libero/Base/object \
        --task "pick up the red bowl" \
        --initial-image data/example_initial.png \
        --output-dir debug/cosmos_rft_out \
        --num-chunks 25 \
        --cosmos-experiment ac_reason_embeddings_rectified_flow_2b_256_320 \
        --cosmos-checkpoint-path /path/to/cosmos/model_ema_bf16.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import mediapy
import numpy as np
import torch
import torchvision
from PIL import Image


def _add_cosmos_to_path(cosmos_root: Path) -> None:
    cosmos_root = cosmos_root.resolve()
    if str(cosmos_root) not in sys.path:
        sys.path.insert(0, str(cosmos_root))
    env_root = os.environ.get('COSMOS_ROOT')
    if env_root and env_root not in sys.path:
        sys.path.insert(0, str(Path(env_root).resolve()))


def _add_vla_rft_to_path() -> None:
    """Add VLA-RFT's openvla-oft to sys.path for model loading."""
    import os
    rft_root = Path(__file__).resolve().parent.parent
    oft_path = rft_root / "train" / "verl" / "vla-adapter" / "openvla-oft"
    if oft_path.exists() and str(oft_path) not in sys.path:
        sys.path.insert(0, str(oft_path))
    verl_path = rft_root / "train" / "verl"
    if verl_path.exists() and str(verl_path) not in sys.path:
        sys.path.insert(0, str(verl_path))


def _load_vla_rft(ckpt_path: Path, cfg_path: Path | None, device: torch.device):
    """
    Load VLA-Adapter model from VLA-RFT checkpoint.
    
    This differs from WMPO's _load_vla which uses standard OpenVLA.
    VLA-RFT uses OpenVLA-OFT with flow-matching action heads.
    """
    _add_vla_rft_to_path()
    from prismatic.models.load import load_vla

    ckpt_path = ckpt_path.resolve()
    cfg_path = cfg_path.resolve() if cfg_path else ckpt_path / "config.json"

    model = load_vla(
        str(ckpt_path),
        load_for_training=False,
    )
    model = model.to(device).eval()
    return model


def _load_vla_processor(ckpt_path: Path):
    """Load processor for VLA-Adapter."""
    _add_vla_rft_to_path()
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(ckpt_path), trust_remote_code=True)
    return processor


@torch.no_grad()
def _generate_action_chunk_rft(
    vla,
    processor,
    image_np: np.ndarray,
    prompt: str,
    device: torch.device,
    temperature: float = 1.0,
    use_flow_matching: bool = True,
) -> np.ndarray:
    """
    Generate an action chunk using VLA-RFT's flow-matching model.
    
    Returns:
        actions: (num_steps, action_dim) numpy array
    """
    image = Image.fromarray(image_np).convert("RGB")
    batch_feature = processor(prompt, image)

    input_ids = batch_feature["input_ids"].to(device)
    attention_mask = batch_feature["attention_mask"].to(device)
    pixel_values = batch_feature["pixel_values"].to(device)

    if input_ids.dim() == 3 and input_ids.shape[-1] == 1:
        input_ids = input_ids.squeeze(-1)
    if attention_mask.dim() == 3 and attention_mask.shape[-1] == 1:
        attention_mask = attention_mask.squeeze(-1)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if hasattr(vla, 'predict_action'):
            actions = vla.predict_action(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
            )
        elif hasattr(vla, 'generate_action_verl'):
            actions, _, _ = vla.generate_action_verl(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                padding_idx=processor.tokenizer.pad_token_id,
                do_sample=True,
                temperature=temperature,
            )
        else:
            raise AttributeError(
                "VLA model has neither 'predict_action' nor 'generate_action_verl'. "
                "Check that VLA-RFT's model is loaded correctly."
            )

    if isinstance(actions, torch.Tensor):
        actions = actions.detach().cpu().numpy()
    if actions.ndim == 3:
        actions = actions[0]

    return actions


def _pad_actions(actions: np.ndarray, target_len: int) -> np.ndarray:
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    if actions.shape[0] >= target_len:
        return actions[:target_len]
    pad = np.tile(actions[-1:], (target_len - actions.shape[0], 1))
    return np.concatenate([actions, pad], axis=0)


def _adapt_actions_to_cosmos(
    actions_chunk: np.ndarray,
    target_chunk_size: int,
    action_scale: float,
    gripper_scale: float,
) -> np.ndarray:
    actions_chunk = actions_chunk.astype(np.float32, copy=False)
    actions_chunk = _pad_actions(actions_chunk, target_chunk_size)
    if actions_chunk.shape[1] >= 7:
        actions_chunk[:, :6] = actions_chunk[:, :6] * action_scale
        actions_chunk[:, 6] = actions_chunk[:, 6] * gripper_scale
    return actions_chunk


def _print_stats(tag: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr)
    if arr.size == 0:
        print(f"[{tag}] empty")
        return
    print(
        f"[{tag}] min={float(np.nanmin(arr)):.6f} "
        f"max={float(np.nanmax(arr)):.6f} "
        f"mean={float(np.nanmean(arr)):.6f} "
        f"shape={arr.shape}"
    )


def _load_cosmos_infer(cosmos_root, experiment, checkpoint_path, config_file):
    _add_cosmos_to_path(cosmos_root)
    from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference

    config_path = Path(config_file)
    if not config_path.is_absolute():
        config_path = cosmos_root / config_path
    if config_path.suffix == ".py":
        if not config_path.exists():
            raise FileNotFoundError(f"Cosmos config not found: {config_path}")
        rel_path = config_path.relative_to(cosmos_root.resolve())
        config_arg = str(rel_path)
    else:
        config_arg = config_file

    return Video2WorldInference(
        experiment_name=experiment,
        ckpt_path=str(checkpoint_path),
        s3_credential_path="",
        context_parallel_size=1,
        config_file=config_arg,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VLA-RFT <-> Cosmos World Model Bridge"
    )
    parser.add_argument("--vla-path", required=True, type=Path,
                        help="Path to VLA-RFT checkpoint")
    parser.add_argument("--vla-cfg-path", type=Path, default=None,
                        help="Path to VLA config.json (defaults to vla-path/config.json)")
    parser.add_argument("--task", required=True,
                        help="Task description for the VLA prompt")
    parser.add_argument("--initial-image", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-chunks", type=int, default=25)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--resolution", type=str, default="256,320")
    parser.add_argument("--guidance", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cosmos-action-chunk-size", type=int, default=12)
    parser.add_argument("--cosmos-action-scale", type=float, default=20.0)
    parser.add_argument("--cosmos-gripper-scale", type=float, default=1.0)
    parser.add_argument("--cosmos-root", type=Path, default=Path("/workspace"))
    parser.add_argument("--cosmos-experiment", required=True)
    parser.add_argument("--cosmos-checkpoint-path", required=True, type=Path)
    parser.add_argument(
        "--cosmos-config-file",
        default="cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py",
    )
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--negative-prompt-file", type=Path, default=None)
    parser.add_argument("--save-fps", type=int, default=14)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA required.")

    print("[1/3] Loading VLA-RFT model...")
    vla = _load_vla_rft(args.vla_path, args.vla_cfg_path, device)
    processor = _load_vla_processor(args.vla_path)

    print("[2/3] Loading Cosmos world model...")
    negative_prompt = args.negative_prompt
    if args.negative_prompt_file and args.negative_prompt_file.exists():
        try:
            neg_data = json.loads(args.negative_prompt_file.read_text(encoding="utf-8"))
            if isinstance(neg_data, dict) and neg_data.get("negative_prompt"):
                negative_prompt = neg_data["negative_prompt"]
        except Exception as exc:
            print(f"[WARN] failed to read negative_prompt: {exc}")

    cosmos_infer = _load_cosmos_infer(
        args.cosmos_root,
        args.cosmos_experiment,
        args.cosmos_checkpoint_path,
        args.cosmos_config_file,
    )

    print("[3/3] Starting VLA <-> Cosmos rollout loop...")
    prompt = f"In: What action should the robot take to {args.task.lower()}?\nOut:"
    current_frame = np.array(Image.open(args.initial_image).convert("RGB"))
    if args.resolution != "none":
        try:
            h, w = map(int, args.resolution.split(","))
            current_frame = np.array(
                Image.fromarray(current_frame).resize((w, h), Image.BICUBIC)
            )
        except Exception:
            pass

    all_frames = [current_frame]
    frame_dir = args.output_dir / "frame_inputs"
    vla_dir = args.output_dir / "vla_actions"
    cosmos_dir = args.output_dir / "cosmos_actions"
    cosmos_frames_dir = args.output_dir / "cosmos_frames"
    for d in [frame_dir, vla_dir, cosmos_dir, cosmos_frames_dir]:
        d.mkdir(parents=True, exist_ok=True)

    mediapy.write_image(str(frame_dir / "chunk_000_input.png"), current_frame)
    all_actions_log = []

    for chunk_idx in range(args.num_chunks):
        t0 = time.time()
        _print_stats(f"chunk {chunk_idx} input_frame", current_frame)

        raw_actions = _generate_action_chunk_rft(
            vla, processor, current_frame, prompt, device,
            temperature=args.temperature,
        )
        _print_stats(f"chunk {chunk_idx} vla_actions_raw", raw_actions)

        (vla_dir / f"chunk_{chunk_idx:03d}_vla_actions.json").write_text(
            json.dumps(raw_actions.astype(float).tolist()), encoding="utf-8"
        )
        all_actions_log.append(raw_actions.tolist())

        actions_cosmos = _adapt_actions_to_cosmos(
            raw_actions,
            target_chunk_size=args.cosmos_action_chunk_size,
            action_scale=args.cosmos_action_scale,
            gripper_scale=args.cosmos_gripper_scale,
        )
        _print_stats(f"chunk {chunk_idx} cosmos_actions", actions_cosmos)

        (cosmos_dir / f"chunk_{chunk_idx:03d}_cosmos_actions.json").write_text(
            json.dumps(actions_cosmos.astype(float).tolist()), encoding="utf-8"
        )

        img_tensor = torchvision.transforms.functional.to_tensor(current_frame).unsqueeze(0)
        num_video_frames = actions_cosmos.shape[0] + 1
        vid_input = torch.cat(
            [img_tensor, torch.zeros_like(img_tensor).repeat(num_video_frames - 1, 1, 1, 1)],
            dim=0,
        )
        vid_input = (vid_input * 255.0).to(torch.uint8)
        vid_input = vid_input.unsqueeze(0).permute(0, 2, 1, 3, 4)

        video = cosmos_infer.generate_vid2world(
            prompt="",
            input_path=vid_input,
            action=torch.from_numpy(actions_cosmos).float(),
            guidance=args.guidance,
            num_video_frames=num_video_frames,
            num_latent_conditional_frames=1,
            resolution=args.resolution,
            seed=args.seed + chunk_idx,
            negative_prompt=negative_prompt,
        )

        video_normalized = torch.clamp((video - (-1)) / 2.0, 0, 1)
        video_clamped = (
            (video_normalized[0] * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
        )

        chunk_frame_dir = cosmos_frames_dir / f"chunk_{chunk_idx:03d}"
        chunk_frame_dir.mkdir(parents=True, exist_ok=True)
        for f_idx, frame in enumerate(video_clamped):
            mediapy.write_image(str(chunk_frame_dir / f"frame_{f_idx:03d}.png"), frame)

        current_frame = video_clamped[-1]
        mediapy.write_image(
            str(frame_dir / f"chunk_{chunk_idx+1:03d}_input.png"), current_frame
        )
        all_frames.extend(video_clamped[1:])

        elapsed = time.time() - t0
        print(
            f"  chunk {chunk_idx}/{args.num_chunks} done in {elapsed:.1f}s, "
            f"generated {video_clamped.shape[0]} frames"
        )

    out_path = args.output_dir / "vla_cosmos_rollout.mp4"
    mediapy.write_video(str(out_path), np.stack(all_frames, axis=0), fps=args.save_fps)

    actions_log_path = args.output_dir / "all_actions.json"
    actions_log_path.write_text(json.dumps(all_actions_log), encoding="utf-8")

    print(f"\nSaved rollout video to: {out_path}")
    print(f"Saved actions log to: {actions_log_path}")
    print(f"Total frames: {len(all_frames)}")


if __name__ == "__main__":
    main()
