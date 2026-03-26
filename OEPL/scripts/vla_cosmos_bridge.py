#!/usr/bin/env python3
"""
Bridge OpenVLA -> Cosmos action-conditioned WM.

Loop:
  1) VLA predicts an action chunk (e.g. 8 steps)
  2) Resample to Cosmos chunk size (e.g. 12) via linear interpolation
  3) Cosmos generates the next video chunk
  4) Feed the last frame back to VLA and repeat
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
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


def _load_vla(vla_path: Path, device: torch.device):
    from transformers import AutoModelForVision2Seq, AutoProcessor

    vla_path = vla_path.resolve()
    checkpoint_root = None
    for parent in vla_path.parents:
        candidate = parent / "checkpoint_files"
        if candidate.exists():
            checkpoint_root = candidate
            break
    if checkpoint_root is not None:
        for filename in ("modeling_prismatic.py", "processing_prismatic.py", "preprocessor_config.json"):
            src = checkpoint_root / filename
            dst = vla_path / filename
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)

    dataset_stats_path = vla_path / "dataset_statistics.json"
    processor = AutoProcessor.from_pretrained(str(vla_path), trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        str(vla_path),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    if dataset_stats_path.exists():
        with dataset_stats_path.open("r", encoding="utf-8") as f:
            model.norm_stats = json.load(f)
    model.eval()
    return model, processor


def _prepare_vla_inputs(processor, image_np: np.ndarray, prompt: str, device: torch.device) -> dict:
    image = Image.fromarray(image_np).convert("RGB")
    batch_feature = processor(prompt, image)

    input_ids = batch_feature["input_ids"]
    attention_mask = batch_feature["attention_mask"]
    pixel_values = batch_feature["pixel_values"]

    if input_ids.dim() == 3 and input_ids.shape[-1] == 1:
        input_ids = input_ids.squeeze(-1)
    if attention_mask.dim() == 3 and attention_mask.shape[-1] == 1:
        attention_mask = attention_mask.squeeze(-1)

    # Keep the special empty token at the end (OpenVLA/WMPO behavior).
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.tensor([[29871]], dtype=input_ids.dtype)), dim=1
        )
        attention_mask = torch.cat(
            (attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype)), dim=1
        )

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "pixel_values": pixel_values.to(device),
    }


def _generate_action_chunk(
    vla,
    processor,
    image_np: np.ndarray,
    prompt: str,
    unnorm_key: str,
    temperature: float,
    device: torch.device,
    use_normalized: bool,
) -> torch.Tensor:
    vla_inputs = _prepare_vla_inputs(processor, image_np, prompt, device)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if use_normalized:
                if unnorm_key not in vla.norm_stats:
                    alt_key = f"{unnorm_key}_no_noops"
                    if alt_key in vla.norm_stats:
                        unnorm_key = alt_key
                    else:
                        available = list(vla.norm_stats.keys())
                        if not available:
                            raise ValueError("No norm_stats available in the VLA model.")
                        print(f"[WARN] unnorm_key '{unnorm_key}' not found; using '{available[0]}'")
                        unnorm_key = available[0]
                _, _, normalized_actions = vla.generate_action_verl(
                    input_ids=vla_inputs["input_ids"],
                    pixel_values=vla_inputs["pixel_values"],
                    attention_mask=vla_inputs["attention_mask"],
                    padding_idx=processor.tokenizer.pad_token_id,
                    do_sample=True,
                    unnorm_key=unnorm_key,
                    temperature=temperature,
                )
                actions = normalized_actions
            else:
                if unnorm_key not in vla.norm_stats:
                    alt_key = f"{unnorm_key}_no_noops"
                    if alt_key in vla.norm_stats:
                        unnorm_key = alt_key
                    else:
                        available = list(vla.norm_stats.keys())
                        if not available:
                            raise ValueError("No norm_stats available in the VLA model.")
                        print(f"[WARN] unnorm_key '{unnorm_key}' not found; using '{available[0]}'")
                        unnorm_key = available[0]
                actions, _, _ = vla.generate_action_verl(
                    input_ids=vla_inputs["input_ids"],
                    pixel_values=vla_inputs["pixel_values"],
                    attention_mask=vla_inputs["attention_mask"],
                    padding_idx=processor.tokenizer.pad_token_id,
                    do_sample=True,
                    unnorm_key=unnorm_key,
                    temperature=temperature,
                )
    return actions[0]


def _pad_actions(actions: np.ndarray, target_len: int) -> np.ndarray:
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    if actions.shape[0] >= target_len:
        return actions[:target_len]
    pad_len = target_len - actions.shape[0]
    pad = np.zeros((pad_len, actions.shape[1]), dtype=actions.dtype)
    return np.concatenate([actions, pad], axis=0)


def _adapt_actions_to_cosmos(
    actions_chunk: torch.Tensor | np.ndarray,
    target_chunk_size: int,
    action_scale: float,
    gripper_scale: float,
) -> np.ndarray:
    if isinstance(actions_chunk, torch.Tensor):
        actions_chunk = actions_chunk.detach().cpu().numpy()
    # Defensive copy to avoid mutating caller buffers/views.
    actions_chunk = actions_chunk.astype(np.float32, copy=True)

    actions_chunk = _pad_actions(actions_chunk, target_chunk_size)

    # Scale to match Cosmos expectations (delta xyz/rpy * action_scale, gripper * gripper_scale).
    if actions_chunk.shape[1] >= 7:
        actions_chunk[:, :6] = actions_chunk[:, :6] * action_scale
        actions_chunk[:, 6] = actions_chunk[:, 6] * gripper_scale
    return actions_chunk


def _print_stats(tag: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr)
    if arr.size == 0:
        print(f"[{tag}] empty")
        return
    finite = np.isfinite(arr)
    finite_ratio = float(finite.mean()) if finite.size else 0.0
    if finite_ratio < 1.0:
        print(f"[{tag}] non-finite detected (finite_ratio={finite_ratio:.4f})")
    min_v = float(np.nanmin(arr))
    max_v = float(np.nanmax(arr))
    mean_v = float(np.nanmean(arr))
    print(f"[{tag}] min={min_v:.6f} max={max_v:.6f} mean={mean_v:.6f} shape={arr.shape}")


def _load_cosmos_infer(
    cosmos_root: Path,
    experiment: str,
    checkpoint_path: str | None,
    model_key: str | None,
    config_file: str,
):
    _add_cosmos_to_path(cosmos_root)
    from cosmos_predict2.config import MODEL_CHECKPOINTS
    from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference

    config_path = Path(config_file)
    if not config_path.is_absolute():
        config_path = cosmos_root / config_path
    if config_path.suffix == ".py":
        if not config_path.exists():
            raise FileNotFoundError(f"Cosmos config file not found: {config_path}")
        rel_path = config_path.relative_to(cosmos_root)
        config_arg = str(rel_path)
    else:
        config_arg = config_file

    if checkpoint_path is None:
        if model_key is None:
            raise ValueError("Either --cosmos-checkpoint-path or --cosmos-model-key must be provided.")
        checkpoint = None
        if model_key in MODEL_CHECKPOINTS:
            checkpoint = MODEL_CHECKPOINTS[model_key]
        else:
            for k, v in MODEL_CHECKPOINTS.items():
                if str(k) == model_key:
                    checkpoint = v
                    break
        if checkpoint is None:
            raise KeyError(f"Unknown model_key: {model_key}.")
        checkpoint_path = checkpoint.s3.uri

    return Video2WorldInference(
        experiment_name=experiment,
        ckpt_path=checkpoint_path,
        s3_credential_path="",
        context_parallel_size=1,
        config_file=config_arg,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vla-path", required=True, type=Path)
    parser.add_argument("--unnorm-key", required=True)
    parser.add_argument("--task", required=True, help="Task description for prompt, e.g. 'make coffee'")
    parser.add_argument("--initial-image", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-chunks", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--resolution", type=str, default="256,320")
    parser.add_argument("--guidance", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
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
    parser.add_argument("--cosmos-experiment", required=True)
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
    parser.add_argument("--save-fps", type=int, default=14)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this script.")

    # Force OpenVLA to pick the desired platform constants based on argv scanning.
    argv_lower = " ".join(sys.argv).lower()
    if args.vla_platform and args.vla_platform not in argv_lower:
        sys.argv.append(args.vla_platform)

    vla, processor = _load_vla(args.vla_path, device)

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

    prompt = f"In: What action should the robot take to {args.task.lower()}?\nOut:"
    current_frame = np.array(Image.open(args.initial_image).convert("RGB"))
    # Match action_conditioned.py behavior: resize initial frame if resolution is set.
    if args.resolution != "none":
        try:
            h, w = map(int, args.resolution.split(","))
            current_frame = np.array(Image.fromarray(current_frame).resize((w, h), Image.BICUBIC))
        except Exception as exc:
            print(f"[WARN] failed to resize initial frame to {args.resolution}: {exc}")
    all_frames = [current_frame]
    frame_dir = args.output_dir / "frame_inputs"
    frame_dir.mkdir(parents=True, exist_ok=True)
    vla_dir = args.output_dir / "vla_actions"
    cosmos_dir = args.output_dir / "cosmos_actions"
    cosmos_frames_dir = args.output_dir / "cosmos_frames"
    vla_dir.mkdir(parents=True, exist_ok=True)
    cosmos_dir.mkdir(parents=True, exist_ok=True)
    cosmos_frames_dir.mkdir(parents=True, exist_ok=True)
    mediapy.write_image(str(frame_dir / "chunk_000_input.png"), current_frame)

    for chunk_idx in range(args.num_chunks):
        # #region agent log
        import time
        debug_log_path = args.output_dir / "debug.log"
        with open(debug_log_path, 'a') as f:
            # Check image data integrity
            cf_dtype = str(current_frame.dtype)
            cf_unique_vals = len(np.unique(current_frame))
            cf_zero_ratio = float((current_frame == 0).mean())
            cf_full_ratio = float((current_frame == 255).mean())
            # Check per-channel statistics
            if len(current_frame.shape) == 3 and current_frame.shape[2] == 3:
                cf_channel_stats = [{"ch":i,"min":float(current_frame[:,:,i].min()),"max":float(current_frame[:,:,i].max()),"mean":float(current_frame[:,:,i].mean())} for i in range(3)]
            else:
                cf_channel_stats = []
            f.write(json.dumps({"sessionId":"debug-session","runId":"initial","hypothesisId":"B,F,G","location":"vla_cosmos_bridge.py:308","message":"chunk_start_detailed","data":{"chunk_idx":chunk_idx,"current_frame_shape":list(current_frame.shape),"current_frame_dtype":cf_dtype,"current_frame_min":float(current_frame.min()),"current_frame_max":float(current_frame.max()),"current_frame_mean":float(current_frame.mean()),"unique_values_count":cf_unique_vals,"zero_ratio":cf_zero_ratio,"full_255_ratio":cf_full_ratio,"channel_stats":cf_channel_stats},"timestamp":int(time.time()*1000)}) + '\n')
        # #endregion
        _print_stats(f"chunk {chunk_idx} input_frame", current_frame)
        actions_chunk = _generate_action_chunk(
            vla,
            processor,
            current_frame,
            prompt,
            args.unnorm_key,
            args.temperature,
            device,
            args.use_normalized_actions,
        )
        # #region agent log
        with open(debug_log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"initial","hypothesisId":"A","location":"vla_cosmos_bridge.py:320","message":"vla_action_generated","data":{"chunk_idx":chunk_idx,"actions_shape":list(actions_chunk.shape) if hasattr(actions_chunk, 'shape') else "unknown","actions_min":float(actions_chunk.min()) if hasattr(actions_chunk, 'min') else "unknown","actions_max":float(actions_chunk.max()) if hasattr(actions_chunk, 'max') else "unknown"},"timestamp":int(time.time()*1000)}) + '\n')
        # #endregion
        if isinstance(actions_chunk, torch.Tensor):
            raw_actions = actions_chunk.detach().cpu().numpy()
        else:
            raw_actions = np.asarray(actions_chunk)
        _print_stats(f"chunk {chunk_idx} vla_actions_raw", raw_actions)
        (vla_dir / f"chunk_{chunk_idx:03d}_vla_actions.json").write_text(
            json.dumps(raw_actions.astype(float).tolist(), ensure_ascii=True),
            encoding="utf-8",
        )
        actions_chunk = _adapt_actions_to_cosmos(
            actions_chunk,
            target_chunk_size=args.cosmos_action_chunk_size,
            action_scale=args.cosmos_action_scale,
            gripper_scale=args.cosmos_gripper_scale,
        )
        # #region agent log
        with open(debug_log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"initial","hypothesisId":"A,D","location":"vla_cosmos_bridge.py:335","message":"actions_adapted_for_cosmos","data":{"chunk_idx":chunk_idx,"actions_shape":list(actions_chunk.shape),"actions_dtype":str(actions_chunk.dtype),"actions_min":float(actions_chunk.min()),"actions_max":float(actions_chunk.max()),"actions_mean":float(actions_chunk.mean()),"first_action":actions_chunk[0].tolist() if len(actions_chunk)>0 else []},"timestamp":int(time.time()*1000)}) + '\n')
        # #endregion
        _print_stats(f"chunk {chunk_idx} cosmos_actions", actions_chunk)
        (cosmos_dir / f"chunk_{chunk_idx:03d}_cosmos_actions.json").write_text(
            json.dumps(actions_chunk.astype(float).tolist(), ensure_ascii=True),
            encoding="utf-8",
        )

        img_tensor = torchvision.transforms.functional.to_tensor(current_frame).unsqueeze(0)
        # #region agent log
        with open(debug_log_path, 'a') as f:
            # Check if to_tensor conversion is correct
            it_expected_max = 1.0  # to_tensor should output [0,1]
            it_actual_max = float(img_tensor.max())
            it_is_correct = it_actual_max <= 1.1  # Allow small tolerance
            f.write(json.dumps({"sessionId":"debug-session","runId":"initial","hypothesisId":"G","location":"vla_cosmos_bridge.py:367","message":"to_tensor_conversion","data":{"chunk_idx":chunk_idx,"img_tensor_shape":list(img_tensor.shape),"img_tensor_dtype":str(img_tensor.dtype),"img_tensor_min":float(img_tensor.min()),"img_tensor_max":it_actual_max,"img_tensor_mean":float(img_tensor.mean()),"expected_range":"[0,1]","is_correct_range":it_is_correct},"timestamp":int(time.time()*1000)}) + '\n')
        # #endregion
        num_video_frames = actions_chunk.shape[0] + 1
        vid_input = torch.cat(
            [img_tensor, torch.zeros_like(img_tensor).repeat(num_video_frames - 1, 1, 1, 1)], dim=0
        )
        vid_input = (vid_input * 255.0).to(torch.uint8)
        vid_input = vid_input.unsqueeze(0).permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        # #region agent log
        with open(debug_log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"initial","hypothesisId":"D","location":"vla_cosmos_bridge.py:361","message":"cosmos_input_prepared","data":{"chunk_idx":chunk_idx,"vid_input_shape":list(vid_input.shape),"vid_input_dtype":str(vid_input.dtype),"num_video_frames":num_video_frames},"timestamp":int(time.time()*1000)}) + '\n')
        # #endregion

        # #region agent log
        with open(debug_log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"initial","hypothesisId":"C,E","location":"vla_cosmos_bridge.py:365","message":"before_cosmos_generate","data":{"chunk_idx":chunk_idx,"seed":args.seed + chunk_idx,"guidance":args.guidance,"num_video_frames":num_video_frames},"timestamp":int(time.time()*1000)}) + '\n')
        # #endregion
        video = cosmos_infer.generate_vid2world(
            prompt="",
            input_path=vid_input,
            action=torch.from_numpy(actions_chunk).float(),
            guidance=args.guidance,
            num_video_frames=num_video_frames,
            num_latent_conditional_frames=1,
            resolution=args.resolution,
            seed=args.seed + chunk_idx,
            negative_prompt=args.negative_prompt,
        )
        # #region agent log
        with open(debug_log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"initial","hypothesisId":"C","location":"vla_cosmos_bridge.py:380","message":"after_cosmos_generate","data":{"chunk_idx":chunk_idx,"video_shape":list(video.shape),"video_min":float(video.min()),"video_max":float(video.max()),"video_mean":float(video.mean())},"timestamp":int(time.time()*1000)}) + '\n')
        # #endregion

        video_normalized = (video - (-1)) / (1 - (-1))
        video_normalized = torch.clamp(video_normalized, 0, 1)
        # #region agent log
        with open(debug_log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"initial","hypothesisId":"B","location":"vla_cosmos_bridge.py:386","message":"video_normalized","data":{"chunk_idx":chunk_idx,"video_norm_min":float(video_normalized.min()),"video_norm_max":float(video_normalized.max()),"video_norm_mean":float(video_normalized.mean())},"timestamp":int(time.time()*1000)}) + '\n')
        # #endregion
        _print_stats(
            f"chunk {chunk_idx} video_norm",
            video_normalized.detach().cpu().numpy(),
        )
        video_clamped = (video_normalized[0] * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
        # #region agent log
        with open(debug_log_path, 'a') as f:
            # Detailed per-frame analysis, especially frame 7 (8th frame, 0-indexed)
            frame_stats = []
            for i in range(len(video_clamped)):
                frame = video_clamped[i]
                is_black = float(frame.max()) < 5  # Nearly all pixels < 5
                is_white = float(frame.min()) > 250  # Nearly all pixels > 250
                frame_stats.append({
                    "idx":i,
                    "min":float(frame.min()),
                    "max":float(frame.max()),
                    "mean":float(frame.mean()),
                    "std":float(frame.std()),
                    "nonzero_ratio":float((frame>0).mean()),
                    "is_black":is_black,
                    "is_white":is_white,
                    "unique_vals":len(np.unique(frame))
                })
            f.write(json.dumps({"sessionId":"debug-session","runId":"initial","hypothesisId":"B,F","location":"vla_cosmos_bridge.py:413","message":"video_clamped_detailed","data":{"chunk_idx":chunk_idx,"video_clamped_shape":list(video_clamped.shape),"video_clamped_dtype":str(video_clamped.dtype),"frame_stats":frame_stats,"last_frame_is_issue":frame_stats[-1]["is_black"] if frame_stats else False},"timestamp":int(time.time()*1000)}) + '\n')
        # #endregion
        chunk_frame_dir = cosmos_frames_dir / f"chunk_{chunk_idx:03d}"
        chunk_frame_dir.mkdir(parents=True, exist_ok=True)
        for f_idx, frame in enumerate(video_clamped):
            mediapy.write_image(str(chunk_frame_dir / f"frame_{f_idx:03d}.png"), frame)
        current_frame = video_clamped[-1]
        # #region agent log
        with open(debug_log_path, 'a') as f:
            # Critical check: is the next input frame valid?
            nf_is_black = float(current_frame.max()) < 5
            nf_is_valid = float(current_frame.min()) < 250 and float(current_frame.max()) > 5
            nf_dtype_ok = current_frame.dtype == np.uint8
            # Sample a few pixel values for debugging
            if current_frame.size > 0:
                nf_sample_pixels = current_frame.flatten()[:20].tolist()
            else:
                nf_sample_pixels = []
            f.write(json.dumps({"sessionId":"debug-session","runId":"initial","hypothesisId":"B,F,G","location":"vla_cosmos_bridge.py:428","message":"next_input_frame_validation","data":{"chunk_idx":chunk_idx,"next_chunk_will_be":chunk_idx+1,"next_frame_shape":list(current_frame.shape),"next_frame_dtype":str(current_frame.dtype),"next_frame_min":float(current_frame.min()),"next_frame_max":float(current_frame.max()),"next_frame_mean":float(current_frame.mean()),"next_frame_std":float(current_frame.std()),"next_frame_nonzero_ratio":float((current_frame>0).mean()),"is_black":nf_is_black,"is_valid":nf_is_valid,"dtype_is_uint8":nf_dtype_ok,"sample_pixels":nf_sample_pixels},"timestamp":int(time.time()*1000)}) + '\n')
        # #endregion
        mediapy.write_image(str(frame_dir / f"chunk_{chunk_idx+1:03d}_input.png"), current_frame)
        all_frames.extend(video_clamped[1:])

    out_path = args.output_dir / "vla_cosmos_rollout.mp4"
    mediapy.write_video(str(out_path), np.stack(all_frames, axis=0), fps=args.save_fps)
    print(f"Saved rollout video to: {out_path}")


if __name__ == "__main__":
    main()
