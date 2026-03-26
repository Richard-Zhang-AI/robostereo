#!/usr/bin/env python3
"""
Bridge script: WMPO VLA (SFT) -> Cosmos action-conditioned world model.

Given an initial frame, this script loops:
  1) VLA predicts an action chunk from the current frame + task prompt
  2) Cosmos action-conditioned model generates the next video chunk
  3) The last frame becomes the next VLA input

Outputs a single stitched video for quick sanity-checking.
"""

import argparse
import os
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
import mediapy


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

    # Ensure the special empty token appears at the end (WMPO behavior)
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


def _generate_action_chunk(vla, processor, image_np, prompt, unnorm_key, temperature, device, use_normalized):
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
    # actions: (B, chunk_len, action_dim)
    return actions[0]


def _adapt_actions_to_cosmos(
    actions_chunk: torch.Tensor | np.ndarray,
    target_chunk_size: int,
    action_scale: float,
    gripper_scale: float,
    resample_mode: str,
) -> np.ndarray:
    """Adapt WMPO VLA delta actions to Cosmos expected format."""
    if isinstance(actions_chunk, torch.Tensor):
        actions_chunk = actions_chunk.detach().cpu().numpy()
    actions_chunk = actions_chunk.astype(np.float32, copy=False)
    if actions_chunk.ndim == 1:
        actions_chunk = actions_chunk.reshape(1, -1)

    # Align action chunk length with Cosmos model expectation (default 12) via linear interpolation.
    if actions_chunk.shape[0] != target_chunk_size:
        if resample_mode == "mean" and actions_chunk.shape[0] % target_chunk_size == 0:
            factor = actions_chunk.shape[0] // target_chunk_size
            actions_chunk = actions_chunk.reshape(target_chunk_size, factor, -1).mean(axis=1)
        else:
            src_t = np.linspace(0.0, 1.0, actions_chunk.shape[0])
            dst_t = np.linspace(0.0, 1.0, target_chunk_size)
            interpolated = []
            for d in range(actions_chunk.shape[1]):
                interpolated.append(np.interp(dst_t, src_t, actions_chunk[:, d]))
            actions_chunk = np.stack(interpolated, axis=1).astype(np.float32)

    # WMPO outputs delta EEF (xyz + rpy + gripper). Cosmos expects relative deltas scaled by 20x.
    if actions_chunk.shape[1] >= 7:
        actions_chunk[:, :6] = actions_chunk[:, :6] * action_scale
        actions_chunk[:, 6] = actions_chunk[:, 6] * gripper_scale

    return actions_chunk


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
            raise KeyError(f"Unknown model_key: {model_key}. Valid keys: {[str(k) for k in MODEL_CHECKPOINTS.keys()]}")
        checkpoint_path = checkpoint.s3.uri

    return Video2WorldInference(
        experiment_name=experiment,
        ckpt_path=checkpoint_path,
        s3_credential_path="",
        context_parallel_size=1,
        config_file=config_arg,
    )


def _write_cosmos_offline_package(
    export_dir: Path,
    initial_frame: np.ndarray,
    actions: np.ndarray,
    task: str,
    save_fps: int,
) -> Path:
    export_dir = export_dir.resolve()
    input_root = export_dir / "input"
    ann_dir = input_root / "annotation" / "test"
    video_dir = input_root / "videos" / "test" / "0"
    ann_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    image_path = video_dir / "initial.png"
    mediapy.write_image(str(image_path), initial_frame)

    sample = {
        "task": "robot_trajectory_prediction",
        "texts": [task],
        "videos": [{"video_path": str(image_path.relative_to(input_root))}],
        "action": actions.astype(float).tolist(),
        "fps": save_fps,
    }
    ann_path = ann_dir / "0.json"
    ann_path.write_text(json.dumps(sample, indent=4), encoding="utf-8")

    loader_path = export_dir / "action_loader.py"
    loader_path.write_text(
        "\n".join(
            [
                "import numpy as np",
                "import mediapy",
                "",
                "def load_action_from_json():",
                "    def load_fn(json_data: dict, video_path: str, args):",
                "        img = mediapy.read_image(video_path)",
                "        actions = np.array(json_data.get('action', []), dtype=np.float32)",
                "        if actions.ndim == 1:",
                "            actions = actions.reshape(1, -1)",
                "        return {",
                "            'actions': actions,",
                "            'initial_frame': img,",
                "            'video_array': None,",
                "            'video_path': video_path,",
                "        }",
                "    return load_fn",
                "",
            ]
        ),
        encoding="utf-8",
    )

    inference_params = {
        "name": "vla_offline",
        "input_root": str(input_root),
        "input_json_sub_folder": "annotation/test",
        "save_root": str(export_dir / "outputs"),
        "guidance": 0,
        "resolution": "256,320",
        "camera_id": 0,
        "start": 0,
        "end": 1,
        "fps_downsample_ratio": 1,
        "gripper_scale": 1.0,
        "gripper_key": "continuous_gripper_state",
        "state_key": "state",
        "reverse": False,
        "single_chunk": False,
        "start_frame_idx": 0,
        "save_fps": save_fps,
        "num_latent_conditional_frames": 1,
        "action_scaler": 1.0,
        "use_quat": False,
        "action_load_fn": "action_loader.load_action_from_json",
        "negative_prompt": "",
        "seed": 0,
        "prompt": None,
        "visual_condition_source": "rgb",
    }
    params_path = export_dir / "inference_params.json"
    params_path.write_text(json.dumps(inference_params, indent=2), encoding="utf-8")
    return params_path


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
    parser.add_argument(
        "--use-normalized-actions",
        action="store_true",
        help="Use VLA normalized actions (skip unnormalization) before Cosmos adaptation.",
    )
    parser.add_argument(
        "--cosmos-resample",
        choices=("linear", "mean"),
        default="linear",
        help="Resampling method when converting VLA action chunks to Cosmos length.",
    )
    parser.add_argument("--cosmos-root", type=Path, default=Path("/nfs/rczhang/code/cosmos-predict2.5"))
    parser.add_argument("--cosmos-experiment", required=True)
    parser.add_argument("--cosmos-checkpoint-path", default=None)
    parser.add_argument("--cosmos-model-key", default=None)
    parser.add_argument(
        "--cosmos-config-file",
        default="cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py",
    )
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--save-fps", type=int, default=14)
    parser.add_argument(
        "--export-inference-json",
        action="store_true",
        help="Write Cosmos offline inference package (json + loader) instead of running Cosmos generation.",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Export directory for offline package (defaults to output_dir).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this script.")

    vla, processor = _load_vla(args.vla_path, device)

    cosmos_infer = None
    if not args.export_inference_json:
        cosmos_infer = _load_cosmos_infer(
            args.cosmos_root,
            args.cosmos_experiment,
            args.cosmos_checkpoint_path,
            args.cosmos_model_key,
            args.cosmos_config_file,
        )

    prompt = f"In: What action should the robot take to {args.task.lower()}?\nOut:"

    current_frame = np.array(Image.open(args.initial_image).convert("RGB"))
    all_frames = [current_frame]

    chunk_size = None
    all_actions = []
    num_chunks = 1 if args.export_inference_json else args.num_chunks
    for chunk_idx in range(num_chunks):
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
        print(f"[DEBUG] VLA actions raw shape: {actions_chunk.shape}")
        if isinstance(actions_chunk, torch.Tensor):
            raw_np = actions_chunk.detach().cpu().numpy()
        else:
            raw_np = np.asarray(actions_chunk)
        print(
            "[DEBUG] VLA actions raw stats:",
            f"min={raw_np.min():.6f}",
            f"max={raw_np.max():.6f}",
            f"mean={raw_np.mean():.6f}",
        )
        actions_chunk = _adapt_actions_to_cosmos(
            actions_chunk,
            target_chunk_size=args.cosmos_action_chunk_size,
            action_scale=args.cosmos_action_scale,
            gripper_scale=args.cosmos_gripper_scale,
            resample_mode=args.cosmos_resample,
        )
        all_actions.append(actions_chunk)
        print(f"[DEBUG] Cosmos actions shape: {actions_chunk.shape}")
        print(
            "[DEBUG] Cosmos actions stats:",
            f"min={actions_chunk.min():.6f}",
            f"max={actions_chunk.max():.6f}",
            f"mean={actions_chunk.mean():.6f}",
        )
        if chunk_size is None:
            chunk_size = actions_chunk.shape[0]

        if args.export_inference_json:
            continue

        img_tensor = torchvision.transforms.functional.to_tensor(current_frame).unsqueeze(0)
        num_video_frames = actions_chunk.shape[0] + 1
        vid_input = torch.cat(
            [img_tensor, torch.zeros_like(img_tensor).repeat(num_video_frames - 1, 1, 1, 1)], dim=0
        )
        vid_input = (vid_input * 255.0).to(torch.uint8)
        vid_input = vid_input.unsqueeze(0).permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)

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

        video_normalized = (video - (-1)) / (1 - (-1))
        video_normalized = torch.clamp(video_normalized, 0, 1)
        video_clamped = (video_normalized[0] * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
        current_frame = video_clamped[-1]
        all_frames.extend(video_clamped[1:])

    if args.export_inference_json:
        export_dir = args.export_dir if args.export_dir is not None else args.output_dir
        actions_full = np.concatenate(all_actions, axis=0) if all_actions else np.zeros((0, 7), dtype=np.float32)
        params_path = _write_cosmos_offline_package(
            export_dir=export_dir,
            initial_frame=all_frames[0],
            actions=actions_full,
            task=args.task,
            save_fps=args.save_fps,
        )
        print(f"Saved offline inference package to: {params_path}")
        return

    out_path = args.output_dir / "vla_cosmos_rollout.mp4"
    mediapy.write_video(str(out_path), np.stack(all_frames, axis=0), fps=args.save_fps)
    print(f"Saved rollout video to: {out_path}")


if __name__ == "__main__":
    main()
