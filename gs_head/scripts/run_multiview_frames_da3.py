#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch

from depth_anything_3.api import DepthAnything3


'''
  1. 遍历根目录下的所有子文件夹(任务)
  2. 对每个任务提取三个视角视频的24帧
  3. 对每帧运行DA3推理预测depth/depth_conf/ray/ray_conf
  4. 保存结果到对应的任务文件夹

  使用方法
  python scripts/run_multiview_frames_da3.py \
      --input-root /nfs/rczhang/code/cosmos-predict2.5/outputs/robot_multiview-352-part \
      --model /nfs/rczhang/code/Depth-Anything-3/DA3NESTED-GIANT-LARGE
'''

VIEW_ORDER = ["head", "left_hand", "right_hand"]


def set_keep_ray(module: torch.nn.Module, enabled: bool) -> None:
    if hasattr(module, "keep_ray"):
        module.keep_ray = enabled
    if hasattr(module, "da3"):
        set_keep_ray(module.da3, enabled)


def build_frame_map(view_dir: Path) -> dict[int, Path]:
    frame_map = {}
    for img_path in sorted(view_dir.glob("frame_*.png")):
        try:
            frame_idx = int(img_path.stem.split("_")[-1])
        except ValueError:
            continue
        frame_map[frame_idx] = img_path
    return frame_map


def common_frame_indices(view_maps: dict[str, dict[int, Path]]) -> list[int]:
    indices = None
    for view_name in VIEW_ORDER:
        view_indices = set(view_maps[view_name].keys())
        indices = view_indices if indices is None else indices & view_indices
    return sorted(indices or [])


def save_outputs(
    output_root: Path,
    scene_name: str,
    frame_idx: int,
    depth: np.ndarray,
    depth_conf: np.ndarray | None,
    ray: np.ndarray | None,
    ray_conf: np.ndarray | None,
) -> None:
    frame_dir = output_root / scene_name / f"frame_{frame_idx:03d}"
    for view_i, view_name in enumerate(VIEW_ORDER):
        view_dir = frame_dir / view_name
        view_dir.mkdir(parents=True, exist_ok=True)

        np.save(view_dir / f"{view_name}_depth.npy", depth[view_i])
        if depth_conf is not None:
            np.save(view_dir / f"{view_name}_depth_conf.npy", depth_conf[view_i])
        if ray is not None:
            np.save(view_dir / f"{view_name}_ray.npy", ray[view_i])
        if ray_conf is not None:
            np.save(view_dir / f"{view_name}_ray_conf.npy", ray_conf[view_i])


def run_inference(
    model: DepthAnything3,
    image_paths: list[Path],
    process_res: int,
    process_res_method: str,
    ref_view_strategy: str,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    imgs_cpu, extrinsics, intrinsics = model.input_processor(
        [str(p) for p in image_paths],
        None,
        None,
        process_res,
        process_res_method,
        num_workers=1,
        sequential=True,
    )
    imgs, ex_t, in_t = model._prepare_model_inputs(imgs_cpu, extrinsics, intrinsics)
    ex_t_norm = model._normalize_extrinsics(ex_t.clone() if ex_t is not None else None)

    raw_output = model._run_model_forward(
        imgs,
        ex_t_norm,
        in_t,
        export_feat_layers=[],
        infer_gs=False,
        use_ray_pose=False,
        ref_view_strategy=ref_view_strategy,
    )

    depth = raw_output["depth"].squeeze(0).cpu().numpy()
    depth_conf = raw_output.get("depth_conf", None)
    if depth_conf is not None:
        depth_conf = depth_conf.squeeze(0).cpu().numpy()

    ray = raw_output.get("ray", None)
    if ray is not None:
        ray = ray.squeeze(0).cpu().numpy()

    ray_conf = raw_output.get("ray_conf", None)
    if ray_conf is not None:
        ray_conf = ray_conf.squeeze(0).cpu().numpy()

    return depth, depth_conf, ray, ray_conf


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run DA3 on multiview frames and save depth/ray outputs."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(
            "/nfs/rczhang/code/cosmos-predict2.5/outputs/robot_multiview-352-part"
        ),
        help="Root directory with scene folders and view subfolders.",
    )
    # parser.add_argument(
    #     "--output-root",
    #     type=Path,
    #     default=Path("/nfs/rczhang/code/Depth-Anything-3/outputs/robot_multiview-352_da3"),
    #     help="Output directory (default: <input-root>_da3).",
    # )
    parser.add_argument(
        "--model",
        type=str,
        default="/nfs/rczhang/code/Depth-Anything-3/DA3NESTED-GIANT-LARGE",
        help="Model name or path for DepthAnything3.from_pretrained.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (default: cuda).",
    )
    parser.add_argument(
        "--process-res",
        type=int,
        default=504,
        help="Processing resolution (default: 504).",
    )
    parser.add_argument(
        "--process-res-method",
        type=str,
        default="upper_bound_resize",
        help="Resize method for preprocessing.",
    )
    parser.add_argument(
        "--ref-view-strategy",
        type=str,
        default="saddle_balanced",
        help="Reference view strategy for DA3.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any view is missing frames or common frames are empty.",
    )
    parser.add_argument(
        "--no-keep-ray",
        action="store_true",
        help="Disable keeping ray outputs (keeps camera decoder enabled).",
    )
    args = parser.parse_args()

    input_root = args.input_root
    output_root = args.output_root or Path(f"{input_root}_da3")

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    device = torch.device(args.device)
    model = DepthAnything3.from_pretrained(args.model).to(device=device)

    keep_ray = not args.no_keep_ray
    if keep_ray:
        set_keep_ray(model.model, True)

    model.eval()

    scenes = [p for p in sorted(input_root.iterdir()) if p.is_dir()]
    for scene_dir in scenes:
        view_maps = {}
        missing_view = False
        for view_name in VIEW_ORDER:
            view_dir = scene_dir / view_name
            if not view_dir.is_dir():
                missing_view = True
                break
            view_maps[view_name] = build_frame_map(view_dir)
        if missing_view:
            print(f"Skip {scene_dir}: missing views")
            continue

        frame_indices = common_frame_indices(view_maps)
        counts = {v: len(view_maps[v]) for v in VIEW_ORDER}
        if any(counts[v] == 0 for v in VIEW_ORDER):
            msg = f"{scene_dir}: empty views {counts}"
            if args.strict:
                raise RuntimeError(msg)
            print(f"Skip {msg}")
            continue
        if args.strict and not frame_indices:
            raise RuntimeError(f"{scene_dir}: no common frames across views {counts}")
        if not frame_indices:
            print(f"Skip {scene_dir}: no common frames {counts}")
            continue

        for frame_idx in frame_indices:
            missing = [v for v in VIEW_ORDER if frame_idx not in view_maps[v]]
            if missing:
                msg = f"{scene_dir} frame {frame_idx:03d}: missing {missing}"
                if args.strict:
                    raise RuntimeError(msg)
                print(f"Skip {msg}")
                continue
            image_paths = [view_maps[v][frame_idx] for v in VIEW_ORDER]
            with torch.inference_mode():
                depth, depth_conf, ray, ray_conf = run_inference(
                    model,
                    image_paths,
                    args.process_res,
                    args.process_res_method,
                    args.ref_view_strategy,
                )

            save_outputs(
                output_root,
                scene_dir.name,
                frame_idx,
                depth,
                depth_conf,
                ray,
                ray_conf,
            )
            print(f"{scene_dir.name} frame {frame_idx:03d} done")

    print(f"Done. Outputs at: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
