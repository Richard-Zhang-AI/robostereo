#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch

from depth_anything_3.api import DepthAnything3


def set_keep_ray(module: torch.nn.Module, enabled: bool) -> None:
    if hasattr(module, "keep_ray"):
        module.keep_ray = enabled
    if hasattr(module, "da3"):
        set_keep_ray(module.da3, enabled)


def squeeze_single_view(arr: np.ndarray | None) -> np.ndarray | None:
    if arr is None:
        return None
    if arr.ndim >= 1 and arr.shape[0] == 1:
        return arr[0]
    return arr


def run_inference(
    model: DepthAnything3,
    image_path: Path,
    process_res: int,
    process_res_method: str,
    ref_view_strategy: str,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    imgs_cpu, extrinsics, intrinsics = model.input_processor(
        [str(image_path)],
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

    return (
        squeeze_single_view(depth),
        squeeze_single_view(depth_conf),
        squeeze_single_view(ray),
        squeeze_single_view(ray_conf),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run DA3 on a single image and save depth/ray outputs."
    )
    parser.add_argument(
        "--input-image",
        type=Path,
        required=True,
        help="Input image path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <input-image-stem>_da3 next to image).",
    )
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
        "--no-keep-ray",
        action="store_true",
        help="Disable keeping ray outputs (keeps camera decoder enabled).",
    )
    args = parser.parse_args()

    input_image = args.input_image
    if not input_image.is_file():
        raise FileNotFoundError(f"Input image not found: {input_image}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = input_image.parent / f"{input_image.stem}_da3"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = DepthAnything3.from_pretrained(args.model).to(device=device)

    keep_ray = not args.no_keep_ray
    if keep_ray:
        set_keep_ray(model.model, True)

    model.eval()

    with torch.inference_mode():
        depth, depth_conf, ray, ray_conf = run_inference(
            model,
            input_image,
            args.process_res,
            args.process_res_method,
            args.ref_view_strategy,
        )

    np.save(output_dir / "depth.npy", depth)
    if depth_conf is not None:
        np.save(output_dir / "depth_conf.npy", depth_conf)
    if ray is not None:
        np.save(output_dir / "ray.npy", ray)
    if ray_conf is not None:
        np.save(output_dir / "ray_conf.npy", ray_conf)

    print(f"Done. Outputs at: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
