#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_GS_HEAD_ROOT = _SCRIPT_DIR.parent
_PROJECT_ROOT = _GS_HEAD_ROOT.parent
sys.path.insert(0, str(_GS_HEAD_ROOT / "src"))

_DEFAULT_MODEL = str(_PROJECT_ROOT / "checkpoints" / "DA3NESTED-GIANT-LARGE")
_DEFAULT_DATASET = _PROJECT_ROOT / "datasets" / "bridge" / "videos" / "train" / "0"
_DEFAULT_OUTPUT = _GS_HEAD_ROOT / "output_gs_from_xyz"

import cv2
import numpy as np
import torch
from einops import rearrange, repeat

from depth_anything_3.api import DepthAnything3
from depth_anything_3.model.da3 import DepthAnything3Net
from depth_anything_3.model.utils.transform import (
    cam_quat_xyzw_to_world_quat_wxyz,
    pose_encoding_to_extri_intri,
)
from depth_anything_3.specs import Gaussians, Prediction
from depth_anything_3.utils.export.gs import export_to_gs_image, export_to_gs_ply, export_to_gs_video
from depth_anything_3.utils.geometry import (
    affine_inverse,
    as_homogeneous,
    get_world_rays,
    map_pdf_to_opacity,
    sample_image_grid,
)
from depth_anything_3.utils.sh_helpers import rotate_sh

try:
    from safetensors.numpy import load_file as load_safetensors
except Exception:
    load_safetensors = None


def load_first_extrinsic(extr_path: Path) -> np.ndarray:
    ext = np.loadtxt(extr_path, dtype=np.float32).reshape(-1, 3, 4)[0]
    ext4 = np.eye(4, dtype=np.float32)
    ext4[:3, :4] = ext
    return ext4


def resize_depth(depth: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    h, w = target_hw
    return cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)


def align_depth_to_da3(
    depth: np.ndarray, process_res: int, process_res_method: str, patch_size: int = 14
) -> np.ndarray:
    h, w = depth.shape
    if process_res_method in ("upper_bound_resize", "upper_bound_crop"):
        longest = max(w, h)
        scale = process_res / float(longest)
    elif process_res_method in ("lower_bound_resize", "lower_bound_crop"):
        shortest = min(w, h)
        scale = process_res / float(shortest)
    else:
        raise ValueError(f"Unsupported process_res_method: {process_res_method}")

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    depth = cv2.resize(depth, (new_w, new_h), interpolation=interp)

    if process_res_method.endswith("resize"):
        def nearest_multiple(x: int, p: int) -> int:
            down = (x // p) * p
            up = down + p
            return up if abs(up - x) <= abs(x - down) else down

        new_w = max(1, nearest_multiple(depth.shape[1], patch_size))
        new_h = max(1, nearest_multiple(depth.shape[0], patch_size))
        if (new_h, new_w) != depth.shape:
            upscale = (new_w > depth.shape[1]) or (new_h > depth.shape[0])
            interp = cv2.INTER_CUBIC if upscale else cv2.INTER_AREA
            depth = cv2.resize(depth, (new_w, new_h), interpolation=interp)
    elif process_res_method.endswith("crop"):
        new_w = (depth.shape[1] // patch_size) * patch_size
        new_h = (depth.shape[0] // patch_size) * patch_size
        left = (depth.shape[1] - new_w) // 2
        top = (depth.shape[0] - new_h) // 2
        depth = depth[top : top + new_h, left : left + new_w]

    return depth


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run DA3 backbone + GS head using depth from XYZ map and a single RGB frame."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=_DEFAULT_DATASET,
        help="Directory containing geometry.safetensors / intrinsics.txt / extrinsics.txt",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional RGB image path. If omitted, uses first frame from dataset rgb.mp4.",
    )
    parser.add_argument("--frame-idx", type=int, default=0, help="Frame index for xyz.")
    parser.add_argument(
        "--model",
        type=str,
        default=_DEFAULT_MODEL,
        help="Model path or HF repo for DepthAnything3.from_pretrained",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--process-res", type=int, default=504, help="DA3 process resolution")
    parser.add_argument(
        "--process-res-method",
        type=str,
        default="upper_bound_resize",
        help="DA3 resize method",
    )
    parser.add_argument(
        "--use-extrinsics",
        action="store_true",
        help="Treat XYZ as world coords and transform to camera with extrinsics.",
    )
    parser.add_argument(
        "--xyz-is-normalized",
        action="store_true",
        help="If set, de-normalize xyz using --xyz-min/--xyz-max.",
    )
    parser.add_argument("--xyz-min", type=float, default=-0.55)
    parser.add_argument("--xyz-max", type=float, default=1.51)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Output directory.",
    )
    parser.add_argument(
        "--export-video",
        action="store_true",
        help="Export preview video (wander trajectory). Default: only PLY, since this script is single-image.",
    )
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    if load_safetensors is None:
        raise RuntimeError("safetensors not available; install it in this env.")

    geom_path = args.dataset_dir / "geometry.safetensors"
    if not geom_path.is_file():
        raise FileNotFoundError(f"Missing geometry.safetensors: {geom_path}")
    data = load_safetensors(str(geom_path))
    if "xyz" not in data:
        raise RuntimeError("geometry.safetensors missing key 'xyz'")
    xyz = data["xyz"][args.frame_idx].astype(np.float32)  # (H, W, 3)
    print(
        f"[INFO] xyz_is_normalized={args.xyz_is_normalized} "
        f"(min={args.xyz_min}, max={args.xyz_max}), "
        f"use_extrinsics={args.use_extrinsics}"
    )
    if args.xyz_is_normalized:
        xyz = xyz * (args.xyz_max - args.xyz_min) + args.xyz_min

    if args.use_extrinsics:
        extr_path = args.dataset_dir / "extrinsics.txt"
        if not extr_path.is_file():
            raise FileNotFoundError(f"Missing extrinsics.txt: {extr_path}")
        extr = load_first_extrinsic(extr_path)
        xyz_h = np.concatenate([xyz.reshape(-1, 3), np.ones((xyz.size // 3, 1))], axis=1)
        xyz = (extr @ xyz_h.T).T[:, :3].reshape(xyz.shape)

    depth = xyz[..., 2]

    rgb_video = args.dataset_dir / "rgb.mp4"
    if args.image is None:
        if not rgb_video.is_file():
            raise FileNotFoundError(f"Missing rgb.mp4: {rgb_video}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        extracted = args.output_dir / "inputs" / "frame0000.png"
        cap = cv2.VideoCapture(str(rgb_video))
        ok, frame_bgr = cap.read()
        cap.release()
        if not ok or frame_bgr is None:
            raise RuntimeError(f"Failed to read first frame from: {rgb_video}")
        extracted.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(extracted), frame_bgr):
            raise RuntimeError(f"Failed to write frame image: {extracted}")
        image_path = extracted
    else:
        image_path = args.image
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image: {image_path}")

    device = torch.device(args.device)
    model = DepthAnything3.from_pretrained(args.model).to(device=device)
    model.eval()
    da3: DepthAnything3Net = model.model.da3 if hasattr(model.model, "da3") else model.model

    imgs_cpu, extr_t, intr_t = model.input_processor(
        [str(image_path)],
        None,
        None,
        args.process_res,
        args.process_res_method,
        num_workers=1,
        sequential=True,
    )
    imgs, ex_t, in_t = model._prepare_model_inputs(imgs_cpu, extr_t, intr_t)
    H, W = imgs.shape[-2], imgs.shape[-1]
    depth_rs = align_depth_to_da3(
        depth.astype(np.float32), args.process_res, args.process_res_method
    )
    if depth_rs.shape != (H, W):
        depth_rs = resize_depth(depth_rs, (H, W)).astype(np.float32)
    depth_t = torch.from_numpy(depth_rs)[None, None].to(imgs.device)  # B V H W

    with torch.inference_mode():
        cam_token = None
        t0 = time.perf_counter()
        feats, _ = da3.backbone(
            imgs, cam_token=cam_token, export_feat_layers=[], ref_view_strategy="saddle_balanced"
        )
        # Prefer dataset camera params if available; otherwise fall back to DA3 prediction.
        intr_path = args.dataset_dir / "intrinsics.txt"
        extr_path = args.dataset_dir / "extrinsics.txt"
        if intr_path.is_file() and extr_path.is_file():
            intr = np.loadtxt(intr_path, dtype=np.float32).reshape(-1, 3, 3)[args.frame_idx]
            extr = np.loadtxt(extr_path, dtype=np.float32).reshape(-1, 3, 4)[args.frame_idx]
            extr4 = np.eye(4, dtype=np.float32)
            extr4[:3, :4] = extr
            in_t = torch.from_numpy(intr)[None, None].to(imgs.device)
            ex_t = torch.from_numpy(extr4)[None, None].to(imgs.device)
            print("[INFO] using dataset intrinsics/extrinsics")
        else:
            if da3.cam_dec is None:
                raise RuntimeError("cam_dec not available; cannot predict camera parameters.")
            pose_enc = da3.cam_dec(feats[-1][1])
            c2w, ixt = pose_encoding_to_extri_intri(pose_enc, (H, W))
            ex_t = as_homogeneous(affine_inverse(c2w))
            in_t = ixt
            print("[INFO] using DA3-predicted intrinsics/extrinsics")
        print(f"[TIME] backbone: {time.perf_counter() - t0:.1f}s")
        t1 = time.perf_counter()
        gs_outs = da3.gs_head(
            feats=feats, H=H, W=W, patch_start_idx=0, images=imgs
        )
        raw_gaussians = gs_outs.raw_gs
        densities = gs_outs.raw_gs_conf
        # Custom GS adapter: use external depth with DA3-predicted camera params.
        adapter = da3.gs_adapter
        sh_degree = getattr(adapter, "sh_degree", 0)
        pred_color = getattr(adapter, "pred_color", False)
        pred_offset_depth = getattr(adapter, "pred_offset_depth", False)
        pred_offset_xy = getattr(adapter, "pred_offset_xy", True)
        scale_min = getattr(adapter, "gaussian_scale_min", 1e-5)
        scale_max = getattr(adapter, "gaussian_scale_max", 30.0)
        d_sh = 1 if pred_color else (sh_degree + 1) ** 2

        # camera to world & normalized intrinsics
        cam2worlds = affine_inverse(ex_t)
        intr_normed = in_t.clone()
        intr_normed[..., 0, :] /= W
        intr_normed[..., 1, :] /= H

        gs_depths = depth_t
        if pred_offset_depth:
            gs_depths = depth_t + raw_gaussians[..., -1]
            raw_gaussians = raw_gaussians[..., :-1]

        xy_ray, _ = sample_image_grid((H, W), device=depth_t.device)
        xy_ray = xy_ray[None, None, ...].expand(1, 1, -1, -1, -1)  # b v h w xy
        if pred_offset_xy:
            pixel_size = 1 / torch.tensor((W, H), dtype=xy_ray.dtype, device=xy_ray.device)
            offset_xy = raw_gaussians[..., :2]
            xy_ray = xy_ray + offset_xy * pixel_size
            raw_gaussians = raw_gaussians[..., 2:]

        origins, directions = get_world_rays(
            xy_ray,
            repeat(cam2worlds, "b v i j -> b v h w i j", h=H, w=W),
            repeat(intr_normed, "b v i j -> b v h w i j", h=H, w=W),
        )
        gs_means_world = origins + directions * gs_depths[..., None]
        gs_means_world = rearrange(gs_means_world, "b v h w d -> b (v h w) d")

        # split raw gaussians
        scales_raw, rotations_raw, sh_raw = raw_gaussians.split((3, 4, 3 * d_sh), dim=-1)

        # scales
        scales = scale_min + (scale_max - scale_min) * scales_raw.sigmoid()
        pixel_size = 1 / torch.tensor((W, H), dtype=scales.dtype, device=scales.device)
        multiplier = adapter.get_scale_multiplier(intr_normed, pixel_size)
        gs_scales = scales * gs_depths[..., None] * multiplier[..., None, None, None]
        gs_scales = rearrange(gs_scales, "b v h w d -> b (v h w) d")

        # rotations
        rotations_raw = rotations_raw / (rotations_raw.norm(dim=-1, keepdim=True) + 1e-8)
        cam_quat_xyzw = rearrange(rotations_raw, "b v h w c -> b (v h w) c")
        c2w_mat = repeat(cam2worlds, "b v i j -> b (v h w) i j", h=H, w=W)
        gs_rotations_world = cam_quat_xyzw_to_world_quat_wxyz(cam_quat_xyzw, c2w_mat)

        # harmonics / color
        sh = rearrange(sh_raw, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        if not pred_color and getattr(adapter, "sh_mask", None) is not None:
            sh = sh * adapter.sh_mask
        if pred_color or sh_degree == 0:
            gs_sh_world = sh
        else:
            gs_sh_world = rotate_sh(sh, cam2worlds[:, :, None, None, None, :3, :3])
        gs_sh_world = rearrange(gs_sh_world, "b v h w xyz d_sh -> b (v h w) xyz d_sh")

        gs_opacities = map_pdf_to_opacity(densities)
        gs_opacities = rearrange(gs_opacities, "b v h w ... -> b (v h w) ...")

        gs_world = Gaussians(
            means=gs_means_world,
            scales=gs_scales,
            rotations=gs_rotations_world,
            harmonics=gs_sh_world,
            opacities=gs_opacities,
        )
        print(f"[TIME] gs_head+adapter: {time.perf_counter() - t1:.1f}s")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    opacities = gs_world.opacities
    if opacities.dim() > 2:
        opacities = opacities.squeeze(-1)
    gs_world = Gaussians(
        means=gs_world.means,
        scales=gs_world.scales,
        rotations=gs_world.rotations,
        harmonics=gs_world.harmonics,
        opacities=opacities,
    )
    pred = Prediction(
        depth=depth_rs[None, ...],
        is_metric=1,
        extrinsics=ex_t.squeeze(0).cpu().numpy(),
        intrinsics=in_t.squeeze(0).cpu().numpy(),
        gaussians=gs_world,
    )
    export_to_gs_ply(pred, str(args.output_dir))
    t_img = time.perf_counter()
    export_to_gs_image(
        pred,
        str(args.output_dir),
        extrinsics=ex_t,
        intrinsics=in_t,
        out_image_hw=(H, W),
        output_name="gs_render",
        save_depth=True,
    )
    print(f"[TIME] gs_image: {time.perf_counter() - t_img:.1f}s")
    if args.export_video:
        t2 = time.perf_counter()
        export_to_gs_video(
            pred,
            str(args.output_dir),
            extrinsics=ex_t,
            intrinsics=in_t,
            out_image_hw=(H, W),
            trj_mode="wander",
            chunk_size=1,
            video_quality="high",
        )
        print(f"[TIME] gs_video: {time.perf_counter() - t2:.1f}s")
    else:
        print("[INFO] skipped video export (use --export-video to enable)")
    print(f"[DONE] outputs saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
