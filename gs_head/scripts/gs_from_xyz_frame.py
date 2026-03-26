#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from einops import rearrange, repeat

from depth_anything_3.api import DepthAnything3
from depth_anything_3.model.da3 import DepthAnything3Net
from depth_anything_3.model.utils.transform import cam_quat_xyzw_to_world_quat_wxyz
from depth_anything_3.specs import Gaussians
from depth_anything_3.utils.geometry import (
    affine_inverse,
    get_world_rays,
    map_pdf_to_opacity,
    sample_image_grid,
)
from depth_anything_3.utils.sh_helpers import rotate_sh
from depth_anything_3.model.utils.gs_renderer import run_renderer_in_chunk_w_trj_mode

try:
    from safetensors.numpy import load_file as load_safetensors
except Exception:
    load_safetensors = None


def align_depth_to_da3(depth: np.ndarray, process_res: int, process_res_method: str, patch: int = 14) -> np.ndarray:
    h, w = depth.shape
    if process_res_method in ("upper_bound_resize", "upper_bound_crop"):
        scale = process_res / float(max(w, h))
    elif process_res_method in ("lower_bound_resize", "lower_bound_crop"):
        scale = process_res / float(min(w, h))
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
        new_w = max(1, nearest_multiple(depth.shape[1], patch))
        new_h = max(1, nearest_multiple(depth.shape[0], patch))
        if (new_h, new_w) != depth.shape:
            upscale = (new_w > depth.shape[1]) or (new_h > depth.shape[0])
            interp = cv2.INTER_CUBIC if upscale else cv2.INTER_AREA
            depth = cv2.resize(depth, (new_w, new_h), interpolation=interp)
    elif process_res_method.endswith("crop"):
        new_w = (depth.shape[1] // patch) * patch
        new_h = (depth.shape[0] // patch) * patch
        left = (depth.shape[1] - new_w) // 2
        top = (depth.shape[0] - new_h) // 2
        depth = depth[top : top + new_h, left : left + new_w]
    return depth


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a single GS frame with a fixed view.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--frame-idx", type=int, default=0)
    parser.add_argument("--model", type=str, default="/nfs/rczhang/code/Depth-Anything-3/DA3NESTED-GIANT-LARGE")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--process-res-method", type=str, default="upper_bound_resize")
    parser.add_argument("--view-yaw", type=float, default=0.0, help="Yaw deg about Z axis")
    parser.add_argument("--view-pitch", type=float, default=0.0, help="Pitch deg about X axis")
    parser.add_argument("--view-roll", type=float, default=0.0, help="Roll deg about Y axis")
    parser.add_argument(
        "--view-forward",
        type=float,
        default=0.0,
        help="Move camera forward along its -Z axis (world units).",
    )
    parser.add_argument("--view-tx", type=float, default=0.0, help="Translate camera in X (world units)")
    parser.add_argument("--view-ty", type=float, default=0.0, help="Translate camera in Y (world units)")
    parser.add_argument("--view-tz", type=float, default=0.0, help="Translate camera in Z (world units)")
    parser.add_argument("--output", type=Path, default=Path("/nfs/rczhang/code/Depth-Anything-3/output_gs_from_xyz_frame.png"))
    parser.add_argument(
        "--output-rot",
        type=Path,
        default=Path("/nfs/rczhang/code/Depth-Anything-3/output_gs_from_xyz_frame_rot.png"),
        help="Output path for rotated-view image.",
    )
    args = parser.parse_args()

    if load_safetensors is None:
        raise RuntimeError("safetensors not available; install it in this env.")

    rgb_video = args.dataset_dir / "rgb.mp4"
    geom_path = args.dataset_dir / "geometry.safetensors"
    intr_path = args.dataset_dir / "intrinsics.txt"
    extr_path = args.dataset_dir / "extrinsics.txt"
    if not rgb_video.is_file() or not geom_path.is_file():
        raise FileNotFoundError("Missing rgb.mp4 or geometry.safetensors.")
    if not intr_path.is_file() or not extr_path.is_file():
        raise FileNotFoundError("Missing intrinsics.txt or extrinsics.txt.")

    xyz = load_safetensors(str(geom_path))["xyz"].astype(np.float32)  # (T,H,W,3)
    intr_all = np.loadtxt(intr_path, dtype=np.float32).reshape(-1, 3, 3)
    extr_all = np.loadtxt(extr_path, dtype=np.float32).reshape(-1, 3, 4)

    cap = cv2.VideoCapture(str(rgb_video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_idx)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Failed to read frame {args.frame_idx} from {rgb_video}")

    device = torch.device(args.device)
    model = DepthAnything3.from_pretrained(args.model).to(device=device)
    model.eval()
    da3: DepthAnything3Net = model.model.da3 if hasattr(model.model, "da3") else model.model

    imgs_cpu, _, _ = model.input_processor(
        [frame_bgr[..., ::-1]],
        None,
        None,
        args.process_res,
        args.process_res_method,
        num_workers=1,
        sequential=True,
    )
    imgs, _, _ = model._prepare_model_inputs(imgs_cpu, None, None)
    H, W = imgs.shape[-2], imgs.shape[-1]

    depth = xyz[args.frame_idx][..., 2]
    depth_rs = align_depth_to_da3(depth.astype(np.float32), args.process_res, args.process_res_method)
    if depth_rs.shape != (H, W):
        depth_rs = cv2.resize(depth_rs, (W, H), interpolation=cv2.INTER_LINEAR)
    depth_t = torch.from_numpy(depth_rs)[None, None].to(imgs.device)

    with torch.inference_mode():
        feats, _ = da3.backbone(
            imgs, cam_token=None, export_feat_layers=[], ref_view_strategy="saddle_balanced"
        )
        gs_outs = da3.gs_head(feats=feats, H=H, W=W, patch_start_idx=0, images=imgs)
        raw_gaussians = gs_outs.raw_gs
        densities = gs_outs.raw_gs_conf

        intr = intr_all[args.frame_idx]
        extr = extr_all[args.frame_idx]
        extr4 = np.eye(4, dtype=np.float32)
        extr4[:3, :4] = extr
        in_t = torch.from_numpy(intr)[None, None].to(imgs.device)
        ex_t = torch.from_numpy(extr4)[None, None].to(imgs.device)

        adapter = da3.gs_adapter
        sh_degree = getattr(adapter, "sh_degree", 0)
        pred_color = getattr(adapter, "pred_color", False)
        pred_offset_depth = getattr(adapter, "pred_offset_depth", False)
        pred_offset_xy = getattr(adapter, "pred_offset_xy", True)
        scale_min = getattr(adapter, "gaussian_scale_min", 1e-5)
        scale_max = getattr(adapter, "gaussian_scale_max", 30.0)
        d_sh = 1 if pred_color else (sh_degree + 1) ** 2

        cam2worlds = affine_inverse(ex_t)
        intr_normed = in_t.clone()
        intr_normed[..., 0, :] /= W
        intr_normed[..., 1, :] /= H

        gs_depths = depth_t
        if pred_offset_depth:
            gs_depths = depth_t + raw_gaussians[..., -1]
            raw_gaussians = raw_gaussians[..., :-1]

        xy_ray, _ = sample_image_grid((H, W), device=depth_t.device)
        xy_ray = xy_ray[None, None, ...].expand(1, 1, -1, -1, -1)
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

        scales_raw, rotations_raw, sh_raw = raw_gaussians.split((3, 4, 3 * d_sh), dim=-1)
        scales = scale_min + (scale_max - scale_min) * scales_raw.sigmoid()
        pixel_size = 1 / torch.tensor((W, H), dtype=scales.dtype, device=scales.device)
        multiplier = adapter.get_scale_multiplier(intr_normed, pixel_size)
        gs_scales = scales * gs_depths[..., None] * multiplier[..., None, None, None]
        gs_scales = rearrange(gs_scales, "b v h w d -> b (v h w) d")

        rotations_raw = rotations_raw / (rotations_raw.norm(dim=-1, keepdim=True) + 1e-8)
        cam_quat_xyzw = rearrange(rotations_raw, "b v h w c -> b (v h w) c")
        c2w_mat = repeat(cam2worlds, "b v i j -> b (v h w) i j", h=H, w=W)
        gs_rotations_world = cam_quat_xyzw_to_world_quat_wxyz(cam_quat_xyzw, c2w_mat)

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

        gaussians = Gaussians(
            means=gs_means_world,
            scales=gs_scales,
            rotations=gs_rotations_world,
            harmonics=gs_sh_world,
            opacities=gs_opacities.squeeze(-1),
        )

        # Save original view
        ex_rep = ex_t.repeat(1, 2, 1, 1)
        in_rep = in_t.repeat(1, 2, 1, 1)
        color, _ = run_renderer_in_chunk_w_trj_mode(
            gaussians=gaussians,
            extrinsics=ex_rep,
            intrinsics=in_rep,
            image_shape=(H, W),
            trj_mode="original",
            chunk_size=1,
            enable_tqdm=False,
        )
        frame = color[0, 0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        frame = (frame * 255).astype(np.uint8)

        # Apply rotation if specified, then render rotated view
        if (
            args.view_yaw != 0.0
            or args.view_pitch != 0.0
            or args.view_roll != 0.0
            or args.view_forward != 0.0
            or args.view_tx != 0.0
            or args.view_ty != 0.0
            or args.view_tz != 0.0
        ):
            yaw = torch.deg2rad(torch.tensor(args.view_yaw, device=imgs.device))
            pitch = torch.deg2rad(torch.tensor(args.view_pitch, device=imgs.device))
            roll = torch.deg2rad(torch.tensor(args.view_roll, device=imgs.device))
            cy, sy = torch.cos(yaw), torch.sin(yaw)
            cp, sp = torch.cos(pitch), torch.sin(pitch)
            cr, sr = torch.cos(roll), torch.sin(roll)
            Rz = torch.tensor([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], device=imgs.device)
            Rx = torch.tensor([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], device=imgs.device)
            Ry = torch.tensor([[cr, 0.0, sr], [0.0, 1.0, 0.0], [-sr, 0.0, cr]], device=imgs.device)
            R = Rz @ Rx @ Ry
            c2w = affine_inverse(ex_t)
            c2w_rot = c2w.clone()
            c2w_rot[..., :3, :3] = c2w[..., :3, :3] @ R
            if args.view_forward != 0.0:
                forward = c2w_rot[..., :3, 2]  # camera forward (world)
                c2w_rot[..., :3, 3] += forward * args.view_forward
            if args.view_tx != 0.0 or args.view_ty != 0.0 or args.view_tz != 0.0:
                c2w_rot[..., :3, 3] += torch.tensor(
                    [args.view_tx, args.view_ty, args.view_tz],
                    device=imgs.device,
                )
            ex_t_rot = affine_inverse(c2w_rot)
            ex_rep = ex_t_rot.repeat(1, 2, 1, 1)
            in_rep = in_t.repeat(1, 2, 1, 1)
            color, _ = run_renderer_in_chunk_w_trj_mode(
                gaussians=gaussians,
                extrinsics=ex_rep,
                intrinsics=in_rep,
                image_shape=(H, W),
                trj_mode="original",
                chunk_size=1,
                enable_tqdm=False,
            )
            frame_rot = color[0, 0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            frame_rot = (frame_rot * 255).astype(np.uint8)
        else:
            frame_rot = frame

        ex_rep = ex_t.repeat(1, 2, 1, 1)
        in_rep = in_t.repeat(1, 2, 1, 1)
        color, _ = run_renderer_in_chunk_w_trj_mode(
            gaussians=gaussians,
            extrinsics=ex_rep,
            intrinsics=in_rep,
            image_shape=(H, W),
            trj_mode="original",
            chunk_size=1,
            enable_tqdm=False,
        )
        frame = color[0, 0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        frame = (frame * 255).astype(np.uint8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(args.output_rot), cv2.cvtColor(frame_rot, cv2.COLOR_RGB2BGR))
    print(f"[DONE] saved image to: {args.output}")
    print(f"[DONE] saved rotated image to: {args.output_rot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
