#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    from safetensors.numpy import load_file as load_safetensors
except Exception:
    load_safetensors = None


def load_first_extrinsic(extr_path: Path) -> np.ndarray:
    ext = np.loadtxt(extr_path, dtype=np.float32).reshape(-1, 3, 4)[0]
    ext4 = np.eye(4, dtype=np.float32)
    ext4[:3, :4] = ext
    return ext4


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert per-pixel XYZ (H,W,3) to depth map."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/nfs/rczhang/code/cosmos-predict2.5/datasets/bridge/videos/test/0"),
        help="Directory containing geometry.safetensors / depth.npz / extrinsics.txt",
    )
    parser.add_argument(
        "--frame-idx",
        type=int,
        default=0,
        help="Frame index to extract.",
    )
    parser.add_argument(
        "--use-extrinsics",
        action="store_true",
        help="Treat XYZ as world coords and transform to camera with extrinsics.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/nfs/rczhang/code/Depth-Anything-3/output_xyz_to_depth"),
        help="Output directory.",
    )
    parser.add_argument(
        "--xyz-is-normalized",
        action="store_true",
        help="If set, de-normalize xyz using --xyz-min/--xyz-max.",
    )
    parser.add_argument(
        "--xyz-min",
        type=float,
        default=-0.55,
        help="XYZ_MIN used for de-normalization (matches 4dvis.py).",
    )
    parser.add_argument(
        "--xyz-max",
        type=float,
        default=1.51,
        help="XYZ_MAX used for de-normalization (matches 4dvis.py).",
    )
    args = parser.parse_args()

    xyz_path = args.dataset_dir / "geometry.safetensors"
    if not xyz_path.is_file():
        raise FileNotFoundError(f"Missing geometry.safetensors: {xyz_path}")
    if load_safetensors is None:
        raise RuntimeError("safetensors not available; install it in this env.")
    data = load_safetensors(str(xyz_path))
    if "xyz" not in data:
        raise RuntimeError("geometry.safetensors missing key 'xyz'")
    xyz = data["xyz"][args.frame_idx].astype(np.float32)  # (H, W, 3)
    if args.xyz_is_normalized:
        xyz = xyz * (args.xyz_max - args.xyz_min) + args.xyz_min

    if args.use_extrinsics:
        extr_path = args.dataset_dir / "extrinsics.txt"
        if not extr_path.is_file():
            raise FileNotFoundError(f"Missing extrinsics.txt: {extr_path}")
        extr = load_first_extrinsic(extr_path)
        xyz_h = np.concatenate([xyz.reshape(-1, 3), np.ones((xyz.size // 3, 1))], axis=1)
        xyz_cam = (extr @ xyz_h.T).T[:, :3].reshape(xyz.shape)
        depth = xyz_cam[..., 2]
    else:
        depth = xyz[..., 2]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "depth_from_xyz.npy", depth)
    plt.imsave(args.output_dir / "depth_from_xyz.png", depth, cmap="magma")

    # Optional compare with provided depth.npz if available.
    depth_path = args.dataset_dir / "depth.npz"
    if depth_path.is_file():
        with np.load(depth_path) as data:
            depth_ref = data["data"][args.frame_idx]
        diff = depth - depth_ref
        stats = {
            "mean_abs_err": float(np.mean(np.abs(diff))),
            "mean_err": float(np.mean(diff)),
            "max_abs_err": float(np.max(np.abs(diff))),
        }
        np.save(args.output_dir / "depth_ref.npy", depth_ref)
        np.save(args.output_dir / "depth_diff.npy", diff)
        plt.imsave(args.output_dir / "depth_ref.png", depth_ref, cmap="magma")
        plt.imsave(args.output_dir / "depth_diff.png", diff, cmap="coolwarm")
        print("[INFO] compare stats:", stats)

    print(f"[DONE] outputs saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
