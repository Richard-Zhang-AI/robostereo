#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from depth_anything_3.api import DepthAnything3


def read_first_frame(video_path: Path, out_path: Path) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Failed to read first frame from: {video_path}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"Failed to write frame image: {out_path}")
    return out_path


def load_first_camera(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    intr_path = dataset_dir / "intrinsics.txt"
    extr_path = dataset_dir / "extrinsics.txt"

    intr_all = np.loadtxt(intr_path, dtype=np.float32).reshape(-1, 3, 3)
    extr_all = np.loadtxt(extr_path, dtype=np.float32).reshape(-1, 3, 4)

    intr = intr_all[0]
    extr_3x4 = extr_all[0]
    extr = np.eye(4, dtype=np.float32)
    extr[:3, :4] = extr_3x4
    return intr, extr


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract first frame from bridge rgb.mp4 and test DA3 GS head."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/nfs/rczhang/code/cosmos-predict2.5/datasets/bridge/videos/train/0"),
        help="Directory containing rgb.mp4 / intrinsics.txt / extrinsics.txt",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/nfs/rczhang/code/Depth-Anything-3/DA3NESTED-GIANT-LARGE",
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
        "--export-format",
        type=str,
        default="gs_ply-gs_video-depth_vis",
        help="DA3 export format string",
    )
    parser.add_argument(
        "--align-to-input-ext-scale",
        action="store_true",
        help="Align predicted poses to input extrinsics scale (disable for single frame).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/nfs/rczhang/code/Depth-Anything-3/output_bridge_first_frame_gs"),
        help="Output directory",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    video_path = dataset_dir / "rgb.mp4"
    if not video_path.is_file():
        raise FileNotFoundError(f"Missing video: {video_path}")

    frame_path = args.output_dir / "inputs" / "frame0000.png"
    frame_path = read_first_frame(video_path, frame_path)
    print(f"[INFO] frame: {frame_path}")
    print("[INFO] using RGB only (no intrinsics/extrinsics)")

    device = torch.device(args.device)
    model = DepthAnything3.from_pretrained(args.model).to(device=device)
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        prediction = model.inference(
            image=[str(frame_path)],
            extrinsics=None,
            intrinsics=None,
            align_to_input_ext_scale=False,
            infer_gs=True,
            use_ray_pose=False,
            process_res=args.process_res,
            process_res_method=args.process_res_method,
            export_dir=str(args.output_dir),
            export_format=args.export_format,
            export_kwargs={
                "gs_ply": {"gs_views_interval": 1},
                "gs_video": {
                    "trj_mode": "wander",
                    "video_quality": "high",
                    "chunk_size": 1,
                    "vis_depth": "hcat",
                },
            },
        )

    np.save(args.output_dir / "depth.npy", prediction.depth)
    if prediction.conf is not None:
        np.save(args.output_dir / "conf.npy", prediction.conf)
    print(f"[DONE] outputs saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
