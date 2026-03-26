#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_GS_HEAD_ROOT = _SCRIPT_DIR.parent
_PROJECT_ROOT = _GS_HEAD_ROOT.parent
sys.path.insert(0, str(_GS_HEAD_ROOT / "src"))

_DEFAULT_MODEL = str(_PROJECT_ROOT / "checkpoints" / "DA3NESTED-GIANT-LARGE")
_DEFAULT_DATASET = _PROJECT_ROOT / "datasets" / "bridge" / "videos" / "train" / "0"
_DEFAULT_OUTPUT = _GS_HEAD_ROOT / "output_da3_first_frame"

import cv2
import numpy as np
import torch

from depth_anything_3.api import DepthAnything3


def load_first_extrinsic(extr_path: Path) -> np.ndarray:
    ext = np.loadtxt(extr_path, dtype=np.float32).reshape(-1, 3, 4)[0]
    ext4 = np.eye(4, dtype=np.float32)
    ext4[:3, :4] = ext
    return ext4


def load_first_intrinsic(intr_path: Path) -> np.ndarray:
    return np.loadtxt(intr_path, dtype=np.float32).reshape(-1, 3, 3)[0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run original DA3 on the first frame from rgb.mp4."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=_DEFAULT_DATASET,
        help="Directory containing rgb.mp4 and optional intrinsics/extrinsics.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=_DEFAULT_MODEL,
        help="Model path or HF repo for DepthAnything3.from_pretrained",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument(
        "--process-res-method",
        type=str,
        default="upper_bound_resize",
    )
    parser.add_argument(
        "--export-format",
        type=str,
        default="gs_ply-gs_image-depth_vis",
        help="Export format string. Default: gs_ply+gs_image+depth_vis (single-image). Use gs_video for preview video.",
    )
    parser.add_argument(
        "--use-camera-params",
        action="store_true",
        help="Use intrinsics/extrinsics from dataset (single frame may be unstable).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Output directory.",
    )
    args = parser.parse_args()

    rgb_video = args.dataset_dir / "rgb.mp4"
    if not rgb_video.is_file():
        raise FileNotFoundError(f"Missing rgb.mp4: {rgb_video}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.output_dir / "inputs" / "frame0000.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(rgb_video))
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Failed to read first frame from: {rgb_video}")
    if not cv2.imwrite(str(image_path), frame_bgr):
        raise RuntimeError(f"Failed to write frame image: {image_path}")

    intrinsics = None
    extrinsics = None
    if args.use_camera_params:
        intr_path = args.dataset_dir / "intrinsics.txt"
        extr_path = args.dataset_dir / "extrinsics.txt"
        if intr_path.is_file() and extr_path.is_file():
            intr = load_first_intrinsic(intr_path)
            extr = load_first_extrinsic(extr_path)
            intrinsics = np.stack([intr], axis=0)
            extrinsics = np.stack([extr], axis=0)
        else:
            print("[WARN] intrinsics/extrinsics not found, running without camera params.")

    device = torch.device(args.device)
    model = DepthAnything3.from_pretrained(args.model).to(device=device)
    model.eval()

    if extrinsics is not None and extrinsics.shape[0] == 1:
        print("[WARN] single frame: disabling align_to_input_ext_scale to avoid Umeyama failure.")

    # Avoid passing single-frame extrinsics to inference to skip Umeyama alignment.
    ex_for_infer = None if extrinsics is not None and extrinsics.shape[0] == 1 else extrinsics

    with torch.inference_mode():
        model.inference(
            image=[str(image_path)],
            extrinsics=ex_for_infer,
            intrinsics=intrinsics,
            align_to_input_ext_scale=False,
            infer_gs=True,
            process_res=args.process_res,
            process_res_method=args.process_res_method,
            export_dir=str(args.output_dir),
            export_format=args.export_format,
            export_kwargs={
                "gs_ply": {"gs_views_interval": 1},
                "gs_image": {"output_name": "gs_render", "save_depth": True},
                "gs_video": {
                    "trj_mode": "wander",
                    "video_quality": "high",
                    "chunk_size": 1,
                    "vis_depth": "hcat",
                },
            },
        )

    print(f"[DONE] outputs saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
