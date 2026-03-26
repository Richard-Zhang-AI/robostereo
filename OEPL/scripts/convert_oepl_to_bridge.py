#!/usr/bin/env python3
import argparse
import io
import json
import os
import tarfile
from glob import glob

import imageio
import numpy as np
from tqdm import tqdm


'''

  python scripts/convert_wmpo_to_bridge.py \
    --shards "/nfs/rczhang/code/WMPO/data_files/example_rollouts/square_1280_demos/**/*.tar" \
    --out-root "/nfs/rczhang/code/cosmos-predict2.5/datasets/train_openvla/square_1280" \
    --fps 14 \
    --start-id 0

'''

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _as_uint8_video(video_np: np.ndarray) -> np.ndarray:
    if video_np.ndim != 4:
        raise ValueError(f"video.npy must be 4D, got shape {video_np.shape}")

    # Accept (T, C, H, W) or (T, H, W, C)
    if video_np.shape[1] in (1, 3, 4) and video_np.shape[-1] not in (1, 3, 4):
        video_np = np.transpose(video_np, (0, 2, 3, 1))
    elif video_np.shape[-1] in (1, 3, 4):
        pass
    else:
        raise ValueError(f"Unrecognized video shape {video_np.shape}")

    if video_np.dtype != np.uint8:
        video_np = np.clip(video_np, 0, 255)
        video_np = video_np.astype(np.uint8)

    return video_np


def _write_mp4(video_np: np.ndarray, out_path: str, fps: int = 30) -> None:
    _ensure_dir(os.path.dirname(out_path))
    writer = imageio.get_writer(out_path, fps=fps)
    try:
        for frame in video_np:
            writer.append_data(frame)
    finally:
        writer.close()


def _iter_samples_from_tar(tar_path: str):
    cache = {}
    with tarfile.open(tar_path, "r") as tf:
        for member in tf:
            if not member.isfile():
                continue
            name = member.name
            parts = name.split(".")
            if len(parts) < 3:
                continue
            prefix, kind, ext = parts[0], parts[1], parts[2]
            if kind not in ("video", "action", "meta"):
                continue

            f = tf.extractfile(member)
            if f is None:
                continue
            raw = f.read()
            entry = cache.setdefault(prefix, {})
            if kind == "meta":
                entry["meta"] = json.loads(raw.decode("utf-8"))
            else:
                entry[kind] = np.load(io.BytesIO(raw), allow_pickle=False)

            if "video" in entry and "action" in entry and "meta" in entry:
                yield entry["video"], entry["action"], entry["meta"]
                del cache[prefix]


def convert_dataset(shards_glob: str, out_root: str, fps: int = 30, start_id: int = 0) -> int:
    tar_files = sorted(glob(shards_glob, recursive=True))
    if not tar_files:
        raise FileNotFoundError(f"No .tar files matched: {shards_glob}")

    videos_root = os.path.join(out_root, "videos", "train")
    ann_root = os.path.join(out_root, "annotation", "train")
    _ensure_dir(videos_root)
    _ensure_dir(ann_root)

    sample_id = start_id
    for tar_path in tqdm(tar_files, desc="Shards"):
        for video_np, action_np, meta in _iter_samples_from_tar(tar_path):
            case_dir = os.path.join(videos_root, str(sample_id))
            _ensure_dir(case_dir)

            video_np = _as_uint8_video(video_np)
            video_path = os.path.join(case_dir, "rgb.mp4")
            _write_mp4(video_np, video_path, fps=fps)

            meta_path = os.path.join(case_dir, "meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False)

            ann = {
                "task": "robot_trajectory_prediction",
                "texts": [],
                "videos": [{"video_path": f"videos/train/{sample_id}/rgb.mp4"}],
                "action": action_np.tolist(),
                "state": [],
                "continuous_gripper_state": [],
                "episode_id": str(sample_id),
                "latent_videos": [],
            }
            ann_path = os.path.join(ann_root, f"{sample_id}.json")
            with open(ann_path, "w", encoding="utf-8") as f:
                json.dump(ann, f, ensure_ascii=False)

            sample_id += 1

    return sample_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert WMPO rollouts to bridge-style dataset.")
    parser.add_argument(
        "--shards",
        default="/nfs/rczhang/code/WMPO/data_files/example_rollouts/coffee_128_demos/**/*.tar",
        help="Glob pattern for WMPO webdataset shards.",
    )
    parser.add_argument(
        "--out-root",
        default="/nfs/rczhang/code/cosmos-predict2.5/datasets/bridge",
        help="Output dataset root (will write videos/train and annotation/train).",
    )
    parser.add_argument("--fps", type=int, default=30, help="FPS for output mp4 files.")
    parser.add_argument("--start-id", type=int, default=0, help="Starting id for cases.")
    args = parser.parse_args()

    end_id = convert_dataset(args.shards, args.out_root, fps=args.fps, start_id=args.start_id)
    print(f"Done. Wrote cases {args.start_id}..{end_id - 1} to {args.out_root}.")


if __name__ == "__main__":
    main()
