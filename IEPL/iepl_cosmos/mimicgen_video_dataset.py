"""
MimicGen video dataset reader for Cosmos RFT.

Loads annotation JSON + mp4 video and returns:
- init_frames: (H, W, C) uint8
- gt_actions: (T, 7) float32 (unnormalized actions from annotations)
- gt_lengths: scalar length (T+1, includes init frame)
- state_id: numeric id (for compatibility)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import torch
from torch.utils.data import Dataset


class MimicGenVideoDataset(Dataset):
    def __init__(self, root: str, split: str = "train", max_samples: int = -1):
        self.root = Path(root)
        self.split = split
        self.max_samples = max_samples

        ann_dir = self.root / "annotation" / split
        if not ann_dir.exists():
            raise FileNotFoundError(f"annotation dir not found: {ann_dir}")

        json_paths = sorted(ann_dir.glob("*.json"))
        if max_samples and max_samples > 0:
            json_paths = json_paths[: max_samples]

        self.samples: list[dict[str, Any]] = []
        for p in json_paths:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)

            actions = data.get("action") or data.get("actions")
            if actions is None or len(actions) == 0:
                raise RuntimeError(f"Empty or missing actions in annotation: {p}")

            video_rel = data.get("video_path")
            if video_rel is None:
                videos = data.get("videos") or []
                if videos:
                    video_rel = videos[0].get("video_path")
            if video_rel is None:
                continue

            video_path = Path(video_rel)
            if not video_path.is_absolute():
                video_path = (self.root / video_path).resolve()

            self.samples.append(
                {
                    "json_path": str(p),
                    "video_path": str(video_path),
                    "action_len": int(len(actions)),
                    "actions": actions,
                    "sample_id": p.stem,
                }
            )

        if not self.samples:
            raise RuntimeError(f"No valid samples found under {ann_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        video_path = sample["video_path"]

        # Read only the first frame to avoid decoding the full video.
        try:
            reader = imageio.v2.get_reader(video_path)
            init_frame_np = reader.get_data(0)
            reader.close()
        except Exception:
            frames = imageio.v2.mimread(video_path)
            if not frames:
                raise RuntimeError(f"Empty video: {video_path}")
            init_frame_np = frames[0]

        init_frame = torch.from_numpy(np.asarray(init_frame_np, dtype=np.uint8))
        gt_actions = torch.tensor(
            self.samples[idx].get("actions", []), dtype=torch.float32
        )

        return {
            "state_id": torch.tensor(idx, dtype=torch.long),
            "init_frames": init_frame,
            "gt_actions": gt_actions,
            "gt_lengths": torch.tensor(gt_actions.shape[0] + 1, dtype=torch.long),
            "action_len": torch.tensor(sample["action_len"], dtype=torch.long),
            "sample_id": sample["sample_id"],
        }


def mimicgen_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        return {}

    action_list = [b["gt_actions"] for b in batch]
    action_lens = torch.tensor([int(b["action_len"]) for b in batch], dtype=torch.long)
    max_act_len = int(action_lens.max().item())
    act_dim = action_list[0].shape[-1] if action_list else 0
    padded_actions = torch.zeros((len(batch), max_act_len, act_dim), dtype=torch.float32)
    for i, acts in enumerate(action_list):
        padded_actions[i, : acts.shape[0]] = acts

    lengths = torch.tensor([int(b["gt_lengths"]) for b in batch], dtype=torch.long)
    init_frames = torch.stack([b["init_frames"] for b in batch], dim=0)
    state_ids = torch.stack([b["state_id"] for b in batch], dim=0)

    output: dict[str, Any] = {
        "state_id": state_ids,
        "init_frames": init_frames,
        "gt_actions": padded_actions,
        "gt_lengths": lengths,
        "action_len": action_lens,
        "sample_id": np.array([b["sample_id"] for b in batch], dtype=object),
    }
    return output
