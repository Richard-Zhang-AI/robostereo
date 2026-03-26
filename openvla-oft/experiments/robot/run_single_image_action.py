#!/usr/bin/env python3
"""
Single-image, single-step OpenVLA inference.

Example:
  python experiments/robot/run_single_image_action.py \
    --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
    --image /path/to/image.png \
    --task "move the red block to the right"
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image
import torch

# Append repo root so experiments.* imports work.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from experiments.robot.openvla_utils import (  # noqa: E402
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (  # noqa: E402
    get_action,
    get_image_resize_size,
    get_model,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class SimpleConfig:
    model_family: str = "openvla"
    pretrained_checkpoint: str = ""
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50
    use_film: bool = False
    num_images_in_input: int = 1
    use_proprio: bool = False
    center_crop: bool = True
    lora_rank: int = 32
    unnorm_key: str = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    task_suite_name: str = "libero_spatial"


def _check_unnorm_key(cfg: SimpleConfig, model) -> None:
    unnorm_key = cfg.task_suite_name
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"
    if unnorm_key not in model.norm_stats:
        raise ValueError(f"Action un-norm key {unnorm_key} not found in model norm_stats.")
    cfg.unnorm_key = unnorm_key


def _load_image(path: Path) -> np.ndarray:
    pil = Image.open(path).convert("RGB")
    return np.asarray(pil)


def _parse_proprio(values: Optional[Sequence[float]]) -> np.ndarray:
    if values is None:
        return np.zeros(8, dtype=np.float32)
    if len(values) != 8:
        raise ValueError("Expected 8 proprio values: [eef_pos(3), eef_axis_angle(3), gripper_qpos(2)].")
    return np.asarray(values, dtype=np.float32)


def build_cfg(args: argparse.Namespace) -> SimpleConfig:
    return SimpleConfig(
        pretrained_checkpoint=args.pretrained_checkpoint,
        use_l1_regression=not args.use_diffusion,
        use_diffusion=args.use_diffusion,
        num_diffusion_steps_train=args.num_diffusion_steps,
        num_diffusion_steps_inference=args.num_diffusion_steps,
        num_images_in_input=2 if args.wrist_image else 1,
        use_proprio=args.use_proprio,
        center_crop=not args.no_center_crop,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        task_suite_name=args.task_suite_name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-image OpenVLA inference.")
    parser.add_argument("--pretrained_checkpoint", required=True)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--task", required=True, help="Task description for the prompt.")
    parser.add_argument("--wrist_image", type=Path, default=None)
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--use_proprio", action="store_true", help="Enable 8D proprio input.")
    parser.add_argument("--proprio", type=float, nargs=8, default=None)
    parser.add_argument("--use_diffusion", action="store_true")
    parser.add_argument("--num_diffusion_steps", type=int, default=50)
    parser.add_argument("--no_center_crop", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    args = parser.parse_args()

    cfg = build_cfg(args)

    logger.info("Loading model from %s", cfg.pretrained_checkpoint)
    model = get_model(cfg)
    processor = get_processor(cfg)
    _check_unnorm_key(cfg, model)

    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)

    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, model.llm_dim)

    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, model.llm_dim)

    resize_size = get_image_resize_size(cfg)
    full_image = resize_image_for_policy(_load_image(args.image), resize_size)

    observation = {"full_image": full_image}
    if args.wrist_image:
        wrist_image = resize_image_for_policy(_load_image(args.wrist_image), resize_size)
        observation["wrist_image"] = wrist_image

    if cfg.use_proprio:
        observation["state"] = _parse_proprio(args.proprio)

    actions = get_action(
        cfg,
        model,
        observation,
        args.task,
        processor=processor,
        action_head=action_head,
        proprio_projector=proprio_projector,
        noisy_action_projector=noisy_action_projector,
        use_film=cfg.use_film,
    )

    action_chunk = np.stack(actions, axis=0)
    logger.info("action_chunk shape=%s dtype=%s", action_chunk.shape, action_chunk.dtype)
    print("raw_action_chunk:")
    print(np.array2string(action_chunk, precision=3, floatmode="fixed"))
    print("raw_action_first:")
    print(np.array2string(action_chunk[0], precision=3, floatmode="fixed"))


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
