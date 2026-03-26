# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from pathlib import Path

import torch

from cosmos_predict2._src.imaginaire.lazy_config import instantiate
from cosmos_predict2._src.imaginaire.utils import log, misc
from cosmos_predict2._src.imaginaire.utils.config_helper import get_config_module, override
from cosmos_predict2._src.predict2.checkpointer.dcp import DefaultLoadPlanner, DistributedCheckpointer, dcp_load_state_dict


def _remap_for_ema(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("rgb.net."):
            remapped["rgb.net_ema." + key[len("rgb.net.") :]] = value
        elif key.startswith("xyz.net."):
            remapped["xyz.net_ema." + key[len("xyz.net.") :]] = value
        elif key.startswith("net."):
            remapped["net_ema." + key[len("net.") :]] = value
        else:
            remapped[key] = value
    return remapped


def _remap_from_ema(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("rgb.net_ema."):
            remapped["rgb.net." + key[len("rgb.net_ema.") :]] = value
        elif key.startswith("xyz.net_ema."):
            remapped["xyz.net." + key[len("xyz.net_ema.") :]] = value
        elif key.startswith("net_ema."):
            remapped["net." + key[len("net_ema.") :]] = value
        else:
            remapped[key] = value
    return remapped


def _remap_host_repo_path(path: Path) -> Path:
    """Remap a host-side repo path to the current repo root inside container."""
    repo_root = Path(__file__).resolve().parents[4]  # .../cosmos_predict2/_src/predict2/utils -> repo root
    host_roots = []
    env_root = os.getenv("COSMOS_HOST_REPO_ROOT")
    if env_root:
        host_roots.append(env_root)
    default_root = "/nfs/rczhang/code/cosmos-predict2.5"
    if default_root not in host_roots:
        host_roots.append(default_root)

    for root in host_roots:
        if str(path).startswith(root):
            candidate = Path(str(path).replace(root, str(repo_root), 1))
            if candidate != path:
                log.warning(f"Remapped path from {path} to {candidate}")
            return candidate
    return path


def _resolve_ckpt_dir(ckpt_dir: str) -> str:
    """Resolve checkpoint path inside container by remapping host repo root if needed."""
    path = _remap_host_repo_path(Path(ckpt_dir).expanduser())
    return str(path)


def _resolve_output_pt(output_pt: str) -> str:
    """Resolve output path inside container by remapping host repo root if needed."""
    path = Path(output_pt).expanduser()
    repo_root = Path(__file__).resolve().parents[4]

    if path.is_absolute():
        return str(_remap_host_repo_path(path))

    # If a relative path is provided, resolve under repo root for consistency.
    return str((repo_root / path).resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DCP checkpoint to pt state_dict.")
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to iter_xxxxx/model directory.")
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--output_pt", type=str, required=True)
    parser.add_argument("--use_ema", action="store_true", help="Load EMA weights into regular net keys.")
    args = parser.parse_args()

    config_module = get_config_module(args.config_file)
    config = override(importlib.import_module(config_module).make_config(), ["--", f"experiment={args.experiment}"])
    # Ensure single-process instantiation to avoid distributed init.
    config.model.config.fsdp_shard_size = 1
    if hasattr(config, "model_parallel"):
        config.model_parallel.context_parallel_size = 1
    config.validate()
    config.freeze()  # type: ignore
    misc.set_random_seed(seed=0, by_rank=True)
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True
    model = instantiate(config.model)
    model.on_train_start()

    checkpointer = DistributedCheckpointer(config.checkpoint, config.job, callbacks=None, disable_async=True)
    state_dict = model.state_dict()
    if args.use_ema:
        state_dict = _remap_for_ema(state_dict)

    ckpt_dir = _resolve_ckpt_dir(args.ckpt_dir)
    storage_reader = checkpointer.get_storage_reader(ckpt_dir)
    load_planner = DefaultLoadPlanner(allow_partial_load=True)
    dcp_load_state_dict(state_dict, storage_reader, load_planner)

    if args.use_ema:
        state_dict = _remap_from_ema(state_dict)

    model.load_state_dict(state_dict, strict=False)
    output_pt = _resolve_output_pt(args.output_pt)
    os.makedirs(os.path.dirname(output_pt), exist_ok=True)
    torch.save(model.state_dict(), output_pt)
    log.info(f"Saved pt checkpoint to {output_pt}")


if __name__ == "__main__":
    import importlib

    main()
