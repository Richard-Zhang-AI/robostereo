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

import json
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING
import warnings


def _load_task_config(task_config_path: str) -> tuple[list[str], dict[str, str]]:
    path = Path(task_config_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Task config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Task config must be a JSON object: {path}")

    env = payload.get("env", {})
    if not isinstance(env, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in env.items()):
        raise ValueError(f"Task config field 'env' must be a dict[str,str]: {path}")

    opts = payload.get("opts", [])
    if not isinstance(opts, list) or any(not isinstance(opt, str) for opt in opts):
        raise ValueError(f"Task config field 'opts' must be a list[str]: {path}")

    return opts, env


def _parse_task_config_arg(argv: list[str]) -> str | None:
    for idx, arg in enumerate(argv):
        if arg == "--task-config":
            if idx + 1 < len(argv):
                return argv[idx + 1]
            return None
        if arg.startswith("--task-config="):
            return arg.split("=", 1)[1]
    return None


def _preload_task_env_from_argv(argv: list[str]) -> None:
    task_config_path = _parse_task_config_arg(argv)
    if not task_config_path:
        return
    _, env = _load_task_config(task_config_path)
    for key, value in env.items():
        os.environ[key] = value


_preload_task_env_from_argv(sys.argv)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")

warnings.filterwarnings(
    "ignore",
    message=r".*doesn't match a supported version!.*",
    module=r"requests\.__init__",
)

import argparse
import importlib

from cosmos_predict2._src.imaginaire.config import Config, pretty_print_overrides
from cosmos_predict2._src.imaginaire.flags import SMOKE
from cosmos_predict2._src.imaginaire.lazy_config import instantiate
from cosmos_predict2._src.imaginaire.lazy_config.lazy import LazyConfig
from cosmos_predict2._src.imaginaire.utils.config_helper import get_config_module, override
from cosmos_predict2._src.imaginaire.utils.launch import log_reproducible_setup
from cosmos_predict2._src.predict2.utils.model_loader import create_model_from_consolidated_checkpoint_with_fsdp
from loguru import logger as logging

from cosmos_oss.init import init_environment, init_output_dir, is_rank0


if TYPE_CHECKING:
    from cosmos_predict2._src.imaginaire.config import Config


@logging.catch(reraise=True)
def launch(config: Config, args: argparse.Namespace) -> None:
    # Check that the config is valid
    config.validate()
    # Freeze the config so developers don't change it during training.
    config.freeze()  # type: ignore
    trainer = config.trainer.type(config)
    log_reproducible_setup(config, args)

    # Create the model and load the consolidated checkpoint if provided.
    # If the checkpoint is in DCP format, checkpoint loading will be handled by the DCP checkpointer.
    if isinstance(config.checkpoint.load_path, str) and config.checkpoint.load_path.endswith(".pt"):
        model = create_model_from_consolidated_checkpoint_with_fsdp(config)
    else:
        model = instantiate(config.model)

    # Create dataloaders. Validation loader is only needed when validation is enabled.
    dataloader_train = instantiate(config.dataloader_train)
    dataloader_val = instantiate(config.dataloader_val) if config.trainer.run_validation else None
    # Start training
    trainer.train(
        model,
        dataloader_train,
        dataloader_val,
    )


def main():
    init_environment()

    # Get the config file from the input arguments.
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--config", help="Path to the config file", required=True)
    parser.add_argument(
        "--task-config",
        default=None,
        help="Path to /config/tasks/<task>/train.json that provides default override opts.",
    )
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Do a dry run without training. Useful for debugging the config.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run profiler and save report to output directory.",
    )
    args = parser.parse_args()
    manual_overrides = [opt for opt in args.opts if opt != "--"]
    task_overrides: list[str] = []
    if args.task_config:
        task_overrides, task_env = _load_task_config(args.task_config)
        for key, value in task_env.items():
            os.environ[key] = value

    config_module = get_config_module(args.config)
    config = importlib.import_module(config_module).make_config()
    # config_helper.override requires a leading "--" token
    overrides = ["--"] + task_overrides + manual_overrides
    if SMOKE:
        overrides.append("trainer.max_iter=2")
        overrides.append("trainer.logging_iter=1")
        overrides.append("trainer.validation_iter=1")
    config = override(config, overrides)

    # If loading from a consolidated .pt, treat as model init only (no training-state resume).
    if isinstance(config.checkpoint.load_path, str) and config.checkpoint.load_path.endswith(".pt"):
        config.checkpoint.resume_from_checkpoint = False
        config.checkpoint.load_training_state = False

    if is_rank0():
        output_dir = Path(config.job.path_local)
        init_output_dir(output_dir, profile=args.profile)

    if args.dryrun:
        logging.info(
            "Config:\n" + config.pretty_print(use_color=True) + "\n" + pretty_print_overrides(overrides, use_color=True)
        )
        os.makedirs(config.job.path_local, exist_ok=True)
        LazyConfig.save_yaml(config, f"{config.job.path_local}/config.yaml")
        print(f"{config.job.path_local}/config.yaml")
    else:
        # Launch the training job.
        launch(config, args)


if __name__ == "__main__":
    main()
