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

import os
from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

try:
    from omegaconf import DictConfig, ListConfig
except Exception:
    DictConfig = None  # type: ignore
    ListConfig = None  # type: ignore


from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.action.datasets.dataset_dual_local import DualDataset_3D
from cosmos_predict2._src.predict2.action.datasets.dataset_local import Dataset_3D

try:
    from cosmos_predict2._src.predict2.action.configs.action_conditioned.experiment.gr00t_customized_gr1 import (
        register_gr00t_customized_gr1_data,
    )
except ImportError:
    register_gr00t_customized_gr1_data = None

# bridge dataset path
base_path = "${data_root}"
train_annotation_path = "${data_root}/annotation/train"
val_annotation_path = "${data_root}/annotation/val"
test_annotation_path = "${data_root}/annotation/test"


# experiment for next-frame prediction
bridge_train_dataset = L(Dataset_3D)(
    train_annotation_path=train_annotation_path,
    val_annotation_path=val_annotation_path,
    test_annotation_path=test_annotation_path,
    video_path=base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=1,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[256, 320],
    val_start_frame_interval=1,
    mode="train",
    video_source="${video_source}",  # Options: "rgb" for rgb.mp4, "geometry" for geometry.safetensors
    geometry_normalize="${geometry_normalize}",
    geometry_min="${geometry_min}",
    geometry_max="${geometry_max}",
)
bridge_val_dataset = L(Dataset_3D)(
    train_annotation_path=train_annotation_path,
    val_annotation_path=val_annotation_path,
    test_annotation_path=test_annotation_path,
    video_path=base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=1,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[256, 320],
    val_start_frame_interval=1,
    mode="val",
    video_source="${video_source}",  # Options: "rgb" for rgb.mp4, "geometry" for geometry.safetensors
    geometry_normalize="${geometry_normalize}",
    geometry_min="${geometry_min}",
    geometry_max="${geometry_max}",
)

# experiment for action-sequence video prediction
bridge_13frame_480_640_train_dataset = L(Dataset_3D)(
    train_annotation_path=train_annotation_path,
    val_annotation_path=val_annotation_path,
    test_annotation_path=test_annotation_path,
    video_path=base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=12,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[480, 640],
    val_start_frame_interval=1,
    mode="train",
    video_source="${video_source}",  # Options: "rgb" for rgb.mp4, "geometry" for geometry.safetensors
    geometry_normalize="${geometry_normalize}",
    geometry_min="${geometry_min}",
    geometry_max="${geometry_max}",
)
bridge_13frame_480_640_val_dataset = L(Dataset_3D)(
    train_annotation_path=train_annotation_path,
    val_annotation_path=val_annotation_path,
    test_annotation_path=test_annotation_path,
    video_path=base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=12,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[480, 640],
    val_start_frame_interval=1,
    mode="val",
    video_source="${video_source}",  # Options: "rgb" for rgb.mp4, "geometry" for geometry.safetensors
    geometry_normalize="${geometry_normalize}",
    geometry_min="${geometry_min}",
    geometry_max="${geometry_max}",
)

# Dual RGB+geometry dataset
bridge_13frame_480_640_train_dual_dataset = L(DualDataset_3D)(
    train_annotation_path=train_annotation_path,
    val_annotation_path=val_annotation_path,
    test_annotation_path=test_annotation_path,
    video_path=base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=12,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[480, 640],
    val_start_frame_interval=1,
    mode="train",
    geometry_normalize="${geometry_normalize}",
    geometry_min="${geometry_min}",
    geometry_max="${geometry_max}",
)
bridge_13frame_480_640_val_dual_dataset = L(DualDataset_3D)(
    train_annotation_path=train_annotation_path,
    val_annotation_path=val_annotation_path,
    test_annotation_path=test_annotation_path,
    video_path=base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=12,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[480, 640],
    val_start_frame_interval=1,
    mode="val",
    geometry_normalize="${geometry_normalize}",
    geometry_min="${geometry_min}",
    geometry_max="${geometry_max}",
)


# ------------------------------------------------------------


# create dataloader for each dataset
def get_sampler(dataset):
    world_size = parallel_state.get_data_parallel_world_size()
    rank = parallel_state.get_data_parallel_rank()
    if os.getenv("COSMOS_DATALOADER_DEBUG", "0") == "1":
        try:
            ds_len = len(dataset)
        except Exception as exc:
            ds_len = f"error: {exc}"
        print(
            f"[DatasetSampler] world_size={world_size} rank={rank} dataset_len={ds_len}",
            flush=True,
        )
    return DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=0,
    )


def _is_omegaconf_container(value) -> bool:
    if DictConfig is None and ListConfig is None:
        return False
    return isinstance(value, (DictConfig, ListConfig))


def build_dataloader(
    dataset,
    batch_size=1,
    drop_last=True,
    sampler=None,
    num_workers=None,
    pin_memory=None,
    prefetch_factor=None,
    persistent_workers=None,
    **kwargs,
):
    """Build a DataLoader from a single dataset instance to avoid duplicate scans.

    If a sampler object is provided, use it; otherwise build a DistributedSampler.
    Supports env defaults for worker settings to prevent stalls on slow video I/O.
    """
    if sampler is None or isinstance(sampler, dict) or _is_omegaconf_container(sampler):
        sampler = get_sampler(dataset)

    if num_workers is None:
        num_workers = int(os.getenv("COSMOS_NUM_WORKERS", "4"))
    num_workers = max(0, num_workers)

    dl_kwargs = dict(
        dataset=dataset,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    if num_workers > 0:
        if prefetch_factor is None:
            prefetch_factor = int(os.getenv("COSMOS_PREFETCH_FACTOR", "2"))
        if persistent_workers is None:
            persistent_workers = os.getenv("COSMOS_PERSISTENT_WORKERS", "1") != "0"
        if pin_memory is None:
            pin_memory = os.getenv("COSMOS_PIN_MEMORY", "1") != "0"
        dl_kwargs["prefetch_factor"] = prefetch_factor
        dl_kwargs["persistent_workers"] = persistent_workers
        dl_kwargs["pin_memory"] = pin_memory
    else:
        if pin_memory is None:
            pin_memory = False
        dl_kwargs["pin_memory"] = pin_memory

    dl_kwargs.update(kwargs)
    return DataLoader(**dl_kwargs)


def build_webdataset(webdataset_instance, **kwargs):
    """Helper function to build WebDataset from a WebDataset instance.

    WebDatasets need to call build_dataset() to get the actual iterable dataset
    that can be used with DataLoader.

    Args:
        webdataset_instance: An instantiated WebDataset object.
        **kwargs: Additional parameters to override on the webdataset instance
            before building. This allows experiment configs to override parameters
            like gripper_rescale_factor, num_action_per_chunk, etc.
    """
    # Apply any parameter overrides to the webdataset instance
    for key, value in kwargs.items():
        if hasattr(webdataset_instance, key):
            setattr(webdataset_instance, key, value)
    return webdataset_instance.build_dataset()


bridge_train_dataloader = L(build_dataloader)(
    dataset=bridge_train_dataset,
    batch_size=1,
    drop_last=True,
)
bridge_val_dataloader = L(build_dataloader)(
    dataset=bridge_val_dataset,
    batch_size=1,
    drop_last=True,
)

bridge_13frame_480_640_train_dataloader = L(build_dataloader)(
    dataset=bridge_13frame_480_640_train_dataset,
    batch_size=1,
    drop_last=True,
)
bridge_13frame_480_640_val_dataloader = L(build_dataloader)(
    dataset=bridge_13frame_480_640_val_dataset,
    batch_size=1,
    drop_last=True,
)

bridge_13frame_480_640_train_dual_dataloader = L(build_dataloader)(
    dataset=bridge_13frame_480_640_train_dual_dataset,
    batch_size=1,
    drop_last=True,
)
bridge_13frame_480_640_val_dual_dataloader = L(build_dataloader)(
    dataset=bridge_13frame_480_640_val_dual_dataset,
    batch_size=1,
    drop_last=True,
)


def register_training_and_val_data():
    cs = ConfigStore.instance()
    from cosmos_predict2._src.predict2.configs.common.mock_data import MOCK_DATA_INTERLEAVE_CONFIG

    # Always register mock dataloaders to satisfy defaults when not overridden
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="mock",
        node=MOCK_DATA_INTERLEAVE_CONFIG,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="mock",
        node=MOCK_DATA_INTERLEAVE_CONFIG,
    )

    cs.store(
        group="data_train",
        package="dataloader_train",
        name="bridge_train",
        node=bridge_train_dataloader,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="bridge_val",
        node=bridge_val_dataloader,
    )

    # 13 frame 480 640
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="bridge_13frame_480_640_train",
        node=bridge_13frame_480_640_train_dataloader,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="bridge_13frame_480_640_val",
        node=bridge_13frame_480_640_val_dataloader,
    )
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="bridge_13frame_480_640_train_dual",
        node=bridge_13frame_480_640_train_dual_dataloader,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="bridge_13frame_480_640_val_dual",
        node=bridge_13frame_480_640_val_dual_dataloader,
    )

    # Register gr00t_customized_gr1 data
    if register_gr00t_customized_gr1_data is not None:
        register_gr00t_customized_gr1_data()
