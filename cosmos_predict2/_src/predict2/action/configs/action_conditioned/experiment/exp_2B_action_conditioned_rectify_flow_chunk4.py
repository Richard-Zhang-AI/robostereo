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

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyDict

AC_REASON_EMBEDDINGS_RECTIFIED_FLOW_2B_BRIDGE_5FRAME_256X320_CHUNK4 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_bridge_13frame_480_640_",
            {"override /net": "cosmos_v1_2B_action_chunk_conditioned"},
            {"override /data_train": "bridge_13frame_480_640_train"},
            {"override /data_val": "bridge_13frame_480_640_val"},
        ],
        job=dict(
            group="official_runs_vid2vid",
            name="cosmos_predict2p5_2B_reason_embeddings_action_conditioned_rectified_flow_bridge_5frame_256x320_chunk4",
            project="cosmos_predict2_action_conditioned",
        ),
        optimizer=dict(
            lr=32e-5,
            weight_decay=0.1,
        ),
        model=dict(
            config=dict(
                state_t=1 + 4 // 4,
                net=dict(
                    action_dim=7,
                    num_action_per_chunk=4,
                    temporal_compression_ratio=4,
                ),
            ),
        ),
        dataloader_train=dict(
            batch_size=8,
            sampler=dict(
                dataset=dict(
                    gripper_rescale_factor=1,
                    num_action_per_chunk=4,
                    fps_downsample_ratio=1,
                    video_size=[256, 320],
                ),
            ),
            dataset=dict(
                gripper_rescale_factor=1,
                num_action_per_chunk=4,
                fps_downsample_ratio=1,
                video_size=[256, 320],
            ),
        ),
    ),
    flags={"allow_objects": True},
)

cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name=AC_REASON_EMBEDDINGS_RECTIFIED_FLOW_2B_BRIDGE_5FRAME_256X320_CHUNK4["job"]["name"],
    node=AC_REASON_EMBEDDINGS_RECTIFIED_FLOW_2B_BRIDGE_5FRAME_256X320_CHUNK4,
)
