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

from typing import Callable, Dict, Tuple

import attrs
import torch
from einops import rearrange
from megatron.core import parallel_state
from torch import Tensor

from cosmos_predict2._src.imaginaire.utils import misc
from cosmos_predict2._src.imaginaire.utils.context_parallel import (
    broadcast_split_tensor,
    cat_outputs_cp,
)
from cosmos_predict2._src.predict2.camera.configs.multiview_camera.conditioner import CameraConditionedCondition
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.models.video2world_model_rectified_flow import (
    NUM_CONDITIONAL_FRAMES_KEY,
    Video2WorldModelRectifiedFlow,
    Video2WorldModelRectifiedFlowConfig,
)

IS_PREPROCESSED_KEY = "is_preprocessed"


@attrs.define(slots=False)
class CameraConditionedVideo2WorldRectifiedFlowConfig(Video2WorldModelRectifiedFlowConfig):
    pass


class CameraConditionedVideo2WorldModelRectifiedFlow(Video2WorldModelRectifiedFlow):
    def get_data_and_condition(
        self, data_batch: dict[str, torch.Tensor]
    ) -> Tuple[Tensor, Tensor, CameraConditionedCondition]:
        """
        训练阶段的数据准备函数。
        目的：构建 3 个视角的完整视频作为 Ground Truth，并设置首帧作为条件。
        
        假设：
        - B: Batch Size
        - C: Channels (3)
        - T_view: 单个视角的帧数 (例如 25)
        - H, W: 图像高宽
        - c, t, h, w: VAE 压缩后的 Latent 维度
        
        数据加载器将 3 个视角分成了两部分：
        - input_key + "_cond": 包含 1 个视角 (通常是中间视角 View 1)
        - input_key: 包含 2 个视角 (通常是两侧视角 View 0, View 2)
        """
        
        # 1. 数据归一化和预处理
        self._normalize_multicam_video_databatch_inplace(data_batch)
        self._augment_multicam_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)

        # -------------------------------------------------------------------------
        # 处理 View 1 (中间视角) 的 Latent
        # -------------------------------------------------------------------------
        split_size = data_batch["num_frames"].item() # 获取单个视角的帧数 T_view
        
        # raw_state_cond shape: (B, C, T_view, H, W)
        raw_state_cond = data_batch[self.input_data_key + "_cond"]
        
        # 按帧数切分，这里通常只有 1 个 chunk，因为这个 key 下只有 1 个视角的视频
        # chunks tuple: ((B, C, T_view, H, W), )
        raw_state_cond_chunks = torch.split(raw_state_cond, split_size_or_sections=split_size, dim=2)
        
        latent_state_cond_list = []
        for raw_state_cond_chunk in raw_state_cond_chunks:
            # VAE 编码: (B, C, T_view, H, W) -> (B, c, t_view, h, w)
            latent_state_cond_chunk = self.encode(raw_state_cond_chunk).contiguous().float()
            latent_state_cond_list.append(latent_state_cond_chunk)

        # -------------------------------------------------------------------------
        # 处理 View 0 & View 2 (两侧视角) 的 Latent
        # -------------------------------------------------------------------------
        # raw_state_src shape: (B, C, 2 * T_view, H, W) -> 两个视角在时间维堆叠
        raw_state_src = data_batch[self.input_data_key]
        
        # 按 T_view 切分，得到两个视角的视频
        # raw_state_src_chunks[0]: View 0 (左侧), Shape (B, C, T_view, H, W)
        # raw_state_src_chunks[1]: View 2 (右侧), Shape (B, C, T_view, H, W)
        raw_state_src_chunks = torch.split(raw_state_src, split_size_or_sections=split_size, dim=2)
        
        latent_state_src_list = []
        for raw_state_src_chunk in raw_state_src_chunks:
            # VAE 编码: (B, C, T_view, H, W) -> (B, c, t_view, h, w)
            latent_state_src_chunk = self.encode(raw_state_src_chunk).contiguous().float()
            latent_state_src_list.append(latent_state_src_chunk)

        # -------------------------------------------------------------------------
        # 拼接 Ground Truth
        # 目标结构: [View 0, View 1, View 2] 也就是 [左, 中, 右]
        # -------------------------------------------------------------------------
        # 像素空间拼接: (B, C, 3 * T_view, H, W)
        # 这里 dim=2 是时间维度，将三个视角的视频串联成一个长序列供 Transformer 处理
        raw_state = torch.cat(
            (raw_state_src_chunks[0], raw_state_cond_chunks[0], raw_state_src_chunks[1]),
            dim=2,
        )
        
        # Latent 空间拼接: (B, c, 3 * t_view, h, w)
        latent_state = torch.cat(
            (latent_state_src_list[0], latent_state_cond_list[0], latent_state_src_list[1]),
            dim=2,
        )

        # -------------------------------------------------------------------------
        # 调整相机参数顺序
        # -------------------------------------------------------------------------
        # 原始 extrinsics 可能是 [View 1, View 0, View 2] 的顺序 (根据 list 索引推断)
        # 我们需要将其重排为 [View 0, View 1, View 2] 以匹配 raw_state 的顺序
        chunk_size = len(latent_state_cond_list) + len(latent_state_src_list) # = 3
        extr_list = torch.chunk(data_batch["extrinsics"], chunk_size, dim=1)
        intr_list = torch.chunk(data_batch["intrinsics"], chunk_size, dim=1)
        
        # extr_list[1] -> 原数据的第2部分 (View 0)
        # extr_list[0] -> 原数据的第1部分 (View 1)
        # extr_list[2] -> 原数据的第3部分 (View 2)
        data_batch["extrinsics"] = torch.cat((extr_list[1], extr_list[0], extr_list[2]), dim=1)
        data_batch["intrinsics"] = torch.cat((intr_list[1], intr_list[0], intr_list[2]), dim=1)

        # -------------------------------------------------------------------------
        # 构建条件 (Condition)
        # -------------------------------------------------------------------------
        condition = self.conditioner(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        
        # 将完整的 3 视角 GT 传入。
        # 内部逻辑会根据 num_conditional_frames (例如1) 提取这 3 个视角的首帧作为条件。
        # 训练时，模型学习从 (3视角首帧 + 噪声) -> 恢复 (3视角完整视频)
        condition = condition.set_camera_conditioned_video_condition(
            gt_frames=latent_state.to(**self.tensor_kwargs),
            num_conditional_frames=data_batch.get(NUM_CONDITIONAL_FRAMES_KEY, None),
        )

        # torch.distributed.breakpoint()
        # raw_state: 像素级GT (B, C, 3*T, H, W)
        # latent_state: Latent级GT (B, c, 3*t, h, w)
        # condition: 包含首帧信息和相机参数的条件对象
        return raw_state, latent_state, condition

    def _normalize_multicam_video_databatch_inplace(
        self, data_batch: dict[str, torch.Tensor], input_key: str = None
    ) -> None:
        """
        原地归一化视频数据。
        Input: uint8 [0, 255]
        Output: float32 [-1, 1]
        """
        input_key = self.input_data_key if input_key is None else input_key
        # only handle video batch
        if input_key in data_batch:
            # Check if the data has already been normalized and avoid re-normalizing
            if IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True:
                assert torch.is_floating_point(data_batch[input_key]), "Video data is not in float format."
                assert torch.all(
                    (data_batch[input_key] >= -1.0001)
                    & (data_batch[input_key] <= 1.0001)
                    & (data_batch[input_key + "_cond"] >= -1.0001)
                    & (data_batch[input_key + "_cond"] <= 1.0001)
                ), (
                    f"Video data is not in the range [-1, 1]. get data range [{data_batch[input_key].min()}, {data_batch[input_key].max()}]"
                )
            else:
                assert data_batch[input_key].dtype == torch.uint8, "Video data is not in uint8 format."
                data_batch[input_key] = data_batch[input_key].to(**self.tensor_kwargs) / 127.5 - 1.0
                data_batch[input_key + "_cond"] = data_batch[input_key + "_cond"].to(**self.tensor_kwargs) / 127.5 - 1.0
                data_batch[IS_PREPROCESSED_KEY] = True

    def _augment_multicam_image_dim_inplace(self, data_batch: dict[str, torch.Tensor], input_key: str = None) -> None:
        input_key = self.input_image_key if input_key is None else input_key
        if input_key in data_batch:
            # Check if the data has already been augmented and avoid re-augmenting
            if IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True:
                assert data_batch[input_key].shape[2] == 1, (
                    f"Image data is claimed be augmented while its shape is {data_batch[input_key].shape}"
                )
                return
            else:
                data_batch[input_key] = rearrange(data_batch[input_key], "b c h w -> b c 1 h w").contiguous()
                data_batch[input_key + "_cond"] = rearrange(
                    data_batch[input_key + "_cond"], "b c h w -> b c 1 h w"
                ).contiguous()
                data_batch[IS_PREPROCESSED_KEY] = True

    @torch.no_grad()
    def get_velocity_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        num_input_video: int = 1,
        num_output_video: int = 2,
        is_negative_prompt: bool = False,
    ) -> Callable:
        """
        构造推理时的速度预测函数 (Velocity Function)。
        Input: 首帧图像 (包含在 data_batch 中)。
        Output: 定义好的 velocity_fn，用于 ODE Solver 生成 3 个视角的视频。
        """

        if NUM_CONDITIONAL_FRAMES_KEY in data_batch:
            num_conditional_frames = data_batch[NUM_CONDITIONAL_FRAMES_KEY]
        else:
            # 默认使用 1 帧（首帧）作为条件
            num_conditional_frames = 1

        # -------------------------------------------------------------------------
        # 调整相机参数 (同训练逻辑)
        # 将参数重排为 [View 0, View 1, View 2] 顺序
        # -------------------------------------------------------------------------
        extr_list = torch.chunk(data_batch["extrinsics"], num_input_video + num_output_video, dim=1)
        intr_list = torch.chunk(data_batch["intrinsics"], num_input_video + num_output_video, dim=1)
        data_batch["extrinsics"] = torch.cat((extr_list[1], extr_list[0], extr_list[2]), dim=1)
        data_batch["intrinsics"] = torch.cat((intr_list[1], intr_list[0], intr_list[2]), dim=1)

        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        is_image_batch = self.is_image_batch(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)

        # -------------------------------------------------------------------------
        # 准备初始条件
        # 注意：这里虽然取名叫 x0_cond_chunks，但在推理时，它通常包含了用于参考的初始信息
        # -------------------------------------------------------------------------
        x0_cond_chunks = torch.chunk(data_batch[self.input_data_key], num_input_video, dim=2)
        x0_cond_list = []
        for x0_cond_chunk in x0_cond_chunks:
            x0_cond = self.encode(x0_cond_chunk).contiguous().float()
            x0_cond_list.append(x0_cond)

        # -------------------------------------------------------------------------
        # 构建 Latent Canvas (画布)
        # 这里的结构看似是 [0, Cond, 0]，但在 "3视角首帧生成" 任务中：
        # condition 对象会负责携带 3 个视角的首帧信息。
        # 这里的 x0 主要用于占位和提供形状信息给 conditioner。
        # -------------------------------------------------------------------------
        x0 = torch.cat([torch.zeros_like(x0_cond), x0_cond_list[0], torch.zeros_like(x0_cond)], dim=2)
        
        # 将 x0 传入 conditioner。
        # 关键点：num_conditional_frames 会指示模型去 data_batch 或 gt_frames 中提取
        # 所有 3 个视角的第 0 帧作为已知条件。
        condition = condition.set_camera_conditioned_video_condition(
            gt_frames=x0,
            num_conditional_frames=num_conditional_frames,
        )
        uncondition = uncondition.set_camera_conditioned_video_condition(
            gt_frames=x0,
            num_conditional_frames=num_conditional_frames,
        )

        # -------------------------------------------------------------------------
        # 模型并行处理 (Context Parallelism)
        # -------------------------------------------------------------------------
        # torch.distributed.breakpoint()
        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(x0, condition, None, None)
        _, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(x0, uncondition, None, None)

        if parallel_state.is_initialized():
            pass
        else:
            assert not self.net.is_context_parallel_enabled, (
                "parallel_state is not initialized, context parallel should be turned off."
            )

        def velocity_fn(noise: torch.Tensor, noise_x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            cond_v = self.denoise(noise, noise_x, timestep, condition)
            uncond_v = self.denoise(noise, noise_x, timestep, uncondition)
            velocity_pred = cond_v + guidance * (cond_v - uncond_v)
            return velocity_pred

        return velocity_fn, x0_cond_list

    @torch.no_grad()
    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        num_input_video: int = 1,
        num_output_video: int = 2,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        shift: float = 5.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        采样主函数：生成 3 个视角的视频。
        """

        is_image_batch = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image_batch else self.input_data_key
        if n_sample is None:
            n_sample = data_batch[input_key].shape[0] # B
            
        # 计算 Latent 形状
        if state_shape is None:
            _T, _H, _W = data_batch[input_key].shape[-3:] # 获取 T, H, W (像素空间)
            _T = _T // num_input_video # 计算单个视角的帧数 T_view
            state_shape = [
                self.config.state_ch, # c
                self.tokenizer.get_latent_num_frames(_T), # t_view
                _H // self.tokenizer.spatial_compression_factor, # h
                _W // self.tokenizer.spatial_compression_factor, # w
            ]

        velocity_fn, x0_cond_list = self.get_velocity_fn_from_batch(
            data_batch, guidance, num_input_video, num_output_video, is_negative_prompt=is_negative_prompt
        )

        # -------------------------------------------------------------------------
        # 初始化高斯噪声
        # 我们需要生成 3 个视角的视频内容。
        # 代码逻辑：生成两侧视角的噪声，中间视角可能复用 x0_cond_list[0] (或其形状)
        # -------------------------------------------------------------------------
        noise_list = []
        for i in range(num_output_video): # num_output_video = 2 (View 0 & 2)
            noise = misc.arch_invariant_rand(
                (n_sample,) + tuple(state_shape), # (B, c, t_view, h, w)
                torch.float32,
                self.tensor_kwargs["device"],
                seed,
            )
            noise_list.append(noise)

        # 拼接噪声: [Noise_View0, Latent_View1, Noise_View2]
        # 注意：这里的 Latent_View1 (x0_cond_list[0]) 在推理开始时可能只是占位符或首帧扩展
        # 真正的“首帧条件”是通过 condition 变量注入到网络中的。
        # 这里的 noise 变量代表了Rectified Flow的起点 (X_1)。
        noise = torch.cat([noise_list[0], x0_cond_list[0], noise_list[1]], dim=2)
        # Total Noise Shape: (B, c, 3 * t_view, h, w)

        seed_g = torch.Generator(device=self.tensor_kwargs["device"])
        seed_g.manual_seed(seed)

        self.sample_scheduler.set_timesteps(num_steps, device=self.tensor_kwargs["device"], shift=shift)

        timesteps = self.sample_scheduler.timesteps

        if self.net.is_context_parallel_enabled:
            # 如果开启并行，将 3 * t_view 的时间轴切分到不同 GPU
            noise = broadcast_split_tensor(tensor=noise, seq_dim=2, process_group=self.get_context_parallel_group())
        latents = noise

        # -------------------------------------------------------------------------
        # ODE 采样循环
        # 逐步从噪声中恢复出 3 个视角的连续视频
        # -------------------------------------------------------------------------
        for _, t in enumerate(timesteps):
            latent_model_input = latents
            timestep = [t]

            timestep = torch.stack(timestep)

            velocity_pred = velocity_fn(noise, latent_model_input, timestep.unsqueeze(0))
            temp_x0 = self.sample_scheduler.step(
                velocity_pred.unsqueeze(0), t, latents[0].unsqueeze(0), return_dict=False, generator=seed_g
            )[0]
            latents = temp_x0.squeeze(0)

        if self.net.is_context_parallel_enabled:
            latents = cat_outputs_cp(latents, seq_dim=2, cp_group=self.get_context_parallel_group())

        # 将生成的长序列切分回各个视角
        # sample_chunks[0]: View 0
        # sample_chunks[1]: View 1 (Middle)
        # sample_chunks[2]: View 2
        sample_chunks = torch.chunk(latents, num_input_video + num_output_video, dim=2)
        
        # 返回 View 0 和 View 2 (View 1 在此列表中被省略，具体取决于上层调用需求)
        sample_list = [sample_chunks[0], sample_chunks[2]]

        return sample_list