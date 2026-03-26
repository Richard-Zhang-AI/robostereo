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

from __future__ import annotations
import os

from typing import Any, Callable, Dict, Mapping, Tuple

import torch
import tqdm
import torch.amp as amp
import attrs
from einops import rearrange
from torch.distributed._tensor.api import DTensor

from cosmos_predict2._src.imaginaire.lazy_config import LazyDict, instantiate
from cosmos_predict2._src.imaginaire.model import ImaginaireModel
from cosmos_predict2._src.imaginaire.utils import log, misc
from cosmos_predict2._src.imaginaire.flags import INTERNAL
from cosmos_predict2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path
from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io
from cosmos_predict2._src.imaginaire.utils.optim_instantiate import get_base_scheduler
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.action.models.action_conditioned_video2world_rectified_flow_model import (
    ActionVideo2WorldModelRectifiedFlow,
    Video2WorldModelRectifiedFlowConfig,
)


@attrs.define(slots=False)
class DualActionConditionedConfig(Video2WorldModelRectifiedFlowConfig):
    rgb_init_checkpoint_path: str | None = None
    xyz_init_checkpoint_path: str | None = None
    rgb_init_load_ema_to_reg: bool = True
    xyz_init_load_ema_to_reg: bool = True
    rgb_loss_weight: float = 1.0
    xyz_loss_weight: float = 1.0
    cross_attn_interval: int = 1
    cross_attn_warmup_iters: int = 0
    cross_attn_dropout: float = 0.0
    cross_attn_weight: float = 1.0


class DualActionVideo2WorldModelRectifiedFlow(ImaginaireModel):
    def __init__(self, config: DualActionConditionedConfig):
        super().__init__()
        self.config = config
        self.rgb_model = ActionVideo2WorldModelRectifiedFlow(config=config)
        self.xyz_model = ActionVideo2WorldModelRectifiedFlow(config=config)
        self.precision = self.rgb_model.precision
        self.tokenizer = self.rgb_model.tokenizer
        self.tensor_kwargs = self.rgb_model.tensor_kwargs
        self.input_data_key = self.rgb_model.input_data_key
        self.input_image_key = self.rgb_model.input_image_key
        self.input_caption_key = self.rgb_model.input_caption_key
        self._init_cross_tower_modules()
        self._cross_attn_dropout = torch.nn.Dropout(p=float(self.config.cross_attn_dropout))
        self._global_step = 0

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            modules = object.__getattribute__(self, "_modules")
            if name == "rgb":
                return modules.get("rgb_model")
            if name == "xyz":
                return modules.get("xyz_model")
            if name == "gates":
                return modules.get("gates")
            if name == "cross_attn":
                return modules.get("cross_attn")
            rgb_model = modules.get("rgb_model")
            if rgb_model is not None and hasattr(rgb_model, name):
                return getattr(rgb_model, name)
            xyz_model = modules.get("xyz_model")
            if xyz_model is not None and hasattr(xyz_model, name):
                return getattr(xyz_model, name)
            raise

    def _init_cross_tower_modules(self) -> None:
        num_blocks = len(self.rgb_model.net.blocks)
        hidden_dim = self.rgb_model.net.model_channels
        num_heads = self.rgb_model.net.num_heads
        rgb_to_xyz_gates = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, 1, bias=True) for _ in range(num_blocks)])
        xyz_to_rgb_gates = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, 1, bias=True) for _ in range(num_blocks)])
        self.gates = torch.nn.ModuleDict(
            {
                "rgb_to_xyz": rgb_to_xyz_gates,
                "xyz_to_rgb": xyz_to_rgb_gates,
            }
        )
        self.cross_attn = torch.nn.ModuleDict(
            {
                "rgb_from_xyz": torch.nn.ModuleList(
                    [torch.nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True) for _ in range(num_blocks)]
                ),
                "xyz_from_rgb": torch.nn.ModuleList(
                    [torch.nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True) for _ in range(num_blocks)]
                ),
                "rgb_norm": torch.nn.ModuleList([torch.nn.LayerNorm(hidden_dim) for _ in range(num_blocks)]),
                "xyz_norm": torch.nn.ModuleList([torch.nn.LayerNorm(hidden_dim) for _ in range(num_blocks)]),
            }
        )
        for gate in list(self.gates["rgb_to_xyz"]) + list(self.gates["xyz_to_rgb"]):
            torch.nn.init.zeros_(gate.weight)
            torch.nn.init.zeros_(gate.bias)
        self._log_crosstower_init_status()

    def _cross_attn_scale(self, iteration: int) -> float:
        if self.config.cross_attn_warmup_iters <= 0:
            return float(self.config.cross_attn_weight)
        return float(self.config.cross_attn_weight) * min(
            1.0, float(iteration) / float(self.config.cross_attn_warmup_iters)
        )

    def _log_crosstower_init_status(self) -> None:
        gate_weights = []
        gate_biases = []
        for gate in list(self.gates["rgb_to_xyz"]) + list(self.gates["xyz_to_rgb"]):
            gate_weights.append(gate.weight.detach().abs().max().item())
            if gate.bias is not None:
                gate_biases.append(gate.bias.detach().abs().max().item())
        gate_w_max = max(gate_weights) if gate_weights else 0.0
        gate_b_max = max(gate_biases) if gate_biases else 0.0
        attn_weights = []
        for attn in list(self.cross_attn["rgb_from_xyz"]) + list(self.cross_attn["xyz_from_rgb"]):
            attn_weights.append(attn.in_proj_weight.detach().abs().mean().item())
        attn_mean = sum(attn_weights) / len(attn_weights) if attn_weights else 0.0
        log.info(
            f"[dual] Crosstower init: gate_weight_abs_max={gate_w_max:.6f} "
            f"gate_bias_abs_max={gate_b_max:.6f} cross_attn_in_proj_abs_mean={attn_mean:.6f}"
        )

    def init_optimizer_scheduler(
        self, optimizer_config: LazyDict, scheduler_config: LazyDict
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        optimizer = instantiate(optimizer_config, model=self)
        scheduler = get_base_scheduler(optimizer, self, scheduler_config)
        return optimizer, scheduler

    def on_train_start(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        self.rgb_model.on_train_start(memory_format)
        self.xyz_model.on_train_start(memory_format)
        gate_kwargs = {"device": self.tensor_kwargs["device"], "dtype": self.precision}
        self.gates.to(**gate_kwargs)
        self.cross_attn.to(**gate_kwargs)
        log.info(
            f"[dual] on_train_start: rgb_init_checkpoint_path={self.config.rgb_init_checkpoint_path} "
            f"xyz_init_checkpoint_path={self.config.xyz_init_checkpoint_path}"
        )
        self._log_crosstower_init_status()
        self._maybe_load_init_weights()

    def is_image_batch(self, data_batch: dict[str, torch.Tensor]) -> bool:
        return self.rgb_model.is_image_batch(data_batch["rgb"])

    def get_data_and_condition(self, data_batch: dict[str, torch.Tensor]):
        data_batch = self._cast_batch_to_precision(data_batch)
        return self.rgb_model.get_data_and_condition(data_batch["rgb"])

    def generate_samples_from_batch(self, data_batch: dict[str, torch.Tensor], **kwargs):
        data_batch = self._cast_batch_to_precision(data_batch)
        return self.rgb_model.generate_samples_from_batch(data_batch["rgb"], **kwargs)

    def _cast_batch_to_precision(self, batch: dict[str, Any]) -> dict[str, Any]:
        def _cast(obj: Any) -> Any:
            if isinstance(obj, torch.Tensor) and torch.is_floating_point(obj):
                return obj.to(dtype=self.precision)
            if isinstance(obj, dict):
                return {k: _cast(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_cast(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(_cast(v) for v in obj)
            return obj

        return _cast(batch)

    def _prepare_forward_inputs(self, model: ActionVideo2WorldModelRectifiedFlow, data_batch: dict[str, Any]):
        if model.config.text_encoder_config is not None and model.config.text_encoder_config.compute_online:
            text_embeddings = model.text_encoder.compute_text_embeddings_online(data_batch, model.input_caption_key)
            data_batch["t5_text_embeddings"] = text_embeddings
            data_batch["t5_text_mask"] = torch.ones(text_embeddings.shape[0], text_embeddings.shape[1], device="cuda")

        _, x0_B_C_T_H_W, condition = model.get_data_and_condition(data_batch)
        epsilon_B_C_T_H_W = torch.randn(x0_B_C_T_H_W.size(), **model.tensor_kwargs_fp32)
        batch_size = x0_B_C_T_H_W.size()[0]
        t_B = model.rectified_flow.sample_train_time(batch_size).to(**model.tensor_kwargs_fp32)
        t_B = rearrange(t_B, "b -> b 1")

        x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, t_B = model.broadcast_split_for_model_parallelsim(
            x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, t_B
        )
        timesteps = model.rectified_flow.get_discrete_timestamp(t_B, model.tensor_kwargs_fp32)

        if model.config.use_high_sigma_strategy:
            mask = torch.rand(timesteps.shape, device=timesteps.device) < model.config.high_sigma_ratio
            candidate_timesteps = model.rectified_flow.noise_scheduler.timesteps.to(device=timesteps.device)
            candidate_timesteps = candidate_timesteps[
                (candidate_timesteps >= model.config.high_sigma_timesteps_min)
                & (candidate_timesteps <= model.config.high_sigma_timesteps_max)
            ]
            if len(candidate_timesteps) > 0:
                new_timesteps = candidate_timesteps[torch.randint(0, len(candidate_timesteps), timesteps.shape)]
                timesteps = torch.where(mask, new_timesteps, timesteps)
            else:
                raise ValueError("No candidate timesteps found for high sigma strategy")

        sigmas = model.rectified_flow.get_sigmas(
            timesteps,
            model.tensor_kwargs_fp32,
        )
        timesteps = rearrange(timesteps, "b -> b 1")
        sigmas = rearrange(sigmas, "b -> b 1")
        xt_B_C_T_H_W, vt_B_C_T_H_W = model.rectified_flow.get_interpolation(
            epsilon_B_C_T_H_W, x0_B_C_T_H_W, sigmas
        )
        time_weights_B = model.rectified_flow.train_time_weight(timesteps, model.tensor_kwargs_fp32)

        return dict(
            x0=x0_B_C_T_H_W,
            xt=xt_B_C_T_H_W,
            vt=vt_B_C_T_H_W,
            timesteps=timesteps,
            sigmas=sigmas,
            condition=condition,
            noise=epsilon_B_C_T_H_W,
            time_weights=time_weights_B,
        )

    def _prepare_net_state(self, net, x_B_C_T_H_W, timesteps_B_T, condition):
        data_type = condition.data_type
        if data_type == DataType.VIDEO:
            mask = condition.condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, mask], dim=1)
        else:
            B, _, T, H, W = x_B_C_T_H_W.shape
            x_B_C_T_H_W = torch.cat(
                [
                    x_B_C_T_H_W,
                    torch.zeros((B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device),
                ],
                dim=1,
            )

        timesteps_B_T = timesteps_B_T * net.timestep_scale
        action = condition.action
        assert action is not None, "action must be provided"
        action = rearrange(action, "b t d -> b 1 (t d)")
        action = action.to(dtype=net.action_embedder_B_D.fc1.weight.dtype)
        action_emb_B_D = net.action_embedder_B_D(action)
        action_emb_B_3D = net.action_embedder_B_3D(action)

        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D = net.prepare_embedded_sequence(
            x_B_C_T_H_W,
            fps=condition.fps,
            padding_mask=condition.padding_mask,
        )

        crossattn_emb = condition.crossattn_emb
        if net.use_crossattn_projection:
            crossattn_emb = net.crossattn_proj(crossattn_emb)

        img_context_emb = getattr(condition, "img_context_emb", None)
        if img_context_emb is not None:
            img_context_emb = net.img_context_proj(img_context_emb)
            context_input = (crossattn_emb, img_context_emb)
        else:
            context_input = crossattn_emb

        with amp.autocast("cuda", enabled=net.use_wan_fp32_strategy, dtype=torch.float32):
            if timesteps_B_T.ndim == 1:
                timesteps_B_T = timesteps_B_T.unsqueeze(1)
            t_embedding_B_T_D, adaln_lora_B_T_3D = net.t_embedder(timesteps_B_T)
            t_embedding_B_T_D = t_embedding_B_T_D + action_emb_B_D
            adaln_lora_B_T_3D = adaln_lora_B_T_3D + action_emb_B_3D
            t_embedding_B_T_D = net.t_embedding_norm(t_embedding_B_T_D)

        net.affline_scale_log_info = {"t_embedding_B_T_D": t_embedding_B_T_D.detach()}
        net.affline_emb = t_embedding_B_T_D
        net.crossattn_emb = crossattn_emb

        return dict(
            x=x_B_T_H_W_D,
            rope_emb=rope_emb_L_1_1_D,
            extra_pos_emb=extra_pos_emb_B_T_H_W_D,
            t_embedding=t_embedding_B_T_D,
            adaln_lora=adaln_lora_B_T_3D,
            context=context_input,
        )

    def _apply_video_condition(self, xt_B_C_T_H_W, condition):
        if not condition.is_video:
            return xt_B_C_T_H_W, None
        condition_state = condition.gt_frames.type_as(xt_B_C_T_H_W)
        if not condition.use_video_condition:
            condition_state = condition_state * 0
        _, C, _, _, _ = xt_B_C_T_H_W.shape
        mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(1, C, 1, 1, 1).type_as(xt_B_C_T_H_W)
        xt_B_C_T_H_W = condition_state * mask + xt_B_C_T_H_W * (1 - mask)
        return xt_B_C_T_H_W, mask

    def _forward_nets_with_crosstower(
        self,
        rgb_net,
        xyz_net,
        rgb_state: dict[str, torch.Tensor],
        xyz_state: dict[str, torch.Tensor],
        iteration: int = 0,
    ):
        rgb_x = rgb_state["x"]
        xyz_x = xyz_state["x"]
        interval = max(1, int(self.config.cross_attn_interval))
        scale = self._cross_attn_scale(iteration)
        for i, (rgb_block, xyz_block) in enumerate(zip(rgb_net.blocks, xyz_net.blocks)):
            rgb_out = rgb_block(
                rgb_x,
                rgb_state["t_embedding"],
                rgb_state["context"],
                rope_emb_L_1_1_D=rgb_state["rope_emb"],
                adaln_lora_B_T_3D=rgb_state["adaln_lora"],
                extra_per_block_pos_emb=rgb_state["extra_pos_emb"],
            )
            xyz_out = xyz_block(
                xyz_x,
                xyz_state["t_embedding"],
                xyz_state["context"],
                rope_emb_L_1_1_D=xyz_state["rope_emb"],
                adaln_lora_B_T_3D=xyz_state["adaln_lora"],
                extra_per_block_pos_emb=xyz_state["extra_pos_emb"],
            )
            if i % interval == 0:
                rgb_tokens = rearrange(rgb_out, "b t h w d -> b (t h w) d")
                xyz_tokens = rearrange(xyz_out, "b t h w d -> b (t h w) d")
                rgb_tokens = self.cross_attn["rgb_norm"][i](rgb_tokens)
                xyz_tokens = self.cross_attn["xyz_norm"][i](xyz_tokens)
                rgb_dtype = self.cross_attn["rgb_from_xyz"][i].in_proj_weight.dtype
                xyz_dtype = self.cross_attn["xyz_from_rgb"][i].in_proj_weight.dtype
                rgb_attn, _ = self.cross_attn["rgb_from_xyz"][i](
                    rgb_tokens.to(dtype=rgb_dtype),
                    xyz_tokens.to(dtype=rgb_dtype),
                    xyz_tokens.to(dtype=rgb_dtype),
                )
                xyz_attn, _ = self.cross_attn["xyz_from_rgb"][i](
                    xyz_tokens.to(dtype=xyz_dtype),
                    rgb_tokens.to(dtype=xyz_dtype),
                    rgb_tokens.to(dtype=xyz_dtype),
                )
                rgb_attn = rearrange(rgb_attn, "b (t h w) d -> b t h w d", t=rgb_out.shape[1], h=rgb_out.shape[2])
                xyz_attn = rearrange(xyz_attn, "b (t h w) d -> b t h w d", t=xyz_out.shape[1], h=xyz_out.shape[2])
                rgb_attn = self._cross_attn_dropout(rgb_attn).to(dtype=rgb_out.dtype)
                xyz_attn = self._cross_attn_dropout(xyz_attn).to(dtype=xyz_out.dtype)
                rgb_gate = torch.tanh(self.gates["xyz_to_rgb"][i](rgb_attn)).type_as(rgb_out)
                xyz_gate = torch.tanh(self.gates["rgb_to_xyz"][i](xyz_attn)).type_as(xyz_out)
                rgb_x = rgb_out + rgb_gate * rgb_attn * scale
                xyz_x = xyz_out + xyz_gate * xyz_attn * scale
            else:
                rgb_x = rgb_out
                xyz_x = xyz_out

        rgb_tokens = rgb_net.final_layer(rgb_x, rgb_state["t_embedding"], adaln_lora_B_T_3D=rgb_state["adaln_lora"])
        xyz_tokens = xyz_net.final_layer(xyz_x, xyz_state["t_embedding"], adaln_lora_B_T_3D=xyz_state["adaln_lora"])
        rgb_out = rgb_net.unpatchify(rgb_tokens)
        xyz_out = xyz_net.unpatchify(xyz_tokens)
        return rgb_out, xyz_out

    def _dual_forward(self, data_batch: dict[str, Any]):
        rgb_inputs = self._prepare_forward_inputs(self.rgb_model, data_batch["rgb"])
        xyz_inputs = self._prepare_forward_inputs(self.xyz_model, data_batch["xyz"])
        xyz_inputs["timesteps"] = rgb_inputs["timesteps"]
        xyz_inputs["sigmas"] = rgb_inputs["sigmas"]
        xyz_inputs["xt"], xyz_inputs["vt"] = self.xyz_model.rectified_flow.get_interpolation(
            xyz_inputs["noise"], xyz_inputs["x0"], xyz_inputs["sigmas"]
        )
        xyz_inputs["time_weights"] = self.xyz_model.rectified_flow.train_time_weight(
            xyz_inputs["timesteps"], self.xyz_model.tensor_kwargs_fp32
        )

        rgb_xt, rgb_mask = self._apply_video_condition(rgb_inputs["xt"], rgb_inputs["condition"])
        xyz_xt, xyz_mask = self._apply_video_condition(xyz_inputs["xt"], xyz_inputs["condition"])

        rgb_state = self._prepare_net_state(
            self.rgb_model.net,
            rgb_xt.to(**self.rgb_model.tensor_kwargs),
            rgb_inputs["timesteps"],
            rgb_inputs["condition"],
        )
        xyz_state = self._prepare_net_state(
            self.xyz_model.net,
            xyz_xt.to(**self.xyz_model.tensor_kwargs),
            xyz_inputs["timesteps"],
            xyz_inputs["condition"],
        )

        rgb_pred, xyz_pred = self._forward_nets_with_crosstower(
            self.rgb_model.net,
            self.xyz_model.net,
            rgb_state,
            xyz_state,
            iteration=self._global_step,
        )
        rgb_pred = rgb_pred.float()
        xyz_pred = xyz_pred.float()

        if rgb_inputs["condition"].is_video and self.rgb_model.config.denoise_replace_gt_frames:
            gt_frames_x0 = rgb_inputs["condition"].gt_frames.type_as(rgb_pred)
            gt_frames_velocity = rgb_inputs["noise"] - gt_frames_x0
            rgb_pred = gt_frames_velocity * rgb_mask + rgb_pred * (1 - rgb_mask)
        if xyz_inputs["condition"].is_video and self.xyz_model.config.denoise_replace_gt_frames:
            gt_frames_x0 = xyz_inputs["condition"].gt_frames.type_as(xyz_pred)
            gt_frames_velocity = xyz_inputs["noise"] - gt_frames_x0
            xyz_pred = gt_frames_velocity * xyz_mask + xyz_pred * (1 - xyz_mask)

        def _compute_loss(pred, target, time_weights):
            per_instance = torch.mean((pred - target) ** 2, dim=list(range(1, pred.dim())))
            return torch.mean(time_weights * per_instance)

        rgb_loss = _compute_loss(rgb_pred, rgb_inputs["vt"], rgb_inputs["time_weights"])
        xyz_loss = _compute_loss(xyz_pred, xyz_inputs["vt"], xyz_inputs["time_weights"])

        rgb_output = {
            "x0": rgb_inputs["x0"],
            "xt": rgb_inputs["xt"],
            "sigma": rgb_inputs["sigmas"],
            "condition": rgb_inputs["condition"],
            "model_pred": rgb_pred,
            "edm_loss": rgb_loss,
        }
        xyz_output = {
            "x0": xyz_inputs["x0"],
            "xt": xyz_inputs["xt"],
            "sigma": xyz_inputs["sigmas"],
            "condition": xyz_inputs["condition"],
            "model_pred": xyz_pred,
            "edm_loss": xyz_loss,
        }

        return rgb_output, rgb_loss, xyz_output, xyz_loss

    def _dual_denoise(
        self,
        noise_rgb: torch.Tensor,
        noise_xyz: torch.Tensor,
        xt_rgb: torch.Tensor,
        xt_xyz: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        cond_rgb: Any,
        cond_xyz: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb_xt, rgb_mask = self._apply_video_condition(xt_rgb, cond_rgb)
        xyz_xt, xyz_mask = self._apply_video_condition(xt_xyz, cond_xyz)

        rgb_state = self._prepare_net_state(
            self.rgb_model.net,
            rgb_xt.to(**self.rgb_model.tensor_kwargs),
            timesteps_B_T,
            cond_rgb,
        )
        xyz_state = self._prepare_net_state(
            self.xyz_model.net,
            xyz_xt.to(**self.xyz_model.tensor_kwargs),
            timesteps_B_T,
            cond_xyz,
        )

        rgb_pred, xyz_pred = self._forward_nets_with_crosstower(
            self.rgb_model.net,
            self.xyz_model.net,
            rgb_state,
            xyz_state,
        )

        rgb_pred = rgb_pred.float()
        xyz_pred = xyz_pred.float()

        if cond_rgb.is_video and self.rgb_model.config.denoise_replace_gt_frames:
            gt_frames_x0 = cond_rgb.gt_frames.type_as(rgb_pred)
            gt_frames_velocity = noise_rgb - gt_frames_x0
            rgb_pred = gt_frames_velocity * rgb_mask + rgb_pred * (1 - rgb_mask)
        if cond_xyz.is_video and self.xyz_model.config.denoise_replace_gt_frames:
            gt_frames_x0 = cond_xyz.gt_frames.type_as(xyz_pred)
            gt_frames_velocity = noise_xyz - gt_frames_x0
            xyz_pred = gt_frames_velocity * xyz_mask + xyz_pred * (1 - xyz_mask)

        return rgb_pred, xyz_pred

    def get_dual_velocity_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
    ) -> Callable:
        data_batch = self._cast_batch_to_precision(data_batch)
        _, x0_rgb, _ = self.rgb_model.get_data_and_condition(data_batch["rgb"])
        _, x0_xyz, _ = self.xyz_model.get_data_and_condition(data_batch["xyz"])

        if is_negative_prompt:
            cond_rgb, uncond_rgb = self.rgb_model.conditioner.get_condition_with_negative_prompt(data_batch["rgb"])
            cond_xyz, uncond_xyz = self.xyz_model.conditioner.get_condition_with_negative_prompt(data_batch["xyz"])
        else:
            cond_rgb, uncond_rgb = self.rgb_model.conditioner.get_condition_uncondition(data_batch["rgb"])
            cond_xyz, uncond_xyz = self.xyz_model.conditioner.get_condition_uncondition(data_batch["xyz"])

        cond_rgb = cond_rgb.edit_data_type(
            DataType.IMAGE if self.rgb_model.is_image_batch(data_batch["rgb"]) else DataType.VIDEO
        )
        uncond_rgb = uncond_rgb.edit_data_type(
            DataType.IMAGE if self.rgb_model.is_image_batch(data_batch["rgb"]) else DataType.VIDEO
        )
        cond_xyz = cond_xyz.edit_data_type(
            DataType.IMAGE if self.xyz_model.is_image_batch(data_batch["xyz"]) else DataType.VIDEO
        )
        uncond_xyz = uncond_xyz.edit_data_type(
            DataType.IMAGE if self.xyz_model.is_image_batch(data_batch["xyz"]) else DataType.VIDEO
        )

        num_conditional_frames = data_batch["rgb"].get("num_conditional_frames", None)
        cond_rgb = cond_rgb.set_video_condition(
            gt_frames=x0_rgb.to(**self.rgb_model.tensor_kwargs),
            random_min_num_conditional_frames=self.rgb_model.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.rgb_model.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            conditional_frames_probs=self.rgb_model.config.conditional_frames_probs,
        )
        uncond_rgb = uncond_rgb.set_video_condition(
            gt_frames=x0_rgb.to(**self.rgb_model.tensor_kwargs),
            random_min_num_conditional_frames=self.rgb_model.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.rgb_model.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            conditional_frames_probs=self.rgb_model.config.conditional_frames_probs,
        )

        num_conditional_frames_xyz = data_batch["xyz"].get("num_conditional_frames", None)
        cond_xyz = cond_xyz.set_video_condition(
            gt_frames=x0_xyz.to(**self.xyz_model.tensor_kwargs),
            random_min_num_conditional_frames=self.xyz_model.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.xyz_model.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames_xyz,
            conditional_frames_probs=self.xyz_model.config.conditional_frames_probs,
        )
        uncond_xyz = uncond_xyz.set_video_condition(
            gt_frames=x0_xyz.to(**self.xyz_model.tensor_kwargs),
            random_min_num_conditional_frames=self.xyz_model.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.xyz_model.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames_xyz,
            conditional_frames_probs=self.xyz_model.config.conditional_frames_probs,
        )

        _, cond_rgb, _, _ = self.rgb_model.broadcast_split_for_model_parallelsim(x0_rgb, cond_rgb, None, None)
        _, uncond_rgb, _, _ = self.rgb_model.broadcast_split_for_model_parallelsim(x0_rgb, uncond_rgb, None, None)
        _, cond_xyz, _, _ = self.xyz_model.broadcast_split_for_model_parallelsim(x0_xyz, cond_xyz, None, None)
        _, uncond_xyz, _, _ = self.xyz_model.broadcast_split_for_model_parallelsim(x0_xyz, uncond_xyz, None, None)

        def velocity_fn(
            noise_rgb: torch.Tensor,
            noise_xyz: torch.Tensor,
            xt_rgb: torch.Tensor,
            xt_xyz: torch.Tensor,
            timestep: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            rgb_cond, xyz_cond = self._dual_denoise(noise_rgb, noise_xyz, xt_rgb, xt_xyz, timestep, cond_rgb, cond_xyz)
            rgb_uncond, xyz_uncond = self._dual_denoise(
                noise_rgb, noise_xyz, xt_rgb, xt_xyz, timestep, uncond_rgb, uncond_xyz
            )
            rgb_pred = rgb_uncond + guidance * (rgb_cond - rgb_uncond)
            xyz_pred = xyz_uncond + guidance * (xyz_cond - xyz_uncond)
            return rgb_pred, xyz_pred

        return velocity_fn

    @torch.no_grad()
    def generate_samples_with_latents_from_batch_dual(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        shift: float = 5.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        data_batch = self._cast_batch_to_precision(data_batch)
        self.rgb_model._normalize_video_databatch_inplace(data_batch["rgb"])
        self.rgb_model._augment_image_dim_inplace(data_batch["rgb"])
        self.xyz_model._normalize_video_databatch_inplace(data_batch["xyz"])
        self.xyz_model._augment_image_dim_inplace(data_batch["xyz"])

        is_image_batch = self.rgb_model.is_image_batch(data_batch["rgb"])
        input_key = self.rgb_model.input_image_key if is_image_batch else self.rgb_model.input_data_key
        if n_sample is None:
            n_sample = data_batch["rgb"][input_key].shape[0]
        if state_shape is None:
            _T, _H, _W = data_batch["rgb"][input_key].shape[-3:]
            state_shape = [
                self.rgb_model.config.state_ch,
                self.rgb_model.tokenizer.get_latent_num_frames(_T),
                _H // self.rgb_model.tokenizer.spatial_compression_factor,
                _W // self.rgb_model.tokenizer.spatial_compression_factor,
            ]

        noise_rgb = misc.arch_invariant_rand(
            (n_sample,) + tuple(state_shape),
            torch.float32,
            self.rgb_model.tensor_kwargs["device"],
            seed,
        )
        noise_xyz = misc.arch_invariant_rand(
            (n_sample,) + tuple(state_shape),
            torch.float32,
            self.xyz_model.tensor_kwargs["device"],
            seed + 1,
        )

        seed_g = torch.Generator(device=self.rgb_model.tensor_kwargs["device"])
        seed_g.manual_seed(seed)

        self.rgb_model.sample_scheduler.set_timesteps(
            num_steps,
            device=self.rgb_model.tensor_kwargs["device"],
            shift=shift,
            use_kerras_sigma=self.rgb_model.config.use_kerras_sigma_at_inference,
        )
        self.xyz_model.sample_scheduler.set_timesteps(
            num_steps,
            device=self.xyz_model.tensor_kwargs["device"],
            shift=shift,
            use_kerras_sigma=self.xyz_model.config.use_kerras_sigma_at_inference,
        )
        timesteps = self.rgb_model.sample_scheduler.timesteps

        velocity_fn = self.get_dual_velocity_fn_from_batch(
            data_batch, guidance=guidance, is_negative_prompt=is_negative_prompt
        )
        latents_rgb = noise_rgb
        latents_xyz = noise_xyz

        if INTERNAL:
            timesteps_iter = timesteps
        else:
            disable_tqdm = os.environ.get("COSMOS_PROGRESS", "0") != "1"
            timesteps_iter = tqdm.tqdm(
                timesteps,
                desc="Generating samples (dual)",
                total=len(timesteps),
                disable=disable_tqdm,
                leave=False,
                dynamic_ncols=True,
            )

        for t in timesteps_iter:
            timestep = torch.stack([t]).unsqueeze(0)
            rgb_pred, xyz_pred = velocity_fn(noise_rgb, noise_xyz, latents_rgb, latents_xyz, timestep)
            temp_x0_rgb = self.rgb_model.sample_scheduler.step(
                rgb_pred.unsqueeze(0), t, latents_rgb[0].unsqueeze(0), return_dict=False, generator=seed_g
            )[0]
            temp_x0_xyz = self.xyz_model.sample_scheduler.step(
                xyz_pred.unsqueeze(0), t, latents_xyz[0].unsqueeze(0), return_dict=False, generator=seed_g
            )[0]
            latents_rgb = temp_x0_rgb.squeeze(0)
            latents_xyz = temp_x0_xyz.squeeze(0)

        return latents_rgb, latents_xyz

    def clip_grad_norm_(
        self,
        max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
        foreach: bool | None = None,
    ):
        return torch.nn.utils.clip_grad_norm_(
            self.parameters(),
            max_norm,
            norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite,
            foreach=foreach,
        )

    def on_before_zero_grad(
        self, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, iteration: int
    ) -> None:
        self.rgb_model.on_before_zero_grad(optimizer, scheduler, iteration)
        self.xyz_model.on_before_zero_grad(optimizer, scheduler, iteration)

    def on_after_backward(self, iteration: int = 0) -> None:
        self.rgb_model.on_after_backward(iteration)
        self.xyz_model.on_after_backward(iteration)

    def apply_fsdp(self, dp_mesh) -> None:
        self.rgb_model.apply_fsdp(dp_mesh)
        self.xyz_model.apply_fsdp(dp_mesh)

    def training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        self._global_step = iteration
        data_batch = self._cast_batch_to_precision(data_batch)
        rgb_output, rgb_loss, xyz_output, xyz_loss = self._dual_forward(data_batch)
        rgb_w = float(self.config.rgb_loss_weight)
        xyz_w = float(self.config.xyz_loss_weight)
        weighted_loss = rgb_loss * rgb_w + xyz_loss * xyz_w
        output = {
            "rgb": rgb_output,
            "xyz": xyz_output,
            "rgb_loss": rgb_loss.detach(),
            "xyz_loss": xyz_loss.detach(),
            "rgb_loss_weight": rgb_w,
            "xyz_loss_weight": xyz_w,
            "edm_loss": rgb_output.get("edm_loss", rgb_loss.detach()) * rgb_w
            + xyz_output.get("edm_loss", xyz_loss.detach()) * xyz_w,
        }
        return output, weighted_loss

    @torch.no_grad()
    def validation_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        self._global_step = iteration
        data_batch = self._cast_batch_to_precision(data_batch)
        rgb_output, rgb_loss, xyz_output, xyz_loss = self._dual_forward(data_batch)
        rgb_w = float(self.config.rgb_loss_weight)
        xyz_w = float(self.config.xyz_loss_weight)
        weighted_loss = rgb_loss * rgb_w + xyz_loss * xyz_w
        output = {
            "rgb": rgb_output,
            "xyz": xyz_output,
            "rgb_loss": rgb_loss.detach(),
            "xyz_loss": xyz_loss.detach(),
            "rgb_loss_weight": rgb_w,
            "xyz_loss_weight": xyz_w,
        }
        return output, weighted_loss

    @torch.inference_mode()
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Use per-tower inference via the dedicated inference entrypoints.")

    def state_dict(self) -> Dict[str, Any]:  # noqa: F821
        rgb_state = self.rgb_model.state_dict()
        xyz_state = self.xyz_model.state_dict()
        state = {}
        state.update({f"rgb.{k}": v for k, v in rgb_state.items()})
        state.update({f"xyz.{k}": v for k, v in xyz_state.items()})
        state.update({f"gates.rgb_to_xyz.{k}": v for k, v in self.gates["rgb_to_xyz"].state_dict().items()})
        state.update({f"gates.xyz_to_rgb.{k}": v for k, v in self.gates["xyz_to_rgb"].state_dict().items()})
        state.update(
            {f"cross_attn.rgb_from_xyz.{k}": v for k, v in self.cross_attn["rgb_from_xyz"].state_dict().items()}
        )
        state.update(
            {f"cross_attn.xyz_from_rgb.{k}": v for k, v in self.cross_attn["xyz_from_rgb"].state_dict().items()}
        )
        return state

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):  # noqa: F821
        rgb_state = {k[len("rgb.") :]: v for k, v in state_dict.items() if k.startswith("rgb.")}
        xyz_state = {k[len("xyz.") :]: v for k, v in state_dict.items() if k.startswith("xyz.")}
        gate_rgb_state = {
            k[len("gates.rgb_to_xyz.") :]: v for k, v in state_dict.items() if k.startswith("gates.rgb_to_xyz.")
        }
        gate_xyz_state = {
            k[len("gates.xyz_to_rgb.") :]: v for k, v in state_dict.items() if k.startswith("gates.xyz_to_rgb.")
        }
        attn_rgb_state = {
            k[len("cross_attn.rgb_from_xyz.") :]: v
            for k, v in state_dict.items()
            if k.startswith("cross_attn.rgb_from_xyz.")
        }
        attn_xyz_state = {
            k[len("cross_attn.xyz_from_rgb.") :]: v
            for k, v in state_dict.items()
            if k.startswith("cross_attn.xyz_from_rgb.")
        }

        if rgb_state:
            self.rgb_model.load_state_dict(rgb_state, strict=strict, assign=assign)
        if xyz_state:
            self.xyz_model.load_state_dict(xyz_state, strict=strict, assign=assign)
        if gate_rgb_state:
            self.gates["rgb_to_xyz"].load_state_dict(gate_rgb_state, strict=False)
        if gate_xyz_state:
            self.gates["xyz_to_rgb"].load_state_dict(gate_xyz_state, strict=False)
        if attn_rgb_state:
            self.cross_attn["rgb_from_xyz"].load_state_dict(attn_rgb_state, strict=False)
        if attn_xyz_state:
            self.cross_attn["xyz_from_rgb"].load_state_dict(attn_xyz_state, strict=False)

        if not rgb_state and not xyz_state:
            self.rgb_model.load_state_dict(state_dict, strict=strict, assign=assign)
            self.xyz_model.load_state_dict(state_dict, strict=strict, assign=assign)

    def _maybe_load_init_weights(self) -> None:
        if self.config.rgb_init_checkpoint_path:
            self._load_init_checkpoint(
                self.rgb_model,
                self.config.rgb_init_checkpoint_path,
                self.config.rgb_init_load_ema_to_reg,
                tower_name="rgb",
            )
        else:
            log.info("[dual] rgb init checkpoint path not set; skipping rgb init load")
        if self.config.xyz_init_checkpoint_path:
            self._load_init_checkpoint(
                self.xyz_model,
                self.config.xyz_init_checkpoint_path,
                self.config.xyz_init_load_ema_to_reg,
                tower_name="xyz",
            )
        else:
            log.info("[dual] xyz init checkpoint path not set; skipping xyz init load")

    def _align_checkpoint_tensor_to_target(self, value: Any, target: Any) -> Any:
        if isinstance(target, DTensor):
            target_local = misc.get_local_tensor_if_DTensor(target)
            local_value = misc.get_local_tensor_if_DTensor(value) if isinstance(value, DTensor) else value
            if not isinstance(local_value, torch.Tensor):
                return value
            local_value = local_value.to(device=target_local.device, dtype=target_local.dtype)
            return DTensor.from_local(
                local_value,
                device_mesh=target.device_mesh,
                placements=target.placements,
            )
        if isinstance(value, DTensor):
            return misc.get_local_tensor_if_DTensor(value)
        return value

    def _normalize_checkpoint_state_for_model(self, model: torch.nn.Module, state: dict[str, Any], tower_name: str) -> dict[str, Any]:
        model_state = model.state_dict()
        normalized_state = {}
        converted_keys = []
        for key, value in state.items():
            target = model_state.get(key)
            if target is None or not isinstance(value, torch.Tensor):
                normalized_state[key] = value
                continue

            needs_alignment = isinstance(value, DTensor) or isinstance(target, DTensor)
            if not needs_alignment:
                normalized_state[key] = value
                continue

            normalized_state[key] = self._align_checkpoint_tensor_to_target(value, target)
            converted_keys.append(key)

        if converted_keys:
            preview = ", ".join(converted_keys[:5])
            suffix = "" if len(converted_keys) <= 5 else ", ..."
            log.warning(
                f"[dual] {tower_name} init aligned {len(converted_keys)} checkpoint tensors between "
                f"Tensor/DTensor formats ({preview}{suffix})"
            )
        return normalized_state

    def _load_init_checkpoint(
        self,
        model: ActionVideo2WorldModelRectifiedFlow,
        checkpoint_path: str,
        load_ema_to_reg: bool,
        tower_name: str,
    ) -> None:
        resolved_path = get_checkpoint_path(str(checkpoint_path))
        log.info(f"[dual] Loading {tower_name} init checkpoint from {resolved_path}")
        state = easy_io.load(resolved_path, weights_only=False)

        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        if not isinstance(state, dict):
            raise ValueError(f"Unexpected checkpoint format for {tower_name} init: {type(state)}")

        keys = list(state.keys())
        has_net = any(k.startswith("net.") for k in keys)
        has_net_ema = any(k.startswith("net_ema.") for k in keys)

        if load_ema_to_reg and not has_net and has_net_ema:
            state = {k.replace("net_ema.", "net.", 1): v for k, v in state.items()}
            has_net = True
            has_net_ema = False

        if not has_net and not has_net_ema:
            state = {f"net.{k}": v for k, v in state.items()}

        state = self._normalize_checkpoint_state_for_model(model, state, tower_name)

        log.info(f"[dual] {tower_name} init checkpoint tensors={len(state)}")
        load_result = model.load_state_dict(state, strict=False)
        missing = getattr(load_result, "missing_keys", [])
        unexpected = getattr(load_result, "unexpected_keys", [])
        log.info(
            f"[dual] {tower_name} init load: missing={len(missing)} unexpected={len(unexpected)}"
        )
