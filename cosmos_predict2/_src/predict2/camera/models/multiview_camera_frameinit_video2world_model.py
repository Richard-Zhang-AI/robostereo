from typing import Callable, Dict, Tuple, List, Any

import attrs
import torch
from einops import rearrange
from megatron.core import parallel_state
from torch import Tensor
from tqdm import tqdm

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
class CameraConditionedFrameinitVideo2WorldRectifiedFlowConfig(Video2WorldModelRectifiedFlowConfig):
    pass


class CameraConditionedFrameinitVideo2WorldModelRectifiedFlow(Video2WorldModelRectifiedFlow):
    def get_data_and_condition(
        self, data_batch: dict[str, torch.Tensor]
    ) -> Tuple[Tensor, Tensor, CameraConditionedCondition]:
        self._normalize_multicam_video_databatch_inplace(data_batch)
        self._augment_multicam_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)

        # Latent cond state
        split_size = data_batch["num_frames"].item()
        raw_state_cond = data_batch[self.input_data_key + "_cond"]
        raw_state_cond_chunks = torch.split(raw_state_cond, split_size_or_sections=split_size, dim=2)
        latent_state_cond_list = []
        for raw_state_cond_chunk in raw_state_cond_chunks:
            latent_state_cond_chunk = self.encode(raw_state_cond_chunk).contiguous().float()
            latent_state_cond_list.append(latent_state_cond_chunk)

        # Latent tgt state
        raw_state_src = data_batch[self.input_data_key]
        raw_state_src_chunks = torch.split(raw_state_src, split_size_or_sections=split_size, dim=2)
        latent_state_src_list = []
        for raw_state_src_chunk in raw_state_src_chunks:
            latent_state_src_chunk = self.encode(raw_state_src_chunk).contiguous().float()
            latent_state_src_list.append(latent_state_src_chunk)

        raw_state = torch.cat(
            (raw_state_src_chunks[0], raw_state_cond_chunks[0], raw_state_src_chunks[1]),
            dim=2,
        )
        latent_state = torch.cat(
            (latent_state_src_list[0], latent_state_cond_list[0], latent_state_src_list[1]),
            dim=2,
        )

        # Condition: reorder camera parameters; Plücker rays are computed in the conditioner
        chunk_size = len(latent_state_cond_list) + len(latent_state_src_list)
        extr_list = torch.chunk(data_batch["extrinsics"], chunk_size, dim=1)
        intr_list = torch.chunk(data_batch["intrinsics"], chunk_size, dim=1)

        data_batch["extrinsics"] = torch.cat((extr_list[1], extr_list[0], extr_list[2]), dim=1)
        data_batch["intrinsics"] = torch.cat((intr_list[1], intr_list[0], intr_list[2]), dim=1)

        condition = self.conditioner(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        condition = condition.set_camera_conditioned_video_condition(
            gt_frames=latent_state.to(**self.tensor_kwargs),
            num_conditional_frames=data_batch.get(NUM_CONDITIONAL_FRAMES_KEY, None),
        )

        # torch.distributed.breakpoint()
        return raw_state, latent_state, condition

    def _normalize_multicam_video_databatch_inplace(
        self, data_batch: dict[str, torch.Tensor], input_key: str = None
    ) -> None:
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

    # --------------------------------------------------------------------------------
    # Custom denoise method to extract intermediate features
    # --------------------------------------------------------------------------------
    def denoise(
        self,
        noise: torch.Tensor,
        xt_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        condition: Any,
    ) -> Tensor:
        """
        Custom denoise to extract intermediate features from self.net
        It appends features to self._feature_buffer to handle multiple calls (CFG).
        """
        # 1. Video Condition Masking Logic (Copied from base logic)
        if condition.is_video:
            condition_state_in_B_C_T_H_W = condition.gt_frames.type_as(xt_B_C_T_H_W)
            if not condition.use_video_condition:
                condition_state_in_B_C_T_H_W = condition_state_in_B_C_T_H_W * 0

            _, C, _, _, _ = xt_B_C_T_H_W.shape
            condition_video_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(1, C, 1, 1, 1).type_as(
                xt_B_C_T_H_W
            )

            # Make the first few frames of x_t be the ground truth frames
            xt_B_C_T_H_W = condition_state_in_B_C_T_H_W * condition_video_mask + xt_B_C_T_H_W * (
                1 - condition_video_mask
            )

            if self.config.conditional_frame_timestep >= 0:
                condition_video_mask_B_1_T_1_1 = condition_video_mask.mean(dim=[1, 3, 4], keepdim=True)
                timestep_cond_B_1_T_1_1 = (
                    torch.ones_like(condition_video_mask_B_1_T_1_1) * self.config.conditional_frame_timestep
                )
                timesteps_B_1_T_1_1 = timestep_cond_B_1_T_1_1 * condition_video_mask_B_1_T_1_1 + timesteps_B_T * (
                    1 - condition_video_mask_B_1_T_1_1
                )
                timesteps_B_T = timesteps_B_1_T_1_1.squeeze()
                timesteps_B_T = (
                    timesteps_B_T.unsqueeze(0) if timesteps_B_T.ndim == 1 else timesteps_B_T
                )

        # 2. Construct Forward Arguments
        call_kwargs = dict(
            x_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=timesteps_B_T,
            **condition.to_dict(),
        )

        # Inject feature_ids if configured
        if hasattr(self, "extract_layer_ids") and self.extract_layer_ids is not None:
            call_kwargs["intermediate_feature_ids"] = self.extract_layer_ids

        # 3. Forward Pass
        # self.net outputs either Tensor or (Tensor, List[Tensor])
        net_output = self.net(**call_kwargs)

        # 4. Handle Output & Capture Features
        if isinstance(net_output, tuple):
            net_output_B_C_T_H_W, intermediate_features = net_output

            # Initialize buffer if not exists (defensive programming)
            if not hasattr(self, "_feature_buffer"):
                self._feature_buffer = []

            # Append to buffer (to support CFG multiple calls per step)
            self._feature_buffer.append(intermediate_features)
        else:
            net_output_B_C_T_H_W = net_output
            # If features were requested but not returned (e.g. model mismatch), we proceed without crashing
            pass

        net_output_B_C_T_H_W = net_output_B_C_T_H_W.float()

        # 5. GT Replacement Logic (Copied from base logic)
        if condition.is_video and self.config.denoise_replace_gt_frames:
            gt_frames_x0 = condition.gt_frames.type_as(net_output_B_C_T_H_W)
            gt_frames_velocity = noise - gt_frames_x0
            net_output_B_C_T_H_W = gt_frames_velocity * condition_video_mask + net_output_B_C_T_H_W * (
                1 - condition_video_mask
            )

        return net_output_B_C_T_H_W

    def _parse_step_indices(self, step_spec, num_steps):
        """
        Parse step specification into a list of indices.

        Args:
            step_spec: Can be None, 'first', 'last', or a list of indices
            num_steps: Total number of denoising steps

        Returns:
            List of step indices to extract features from
        """
        if step_spec is None:
            return None  # Extract from all steps
        elif step_spec == 'first':
            return [0]
        elif step_spec == 'last':
            return [num_steps - 1]
        elif isinstance(step_spec, list):
            # Validate indices
            return [i for i in step_spec if 0 <= i < num_steps]
        else:
            raise ValueError(f"Invalid step_spec: {step_spec}. Must be None, 'first', 'last', or a list of indices")

    @torch.no_grad()
    def get_velocity_fn_from_batch(
        self,
        data_batch: Dict,
        guidance: float = 1.5,
        num_output_video: int = 3,
        is_negative_prompt: bool = False,
    ) -> Callable:
        """
        Generates a callable function `x0_fn` based on the provided data batch and guidance factor.
        """

        if NUM_CONDITIONAL_FRAMES_KEY in data_batch:
            num_conditional_frames = data_batch[NUM_CONDITIONAL_FRAMES_KEY]
        else:
            num_conditional_frames = 1

        extr_list = torch.chunk(data_batch["extrinsics"], num_output_video, dim=1)
        intr_list = torch.chunk(data_batch["intrinsics"], num_output_video, dim=1)

        data_batch["extrinsics"] = torch.cat((extr_list[1], extr_list[0], extr_list[2]), dim=1)
        data_batch["intrinsics"] = torch.cat((intr_list[1], intr_list[0], intr_list[2]), dim=1)

        if is_negative_prompt:
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)

        is_image_batch = self.is_image_batch(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)

        x0_cond_chunks = torch.chunk(data_batch[self.input_data_key], num_output_video, dim=2)
        x0_cond_list = []
        for x0_cond_chunk in x0_cond_chunks:
            x0_cond = self.encode(x0_cond_chunk).contiguous().float()
            x0_cond_list.append(x0_cond)

        x0 = torch.cat([x0_cond_list[1], x0_cond_list[0], x0_cond_list[2]], dim=2)
        # override condition with inference mode; num_conditional_frames used Here!
        condition = condition.set_camera_conditioned_video_condition(
            gt_frames=x0,
            num_conditional_frames=num_conditional_frames,
        )
        uncondition = uncondition.set_camera_conditioned_video_condition(
            gt_frames=x0,
            num_conditional_frames=num_conditional_frames,
        )

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
        num_output_video: int = 3,
        is_negative_prompt: bool = False,
        num_steps: int = 35,
        shift: float = 5.0,
        **kwargs,
    ) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]], List[Dict]]:
        """
        Generate samples and extract spatially restored features, saving them to .pt files per step.
        """
        import os
        import torch
        from einops import rearrange
        from tqdm import tqdm
        
        # --- Visualization Helper Function (保持原有的可视化逻辑) ---
        def visualize_and_save_features(feature_tensor, step, layer_id, save_root):
            """
            Visualizes features by saving them as images.
            feature_tensor: [B, C, T, H, W] (already reshaped)
            """
            try:
                from torchvision.utils import save_image
                import numpy as np
                
                os.makedirs(save_root, exist_ok=True)
                
                # Select the first sample and middle frame
                feat = feature_tensor[0].float() # [C, T, H, W]
                C, T, H, W = feat.shape
                mid_t = T // 2
                feat_frame = feat[:, mid_t, :, :] # [C, H, W]
                
                # 1. Save Raw Channels (First 3 channels)
                if C >= 3:
                    raw_rgb = feat_frame[:3, :, :].clone()
                    for c in range(3):
                        c_min, c_max = raw_rgb[c].min(), raw_rgb[c].max()
                        if c_max - c_min > 1e-6:
                            raw_rgb[c] = (raw_rgb[c] - c_min) / (c_max - c_min)
                        else:
                            raw_rgb[c] = 0.0
                    save_image(raw_rgb, os.path.join(save_root, f"step_{step:02d}_layer_{layer_id}_raw.png"))

                # 2. PCA Visualization
                try:
                    from sklearn.decomposition import PCA
                    feat_flat = feat_frame.permute(1, 2, 0).reshape(-1, C).cpu().numpy()
                    
                    # Subsample if too large for speed
                    if feat_flat.shape[0] > 10000:
                        indices = np.random.choice(feat_flat.shape[0], 10000, replace=False)
                        pca = PCA(n_components=3)
                        pca.fit(feat_flat[indices])
                        feat_pca = pca.transform(feat_flat)
                    else:
                        pca = PCA(n_components=3)
                        feat_pca = pca.fit_transform(feat_flat)
                    
                    f_min, f_max = feat_pca.min(axis=0), feat_pca.max(axis=0)
                    feat_pca_norm = (feat_pca - f_min) / (f_max - f_min + 1e-6) if (f_max - f_min).any() > 1e-6 else feat_pca
                    
                    feat_rgb = torch.from_numpy(feat_pca_norm).reshape(H, W, 3).permute(2, 0, 1)
                    save_image(feat_rgb, os.path.join(save_root, f"step_{step:02d}_layer_{layer_id}_pca.png"))
                    
                except Exception:
                    pass # Skip PCA if failed
            except Exception as e:
                print(f"Vis error: {e}")

        # --- Main Logic ---

        is_image_batch = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image_batch else self.input_data_key
        if n_sample is None:
            n_sample = data_batch[input_key].shape[0]

        if state_shape is None:
            _T, _H, _W = data_batch[input_key].shape[-3:]
            state_shape = [
                self.config.state_ch,
                self.tokenizer.get_latent_num_frames(_T // num_output_video),
                _H // self.tokenizer.spatial_compression_factor,
                _W // self.tokenizer.spatial_compression_factor,
            ]

        # --- Configuration ---
        self.extract_layer_ids = kwargs.get('extract_layer_ids', [4, 12, 20])
        # Default extract steps: [Middle Step]
        self.extract_at_steps = kwargs.get('extract_at_steps', [num_steps // 2]) 
        self._step_indices = self._parse_step_indices(self.extract_at_steps, num_steps)
        
        # 定义路径
        vis_save_path = "./outputs/vis_features"    # 可视化图片路径
        pt_save_path = "./outputs/saved_features"   # 特征PT文件路径
        os.makedirs(pt_save_path, exist_ok=True)
        
        # Patch sizes (根据你的模型架构确认，DiT通常是2)
        PATCH_SPATIAL = 2
        PATCH_TEMPORAL = 1

        velocity_fn, x0_cond_list = self.get_velocity_fn_from_batch(
            data_batch, guidance, num_output_video, is_negative_prompt=is_negative_prompt
        )

        noise_list = []
        for i in range(num_output_video):
            noise = misc.arch_invariant_rand(
                (n_sample,) + tuple(state_shape),
                torch.float32,
                self.tensor_kwargs["device"],
                seed,
            )
            noise[:, :, 0, :, :] = x0_cond_list[i][:, :, 0, :, :]
            noise_list.append(noise)
        noise = torch.cat(noise_list, dim=2)

        seed_g = torch.Generator(device=self.tensor_kwargs["device"])
        seed_g.manual_seed(seed)
        self.sample_scheduler.set_timesteps(num_steps, device=self.tensor_kwargs["device"], shift=shift)
        timesteps = self.sample_scheduler.timesteps

        if self.net.is_context_parallel_enabled:
            noise = broadcast_split_tensor(tensor=noise, seq_dim=2, process_group=self.get_context_parallel_group())
        latents = noise

        all_timesteps_features = []
        all_timesteps_step_info = []

        # --- Denoising Loop ---
        for step_idx, t in enumerate(tqdm(timesteps, desc="Denoising", unit="step", ncols=100)):
            latent_model_input = latents
            timestep = [t]
            timestep = torch.stack(timestep)

            # Clear buffer before forward pass
            self._feature_buffer = []

            # Forward Pass (features captured in self._feature_buffer)
            velocity_pred = velocity_fn(noise, latent_model_input, timestep.unsqueeze(0))

            # --- Feature Extraction & Saving Logic ---
            if self._step_indices is None or step_idx in self._step_indices:
                if hasattr(self, "_feature_buffer") and len(self._feature_buffer) > 0:
                    # Get features from the Conditional Pass
                    cond_features_gpu = self._feature_buffer[0] # List[Tensor]
                    
                    # Calculate dimensions for reshaping
                    B, C_latent, T_latent, H_latent, W_latent = latent_model_input.shape
                    T_feat = T_latent // PATCH_TEMPORAL
                    H_feat = H_latent // PATCH_SPATIAL
                    W_feat = W_latent // PATCH_SPATIAL
                    
                    current_step_features_dict = {} # 用于存储当前step的所有层特征
                    restored_features_list = []     # 用于保持原有返回格式

                    print(f"Extracting & Saving features for Step {step_idx}...")

                    for i, feat in enumerate(cond_features_gpu):
                        # feat shape: [B, L, D] (e.g. [1, T*H*W, C])
                        
                        # 1. Move to CPU immediately to save GPU memory
                        feat_cpu = feat.detach().cpu()
                        
                        # Determine Layer ID
                        layer_id = self.extract_layer_ids[i] if i < len(self.extract_layer_ids) else f"layer_{i}"
                        
                        try:
                            # 2. Restore Spatiotemporal Structure: [B, (T H W), D] -> [B, D, T, H, W]
                            # Assuming sequence order is flattened T -> H -> W
                            feat_restored = rearrange(
                                feat_cpu, 
                                "b (t h w) d -> b d t h w", 
                                t=T_feat, 
                                h=H_feat, 
                                w=W_feat
                            )
                            
                            # Add to dictionary for saving
                            current_step_features_dict[f"layer_{layer_id}"] = feat_restored
                            restored_features_list.append(feat_restored)
                            
                            # 3. Visualization (Optional)
                            visualize_and_save_features(
                                feat_restored, 
                                step=step_idx, 
                                layer_id=layer_id, 
                                save_root=vis_save_path
                            )

                        except Exception as e:
                            print(f"Error reshaping layer {layer_id}: {e}")
                            # Fallback: save flat feature if reshape fails
                            current_step_features_dict[f"layer_{layer_id}_flat"] = feat_cpu
                            restored_features_list.append(feat_cpu)

                    # --- 4. Save to Disk (.pt file) ---
                    # 文件名格式: features_step_{step_idx}.pt
                    # 内容: Dict {'layer_4': tensor, 'layer_12': tensor, ...}
                    save_name = f"features_step_{step_idx:03d}.pt"
                    save_full_path = os.path.join(pt_save_path, save_name)
                    
                    try:
                        torch.save(current_step_features_dict, save_full_path)
                        print(f"Saved extracted features to {save_full_path}")
                    except Exception as e:
                        print(f"Failed to save .pt file: {e}")

                    # Append to list for return values (backward compatibility)
                    all_timesteps_features.append(restored_features_list)
                    all_timesteps_step_info.append({
                        'step_idx': step_idx,
                        'timestep': t.item(),
                        'saved_path': save_full_path
                    })

            # Sample next step
            temp_x0 = self.sample_scheduler.step(
                velocity_pred.unsqueeze(0), t, latents[0].unsqueeze(0), return_dict=False, generator=seed_g
            )[0]
            latents = temp_x0.squeeze(0)

        if self.net.is_context_parallel_enabled:
            latents = cat_outputs_cp(latents, seq_dim=2, cp_group=self.get_context_parallel_group())

        sample_chunks = torch.chunk(latents, num_output_video, dim=2)
        sample_list = [sample_chunks[1], sample_chunks[0], sample_chunks[2]]

        return sample_list, all_timesteps_features, all_timesteps_step_info