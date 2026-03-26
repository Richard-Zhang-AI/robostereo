# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Rollout with huggingface models.
TODO: refactor this class. Currently, it will hang when using FSDP HybridShard. We should actually create a single GPU model.
Then, get full state_dict and bind the state_dict to the single GPU model. Then, use the single GPU model to perform generation.
"""
import os
import imageio
import contextlib
import time
import json
import copy
import torch
import torch.distributed
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.utils.rnn import pad_sequence

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask
import verl.utils.torch_functional as verl_F
from verl.utils.cosmos_utils import (
    adapt_actions_to_cosmos,
    postprocess_cosmos_video,
    prepare_video_input,
    resize_frame_to_resolution,
    suppress_cosmos_inference_output,
)
from .base import BaseRollout

from transformers import GenerationConfig, AutoProcessor
import tensorflow as tf
import numpy as np
from PIL import Image
from verl import DataProto
try:
    from libero.libero import benchmark
    from verl.utils.libero_utils import get_libero_env, get_libero_dummy_action, get_image_resize_size, get_libero_image, get_libero_wrist_image, quat2axisangle, normalize_gripper_action, invert_gripper_action, save_rollout_video
except:
    print("please install libero")

try:
    from robomimic.config import config_factory
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils
    import mimicgen.envs.robosuite  # noqa: F401
except:
    print("please install robomimic")

from codetiming import Timer
from collections import deque
import random

import multiprocessing
import gc
from multiprocessing import Process, Queue
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

from collections import defaultdict

from verl.utils.libero_utils import resize_image

__all__ = ['RobHFRollout']

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_FILES_ROOT = os.path.join(REPO_ROOT, "data_files")

OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def _try_claim_preview_slot(save_dir: str, epoch: int) -> bool:
    os.makedirs(save_dir, exist_ok=True)
    marker = os.path.join(save_dir, f".epoch_{epoch:04d}.claimed")
    try:
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False

def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image

def center_crop_image(image):
    batch_size = 1
    crop_scale = 0.9

    # Convert to TF Tensor and record original data type (should be tf.uint8)
    image = tf.convert_to_tensor(np.array(image))
    orig_dtype = image.dtype

    # Convert to data type tf.float32 and values between [0,1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Crop and then resize back to original size
    image = crop_and_resize(image, crop_scale, batch_size)

    # Convert back to original data type
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

    # Convert back to PIL Image
    image = Image.fromarray(image.numpy())
    image = image.convert("RGB")
    return image

def _create_env(cfg):
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=cfg.train.data)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=cfg.train.data,
        all_obs_keys=cfg.all_obs_keys,
        verbose=False,
    )
    if cfg.experiment.env is not None:
        env_meta["env_name"] = cfg.experiment.env
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=env_meta["env_name"],
        render=False,
        render_offscreen=True,
        use_image_obs=shape_meta["use_images"],
        use_depth_obs=shape_meta["use_depths"],
    )
    return EnvUtils.wrap_env_from_config(env, config=cfg)

def sample_state(cfg_dict, n_samples):
    cfg = config_factory(cfg_dict["algo_name"])
    with cfg.values_unlocked():
        cfg.update(cfg_dict)
    cfg.lock()
    ObsUtils.initialize_obs_utils_with_config(cfg)
    env = _create_env(cfg)  # 确保这里不再读数据集文件
    state_list = []
    for i in range(n_samples):
        env.reset()
        state = env.get_state()
        state_list.append(state)
    env.env.close()
    return state_list



class RobWMHFRollout(BaseRollout):

    def __init__(self, module: nn.Module, world_model_mapping, config):
        super().__init__()
        self.config = config
        self.module = module
        self.world_model_mapping = world_model_mapping
        self.processor = AutoProcessor.from_pretrained(config.pretrained_checkpoint, trust_remote_code=True)
        self.vla_preprocess()

        self.task = self.config.unnorm_key.split('_d0')[0]
        if "aloha" in self.task:
            self.task = "aloha"
        if self.task == "square":
            self.task_description = "Insert the square into the stick"
        elif self.task == "aloha":
            self.task_description = "Insert the square into the stick"
        else:
            self.task_description = self.task
            assert self.task in ["coffee", "stack_three", "three_piece_assembly"]
        if self.task != "aloha":
            env_config = os.path.join(DATA_FILES_ROOT, "core_train_configs", f"bc_rnn_image_ds_{self.task}_D0_seed_101.json")
            ext_cfg = json.load(open(env_config, "r"))
            self.ext_cfg = ext_cfg
        
        if self.task == "coffee":
            self.max_steps = 256
        elif self.task == "stack_three":
            self.max_steps = 320
        elif self.task == "three_piece_assembly":
            self.max_steps = 384
        elif self.task == "square":
            self.max_steps = 184
        elif self.task == "aloha":
            self.max_steps = 224
        else:
            assert False

        self.wm_backend = world_model_mapping.get("backend", "opensora")
        self.rm_model = world_model_mapping["rm_model"]
        self.rm_threshold = world_model_mapping["rm_threshold"]
        self.rm_feature_extractor = world_model_mapping["feature_extractor"]
        self.queue_len = 4
        if self.wm_backend == "cosmos":
            self.cosmos_infer = world_model_mapping["cosmos_infer"]
            self.cosmos_resolution = world_model_mapping["cosmos_resolution"]
            self.cosmos_guidance = world_model_mapping["cosmos_guidance"]
            self.cosmos_seed = world_model_mapping["cosmos_seed"]
            self.cosmos_action_chunk_size = world_model_mapping["cosmos_action_chunk_size"]
            self.cosmos_action_scale = world_model_mapping["cosmos_action_scale"]
            self.cosmos_gripper_scale = world_model_mapping["cosmos_gripper_scale"]
            self.use_normalized_actions = world_model_mapping["use_normalized_actions"]
            self.cosmos_negative_prompt = world_model_mapping["negative_prompt"]
            self.device = torch.device("cuda", torch.cuda.current_device())
            self.dtype = torch.bfloat16
            self.vae = None
            self.world_model = None
            self.scheduler = None
            self.model_args = None
            self.latent_size = None
        else:
            self.vae = world_model_mapping["vae"]
            self.world_model = world_model_mapping["model"]
            self.scheduler = world_model_mapping["scheduler"]
            self.model_args = world_model_mapping["model_args"]
            self.device = self.vae.device
            self.dtype = self.vae.dtype
            self.latent_size = self.vae.get_latent_size(input_size = [12, 256, 256])
        
        
    
    @torch.no_grad()
    def predict_success(self, videos, batch_size = 128):
        self.rm_model.eval()
        # videos B T H W C
        total_frames = videos.shape[1]
        window_size = 8
        stride = 1
        min_steps = 100
        beta = float(self.config.get("success_prob_beta", 0.1))
        results = []
        # start_time = time.time()

        for video_idx, video in enumerate(videos):
            clips = []
            for end in range(total_frames, min_steps + window_size - 1, -stride):
                clip = video[end - window_size:end]
                clips.append((clip, end - window_size, end))
            clips = clips[::-1]
            clip_batches = [clips[i:i+batch_size] for i in range(0, len(clips), batch_size)]
            
            finish_step = total_frames - 1
            complete = 0
            clip_success_probs = []
            for batch_idx, batch in enumerate(clip_batches):
                # current_time = time.time()
                # elapsed_time = current_time - start_time
                # print(f"Rank {dist.get_rank()}: Elapsed time: {elapsed_time:.2f} seconds : video {video_idx}/{len(videos)} batch {batch_idx}/{len(clip_batches)}")
                ranges = [(c[1], c[2]) for c in batch]
                clip_imgs = [c[0] for c in batch]
                # clip_imgs = [[Image.fromarray(frame).convert("RGB") for frame in clip ] for clip in clip_imgs]
                clip_imgs = [[img for img in clip] for clip in clip_imgs]
                # if video_idx == 0 and batch_idx == 0:
                #     print(f"[PID {os.getpid()}] before TF feature_extractor")
                inputs = self.rm_feature_extractor(clip_imgs, return_tensors="pt")["pixel_values"].to(self.device) # device(type='cuda', index=7)
                # if video_idx == 0 and batch_idx == 0:
                #     print(f"[PID {os.getpid()}] after TF feature_extractor")
                logits = self.rm_model(pixel_values=inputs).logits
                probs = torch.sigmoid(logits).cpu().numpy()
                clip_success_probs.extend(float(p[1]) for p in probs)
                preds = [1 if p[1] >= self.rm_threshold else 0 for p in probs]
                
                for (start, end), prob, pred in zip(ranges, probs, preds):
                    if pred == 1 and end - 1 < finish_step:
                        finish_step = end - 1
                        complete = 1
                        break
            if clip_success_probs:
                positions = np.arange(1, len(clip_success_probs) + 1, dtype=np.float32)
                weights = np.exp(beta * positions)
                weights = weights / weights.sum()
                success_prob = float(np.dot(weights, np.array(clip_success_probs, dtype=np.float32)))
            else:
                success_prob = 0.0
            results.append({"complete": complete, 'finish_step': finish_step, 'success_prob': success_prob})

        complete = torch.from_numpy(np.array([r['complete'] for r in results]))
        finish_step = torch.from_numpy(np.array([r['finish_step'] for r in results]))
        success_prob = torch.from_numpy(np.array([r['success_prob'] for r in results], dtype=np.float32))

        return {
            "complete": complete,
            "finish_step": finish_step,
            "success_prob": success_prob,
        }
    
    def vla_preprocess(self):
        if self.config.vla in ["openvla","openvla-oft"]:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:  
                    tf.config.experimental.set_memory_growth(gpu, True)
        
        if self.config.vla in ["openvla-oft"]:
            if  self.config.unnorm_key not in self.module.norm_stats and f"{self.config.unnorm_key}_no_noops" in self.module.norm_stats:
                self.config.unnorm_key = f"{self.config.unnorm_key}_no_noops"
            assert self.config.unnorm_key in self.module.norm_stats, f"Action un-norm key {self.config.unnorm_key} not found in VLA `norm_stats`!"

    def generate_wm_sequences(self, prompts):
        # breakpoint()
        batch_size = prompts.batch.batch_size[0]
        if prompts.meta_info.get('n_samples') is None:
            micro_batch_size = self.config.val_micro_batch_size if self.config.val_micro_batch_size is not None else 1
        else:
            micro_batch_size = self.config.get('micro_batch_size', batch_size)
        
        num_chunks = max(batch_size // micro_batch_size, 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        output = [self._generate_wm_minibatch(p) for p in batch_prompts]
        output = DataProto.concat(output)
        return output
    
    def _prepare_data(self, image_paths, repeat):
        """
        一个私有的生成器方法，用于加载、重复和批处理初始数据。
        现在将一次性返回所有数据，而不使用 batch_size。
        """
        batch_buffer = []
        
        for image_path in image_paths:
            task_name = os.path.basename(image_path).split('.')[0]
            task_description = self.task_description
            init_frame_np = imageio.v2.imread(image_path)
            # init_frame_np = resize_image(init_frame_np, (224, 224))
            init_frame_tensor = torch.from_numpy(init_frame_np).permute(2, 0, 1).float() / 255.0 * 2 - 1
            init_frame_tensor = init_frame_tensor.to(self.device).to(self.dtype)

            for i in range(repeat):
                video_name = f"{task_name}_repeat_{i}"
                batch_buffer.append((init_frame_tensor.clone(), init_frame_np.copy(), task_description, video_name))
        
        init_tensors, init_numpys, descs, names = zip(*batch_buffer)
        return torch.stack(init_tensors), list(init_numpys), list(descs), list(names)

    @torch.no_grad()
    def run_cosmos_inference(self, image_paths, max_steps, repeat=1):
        vla_history = []
        _, init_frames_numpys, task_descriptions, _ = self._prepare_data(image_paths, repeat)
        current_frames_np = [
            resize_frame_to_resolution(frame, self.cosmos_resolution)
            for frame in init_frames_numpys
        ]
        predicted_videos = [[np.expand_dims(frame.copy(), axis=0)] for frame in current_frames_np]
        meta_info = getattr(self, "_current_meta_info", {}) or {}
        preview_save_dir = self.world_model_mapping.get("preview_save_dir")
        preview_save_fps = int(self.world_model_mapping.get("preview_save_fps", 6))
        epoch = int(meta_info.get("epoch", meta_info.get("global_steps", 0)))
        save_preview_this_epoch = bool(preview_save_dir) and _try_claim_preview_slot(preview_save_dir, epoch)

        batch_size = len(current_frames_np)
        actor_chunk = int(self.config.action_chunks_len)
        chunk_idx = 0
        frame_num = 1
        while frame_num <= max_steps:
            total_chunks = max((max_steps + actor_chunk - 1) // actor_chunk, 1)
            print(f"[Cosmos] generating chunk {chunk_idx + 1}/{total_chunks}", flush=True)
            current_inputs = [{'full_image': resize_image(frame, (224, 224))} for frame in current_frames_np]
            vla_input = self.process_input(current_inputs, task_descriptions)
            vla_output = self._generate_one_step(vla_input)
            actions = vla_output["action"]
            normalized_actions = vla_output["normalized_actions"]

            step_data = {
                "responses": vla_output["responses"],
                "input_ids": vla_output["input_ids"],
                "attention_mask": vla_output["attention_mask"],
                "pixel_values": vla_output["pixel_values"],
                "action": actions,
                "step": frame_num - 1,
            }
            vla_history.append(step_data)

            next_frames = []
            for sample_idx in range(batch_size):
                action_source = normalized_actions[sample_idx] if self.use_normalized_actions else actions[sample_idx]
                cosmos_actions = adapt_actions_to_cosmos(
                    action_source,
                    target_chunk_size=self.cosmos_action_chunk_size,
                    action_scale=self.cosmos_action_scale,
                    gripper_scale=self.cosmos_gripper_scale,
                )
                num_video_frames = cosmos_actions.shape[0] + 1
                vid_input = prepare_video_input(current_frames_np[sample_idx], num_video_frames)
                with suppress_cosmos_inference_output():
                    video = self.cosmos_infer.generate_vid2world(
                        prompt="",
                        input_path=vid_input,
                        action=torch.from_numpy(cosmos_actions).float(),
                        guidance=self.cosmos_guidance,
                        num_video_frames=num_video_frames,
                        num_latent_conditional_frames=1,
                        resolution=self.cosmos_resolution,
                        seed=self.cosmos_seed + chunk_idx * batch_size + sample_idx,
                        negative_prompt=self.cosmos_negative_prompt,
                    )
                video_clamped = postprocess_cosmos_video(video)
                if save_preview_this_epoch and chunk_idx == 0 and sample_idx == 0:
                    preview_path = os.path.join(preview_save_dir, f"epoch_{epoch:04d}_chunk_{chunk_idx:03d}.mp4")
                    imageio.mimwrite(preview_path, video_clamped, fps=preview_save_fps)
                    print(f"[CosmosPreview] saved first chunk for epoch {epoch} to {preview_path}", flush=True)
                    save_preview_this_epoch = False
                predicted_videos[sample_idx].append(video_clamped[1:])
                next_frames.append(video_clamped[-1])

            current_frames_np = next_frames
            frame_num += actor_chunk
            chunk_idx += 1

        predicted_videos = [np.concatenate(predicted_videos[i], axis=0) for i in range(batch_size)]
        predicted_videos = np.stack(predicted_videos, axis=0)

        if self.config.get('save_vla_data', False):
            self._save_vla_training_data(
                vla_history=vla_history,
                image_paths=image_paths,
                repeat=repeat,
                save_dir=self.config.get('vla_data_save_dir', './tmp_files/vla_training_data')
            )

        return vla_history, predicted_videos

    
    @torch.no_grad()
    def run_wm_inference(self, image_paths, max_steps, repeat=1):
        """
        使用小批量（mini-batch）运行视频生成推理，并返回生成的视频数据。

        Returns:
            list: 一个字典列表。每个字典包含两个键:
                  'name' (str): 视频的名称。
                  'video' (np.ndarray): 视频的Numpy数组，形状为 (T, H, W, C)。
        """
        if self.wm_backend == "cosmos":
            return self.run_cosmos_inference(image_paths, max_steps, repeat=repeat)

        self.world_model.eval()
        vla_history = []
        latent_chunk = self.config.action_chunks_len        
        init_frames_tensors, init_frames_numpys, task_descriptions, video_names = self._prepare_data(image_paths, repeat)

        current_batch_size = init_frames_tensors.shape[0]
        init_frames_for_vae = init_frames_tensors.unsqueeze(2)
        with torch.no_grad():
            latents = self.vae.encode(init_frames_for_vae)
        image_history_tensor = latents.repeat(1, 1, self.queue_len, 1, 1)
        predicted_videos = [[np.expand_dims(frame, axis=0)] for frame in init_frames_numpys]
        current_frames_np = init_frames_numpys
        frame_num = 1
        while frame_num <= max_steps:
            current_inputs = [{'full_image': resize_image(frame, (224, 224))} for frame in current_frames_np]
    
            vla_input = self.process_input(current_inputs, task_descriptions)
            vla_output = self._generate_one_step(vla_input)
            actions = vla_output["action"]
            # breakpoint()
            step_data = {
                "responses": vla_output["responses"],
                "input_ids": vla_output["input_ids"],
                "attention_mask": vla_output["attention_mask"],
                "pixel_values": vla_output["pixel_values"],
                "action": actions,
                "step": frame_num-1
            }
            vla_history.append(step_data)
            
            actions = torch.from_numpy(vla_output['normalized_actions'])
            y = actions.to(self.device).to(self.dtype).reshape(current_batch_size, latent_chunk, -1)
            
            latent_size = self.latent_size

            z = torch.randn(current_batch_size, self.vae.out_channels, latent_chunk, *latent_size[1:], device=self.device, dtype=self.dtype)
            
            # z_combined 的形状: (B, C, T_history + chunk, H, W)
            z_combined = torch.concat([image_history_tensor, z], dim=2)
            
            masks = torch.zeros(current_batch_size, image_history_tensor.shape[2] + latent_chunk, device=self.device, dtype=torch.long)
            masks[:, -latent_chunk:] = 1
            samples = self.scheduler.sample(self.world_model, z=z_combined, y=y, device=self.device, additional_args=self.model_args, progress=False, mask=masks)
            
            # pred_latents 的形状: (B, C_latent, chunk, H_latent, W_latent)
            pred_latents = samples[:, :, -latent_chunk:].to(self.dtype)

            image_history_tensor = pred_latents.clone()[:, :, -self.queue_len:]
            
            # [修正 3] 移除解码前的 permute
            # pred_latents 已经是正确的 (B, C, T, H, W) 格式，可直接输入 vae.decode
            decoded_images = self.vae.decode(pred_latents)
            
            # decoded_images 的输出形状: (B, chunk, C, H, W)
            # permute 以便转换为Numpy: (B, chunk, H, W, C)
            pred_imgs_np = ((decoded_images.to(torch.float32).cpu().permute(0, 2, 3, 4, 1).numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)

            new_current_frames = []
            for i in range(current_batch_size):
                # pred_imgs_np[i] 的形状: (chunk, H, W, C)
                predicted_videos[i].append(pred_imgs_np[i])
                # 更新当前帧为8帧里的最后一帧
                new_current_frames.append(pred_imgs_np[i, -1])
            current_frames_np = new_current_frames
            frame_num += latent_chunk
            print(f"Batch processing frame_num: {frame_num}")
            # 
            # --- 结果处理与保存 (此部分有修改) ---
            # os.makedirs('./debug/wm', exist_ok=True)
            # for i in range(current_batch_size):
            #     final_video_frames = np.concatenate(predicted_videos[i], axis=0)
                
            #     result_item = {
            #         'name': batch_video_names[i],
            #         'video': final_video_frames
            #     }
            #     all_results.append(result_item)
                
            #     imageio.mimwrite(f"./debug/wm/{batch_video_names[i]}.mp4", final_video_frames, fps=30)
            
            # print(f"Finished processing batch. Saved videos: {batch_video_names}")
        predicted_videos = [np.concatenate(predicted_videos[i], axis=0) for i in range(current_batch_size)]
        predicted_videos = np.array(predicted_videos)
        # import pickle
        # debug = {
        #     "vla_history": vla_history,
        #     "predicted_videos": predicted_videos
        # }
        # local_rank = dist.get_rank() % 8
        # os.makedirs('./debug/pickle', exist_ok=True)
        # with open(f'./debug/pickle/debug_{local_rank}.pkl', 'wb') as f:
        #     pickle.dump(debug, f)
        # Save VLA training data if enabled
        if self.config.get('save_vla_data', False):
            self._save_vla_training_data(
                vla_history=vla_history,
                image_paths=image_paths,
                repeat=repeat,
                save_dir=self.config.get('vla_data_save_dir', './tmp_files/vla_training_data')
            )

        return vla_history, predicted_videos

    def _save_vla_training_data(self, vla_history, image_paths, repeat, save_dir):
        """
        保存 VLA 训练数据，方便别人训练使用。

        Args:
            vla_history: VLA 的历史数据，包含每个 step 的输入输出
            image_paths: 初始图像路径列表
            repeat: 每个图像的重复次数
            save_dir: 保存目录

        文件结构: save_dir/{state_id}/{repeat_index}/{step:04d}.npz
        每个 .npz 文件包含:
            - image: 输入图像 (H, W, C), uint8
            - action: VLA 生成的动作 (action_dim,), float32
            - pixel_values: 原始 pixel_values (C, H, W), float32
            - input_ids: 输入 token ids (seq_len,), int64
            - attention_mask: 注意力掩码 (seq_len,), int64
            - response_tokens: VLA 输出的 tokens (action_token_len,), int64
        """
        import os
        from pathlib import Path

        # 计算数据维度
        num_original = len(image_paths)  # 原始状态数量
        total_episodes = num_original * repeat  # 总 episode 数量
        num_steps = len(vla_history)  # 每个 episode 的步数

        # 从 vla_history 中提取每个 batch 的数据
        # vla_history 是一个列表，每个元素是一个 step 的数据
        # 每个 step 的数据维度是 (total_episodes, ...)

        for orig_idx in range(num_original):
            state_id = Path(image_paths[orig_idx]).stem  # 从文件名获取 state_id

            for repeat_idx in range(repeat):
                episode_idx = orig_idx * repeat + repeat_idx

                # 创建保存目录: save_dir/{state_id}/{repeat_index}/
                episode_dir = Path(save_dir) / state_id / str(repeat_idx)
                episode_dir.mkdir(parents=True, exist_ok=True)

                for step_idx, step_data in enumerate(vla_history):
                    # 从 batch 中提取当前 episode 的数据
                    # step_data 的每个 key 的数据维度是 (total_episodes, ...)

                    # 提取图像 (pixel_values)
                    # pixel_values 形状: (total_episodes, C, H, W)
                    pixel_values = step_data['pixel_values'][episode_idx].cpu()

                    # 转换为 numpy 并调整维度: (C, H, W) -> (H, W, C)
                    pixel_values_np = pixel_values.permute(1, 2, 0).float().numpy()
                    # pixel_values 通常在 [-1, 1] 或 [0, 1]，转换到 [0, 255]
                    if pixel_values_np.max() <= 1.0:
                        image_uint8 = (pixel_values_np * 255).clip(0, 255).astype(np.uint8)
                    else:
                        # 假设在 [-1, 1]
                        image_uint8 = ((pixel_values_np + 1) * 127.5).clip(0, 255).astype(np.uint8)

                    # 提取动作
                    # action 形状: (total_episodes, action_chunks, action_dim)
                    action = step_data['action'][episode_idx]
                    action_np = action.astype(np.float32)

                    # 提取 input_ids 和 attention_mask
                    input_ids = step_data['input_ids'][episode_idx].cpu().numpy()
                    attention_mask = step_data['attention_mask'][episode_idx].cpu().numpy()

                    # 提取 response tokens
                    response = step_data['responses'][episode_idx].cpu().numpy()

                    # 保存为 .npz 文件
                    save_path = episode_dir / f"{step_idx:04d}.npz"

                    np.savez_compressed(
                        save_path,
                        image=image_uint8,  # (H, W, C), uint8
                        action=action_np,  # (action_chunks, action_dim), float32
                        pixel_values=pixel_values_np,  # (H, W, C), float32 (原始值)
                        input_ids=input_ids,  # (seq_len,), int64
                        attention_mask=attention_mask,  # (seq_len,), int64
                        response_tokens=response  # (action_token_len,), int64
                    )

        print(f"[VLA Data] Saved VLA training data to {save_dir}")
        print(f"[VLA Data] Total episodes: {total_episodes}, Steps per episode: {num_steps}")

    def _generate_wm_minibatch(self, prompts):        
        self.module.eval()
        meta_info = prompts.meta_info
        self._current_meta_info = meta_info
        n_samples = meta_info.get('n_samples', 1)
        state_ids = prompts.batch['state_id'].cpu().reshape(-1).tolist()
        return_rollouts = meta_info.get('return_rollouts', False)
        max_steps = self.max_steps
        batch_size = len(state_ids) * n_samples
        first_image_dir = os.path.join(DATA_FILES_ROOT, "first_images", self.task)
        if self.task == "square":
            image_paths = [os.path.join(first_image_dir, f"{state_id}.png") for state_id in state_ids]
        elif "aloha" in self.task:
            image_paths = [os.path.join(first_image_dir, f"{state_id}.png") for state_id in state_ids]
        elif self.task in ["coffee", "stack_three", "three_piece_assembly"]:
            image_paths = [os.path.join(first_image_dir, f"{state_id}.png") for state_id in state_ids]
        
        import time
        start_time = time.time()
        vla_history, videos = self.run_wm_inference(image_paths, max_steps, repeat=n_samples) 
        end_time = time.time()
        print(f"Generate video time cost: {end_time-start_time}")
        # import pickle
        # local_rank = dist.get_rank() % 8
        # with open(f'./debug/pickle/debug_{local_rank}.pkl', 'rb') as f:
        #     debug = pickle.load(f)
        # vla_history = debug['vla_history']
        # videos = debug['predicted_videos']
        import time
        start = time.time()
        task_records = self.predict_success(videos, batch_size=512)
        end = time.time()
        print(f"Predict success time: {end-start}")
        
        batch = {
                'responses': [],
                'input_ids': [],  # here input_ids become the whole sentences
                'attention_mask': [],
                'pixel_values': [],
            }
        for k in ["responses", "input_ids", "attention_mask", "pixel_values"]:
            for h in vla_history:
                batch[k].append(h[k])
        
        for k,v in batch.items():
            batch[k] = torch.stack(v,dim=1) 
  
        batch["complete"] = task_records["complete"].to(dtype=torch.bool, device=self.device)
        batch["finish_step"] = task_records["finish_step"].to(dtype=torch.int64, device=self.device)
        batch["success_prob"] = task_records["success_prob"].to(dtype=torch.float32, device=self.device)
        batch['state_id'] = prompts.batch['state_id'].repeat_interleave(n_samples, dim=0)

        batch["action"] = []
        for h in vla_history:
            batch["action"].append(h["action"])
        batch["action"] = torch.tensor(batch["action"], dtype=torch.float32)
        batch["action"] = batch["action"].permute(1, 0, 2, 3).reshape(batch_size, -1, batch["action"].shape[-1])

        print(f"return_rollouts: {return_rollouts}")
        if return_rollouts:
            start_time = time.time()
            H, W, C = videos[0][0].shape 
            T = max_steps + 1
            placeholder = torch.empty((T, H, W, C), dtype=torch.uint8)
            videos_as_tensors = [torch.from_numpy(np.array(v, dtype=np.uint8)) for v in videos]
            # 同样使用 pad_sequence
            padded_with_placeholder = rnn_utils.pad_sequence(
                videos_as_tensors + [placeholder],  # 临时加入占位符
                batch_first=True,
                padding_value=0
            )
            padded_videos = padded_with_placeholder[:-1]
            batch["video"] = padded_videos
            end_time = time.time()
            print(f"Optimized padding time: {end_time - start_time} seconds")
        else:
            del videos

        output_batch = TensorDict(
            batch,
            batch_size=batch_size)

        # import pickle
        # local_rank = dist.get_rank() % 8
        # os.makedirs('./debug/output_batch', exist_ok=True)
        # with open(f'./debug/output_batch/output_batch_{local_rank}.pkl', 'wb') as f:
        #     pickle.dump(output_batch, f)
        # local_rank = dist.get_rank() % 8
        # with open(f'./debug/output_batch/output_batch_{local_rank}.pkl', 'rb') as f:
        #     output_batch = pickle.load(f)
        return DataProto(batch=output_batch)

    def generate_sequences(self, prompts):
        batch_size = prompts.batch.batch_size[0]
        
        if prompts.meta_info.get('n_samples') is None:
            micro_batch_size = self.config.val_micro_batch_size if self.config.val_micro_batch_size is not None else 1
        else:
            micro_batch_size = self.config.get('micro_batch_size', batch_size)
        
        num_chunks = max(batch_size // micro_batch_size, 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        output = [self._generate_minibatch(p) for p in batch_prompts]
        output = DataProto.concat(output)
        return output
    
    def process_input(self,inputs:list, task_descriptions:list):
        
        batchdata = {"input_ids":[],"attention_mask":[],"pixel_values":[]}  
        
        for i in range(len(inputs)):
            input = inputs[i]
            task_description = task_descriptions[i]
           
            image = Image.fromarray(input["full_image"]).convert("RGB")
            if self.config.center_crop:
                image = center_crop_image(image)
            prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
            batch_feature  = self.processor(prompt, image)
            
            if "wrist_image" in input.keys():
                wrist_image = Image.fromarray(input["wrist_image"]).convert("RGB")
                if self.config.center_crop:
                    wrist_image = center_crop_image(wrist_image)
                wrist_batch_feature = self.processor(prompt, wrist_image)
                primary_pixel_values = batch_feature["pixel_values"]
                batch_feature["pixel_values"] = torch.cat([primary_pixel_values] + [wrist_batch_feature["pixel_values"]], dim=1)
                
            input_ids = batch_feature["input_ids"]
            attention_mask = batch_feature["attention_mask"]
            pixel_values = batch_feature["pixel_values"]
            
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
                )
                if self.config.vla in ["openvla-oft"]:
                    attention_mask = torch.cat(
                        (attention_mask, torch.unsqueeze(torch.Tensor([True]).bool(), dim=0).to(attention_mask.device)), dim=1
                    )
            
            batchdata["input_ids"].append(input_ids)    
            batchdata["attention_mask"].append(attention_mask)    
            batchdata["pixel_values"].append(pixel_values)    
        
        
        device = torch.device('cuda') 
        
        if self.config.vla in ["openvla-oft"]:
            batchdata["input_ids"] = [x.transpose(0, 1) for x in batchdata["input_ids"]]
            batchdata["attention_mask"] = [x.transpose(0, 1) for x in batchdata["attention_mask"]]
            batchdata["input_ids"] = pad_sequence(batchdata["input_ids"], batch_first=True, padding_value=self.processor.tokenizer.pad_token_id).squeeze(-1).to(device)
            batchdata["attention_mask"] = pad_sequence(batchdata["attention_mask"], batch_first=True, padding_value=0).squeeze(-1).to(device)
            
            padding_mask = batchdata["input_ids"].ne(self.processor.tokenizer.pad_token_id)
            assert  torch.all(padding_mask==batchdata["attention_mask"].ne(0))
            padding_mask = ~padding_mask
            padding_mask = padding_mask.int() 
            sorted_indices = torch.argsort(padding_mask, dim=1, descending=True, stable=True)
            batchdata["input_ids"] = torch.gather(batchdata["input_ids"], 1, sorted_indices)
            batchdata["attention_mask"] = torch.gather(batchdata["attention_mask"], 1, sorted_indices)
            
            
            batchdata["pixel_values"] = torch.cat(batchdata["pixel_values"] , dim=0).to(device)
            assert torch.all(batchdata["attention_mask"].ne(0) == batchdata["input_ids"].ne(self.processor.tokenizer.pad_token_id))
        else:
            for key in ["input_ids", "attention_mask", "pixel_values"]:
                batchdata[key] = torch.cat(batchdata[key], dim=0).to(device)

        return batchdata
   
    def _generate_minibatch(self, prompts):
        self.module.eval()
        meta_info = prompts.meta_info
        n_samples = meta_info.get('n_samples', 1)
        states = np.array(prompts.batch['states'].cpu())
        models = prompts.non_tensor_batch['model']
        state_list = [{"states": state, "model": model} for state, model in zip(states, models)]
        return_rollouts = meta_info.get('return_rollouts', False)
        max_steps = self.max_steps
        batch_size = prompts.batch.batch_size[0] * n_samples
        is_valid = meta_info.get('n_samples') is None
        global_steps = meta_info.get('global_steps', 0) if is_valid else 0
        is_valid = True

        # --- 初始化多个环境 ---
        envs = []
        inputs = []
        task_descriptions = []
        task_records = []
        valid_video = [[] for _ in range(batch_size)]

        for idx in range(batch_size):
            state = state_list[int(idx / n_samples)]
            cfg = config_factory(self.ext_cfg["algo_name"])
            with cfg.values_unlocked():
                cfg.update(self.ext_cfg)
            cfg.lock()
            ObsUtils.initialize_obs_utils_with_config(cfg)
            env = _create_env(cfg)

            if state:
                env.reset_to(state)
            else:
                env.reset()

            # 预跑 num_steps_wait
            t = 0
            valid_images = []
            obs = None
            while t < self.config.num_steps_wait:
                obs, _, _, _ = env.step(np.zeros(7))
                obs["agentview_image"] = (obs["agentview_image"]*255).astype(np.uint8).transpose(1,2,0)
                t += 1
            if is_valid:
                valid_images.append(obs["agentview_image"])

            envs.append(env)
            task_descriptions.append(self.task_description)
            inputs.append(self._obs_to_input(obs))
            task_records.append({
                "active": True,
                "complete": False,
                "finish_step": 0
            })
            if is_valid:
                valid_video[idx].extend(valid_images)

        # --- 主循环 ---
        vla_history = []
        step = 0
        while step < max_steps:
            print(f"Step = {step}")
            active_indices = [i for i, r in enumerate(task_records) if r['active']]

            current_inputs = inputs
            current_task_descriptions = task_descriptions
            vla_input = self.process_input(current_inputs, current_task_descriptions)
            vla_input.update(meta_info)
            vla_output = self._generate_one_step(vla_input)
            actions = vla_output["action"]

            step_data = {
                "responses": vla_output["responses"],
                "input_ids": vla_output["input_ids"],
                "attention_mask": vla_output["attention_mask"],
                "pixel_values": vla_output["pixel_values"],
                "action": actions,
                "step": step
            }
            vla_history.append(step_data)

            new_inputs = inputs.copy()
            for idx in active_indices:
                env = envs[idx]
                step_images = []

                for a in actions[idx]:
                    obs, reward, done, info = env.step(a.tolist())
                    obs["agentview_image"] = (obs["agentview_image"]*255).astype(np.uint8).transpose(1,2,0)
                    if is_valid:
                        step_images.append(obs["agentview_image"])

                    task_records[idx]['finish_step'] += 1
                    if reward > 0.0 or task_records[idx]['finish_step'] >= max_steps:
                        task_records[idx]['active'] = False
                        task_records[idx]['complete'] = reward > 0.0
                        break

                new_inputs[idx] = self._obs_to_input(obs)
                if is_valid:
                    valid_video[idx].extend(step_images)

            inputs = new_inputs
            step += self.config.action_chunks_len

        # --- 清理环境 ---
        for env in envs:
            env.env.close()
        import gc
        gc.collect()
        torch.cuda.empty_cache()        
        self.module.train()
        
        batch = {
                'responses': [],
                'input_ids': [],  # here input_ids become the whole sentences
                'attention_mask': [],
                'pixel_values': [],
            }
        for k in ["responses", "input_ids", "attention_mask", "pixel_values"]:
            for h in vla_history:
                batch[k].append(h[k])
        
        for k,v in batch.items():
            batch[k] = torch.stack(v,dim=1) 
  
        batch["complete"] = []
        batch["finish_step"] = []

        if return_rollouts:
            batch["action"] = []
            for h in vla_history:
                batch['action'].append(h['action'])
            batch['action'] = torch.tensor(batch['action'], dtype=torch.float32)
            batch['action'] = batch['action'].permute(1, 0, 2, 3).reshape(batch_size, -1, batch['action'].shape[-1])

            start_time = time.time()
            H, W, C = valid_video[0][0].shape 
            T = max_steps + 1
            placeholder = torch.empty((T, H, W, C), dtype=torch.uint8)
            videos_as_tensors = [torch.from_numpy(np.array(v, dtype=np.uint8)) for v in valid_video]
            # 同样使用 pad_sequence
            padded_with_placeholder = rnn_utils.pad_sequence(
                videos_as_tensors + [placeholder],  # 临时加入占位符
                batch_first=True,
                padding_value=0
            )
            padded_videos = padded_with_placeholder[:-1]
            batch["video"] = padded_videos
            end_time = time.time()
            print(f"Optimized padding time: {end_time - start_time} seconds")

        # batch['video'] = valid_video
        for k in task_records:
            batch["complete"].append(k["complete"])
            batch["finish_step"].append(k["finish_step"])
        
        batch["complete"] = torch.tensor(batch["complete"], dtype=torch.bool, device=batch['responses'].device)
        batch["finish_step"] = torch.tensor(batch["finish_step"], dtype=torch.int64, device=batch['responses'].device)
        # f()
        output_batch = TensorDict(
            batch,
            batch_size=batch_size)
        # TODO
        
        return DataProto(batch=output_batch)
    
    @torch.no_grad()
    def _generate_one_step(self, prompts: dict):
        if self.config.vla == "openvla-oft":
            idx = prompts['input_ids']  # (bs, prompt_length)
            attention_mask = prompts['attention_mask']  # left-padded attention_mask
            pixel_values = prompts["pixel_values"]
        
        
            param_ctx = contextlib.nullcontext()

            # make sampling args can be overriden by inputs
            do_sample = prompts.get('do_sample', self.config.do_sample)
        

            temperature = prompts.get('temperature', self.config.temperature)

            #generation_config = GenerationConfig(temperature=temperature, top_p=top_p, top_k=top_k)

            if isinstance(self.module, FSDP):
                # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
                param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
            
            with param_ctx:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    actions, response, normalized_actions = self.module.generate_action_verl(
                        input_ids=idx,
                        pixel_values=pixel_values,
                        attention_mask=attention_mask,
                        padding_idx = self.processor.tokenizer.pad_token_id,
                        do_sample=do_sample,
                        unnorm_key=self.config.unnorm_key,
                        temperature=temperature, )
            
            
            assert self.processor.tokenizer.pad_token_id is not None

            assert idx.ndim == 2
            idx = verl_F.pad_sequence_to_length(idx,max_seq_len=self.config.max_prompt_length,pad_token_id=self.processor.tokenizer.pad_token_id,left_pad=True)
            
            assert attention_mask.ndim == 2
            attention_mask = verl_F.pad_sequence_to_length(attention_mask,max_seq_len=self.config.max_prompt_length,pad_token_id=0,left_pad=True)
            
            
            assert idx.device.type == 'cuda'
            assert response.device.type == 'cuda'
            #assert seq.device.type == 'cuda'
            assert attention_mask.device.type == 'cuda'
            assert pixel_values.device.type == 'cuda'
            batch ={
                    'responses': response,
                    'input_ids': idx,
                    'attention_mask': attention_mask,
                    "pixel_values":pixel_values,
                    "action":actions,
                    "normalized_actions": normalized_actions
                }

            return batch
        
        elif self.config.vla == "openvla": 
            idx = prompts['input_ids']  # (bs, prompt_length)
            attention_mask = prompts['attention_mask']  # left-padded attention_mask
            pixel_values = prompts["pixel_values"]
            
            # used to construct attention_mask
            eos_token_id = prompts['eos_token_id']
            pad_token_id = prompts['pad_token_id']

            batch_size = idx.size(0)
            prompt_length = idx.size(1)
            #self.module.eval()
            param_ctx = contextlib.nullcontext()

            do_sample = prompts.get('do_sample', self.config.do_sample)
            response_length =  self.module.get_action_dim(self.config.unnorm_key)
            top_p = prompts.get('top_p', self.config.get('top_p', 1.0))
            top_k = prompts.get('top_k', self.config.get('top_k', 0))
            if top_k is None:
                top_k = 0
            top_k = max(0, top_k)  # to be compatible with vllm

            temperature = prompts.get('temperature', self.config.temperature)
            generation_config = GenerationConfig(temperature=temperature, top_p=top_p, top_k=top_k)

            if isinstance(self.module, FSDP):
                # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
                param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
            
            with param_ctx:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    
                    output = self.module.generate(
                        input_ids=idx,
                        pixel_values=pixel_values,
                        attention_mask=attention_mask,
                        do_sample=do_sample,
                        max_new_tokens=response_length,
                        # max_length=max_length,
                        eos_token_id=eos_token_id,
                        pad_token_id=pad_token_id,
                        generation_config=generation_config,
                        # renormalize_logits=True,
                        output_scores=False,  # this is potentially very large
                        return_dict_in_generate=True,
                        use_cache=True)
                    
           
            seq = output.sequences
            sequence_length = prompt_length + response_length
            delta_length = sequence_length - seq.shape[1]
            
            assert delta_length == 0
            assert seq.shape[1] == sequence_length

            prompt = seq[:, :prompt_length]  # (bs, prompt_length)
            response = seq[:, prompt_length:]  # (bs, response_length)

            response_length = response.size(1)
            #delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
            #delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
            #response_position_ids = position_ids[:, -1:] + delta_position_id
            #position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

            response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
            attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

            # Extract predicted action tokens and translate into (normalized) continuous actions
            predicted_action_token_ids = response.detach().cpu().numpy()
            discretized_actions = self.module.vocab_size - predicted_action_token_ids
            discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.module.bin_centers.shape[0] - 1)
            normalized_actions = self.module.bin_centers[discretized_actions]

            # Unnormalize actions
            action_norm_stats = self.module.get_action_stats(self.config.unnorm_key)
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
            actions = np.where(
                mask,
                0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
                normalized_actions,
            )
            
            actions = np.expand_dims(actions, axis=1)
            
            assert self.processor.tokenizer.pad_token_id is not None
            assert prompt.ndim == 2
            prompt = verl_F.pad_sequence_to_length(prompt,max_seq_len=self.config.max_prompt_length,pad_token_id=self.processor.tokenizer.pad_token_id,left_pad=True)
            assert seq.ndim == 2
            seq = verl_F.pad_sequence_to_length(seq,max_seq_len=self.config.max_prompt_length,pad_token_id=self.processor.tokenizer.pad_token_id,left_pad=True)
            assert attention_mask.ndim == 2
            attention_mask = verl_F.pad_sequence_to_length(attention_mask,max_seq_len=self.config.max_prompt_length,pad_token_id=0,left_pad=True)
            
            batch ={
                    'prompts': prompt,
                    'responses': response,
                    'input_ids': seq,
                    'attention_mask': attention_mask,
                    "pixel_values":pixel_values,
                    "action":actions,
                    #'position_ids': position_ids
                }
            
            return batch
                    
    def _obs_to_input(self, obs):
        
        if self.config.num_images_in_input > 1:
            return {
                "full_image": get_libero_image(obs, 224),
                "wrist_image": get_libero_wrist_image(obs, 224),
                "state": np.concatenate([
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"]
                ])
            }
        else:
            return {
                "full_image": obs['agentview_image'], # get_libero_image(obs, 224),
                # "state": np.concatenate([
                #     obs["robot0_eef_pos"],
                #     quat2axisangle(obs["robot0_eef_quat"]),
                #     obs["robot0_gripper_qpos"]
                # ])
            }
