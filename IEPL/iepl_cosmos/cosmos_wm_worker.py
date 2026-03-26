"""
Cosmos-based World Model worker for VLA-RFT.

Replaces the iVideoGPT-based WorldModelRolloutWorker with a Cosmos
diffusion-based video generation model while keeping the same
interface expected by RayVLARFTGRPOTrainer.

This file is self-contained and does NOT modify any existing VLA-RFT code.
"""

import json
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf, open_dict
from PIL import Image

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils.debug import log_gpu_memory_usage

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


def _add_cosmos_to_path(cosmos_root: str) -> None:
    """Ensure cosmos_predict2 is importable by adding cosmos_root to sys.path."""
    cosmos_root = str(Path(cosmos_root).resolve())
    if cosmos_root not in sys.path:
        sys.path.insert(0, cosmos_root)
    # Also check COSMOS_ROOT env var as fallback (useful in Docker)
    env_root = os.environ.get('COSMOS_ROOT')
    if env_root and env_root not in sys.path:
        sys.path.insert(0, str(Path(env_root).resolve()))


def _load_cosmos_infer(cosmos_root, experiment, checkpoint_path, config_file):
    _add_cosmos_to_path(cosmos_root)
    from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference

    config_path = Path(config_file)
    if not config_path.is_absolute():
        config_path = Path(cosmos_root) / config_path
    if config_path.suffix == ".py":
        if not config_path.exists():
            raise FileNotFoundError(f"Cosmos config file not found: {config_path}")
        rel_path = config_path.relative_to(Path(cosmos_root).resolve())
        config_arg = str(rel_path)
    else:
        config_arg = config_file

    return Video2WorldInference(
        experiment_name=experiment,
        ckpt_path=checkpoint_path,
        s3_credential_path="",
        context_parallel_size=1,
        config_file=config_arg,
    )


def _load_reward_model(rm_path, rm_img_size=224, device="cuda"):
    """Load VideoMAE reward model for success classification."""
    from transformers import (
        VideoMAEConfig,
        VideoMAEFeatureExtractor,
        VideoMAEForVideoClassification,
    )
    config = VideoMAEConfig.from_pretrained("MCG-NJU/videomae-base")
    config.num_labels = 2
    model = VideoMAEForVideoClassification(config)
    state = torch.load(rm_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base")
    feature_extractor.size = {"shortest_edge": rm_img_size}
    return model, feature_extractor


__all__ = ["CosmosWorldModelWorker"]


class CosmosWorldModelWorker(Worker):
    """
    A Worker that wraps the Cosmos action-conditioned video generation model.
    
    Provides the same interface as VLA-RFT's WorldModelRolloutWorker
    (init_model, generate_sequences) so it can be used as a drop-in
    replacement in the RayVLARFTGRPOTrainer.
    
    Unlike the original WorldModelRolloutWorker which uses an
    autoregressive token-based world model (iVideoGPT + vLLM),
    this worker uses Cosmos diffusion-based video generation.
    """

    def __init__(self, config: DictConfig, role: str):
        super().__init__()
        self.config = config
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group()

        self.role = role
        assert self.role in ['wm_rollout']

        # Accept either config.cosmos or config.wm (actor_rollout_ref.wm).
        self.cosmos_config = getattr(config, "cosmos", None) or getattr(config, "wm", None)
        if self.cosmos_config is None:
            raise ValueError("CosmosWorldModelWorker requires config.cosmos or config.wm")
        self.rm_config = config.get('reward_model', {})

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        device = torch.cuda.current_device()
        cc = self.cosmos_config

        self.cosmos_infer = _load_cosmos_infer(
            cosmos_root=cc.cosmos_root,
            experiment=cc.cosmos_experiment,
            checkpoint_path=cc.cosmos_checkpoint_path,
            config_file=cc.get(
                'cosmos_config_file',
                'cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py',
            ),
        )

        self.action_chunk_size = cc.get('cosmos_action_chunk_size', 12)
        self.action_scale = cc.get('cosmos_action_scale', 20.0)
        self.gripper_scale = cc.get('cosmos_gripper_scale', 1.0)
        self.resolution = cc.get('cosmos_resolution', '256,320')
        self.guidance = cc.get('cosmos_guidance', 0.0)
        self.seed = cc.get('cosmos_seed', 0)

        neg_prompt = ""
        neg_file = cc.get('cosmos_negative_prompt_file', None)
        if neg_file and Path(neg_file).exists():
            try:
                neg_data = json.loads(Path(neg_file).read_text(encoding="utf-8"))
                if isinstance(neg_data, dict) and neg_data.get("negative_prompt"):
                    neg_prompt = neg_data["negative_prompt"]
            except Exception:
                pass
        self.negative_prompt = neg_prompt

        if self.rm_config.get('enable', False):
            rm_path = self.rm_config.path
            rm_img_size = self.rm_config.get('img_size', 224)
            self.rm_model, self.rm_feature_extractor = _load_reward_model(
                rm_path, rm_img_size, device=f"cuda:{device}"
            )
            self.rm_threshold = self.rm_config.get('threshold', 0.5)
        else:
            self.rm_model = None


        log_gpu_memory_usage('After CosmosWorldModelWorker init_model', logger=None)

    def _pad_actions(self, actions: np.ndarray, target_len: int) -> np.ndarray:
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if actions.shape[0] >= target_len:
            return actions[:target_len]
        pad = np.tile(actions[-1:], (target_len - actions.shape[0], 1))
        return np.concatenate([actions, pad], axis=0)

    def _scale_actions(self, actions: np.ndarray) -> np.ndarray:
        actions = actions.astype(np.float32, copy=False)
        if actions.shape[-1] >= 7:
            actions[..., :6] = actions[..., :6] * self.action_scale
            actions[..., 6] = actions[..., 6] * self.gripper_scale
        return actions

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.resolution == "none":
            return frame
        try:
            h, w = map(int, str(self.resolution).split(","))
            return np.array(Image.fromarray(frame).resize((w, h), Image.BICUBIC))
        except Exception:
            return frame

    def _cosmos_generate_chunk(
        self, current_frame: np.ndarray, actions_chunk: np.ndarray, seed: int
    ) -> np.ndarray:
        """
        Generate a video chunk using Cosmos.
        
        Args:
            current_frame: (H, W, C) uint8 or float numpy array
            actions_chunk: (T, action_dim) numpy array, already scaled
            seed: random seed
            
        Returns:
            video_clamped: (T+1, H, W, C) uint8 numpy array
        """
        frame = self._resize_frame(current_frame)
        img_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
        num_video_frames = actions_chunk.shape[0] + 1
        vid_input = torch.cat(
            [img_tensor, torch.zeros_like(img_tensor).repeat(num_video_frames - 1, 1, 1, 1)],
            dim=0,
        )
        vid_input = (vid_input * 255.0).to(torch.uint8)
        vid_input = vid_input.unsqueeze(0).permute(0, 2, 1, 3, 4)

        video = self.cosmos_infer.generate_vid2world(
            prompt="",
            input_path=vid_input,
            action=torch.from_numpy(actions_chunk).float(),
            guidance=self.guidance,
            num_video_frames=num_video_frames,
            num_latent_conditional_frames=1,
            resolution=self.resolution,
            seed=seed,
            negative_prompt=self.negative_prompt,
        )

        video_normalized = torch.clamp((video - (-1)) / 2.0, 0, 1)
        video_clamped = (
            (video_normalized[0] * 255)
            .to(torch.uint8)
            .permute(1, 2, 3, 0)
            .cpu()
            .numpy()
        )
        return video_clamped

    @torch.no_grad()
    def _evaluate_trajectory(self, video_frames: np.ndarray) -> dict:
        """
        Evaluate a trajectory using VideoMAE reward model.
        
        Returns dict with 'success_prob' and 'complete' (binary).
        """
        if self.rm_model is None:
            return {'success_prob': 0.0, 'complete': 0}

        window_size = 8
        if video_frames.shape[0] < window_size:
            return {'success_prob': 0.0, 'complete': 0}

        clips = []
        for end in range(video_frames.shape[0], window_size - 1, -1):
            clip = [Image.fromarray(video_frames[i]) for i in range(end - window_size, end)]
            clips.append(clip)

        max_prob = 0.0
        batch_size = 4
        for i in range(0, len(clips), batch_size):
            batch_clips = clips[i:i+batch_size]
            inputs = self.rm_feature_extractor(batch_clips, return_tensors="pt")["pixel_values"]
            inputs = inputs.to(next(self.rm_model.parameters()).device)
            logits = self.rm_model(pixel_values=inputs).logits
            probs = torch.sigmoid(logits)[:, 1].cpu().numpy()
            max_prob = max(max_prob, float(probs.max()))

        complete = 1 if max_prob >= self.rm_threshold else 0
        return {'success_prob': max_prob, 'complete': complete}


    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        """
        Generate future visual frames using Cosmos world model.
        
        Expected input in prompts.batch:
            - 'pixels': (B, T, H, W, C) or (B, T, C, H, W) raw pixel values
            - 'predicted_actions': (B, T-1, action_dim) actions from VLA
            
        Returns DataProto with:
            - 'responses': token-level responses (for compatibility with reward_fn)
            - 'predicted_frames': (B, T_out, H, W, C) generated frames
            - 'attention_mask': dummy mask for compatibility
        """
        prompts = prompts.to(torch.cuda.current_device())

        pixels = prompts.batch['pixels']
        predicted_actions = prompts.batch['predicted_actions']

        batch_size = pixels.shape[0]
        device = pixels.device

        if pixels.dtype == torch.uint8:
            pixels_np = pixels.cpu().numpy()
        else:
            if pixels.shape[-1] == 3:
                pixels_np = (pixels.cpu().numpy() * 255).astype(np.uint8)
            else:
                pixels_np = (pixels.permute(0, 1, 3, 4, 2).cpu().numpy() * 255).astype(np.uint8)

        actions_np = predicted_actions.cpu().numpy()

        all_generated_frames = []
        all_success_probs = []
        all_completes = []

        for b in range(batch_size):
            current_frame = pixels_np[b, 0]
            current_frame = self._resize_frame(current_frame)
            sample_actions = actions_np[b]
            act_len = sample_actions.shape[0]
            num_chunks_this = (act_len + self.action_chunk_size - 1) // self.action_chunk_size
            print(f"[CosmosRFT][WM] sample {b + 1}/{batch_size} ({num_chunks_this} Cosmos chunks)", flush=True)
            frames = [np.expand_dims(current_frame, axis=0)]
            step = 0
            chunk_idx = 0
            while step < act_len:
                chunk_len = min(self.action_chunk_size, act_len - step)
                actions_chunk = sample_actions[step:step + chunk_len]
                actions_chunk = self._pad_actions(actions_chunk, self.action_chunk_size)
                actions_chunk = self._scale_actions(actions_chunk)
                sample_seed = int(self.seed + b + chunk_idx)
                video_clamped = self._cosmos_generate_chunk(
                    current_frame, actions_chunk, seed=sample_seed
                )
                frames.append(video_clamped[1:chunk_len + 1])
                print(f"[CosmosRFT][WM]   sample {b + 1} chunk {chunk_idx + 1}/{num_chunks_this} done", flush=True)
                current_frame = video_clamped[chunk_len]
                step += chunk_len
                chunk_idx += 1

            video_full = np.concatenate(frames, axis=0)
            all_generated_frames.append(video_full)
            if self.rm_model is not None:
                eval_result = self._evaluate_trajectory(video_full)
                all_success_probs.append(eval_result['success_prob'])
                all_completes.append(eval_result['complete'])


        max_frames = max(v.shape[0] for v in all_generated_frames)
        h, w, c = all_generated_frames[0].shape[1:]
        padded_frames = np.zeros((batch_size, max_frames, h, w, c), dtype=np.uint8)
        for b, frames in enumerate(all_generated_frames):
            padded_frames[b, :frames.shape[0]] = frames

        predicted_frames = torch.from_numpy(padded_frames).to(device)

        output_dict = {
            'predicted_frames': predicted_frames,
        }

        if all_success_probs:
            output_dict['success_prob'] = torch.tensor(
                all_success_probs, dtype=torch.float32, device=device
            )
            output_dict['complete'] = torch.tensor(
                all_completes, dtype=torch.float32, device=device
            )

        output = DataProto.from_single_dict(output_dict)
        return output.to('cpu')

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data: DataProto):
        """No-op for Cosmos: diffusion models don't produce token-level log probs."""
        batch_size = data.batch['responses'].shape[0]
        resp_len = data.batch['responses'].shape[1]
        dummy_log_prob = torch.zeros(batch_size, resp_len, dtype=torch.float32)
        output = DataProto.from_dict(tensors={'wm_log_probs': dummy_log_prob})
        return output.to('cpu')
