"""
VLA-RFT trainer that uses Cosmos as the world model with MimicGen data.

Architecture for MimicGen mode:
  - Actor rollout worker (local VLA-RFT ActorRolloutRefWorker):
    VLA action generation
  - CosmosWorldModelWorker:
    Video rollout from init frames + actions
  - StateDataset + BufferedDataLoader for MimicGen initial states
  - GRPO advantage computation
  - Reward from pixel-level reconstruction (MSE/LPIPS) on predicted vs GT frames

The actor rollout worker handles VLA action generation. This trainer orchestrates:
  1. Loading initial states (state_id) from StateDataset
  2. Calling actor rollout with use_wm=True
  3. Computing reconstruction reward from predicted vs GT frames
  4. GRPO advantage + actor update

Usage:
    python -m iepl_cosmos.main_cosmos_rft <hydra overrides>
"""

import os
import uuid
import time
import json
from collections import defaultdict, Counter
from pprint import pprint

import numpy as np
import torch
from torch.utils.data import DataLoader
from codetiming import Timer
from omegaconf import OmegaConf

from verl import DataProto

def _debug_log(msg: str) -> None:
    path = os.environ.get("VLA_RFT_DEBUG_LOG")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def _filter_metrics_for_logging(metrics: dict) -> dict:
    """Keep only MSE/reconstruction/reward-related metrics; drop timing/noise."""
    keep_substrings = (
        'mse',
        'recon',
        'perceptual',
        'total_loss',
        'reward',
        'score',
        'acc',
    )
    filtered = {}
    for k, v in metrics.items():
        if any(s in k for s in keep_substrings):
            filtered[k] = v
    return filtered
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import (
    Role,
    ResourcePoolManager,
    compute_advantage,
    compute_data_metrics,
    reduce_metrics,
    apply_kl_penalty,
)
try:
    from verl.utils.dataset.rob_dataset import StateDataset, BufferedDataLoader, collate_fn
except ModuleNotFoundError:
    # Fallback for local VLA-RFT (no WMPO rob_dataset). Only MimicGenVideoDataset is supported here.
    from verl.utils.dataset.rl_dataset import collate_fn

    class StateDataset:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "StateDataset is not available in local VLA-RFT. "
                "Please set data.video_root to use MimicGenVideoDataset."
            )

    class BufferedDataLoader:  # type: ignore
        def __init__(self, dataloader):
            self.dataloader = dataloader
            self._iter = None

        def start_new_epoch(self):
            self._iter = iter(self.dataloader)

        def get_next_batch(self):
            if self._iter is None:
                self._iter = iter(self.dataloader)
            try:
                return next(self._iter)
            except StopIteration:
                self._iter = iter(self.dataloader)
                return next(self._iter)
from iepl_cosmos.mimicgen_video_dataset import MimicGenVideoDataset, mimicgen_collate_fn
from iepl_cosmos.cosmos_reward import compute_trajectory_recon_reward


class MimicGenRewardManager:
    """
    Reward manager for MimicGen tasks with Cosmos world model.

    Uses pixel-level reconstruction reward (MSE/LPIPS) from predicted vs GT frames.
    Compatible with WMPO's reward interface (verify + __call__).
    """

    def __init__(self, config, num_examine=0):
        self.config = config
        self.num_examine = num_examine
        self._wm_enable = True
        self._lpips = None

    def verify(self, data):
        batch_size = data.batch['responses'].size(0)

        if 'video' in data.batch and 'gt_frames' in data.batch:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            predicted_frames = data.batch['video'].to(device)
            gt_frames = data.batch['gt_frames'].to(device)
            gt_lengths = data.batch.get('gt_lengths', None)
            if gt_lengths is None:
                gt_lengths = torch.full(
                    (predicted_frames.shape[0],),
                    gt_frames.shape[1],
                    dtype=torch.long,
                    device=device,
                )
            else:
                gt_lengths = gt_lengths.to(device)

            if self._lpips is None:
                from ivideogpt.lpips import LPIPS
                self._lpips = LPIPS().to(device).eval()

            lpips_every = int(self.config.data.get('lpips_every', 4))
            global_steps = int(data.meta_info.get('global_steps', 0))
            compute_lpips = (global_steps + 1) % max(1, lpips_every) == 0

            resize_hw = tuple(self.config.data.get('reward_resize_hw', [224, 224]))
            loss_weight = {
                'mse': float(self.config.data.get('reward_mse_weight', 1.0)),
                'lpips': float(self.config.data.get('reward_lpips_weight', 1.0)),
            }
            reward_tensor, recon_metrics = compute_trajectory_recon_reward(
                predicted_frames=predicted_frames,
                gt_frames=gt_frames,
                gt_lengths=gt_lengths,
                lpips_fn=self._lpips if compute_lpips else None,
                reward_fn_type='mse',
                loss_weight=loss_weight,
                resize_hw=resize_hw,
                compute_lpips=compute_lpips,
            )

            reward_scale = float(self.config.data.get('reward_scale', 1.0))
            score_tensor = reward_tensor * reward_scale
            score_tensor = score_tensor.to(dtype=torch.float32, device=data.batch['responses'].device)
            reward_metrics = recon_metrics
        else:
            # If frames are missing, return zero scores.
            score_tensor = torch.zeros(
                batch_size,
                dtype=torch.float32,
                device=data.batch['responses'].device,
            )
            reward_metrics = {}

        data.batch['acc'] = score_tensor
        data.batch['format_correctness'] = torch.ones(
            batch_size, dtype=torch.float32, device=data.batch['responses'].device
        )

        reward_metrics = {'all': score_tensor.mean().item(), **reward_metrics}
        format_metrics = {'all': 1.0}
        reward_format_metrics = {'all': score_tensor.mean().item()}

        return score_tensor.detach().cpu().tolist(), reward_metrics, format_metrics, reward_format_metrics

    def __call__(self, data: DataProto):
        reward_tensor_dict = {}
        reward_metrics = {}

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        verifier_reward = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_tensor = reward_tensor.reshape((reward_tensor.shape[0], -1))
        verifier_reward = verifier_reward.reshape((verifier_reward.shape[0], -1))

        valid_response_length = data.batch['finish_step'] * self.config.actor_rollout_ref.model.action_token_len
        max_token_len = verifier_reward.shape[1]
        safe_valid_response_length = torch.clamp(valid_response_length, min=1, max=max_token_len)

        if 'acc' in data.batch:
            verifier_score = data.batch['acc'].cpu().numpy().tolist()
        else:
            verifier_score, _, _, _ = self.verify(data)

        score_tensor = torch.tensor(verifier_score, dtype=verifier_reward.dtype, device=verifier_reward.device)
        row_idx = torch.arange(verifier_reward.shape[0], device=verifier_reward.device)
        # Distribute reward uniformly across valid action tokens to reduce sparsity.
        valid_lens = safe_valid_response_length.to(dtype=torch.long)
        max_token_len = verifier_reward.shape[1]
        steps = torch.arange(max_token_len, device=verifier_reward.device).unsqueeze(0)
        mask = steps < valid_lens.unsqueeze(1)
        per_token = (score_tensor / valid_lens.to(dtype=score_tensor.dtype)).unsqueeze(1)
        verifier_reward = verifier_reward + mask.to(dtype=verifier_reward.dtype) * per_token

        reward_tensor_dict['gt_scores'] = verifier_reward

        verifier_coef = self.config.get('verifier', {}).get('reward_coef', 5.0)
        if verifier_coef != 0:
            reward_metrics['verifier'] = (verifier_coef * reward_tensor_dict['gt_scores']).sum(dim=1).mean().item()
            reward_tensor += verifier_coef * reward_tensor_dict['gt_scores']

        reward_tensor_dict['all'] = reward_tensor
        reward_metrics['reward_all'] = reward_tensor.sum(dim=-1).mean(dim=0).item()

        return reward_tensor_dict, reward_metrics


class RayCosmosRFTGRPOTrainer:
    """
    VLA-RFT GRPO trainer with Cosmos world model and MimicGen data.

    Uses local VLA-RFT actor rollout worker + CosmosWorldModelWorker.
    The training loop:
      1. Sample initial states from StateDataset
      2. Actor rollout generates imagined trajectories (VLA → Cosmos)
      3. Compute reconstruction reward from predicted vs GT frames
      4. GRPO advantage estimation
      5. Actor policy update
    """

    def __init__(self, config, tokenizer, role_worker_mapping, resource_pool_manager,
                 reward_fn, val_reward_fn, ray_worker_group_cls=RayWorkerGroup):
        self.config = config
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = (
            Role.RefPolicy in role_worker_mapping
            and config.algorithm.kl_ctrl.kl_coef > 0
        )
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.use_critic = config.algorithm.adv_estimator == 'gae'
        self.ray_worker_group_cls = ray_worker_group_cls

        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                self.kl_ctrl = core_algos.AdaptiveKLController(
                    init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                    target_kl=config.algorithm.kl_ctrl.target_kl,
                    horizon=config.algorithm.kl_ctrl.horizon,
                )
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self.train_mode = str(self.config.trainer.get('train_mode', 'rl')).lower()
        if self.train_mode == 'bc':
            from prismatic.vla.action_tokenizer import ActionTokenizer
            bins = int(self.config.data.get('bc_action_bins', 256))
            min_action = float(self.config.data.get('bc_action_min', -1.0))
            max_action = float(self.config.data.get('bc_action_max', 1.0))
            self.bc_action_tokenizer = ActionTokenizer(
                self.tokenizer, bins=bins, min_action=min_action, max_action=max_action
            )

        self._create_dataloader()

    def _build_bc_responses(self, gt_actions: torch.Tensor, action_len: torch.Tensor,
                            traj_len: int, response_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Build BC target action tokens and mask from GT continuous actions."""
        # gt_actions: (B, T, action_dim) float32
        # action_len: (B,) int number of valid actions
        bsz, max_act, action_dim = gt_actions.shape
        action_chunks_len = int(self.config.actor_rollout_ref.model.action_chunks_len)
        total_steps = traj_len * action_chunks_len
        # Clamp and pad/truncate actions to total_steps.
        actions_np = gt_actions.detach().cpu().numpy()
        if max_act < total_steps:
            pad = np.zeros((bsz, total_steps - max_act, action_dim), dtype=actions_np.dtype)
            actions_np = np.concatenate([actions_np, pad], axis=1)
        else:
            actions_np = actions_np[:, :total_steps, :]

        # Convert continuous actions to token ids.
        bins = self.bc_action_tokenizer.bins
        vocab_size = self.bc_action_tokenizer.tokenizer.vocab_size
        discretized = np.digitize(actions_np, bins).astype(np.int64)
        token_ids = (vocab_size - discretized).astype(np.int64)  # (B, total_steps, action_dim)

        # Reshape to (B, traj_len, action_chunks_len * action_dim)
        token_ids = token_ids.reshape(bsz, traj_len, action_chunks_len * action_dim)

        # Align to response_len (pad or truncate if needed).
        if token_ids.shape[2] < response_len:
            pad = np.full((bsz, traj_len, response_len - token_ids.shape[2]), fill_value=self.tokenizer.pad_token_id, dtype=np.int64)
            token_ids = np.concatenate([token_ids, pad], axis=2)
        elif token_ids.shape[2] > response_len:
            token_ids = token_ids[:, :, :response_len]

        # Build token-level mask based on action_len.
        valid_steps = torch.clamp(action_len, min=0, max=total_steps).to(torch.long)
        valid_tokens = valid_steps * action_dim
        total_tokens = traj_len * response_len
        steps = torch.arange(total_tokens, device=valid_tokens.device).unsqueeze(0).expand(bsz, -1)
        mask = steps < valid_tokens.unsqueeze(1)
        mask = mask.reshape(bsz, traj_len, response_len)

        return torch.from_numpy(token_ids), mask

    def _slice_gt_actions_to_chunk(self, gt_actions: torch.Tensor, max_chunk_len: int = 8) -> torch.Tensor:
        """
        Slice GT actions to match VLA model's action chunk length.
        
        Args:
            gt_actions: (B, T, action_dim) full trajectory actions
            max_chunk_len: maximum chunk length (default 8 for flow-matching)
            
        Returns:
            sliced_actions: (B, max_chunk_len, action_dim) first chunk of actions
        """
        original_shape = gt_actions.shape
        if gt_actions.shape[1] <= max_chunk_len:
            return gt_actions
        
        sliced = gt_actions[:, :max_chunk_len, :]
        print(f"[CosmosRFT] Sliced GT actions from {original_shape} to {sliced.shape}")
        return sliced

    def _create_dataloader(self):
        task_name = self.config.data.task_name
        video_root = self.config.data.get('video_root', None)
        num_workers = int(self.config.data.get('num_workers', 0))
        val_num_workers = int(self.config.data.get('val_num_workers', num_workers))
        rollout_num_workers = int(self.config.data.get('rollout_num_workers', num_workers))
        prefetch_factor = int(self.config.data.get('prefetch_factor', 2))
        pin_memory = bool(self.config.data.get('pin_memory', True))

        def _loader_kwargs(nw: int):
            kwargs = {
                'num_workers': nw,
                'pin_memory': pin_memory,
                'persistent_workers': nw > 0,
            }
            if nw > 0:
                kwargs['prefetch_factor'] = prefetch_factor
            return kwargs

        train_loader_kwargs = _loader_kwargs(num_workers)
        val_loader_kwargs = _loader_kwargs(val_num_workers)
        rollout_loader_kwargs = _loader_kwargs(rollout_num_workers)

        if video_root:
            self.train_dataset = MimicGenVideoDataset(video_root, split="train")
            self.train_dataloader = BufferedDataLoader(DataLoader(
                dataset=self.train_dataset,
                batch_size=int(
                    self.config.data.train_batch_size * self.config.data.get('oversample_factor', 1.0)
                ),
                shuffle=True,
                drop_last=True,
                collate_fn=mimicgen_collate_fn,
                **train_loader_kwargs,
            ))

            val_batch_size = self.config.data.get('val_batch_size', self.config.data.train_batch_size)
            self.val_dataset = MimicGenVideoDataset(video_root, split="train")
            self.val_dataloader = DataLoader(
                dataset=self.val_dataset,
                batch_size=val_batch_size,
                collate_fn=mimicgen_collate_fn,
                **val_loader_kwargs,
            )

            rollout_batch_size = self.config.data.get('rollout_batch_size', val_batch_size)
            self.rollout_dataset = MimicGenVideoDataset(video_root, split="train")
            self.rollout_dataloader = DataLoader(
                dataset=self.rollout_dataset,
                batch_size=rollout_batch_size,
                shuffle=True,
                collate_fn=mimicgen_collate_fn,
                **rollout_loader_kwargs,
            )
            print(f'[CosmosRFT] task={task_name} video_root={video_root}')
        else:
            state_path = self.config.data.get(
                'state_path',
                f'./verl/utils/dataset/{task_name}_d0_states.pkl',
            )

            self.train_dataset = StateDataset(state_path)
            self.train_dataloader = BufferedDataLoader(DataLoader(
                dataset=self.train_dataset,
                batch_size=int(
                    self.config.data.train_batch_size * self.config.data.get('oversample_factor', 1.0)
                ),
                shuffle=True,
                drop_last=True,
                collate_fn=collate_fn,
                **train_loader_kwargs,
            ))

            val_batch_size = self.config.data.get('val_batch_size', self.config.data.train_batch_size)
            self.val_dataset = StateDataset(state_path)
            self.val_dataloader = DataLoader(
                dataset=self.val_dataset,
                batch_size=val_batch_size,
                collate_fn=collate_fn,
                **val_loader_kwargs,
            )

            rollout_batch_size = self.config.data.get('rollout_batch_size', val_batch_size)
            self.rollout_dataset = StateDataset(state_path)
            self.rollout_dataloader = DataLoader(
                dataset=self.rollout_dataset,
                batch_size=rollout_batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                **rollout_loader_kwargs,
            )

            print(f'[CosmosRFT] task={task_name} state_path={state_path}')

        print(f'[CosmosRFT] train_dataset={len(self.train_dataset)} '
              f'val_dataset={len(self.val_dataset)} '
              f'rollout_dataset={len(self.rollout_dataset)}')

    def init_workers(self):
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {
            pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()
        }

        resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        actor_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRollout],
            config=self.config.actor_rollout_ref,
            role='actor_rollout',
        )
        self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls

        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic],
                config=self.config.critic,
            )
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role='ref',
            )
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_cls

        if Role.WorldModelRollout in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.WorldModelRollout)
            wm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.WorldModelRollout],
                config=self.config.actor_rollout_ref,
                role='wm_rollout',
            )
            self.resource_pool_to_cls[resource_pool]['world_model_rollout'] = wm_cls

        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

        # Optional Cosmos world model worker (local, no VideoMAE).
        if Role.WorldModelRollout in self.role_worker_mapping:
            self.world_model_wg = all_wg['world_model_rollout']
            self.world_model_wg.init_model()
        else:
            self.world_model_wg = None

    def _get_dp_world_size(self):
        tp = self.config.actor_rollout_ref.rollout.get('tensor_model_parallel_size', 1)
        return self.actor_rollout_wg.world_size // tp

    def _call_wg(self, wg, method: str, *args, prefixes=None, **kwargs):
        """Call a RayWorkerGroup method with optional prefix fallback."""
        if hasattr(wg, method):
            return getattr(wg, method)(*args, **kwargs)
        # Fallback to prefixed method names produced by colocated workers.
        for prefix in (prefixes or ()):
            prefixed = f"{prefix}_{method}"
            if hasattr(wg, prefixed):
                return getattr(wg, prefixed)(*args, **kwargs)
        raise AttributeError(f"{wg.__class__.__name__} has no method {method} (or prefixed variant)")

    def _generate_actions_in_chunks(self, prompts: DataProto, tag: str):
        """Split one long generate RPC into smaller RPCs and concat outputs."""
        total_size = len(prompts)
        chunk_size_key = f'{tag}_generate_rpc_chunk_size'
        chunk_size = int(self.config.trainer.get('generate_rpc_chunk_size', 0))
        tag_specific = int(self.config.trainer.get(chunk_size_key, 0))
        if tag_specific > 0:
            chunk_size = tag_specific

        dp_world_size = self._get_dp_world_size()
        if chunk_size > 0 and dp_world_size > 1:
            min_per_rank = int(self.config.trainer.get('generate_rpc_min_samples_per_rank', 2))
            min_chunk_size = dp_world_size * max(1, min_per_rank)
            chunk_size = max(chunk_size, min_chunk_size)
            if chunk_size % dp_world_size != 0:
                chunk_size = ((chunk_size + dp_world_size - 1) // dp_world_size) * dp_world_size

        if chunk_size <= 0 or total_size <= chunk_size:
            return self._call_wg(self.actor_rollout_wg, "generate_actions", prompts, prefixes=("actor_rollout",))

        if dp_world_size > 1 and total_size % dp_world_size != 0:
            return self._call_wg(self.actor_rollout_wg, "generate_actions", prompts, prefixes=("actor_rollout",))

        outputs = []
        for start in range(0, total_size, chunk_size):
            end = min(start + chunk_size, total_size)
            chunk = prompts.slice(range(start, end))
            chunk.meta_info = prompts.meta_info
            out = self._call_wg(self.actor_rollout_wg, "generate_actions", chunk, prefixes=("actor_rollout",))
            outputs.append(out)

        return DataProto.concat(outputs)

    def _decode_actions_from_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Decode action tokens into continuous actions."""
        from prismatic.vla.action_tokenizer import ActionTokenizer

        action_token_len = int(self.config.actor_rollout_ref.model.action_token_len)
        action_chunks_len = int(self.config.actor_rollout_ref.model.action_chunks_len)
        bins = int(self.config.data.get('bc_action_bins', 256))
        min_action = float(self.config.data.get('bc_action_min', -1.0))
        max_action = float(self.config.data.get('bc_action_max', 1.0))

        tokenizer = self.tokenizer
        act_tokenizer = ActionTokenizer(tokenizer, bins=bins, min_action=min_action, max_action=max_action)

        bsz, traj_len, resp_len = responses.shape
        flat = responses.reshape(bsz, traj_len * resp_len).detach().cpu().numpy()
        decoded = act_tokenizer.decode_token_ids_to_actions(flat)
        decoded = decoded.reshape(bsz, traj_len * action_chunks_len, action_token_len)
        return torch.from_numpy(decoded)

    def _generate_cosmos_video(self, init_frames: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Generate Cosmos video given init frames and continuous actions."""
        if self.world_model_wg is None:
            raise RuntimeError("WorldModelRollout worker not initialized.")
        if init_frames.dim() == 4:
            init_frames = init_frames.unsqueeze(1)
        wm_input = DataProto.from_dict(
            tensors={
                'pixels': init_frames,
                'predicted_actions': actions,
            }
        )
        wm_output = self._call_wg(self.world_model_wg, "generate_sequences", wm_input, prefixes=("world_model_rollout",))
        return wm_output.batch['predicted_frames']

    def _prepare_vla_input_from_frame_prompt_only(self, frame_batch: torch.Tensor) -> dict:
        """
        Build VLA prompt-only input for Token Generation (no action tokens).
        Used when we want the model to generate action tokens autoregressively.
        """
        import torch.nn.functional as F

        bsz = frame_batch.shape[0]
        device = frame_batch.device

        task_name = str(self.config.data.task_name)
        prompt = f"In: What action should the robot take to {task_name.lower()}?\nOut:"
        tok = self.tokenizer(prompt, return_tensors="pt")
        prompt_ids = tok["input_ids"].to(device)
        prompt_ids = prompt_ids.repeat(bsz, 1)
        prompt_len = prompt_ids.shape[1]

        pad_id = int(self.tokenizer.pad_token_id)
        input_ids = prompt_ids.clone()
        attention_mask = (input_ids != pad_id).to(dtype=torch.long)

        pixels_single = frame_batch.permute(0, 3, 1, 2).float() / 255.0
        pixels_single = F.interpolate(pixels_single, size=(224, 224), mode='bilinear', align_corners=False)
        pixels = torch.cat([pixels_single, pixels_single], dim=1)

        proprio = torch.zeros((bsz, 8), dtype=pixels.dtype, device=device)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixels': pixels,
            'proprio': proprio,
        }

    def _prepare_vla_input_from_frame(self, frame_batch: torch.Tensor, gt_actions_chunk: torch.Tensor) -> dict:
        """
        Build VLA model input from a batch of raw frames and a GT action chunk.

        Args:
            frame_batch: (B, H, W, C) uint8 tensor
            gt_actions_chunk: (B, chunk_len, action_dim) used for building labels/masks
        Returns:
            dict with input_ids, attention_mask, labels, pixels, proprio
        """
        import torch.nn.functional as F

        bsz = frame_batch.shape[0]
        device = frame_batch.device
        action_chunk_len = int(self.config.actor_rollout_ref.model.action_chunks_len)
        action_dim = int(self.config.actor_rollout_ref.model.action_token_len)

        task_name = str(self.config.data.task_name)
        prompt = f"In: What action should the robot take to {task_name.lower()}?\nOut:"
        tok = self.tokenizer(prompt, return_tensors="pt")
        prompt_ids = tok["input_ids"].to(device)
        prompt_ids = prompt_ids.repeat(bsz, 1)
        prompt_len = prompt_ids.shape[1]

        sliced = gt_actions_chunk[:, :action_chunk_len, :]
        actual_len = min(gt_actions_chunk.shape[1], action_chunk_len)
        if sliced.shape[1] < action_chunk_len:
            pad_actions = torch.zeros(
                bsz, action_chunk_len - sliced.shape[1], sliced.shape[2],
                dtype=sliced.dtype, device=device,
            )
            sliced = torch.cat([sliced, pad_actions], dim=1)

        act_len_tensor = torch.full((bsz,), actual_len, dtype=torch.long, device=device)
        action_tokens, _ = self._build_action_tokens_from_gt(sliced, act_len_tensor)
        action_tokens = action_tokens.to(device)

        pad_id = int(self.tokenizer.pad_token_id)
        total_len = prompt_len + action_tokens.shape[1]
        input_ids = torch.full((bsz, total_len), pad_id, dtype=prompt_ids.dtype, device=device)
        input_ids[:, :prompt_len] = prompt_ids
        input_ids[:, prompt_len:] = action_tokens
        attention_mask = (input_ids != pad_id).to(dtype=torch.long)

        IGNORE_INDEX = -100
        labels = input_ids.clone()
        labels[:, :prompt_len] = IGNORE_INDEX

        pixels_single = frame_batch.permute(0, 3, 1, 2).float() / 255.0
        pixels_single = F.interpolate(pixels_single, size=(224, 224), mode='bilinear', align_corners=False)
        pixels = torch.cat([pixels_single, pixels_single], dim=1)

        proprio = torch.zeros((bsz, 8), dtype=pixels.dtype, device=device)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'pixels': pixels,
            'proprio': proprio,
            'gt_actions': sliced,
        }

    def _generate_full_gt_video(self, init_frames: torch.Tensor, gt_actions: torch.Tensor) -> torch.Tensor:
        """
        Generate full GT video using Cosmos with all GT actions.
        The Cosmos worker handles chunking (12-step chunks) internally.

        Args:
            init_frames: (B, H, W, C) uint8 tensor
            gt_actions: (B, T, action_dim) full GT actions
        Returns:
            gt_video: (B, T_frames, H, W, C) uint8 tensor
        """
        return self._generate_cosmos_video(init_frames, gt_actions)

    def _generate_full_predicted_video(
        self,
        init_frames: torch.Tensor,
        gt_actions: torch.Tensor,
        first_pred_actions: torch.Tensor = None,
    ) -> tuple:
        """
        Generate full predicted video by looping VLA -> Cosmos.

        For the first chunk, reuse first_pred_actions (already computed in fit()).
        For subsequent chunks, call VLA to predict actions from the latest frame.

        Args:
            init_frames: (B, H, W, C) uint8 tensor
            gt_actions: (B, T, action_dim) determines number of loop iterations
            first_pred_actions: (B, vla_chunk, action_dim) from the first VLA call
        Returns:
            (full_video, all_predicted_actions) tuple
        """
        vla_chunk_size = int(self.config.actor_rollout_ref.model.action_chunks_len)
        T = gt_actions.shape[1]
        num_vla_chunks = (T + vla_chunk_size - 1) // vla_chunk_size

        all_videos = []
        all_actions = []
        current_frames = init_frames.clone()

        for chunk_idx in range(num_vla_chunks):
            print(f"[CosmosRFT]   Video chunk {chunk_idx + 1}/{num_vla_chunks} (VLA+Cosmos)...", flush=True)
            gt_start = chunk_idx * vla_chunk_size
            gt_end = min(gt_start + vla_chunk_size, T)
            valid_len = gt_end - gt_start
            gt_chunk = gt_actions[:, gt_start:gt_end, :]

            if chunk_idx == 0 and first_pred_actions is not None:
                pred_actions = first_pred_actions
            else:
                vla_input = self._prepare_vla_input_from_frame_prompt_only(current_frames)
                gen_batch = DataProto.from_single_dict(vla_input)
                gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'use_wm': True,
                    'return_rollouts': True,
                }
                gen_output = self._generate_actions_in_chunks(gen_batch, tag='train')
                pred_actions = gen_output.batch['predicted_actions']

            chunk_actions = pred_actions[:, :valid_len, :]
            video_chunk = self._generate_cosmos_video(current_frames, chunk_actions)

            if chunk_idx == 0:
                all_videos.append(video_chunk)
            else:
                all_videos.append(video_chunk[:, 1:])

            all_actions.append(pred_actions[:, :valid_len])
            current_frames = video_chunk[:, -1].clone()

        full_video = torch.cat(all_videos, dim=1)
        all_pred_actions = torch.cat(all_actions, dim=1)
        return full_video, all_pred_actions

    def _ensure_response_placeholders(self, batch: DataProto) -> None:
        """Ensure responses/finish_step exist for reward/advantage computation."""
        if 'responses' in batch.batch:
            return
        if 'predicted_actions' not in batch.batch:
            return
        actions = batch.batch['predicted_actions']
        bsz, t, act_dim = actions.shape
        device = actions.device
        resp = torch.zeros((bsz, t * act_dim), dtype=torch.long, device=device)
        batch.batch['responses'] = resp
        batch.batch['finish_step'] = torch.full((bsz,), t, dtype=torch.long, device=device)

    def _build_action_tokens_from_gt(self, gt_actions: torch.Tensor, action_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Discretize GT actions into token ids for flow-matching masks."""
        from prismatic.vla.action_tokenizer import ActionTokenizer
        bins = int(self.config.data.get('bc_action_bins', 256))
        min_action = float(self.config.data.get('bc_action_min', -1.0))
        max_action = float(self.config.data.get('bc_action_max', 1.0))
        act_tokenizer = ActionTokenizer(self.tokenizer, bins=bins, min_action=min_action, max_action=max_action)

        actions_np = gt_actions.detach().cpu().numpy()
        bsz, max_act, action_dim = actions_np.shape
        discretized = np.digitize(actions_np, act_tokenizer.bins).astype(np.int64)
        token_ids = (act_tokenizer.tokenizer.vocab_size - discretized).astype(np.int64)  # (B, T, action_dim)
        token_ids = token_ids.reshape(bsz, max_act * action_dim)

        valid_tokens = (action_len.detach().cpu().numpy() * action_dim).astype(np.int64)
        pad_id = int(self.tokenizer.pad_token_id)
        for i in range(bsz):
            token_ids[i, valid_tokens[i]:] = pad_id

        return torch.from_numpy(token_ids), torch.from_numpy(valid_tokens)

    def _require_gt_checks(self, batch: DataProto, tag: str) -> None:
        """Fail fast if GT actions/frames required for recon reward are missing."""
        if 'gt_actions' not in batch.batch or 'action_len' not in batch.batch:
            raise RuntimeError(
                f"[CosmosRFT][{tag}] Missing gt_actions/action_len in batch; cannot compute recon reward."
            )
        action_len = batch.batch['action_len']
        if (action_len <= 0).any():
            raise RuntimeError(
                f"[CosmosRFT][{tag}] Found zero-length gt_actions; cannot compute recon reward."
            )
        if 'gt_frames' not in batch.batch or 'video' not in batch.batch:
            raise RuntimeError(
                f"[CosmosRFT][{tag}] Missing gt_frames/video in batch; recon reward will be zero."
            )

    def _validate(self, global_steps=0):
        reward_tensor_lst = []
        data_source_lst = []
        metric_dict = {}

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            if 'init_frames' in test_batch.batch:
                vla_input = self._prepare_vla_input_from_frame_prompt_only(test_batch.batch['init_frames'])
                test_batch.batch['input_ids'] = vla_input['input_ids']
                test_batch.batch['attention_mask'] = vla_input['attention_mask']
                test_batch.batch['pixels'] = vla_input['pixels']
                test_batch.batch['proprio'] = vla_input['proprio']
                test_batch.batch['labels'] = vla_input['input_ids'].clone()
            test_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'return_rollouts': False,
                'validate': True,
                'use_wm': True,
                'global_steps': global_steps,
            }
            if 'gt_frames' in test_batch.batch:
                test_batch.meta_info['return_rollouts'] = True
            noise_batch = self._call_wg(
                self.actor_rollout_wg, "sample_noisy_actions", test_batch, prefixes=("actor_rollout",)
            )
            if 'noise' in noise_batch.batch:
                noise_partial = noise_batch.pop(batch_keys=['noise'])
                test_batch = test_batch.union(noise_partial)

            test_output = self._generate_actions_in_chunks(test_batch, tag='validate')
            test_batch = test_batch.union(test_output)
            self._ensure_response_placeholders(test_batch)
            test_batch.meta_info['global_steps'] = global_steps

            if self.world_model_wg is not None and 'init_frames' in test_batch.batch:
                if 'predicted_actions' in test_batch.batch:
                    pred_actions = test_batch.batch['predicted_actions'].to(
                        test_batch.batch['predicted_actions'].device
                    )
                else:
                    responses = test_batch.batch['responses']
                    pred_actions = self._decode_actions_from_responses(responses).to(
                        test_batch.batch['responses'].device
                    )
                test_batch.batch['action'] = pred_actions
                init_frames = test_batch.batch['init_frames']
                pred_video = self._generate_cosmos_video(init_frames, pred_actions)
                test_batch.batch['video'] = pred_video
                if 'gt_actions' in test_batch.batch:
                    # Use sliced GT actions (8 steps) for fair comparison
                    action_chunk_len = int(self.config.actor_rollout_ref.model.action_chunks_len)
                    gt_actions_sliced = self._slice_gt_actions_to_chunk(
                        test_batch.batch['gt_actions'], max_chunk_len=action_chunk_len
                    )
                    gt_video = self._generate_cosmos_video(init_frames, gt_actions_sliced)
                    test_batch.batch['gt_frames'] = gt_video
                    if 'gt_lengths' not in test_batch.batch:
                        test_batch.batch['gt_lengths'] = torch.full(
                            (gt_actions_sliced.shape[0],),
                            gt_actions_sliced.shape[1] + 1,
                            dtype=torch.long,
                            device=gt_actions_sliced.device,
                        )
            self._require_gt_checks(test_batch, tag='validate')

            verifier_score, reward_metrics, format_metrics, _ = self.val_reward_fn.verify(test_batch)
            reward_tensor = torch.tensor(verifier_score, dtype=torch.float32).unsqueeze(-1)

            for k, v in reward_metrics.items():
                metric_dict[f'test_reward/{k}'] = v
            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append([self.config.data.task_name] * reward_tensor.shape[0])

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()
        data_sources = np.concatenate(data_source_lst, axis=0)

        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            ds = data_sources[i]
            data_source_reward.setdefault(ds, []).append(reward_tensor[i].item())

        result = {}
        for ds, rewards in data_source_reward.items():
            result[f'test_score/{ds}'] = np.mean(rewards)
        result['test_score/all'] = reward_tensor.mean().item()
        return result

    def filter(self, reward_tensor, batch, n_samples):
        if self.config.data.get('filter_accuracy', False):
            reward_matrix = reward_tensor.sum(-1).reshape(-1, n_samples)
            acc_tensor = torch.mean(reward_matrix, dim=-1)

            lower = self.config.data.get('accuracy_lower_bound', 0.0)
            upper = self.config.data.get('accuracy_upper_bound', 1.0)
            acc_mask = (acc_tensor >= lower) & (acc_tensor <= upper)
        else:
            acc_mask = torch.ones(len(batch) // n_samples, dtype=torch.bool, device=reward_tensor.device)

        final_mask = acc_mask.repeat_interleave(n_samples)
        if final_mask.sum() == 0:
            return batch
        return batch.slice(final_mask)

    def add_to_buffer(self, batch, batch_size, n_samples):
        buffer_length = len(batch) // n_samples - batch_size
        buffer_mask = torch.ones(buffer_length + batch_size, dtype=torch.bool)
        buffer_mask[batch_size:] = False
        buffer_mask = buffer_mask.repeat_interleave(n_samples)
        return batch.slice(buffer_mask)

    def fit(self):
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        global_steps = 0
        batch_size = self.config.data.train_batch_size
        n_samples = self.config.data.n_samples
        train_mode = str(self.config.trainer.get('train_mode', 'rl')).lower()

        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', False):
            val_metrics = self._validate(global_steps=global_steps)
            val_metrics = {f'val/{k}': v for k, v in val_metrics.items()}
            print(f'[CosmosRFT] Initial validation: {val_metrics}')
            logger.log(data=val_metrics, step=global_steps)
            if self.config.trainer.get('val_only', False):
                return

        self.train_dataloader.start_new_epoch()

        for epoch in range(self.config.trainer.total_epochs):
            valid_batch = []
            metrics = defaultdict(list)
            metrics['timing/gen'] = 0
            metrics['timing/verify'] = 0
            debug_logged = False
            rollout_batch_idx = 0

            print(f"[CosmosRFT] ========== Epoch {epoch + 1}/{self.config.trainer.total_epochs} ==========", flush=True)

            while len(valid_batch) < batch_size * n_samples:
                rollout_batch_idx += 1
                print(f"[CosmosRFT] Rollout batch {rollout_batch_idx} | 1/5 VLA action generation...", flush=True)

                with Timer(name='gen', text='{name}: {seconds:.1f}s') as timer:
                    newbatch = self.train_dataloader.get_next_batch()
                    newbatch = DataProto.from_single_dict(newbatch)
                    max_act = self.config.data.get('max_action_len', None)
                    if max_act is not None and max_act > 0 and 'gt_actions' in newbatch.batch:
                        ga = newbatch.batch['gt_actions']
                        newbatch.batch['gt_actions'] = ga[:, :max_act]
                        if 'action_len' in newbatch.batch:
                            newbatch.batch['action_len'] = torch.clamp(
                                newbatch.batch['action_len'], max=max_act
                            )
                        if 'gt_lengths' in newbatch.batch:
                            newbatch.batch['gt_lengths'] = torch.clamp(
                                newbatch.batch['gt_lengths'], max=max_act + 1
                            )
                    if 'init_frames' in newbatch.batch:
                        init_frames = newbatch.batch['init_frames']
                        gt_actions_full = newbatch.batch['gt_actions']
                        vla_input = self._prepare_vla_input_from_frame_prompt_only(init_frames)
                        vla_input['gt_actions'] = gt_actions_full
                        vla_input['init_frames'] = init_frames
                        gen_batch = DataProto.from_single_dict(vla_input)
                        gen_batch = gen_batch.repeat(repeat_times=n_samples, interleave=True)
                    else:
                        gen_batch = newbatch.select(batch_keys=['states', 'state_id'], meta_info_keys={})

                    newbatch.non_tensor_batch['uid'] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(newbatch.batch))],
                        dtype=object,
                    )

                    batch_lst = sum(
                        [[newbatch[i:i + 1] for _ in range(n_samples)] for i in range(len(newbatch))],
                        [],
                    )

                    gen_batch.meta_info = {
                        'eos_token_id': self.tokenizer.eos_token_id,
                        'n_samples': n_samples,
                        'pad_token_id': self.tokenizer.pad_token_id,
                        'use_wm': True,
                        'return_rollouts': True,
                        'global_steps': global_steps,
                    }
                    gen_batch_output = self._generate_actions_in_chunks(gen_batch, tag='train')
                    if 'old_log_probs' not in gen_batch_output.batch:
                        log_prob = self._call_wg(
                            self.actor_rollout_wg, "compute_log_prob", gen_batch_output, prefixes=("actor_rollout",)
                        )
                        gen_batch_output = gen_batch_output.union(log_prob)
                    roll_batch = DataProto.concat(batch_lst)
                    roll_batch = roll_batch.union(gen_batch_output)
                    self._ensure_response_placeholders(roll_batch)
                    roll_batch.meta_info['global_steps'] = global_steps

                # Generate full videos: VLA->Cosmos loop for predicted, Cosmos-only for GT.
                if self.world_model_wg is not None and 'init_frames' in roll_batch.batch:
                    try:
                        print(f"[CosmosRFT] Rollout batch {rollout_batch_idx} | 2/5 Cosmos video generation (predicted + GT)...", flush=True)
                        init_frames_rb = roll_batch.batch['init_frames']
                        gt_actions_rb = roll_batch.batch['gt_actions']
                        first_pred_actions = roll_batch.batch.get('predicted_actions', None)
                        full_pred_video, all_pred_actions = self._generate_full_predicted_video(
                            init_frames_rb, gt_actions_rb, first_pred_actions
                        )
                        print(f"[CosmosRFT] Rollout batch {rollout_batch_idx} | 2b/5 Cosmos GT video...", flush=True)
                        full_gt_video = self._generate_full_gt_video(init_frames_rb, gt_actions_rb)

                        min_frames = min(full_pred_video.shape[1], full_gt_video.shape[1])
                        roll_batch.batch['video'] = full_pred_video[:, :min_frames]
                        roll_batch.batch['gt_frames'] = full_gt_video[:, :min_frames]
                        roll_batch.batch['action'] = all_pred_actions

                        if 'action_len' in roll_batch.batch:
                            gt_lengths = roll_batch.batch['action_len'].to(torch.long) + 1
                            gt_lengths = torch.clamp(gt_lengths, max=min_frames)
                        else:
                            gt_lengths = torch.full(
                                (init_frames_rb.shape[0],), min_frames,
                                dtype=torch.long, device=init_frames_rb.device,
                            )
                        roll_batch.batch['gt_lengths'] = gt_lengths
                        print(f"[CosmosRFT] Rollout batch {rollout_batch_idx} | 2/5 done: pred={tuple(full_pred_video.shape)} "
                              f"gt={tuple(full_gt_video.shape)} aligned_frames={min_frames}", flush=True)
                    except Exception as e:
                        raise RuntimeError(f"[CosmosRFT] failed to generate full videos: {e}")

                metrics['timing/gen'] += timer.last
                # Debug: log data/video immediately after rollout (before any drops).
                if not debug_logged and 'gt_actions' in roll_batch.batch:
                    try:
                        act = roll_batch.batch['gt_actions']
                        act_len = roll_batch.batch.get('action_len')
                        gt_len = roll_batch.batch.get('gt_lengths')
                        msg = [
                            f"[DEBUG][Data] gt_actions shape={tuple(act.shape)}",
                            f"gt_actions min={float(act.min().item()):.4f} max={float(act.max().item()):.4f}",
                        ]
                        if act_len is not None:
                            msg.append(f"action_len min={int(act_len.min().item())} max={int(act_len.max().item())}")
                        if gt_len is not None:
                            msg.append(f"gt_lengths min={int(gt_len.min().item())} max={int(gt_len.max().item())}")
                        line = " | ".join(msg)
                        print(line)
                        _debug_log(line)
                    except Exception as e:
                        line = f"[DEBUG][Data] failed to log gt_actions stats: {e}"
                        print(line)
                        _debug_log(line)
                    # Keys/video right after rollout
                    try:
                        keys = list(roll_batch.batch.keys())
                        line = f"[DEBUG][Keys] keys={keys}"
                        print(line)
                        _debug_log(line)
                        has_video = 'video' in roll_batch.batch
                        has_gt = 'gt_frames' in roll_batch.batch
                        line = f"[DEBUG][Keys] has_video={has_video} has_gt_frames={has_gt}"
                        print(line)
                        _debug_log(line)
                    except Exception as e:
                        line = f"[DEBUG][Keys] failed to log keys: {e}"
                        print(line)
                        _debug_log(line)

                    if 'video' in roll_batch.batch:
                        try:
                            vid = roll_batch.batch['video']
                            msg = [
                                f"[DEBUG][Video] video shape={tuple(vid.shape)}",
                                f"video min={int(vid.min().item())} max={int(vid.max().item())}",
                                f"video mean={float(vid.float().mean().item()):.4f}",
                            ]
                            if 'gt_frames' in roll_batch.batch:
                                gt = roll_batch.batch['gt_frames']
                                msg.append(f"gt_frames shape={tuple(gt.shape)}")
                                msg.append(f"gt_frames min={int(gt.min().item())} max={int(gt.max().item())}")
                                msg.append(f"gt_frames mean={float(gt.float().mean().item()):.4f}")
                            line = " | ".join(msg)
                            print(line)
                            _debug_log(line)
                        except Exception as e:
                            line = f"[DEBUG][Video] failed to log video stats: {e}"
                            print(line)
                            _debug_log(line)
                    if 'video' in roll_batch.batch and 'gt_frames' in roll_batch.batch:
                        try:
                            vid = roll_batch.batch['video'].float()
                            gt = roll_batch.batch['gt_frames'].float()
                            t = min(vid.shape[1], gt.shape[1])
                            vid = vid[:, :t]
                            gt = gt[:, :t]
                            diff = (vid - gt).abs()
                            line = (
                                f"[DEBUG][Diff] abs diff mean={float(diff.mean().item()):.6f} "
                                f"max={float(diff.max().item()):.6f}"
                            )
                            print(line)
                            _debug_log(line)
                        except Exception as e:
                            line = f"[DEBUG][Diff] failed to compute diff: {e}"
                            print(line)
                            _debug_log(line)
                    if 'action' in roll_batch.batch and 'gt_actions' in roll_batch.batch:
                        try:
                            act = roll_batch.batch['action'].float()
                            gt_act = roll_batch.batch['gt_actions'].float()
                            t = min(act.shape[1], gt_act.shape[1])
                            act = act[:, :t]
                            gt_act = gt_act[:, :t]
                            diff = (act - gt_act).abs()
                            line = (
                                f"[DEBUG][ActionDiff] mean={float(diff.mean().item()):.6f} "
                                f"max={float(diff.max().item()):.6f}"
                            )
                            print(line)
                            _debug_log(line)
                        except Exception as e:
                            line = f"[DEBUG][ActionDiff] failed: {e}"
                            print(line)
                            _debug_log(line)

                    debug_logged = True

                if train_mode != 'bc':
                    print(f"[CosmosRFT] Rollout batch {rollout_batch_idx} | 3/5 Computing reward...", flush=True)
                    with Timer(name='verify', text='{name}: {seconds:.1f}s') as timer:
                        self._require_gt_checks(roll_batch, tag='train')
                        scores_tensor, reward_metrics, format_metrics, _ = self.reward_fn.verify(roll_batch)
                        for k, v in reward_metrics.items():
                            metrics[f'train_verify_score/{k}'].append(v)
                        for k, v in format_metrics.items():
                            metrics[f'format_score/{k}'].append(v)
                    metrics['timing/verify'] += timer.last
                else:
                    metrics['timing/verify'] += 0

                if train_mode != 'bc' and self.config.data.get('filter_accuracy', False):
                    print(f'[CosmosRFT] before filtering: {len(roll_batch)}')
                    filtered = self.filter(roll_batch.batch['acc'].unsqueeze(1), roll_batch, n_samples)
                    print(f'[CosmosRFT] after filtering: {len(filtered)}')
                else:
                    filtered = roll_batch

                if len(valid_batch) == 0:
                    valid_batch = filtered
                else:
                    valid_batch = DataProto.concat([valid_batch, filtered])
                print(f'[CosmosRFT] collected {len(valid_batch)} / {batch_size * n_samples}')

            if len(valid_batch) > batch_size * n_samples:
                valid_batch = self.add_to_buffer(valid_batch, batch_size, n_samples)

            for k in list(metrics.keys()):
                if isinstance(metrics[k], list) and metrics[k]:
                    metrics[k] = np.mean(metrics[k])

            batch = valid_batch

            print(f"[CosmosRFT] Epoch {epoch + 1} | Step {global_steps} | 4/5 Advantage computation...", flush=True)
            if train_mode != 'bc':
                if self.use_reference_policy:
                    with Timer(name='ref', text='{name}: {seconds:.1f}s') as timer:
                        ref_log_prob = self._call_wg(self.ref_policy_wg, "compute_ref_log_prob", batch, prefixes=("ref",))
                        batch = batch.union(ref_log_prob)
                    metrics['timing/ref'] = timer.last

                with Timer(name='adv', text='{name}: {seconds:.1f}s') as timer:
                    print(f"[CosmosRFT] reward_fn (pred vs GT)...", flush=True)
                    reward_tensor_dict, reward_metrics = self.reward_fn(batch)
                    token_level_scores = reward_tensor_dict['all']
                    olp = batch.batch['old_log_probs']
                    # Align token_level_scores and responses with old_log_probs (apply_kl_penalty requires matching shapes)
                    if token_level_scores.shape != olp.shape:
                        total_reward = token_level_scores.sum()
                        n = olp.numel()
                        token_level_scores = (total_reward / max(1, n)) * torch.ones_like(olp, dtype=token_level_scores.dtype, device=olp.device)
                        # responses must be 2D (batch, response_length) for apply_kl_penalty
                        resp = torch.zeros(olp.shape, dtype=torch.long, device=olp.device)
                        if resp.dim() == 1:
                            resp = resp.unsqueeze(-1)
                        batch.batch['responses'] = resp
                    batch.batch['token_level_scores'] = token_level_scores
                    for k, v in reward_metrics.items():
                        metrics[f'train_reward/{k}'] = v
                    for k, v in reward_tensor_dict.items():
                        batch.batch[k] = v

                    # When use_reference_policy=False (e.g. kl_coef=0), ref_log_prob is never computed.
                    # Use old_log_probs so kld=0 and token_level_rewards = token_level_scores.
                    if 'ref_log_prob' not in batch.batch:
                        batch.batch['ref_log_prob'] = batch.batch['old_log_probs'].clone()

                    batch, kl_metrics = apply_kl_penalty(
                        batch,
                        kl_ctrl=self.kl_ctrl,
                        kl_penalty=self.config.algorithm.kl_penalty,
                    )
                    metrics.update(kl_metrics)

                    batch = compute_advantage(
                        batch,
                        self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                    )
                metrics['timing/adv'] = timer.last
            else:
                metrics['timing/adv'] = 0

                # Build BC targets from GT actions.
                if 'gt_actions' in batch.batch and 'action_len' in batch.batch:
                    bsz, traj_len, response_len = batch.batch['responses'].shape
                    bc_responses, bc_mask = self._build_bc_responses(
                        batch.batch['gt_actions'],
                        batch.batch['action_len'],
                        traj_len=traj_len,
                        response_len=response_len,
                    )
                    batch.batch['responses'] = bc_responses.to(batch.batch['responses'].device)
                    batch.batch['bc_mask'] = bc_mask.to(batch.batch['responses'].device)

                # Log frame-level reconstruction loss (no gradients).
                if 'video' in batch.batch and 'gt_frames' in batch.batch and 'gt_lengths' in batch.batch:
                    print(f"[CosmosRFT] Computing recon reward (MAE/LPIPS)...", flush=True)
                    device = batch.batch['video'].device
                    resize_hw = tuple(self.config.data.get('reward_resize_hw', [224, 224]))
                    loss_weight = {
                        'mae': float(self.config.data.get('reward_mae_weight', 1.0)),
                        'lpips': float(self.config.data.get('reward_lpips_weight', 1.0)),
                    }
                    lpips_every = int(self.config.data.get('lpips_every', 1))
                    compute_lpips = (global_steps % lpips_every) == 0
                    lpips_fn = None
                    if compute_lpips and loss_weight.get('lpips', 0) > 0:
                        from ivideogpt.lpips import LPIPS
                        lpips_fn = LPIPS(net="vgg").to(device)
                    _, recon_metrics = compute_trajectory_recon_reward(
                        predicted_frames=batch.batch['video'].to(device),
                        gt_frames=batch.batch['gt_frames'].to(device),
                        gt_lengths=batch.batch['gt_lengths'].to(device),
                        lpips_fn=lpips_fn,
                        loss_weight=loss_weight,
                        resize_hw=resize_hw,
                        compute_lpips=compute_lpips,
                    )
                    for k, v in recon_metrics.items():
                        metrics[f'train_bc_frame/{k}'] = v
                    print(f"[CosmosRFT] Recon reward done.", flush=True)

            if self.config.trainer.get('critic_warmup', 0) <= global_steps:
                print(f"[CosmosRFT] Epoch {epoch + 1} | Step {global_steps} | 5/5 Actor update (loss)...", flush=True)
                with Timer(name='update_actor', text='{name}: {seconds:.1f}s') as timer:
                    batch.meta_info['is_filtered'] = True
                    batch.meta_info['train_mode'] = False
                    # Drop large tensors before actor update to avoid GPU OOM.
                    for drop_key in ('video', 'gt_frames', 'init_frames'):
                        if drop_key in batch.batch:
                            batch.batch.pop(drop_key)
                    if train_mode == 'bc':
                        actor_output = self._call_wg(self.actor_rollout_wg, "update_actor_bc", batch, prefixes=("actor_rollout",))
                        entropy_output = None
                    else:
                        actor_output = self._call_wg(self.actor_rollout_wg, "update_actor", batch, prefixes=("actor_rollout",))
                        try:
                            entropy_output = self._call_wg(self.actor_rollout_wg, "compute_entropy", data=batch, prefixes=("actor_rollout",))
                        except AttributeError:
                            entropy_output = None
                metrics['timing/update_actor'] = timer.last
                metrics.update(reduce_metrics(actor_output.meta_info['metrics']))
                if entropy_output is not None:
                    metrics.update(reduce_metrics(entropy_output.meta_info['metrics']))
                print(f"[CosmosRFT] Epoch {epoch + 1} | Step {global_steps} | 5/5 Actor update done.", flush=True)

            # Post-update debug removed; roll_batch debug above is authoritative.

            if (
                self.val_reward_fn is not None
                and self.config.trainer.get('test_freq', -1) > 0
                and (global_steps + 1) % self.config.trainer.test_freq == 0
            ):
                with Timer(name='testing', text='{name}: {seconds:.1f}s') as timer:
                    val_metrics = self._validate(global_steps=global_steps + 1)
                    val_metrics = {f'val/{k}': v for k, v in val_metrics.items()}
                metrics['timing/testing'] = timer.last
                metrics.update(val_metrics)

            data_metrics = compute_data_metrics(batch=batch, use_critic=False)
            metrics.update(data_metrics)
            logger.log(data=_filter_metrics_for_logging(metrics), step=global_steps)

            if self.config.trainer.save_freq > 0 and (global_steps + 1) % self.config.trainer.save_freq == 0:
                actor_local_path = os.path.join(
                    self.config.trainer.default_local_dir, 'actor', f'global_step_{global_steps}'
                )
                self._call_wg(self.actor_rollout_wg, "save_checkpoint", actor_local_path, None, prefixes=("actor_rollout",))

            global_steps += 1

        if self.val_reward_fn is not None:
            val_metrics = self._validate(global_steps=global_steps)
            pprint(f'[CosmosRFT] Final validation: {val_metrics}')
            logger.log(data=val_metrics, step=global_steps)
