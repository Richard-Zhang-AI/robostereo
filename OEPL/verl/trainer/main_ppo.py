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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import json
import os
import statistics
from functools import partial

from verl import DataProto
from omegaconf import OmegaConf

import torch
from verl.utils.reward_score import gsm8k, math, countdown, multiply, logic
from verl.trainer.ppo.ray_trainer import RayTrainer


def _require_positive_int(config, path: str) -> int:
    value = OmegaConf.select(config, path)
    if value is None:
        raise ValueError(f"[CFG] Missing required config: {path}")
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"[CFG] Config `{path}` must be an integer, got: {value!r}") from exc
    if value <= 0:
        raise ValueError(f"[CFG] Config `{path}` must be > 0, got: {value}")
    return value


def _optional_int(config, path: str, default=None):
    value = OmegaConf.select(config, path)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"[CFG] Config `{path}` must be an integer, got: {value!r}") from exc


def _validate_action_alignment_config(config) -> None:
    """Fail fast on invalid action/chunk configuration before Ray workers start."""
    model_chunk = _require_positive_int(config, "actor_rollout_ref.model.action_chunks_len")
    model_token = _require_positive_int(config, "actor_rollout_ref.model.action_token_len")

    actor_chunk = _optional_int(config, "actor_rollout_ref.actor.action_chunks_len", default=model_chunk)
    rollout_chunk = _optional_int(config, "actor_rollout_ref.rollout.action_chunks_len", default=model_chunk)
    actor_token = _optional_int(config, "actor_rollout_ref.actor.action_token_len", default=model_token)
    rollout_token = _optional_int(config, "actor_rollout_ref.rollout.action_token_len", default=model_token)

    if actor_chunk <= 0 or rollout_chunk <= 0:
        raise ValueError(
            "[CFG] action_chunks_len must be > 0 for actor/rollout. "
            f"actor={actor_chunk}, rollout={rollout_chunk}"
        )

    mismatches = []
    if actor_chunk != model_chunk:
        mismatches.append(f"actor.action_chunks_len={actor_chunk} != model.action_chunks_len={model_chunk}")
    if rollout_chunk != model_chunk:
        mismatches.append(f"rollout.action_chunks_len={rollout_chunk} != model.action_chunks_len={model_chunk}")
    if actor_token != model_token:
        mismatches.append(f"actor.action_token_len={actor_token} != model.action_token_len={model_token}")
    if rollout_token != model_token:
        mismatches.append(f"rollout.action_token_len={rollout_token} != model.action_token_len={model_token}")
    if mismatches:
        raise ValueError("[CFG] Inconsistent actor/model/rollout action config: " + "; ".join(mismatches))

    wm_enable = bool(OmegaConf.select(config, "actor_rollout_ref.wm.enable", default=False))
    if not wm_enable:
        return

    wm_backend = str(OmegaConf.select(config, "actor_rollout_ref.wm.backend", default="opensora")).lower()
    if wm_backend != "cosmos":
        return

    cosmos_chunk = _optional_int(config, "actor_rollout_ref.wm.cosmos_action_chunk_size", default=12)
    if cosmos_chunk is None or cosmos_chunk <= 0:
        raise ValueError(
            "[CFG] actor_rollout_ref.wm.cosmos_action_chunk_size must be a positive integer when WM backend=cosmos. "
            f"Got: {cosmos_chunk!r}"
        )

    align_mode = str(
        OmegaConf.select(config, "actor_rollout_ref.wm.cosmos_action_align_mode", default="interpolate")
    ).lower()
    if align_mode not in {"interpolate"}:
        raise ValueError(
            "[CFG] Unsupported actor->cosmos action alignment mode. "
            f"cosmos_action_align_mode={align_mode!r}, supported={{'interpolate'}}"
        )

    if cosmos_chunk != model_chunk:
        print(
            "[CFG][WM] action chunk mismatch detected; will align actor chunk to cosmos chunk via interpolation: "
            f"actor/model/rollout={model_chunk}, cosmos={cosmos_chunk}"
        )


class RobRewardManager():
    """The reward manager.
    """
    # TODO: we are requiring a reward manager to be much more stronger than this. so this is fully refactored!
    def __init__(self, num_examine,config) -> None:
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.config=config
        self._wm_enable = bool(self.config.actor_rollout_ref.wm.get('enable', False))

    def verify(self, data):
        batch_size = data.batch['responses'].size(0)
        if self._wm_enable and 'success_prob' not in data.batch:
            raise RuntimeError(
                "[Reward] WM is enabled but success_prob is missing in rollout batch. "
                "Refuse to fallback to binary complete reward."
            )
        if 'success_prob' in data.batch:
            # Use continuous success probability directly as reward signal.
            score_tensor = data.batch['success_prob'].detach().to(dtype=torch.float32)
            assert score_tensor.numel() == batch_size
        else:
            completes = data.batch['complete'].tolist()
            assert len(completes) == batch_size
            score_tensor = torch.tensor([float(item) for item in completes],
                                        dtype=torch.float32,
                                        device=data.batch['responses'].device)
        format = [1.0 for _ in range(batch_size)]

        data.batch['acc'] = score_tensor
        data.batch['format_correctness'] = torch.tensor(format, dtype=torch.float32, device=data.batch['responses'].device)
        
        reward_metrics = {}
        format_metrics = {}
        reward_format_metrics = {}
            
        reward_metrics['all'] = score_tensor.mean().item()
        if 'success_prob' in data.batch:
            reward_metrics['success_prob_mean'] = score_tensor.mean().item()
            reward_metrics['success_prob_min'] = score_tensor.min().item()
            reward_metrics['success_prob_max'] = score_tensor.max().item()
        format_metrics['all'] = data.batch['format_correctness'].mean().item()
        reward_format_metrics['all'] = score_tensor.mean().item()

        return score_tensor.detach().cpu().tolist(), reward_metrics, format_metrics, reward_format_metrics

    def __call__(self, data: DataProto):
        
        # aggregate all available reward tensors

        reward_tensor_dict={}
        reward_metrics={}
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32) # batch * 64 * 56
        verifier_reward=torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_tensor = reward_tensor.reshape((reward_tensor.shape[0],-1))
        verifier_reward = verifier_reward.reshape((verifier_reward.shape[0],-1))
        
        valid_response_length = data.batch['finish_step'] * self.config.actor_rollout_ref.model.action_token_len
        max_token_len = verifier_reward.shape[1]
        safe_valid_response_length = torch.clamp(valid_response_length, min=1, max=max_token_len)
        if torch.any(safe_valid_response_length != valid_response_length):
            clipped = int((safe_valid_response_length != valid_response_length).sum().item())
            max_raw = int(valid_response_length.max().item())
            print(
                f"[WARN][Reward] finish_step->token index clipped: {clipped}/{valid_response_length.numel()} "
                f"(max_raw={max_raw}, max_allowed={max_token_len})"
            )
       
        # In WM mode we require continuous success_prob; do not silently fallback to binary complete.
        if self._wm_enable and 'success_prob' not in data.batch:
            raise RuntimeError(
                "[Reward] WM is enabled but success_prob is missing in reward computation. "
                "Refuse to fallback to binary complete reward."
            )
        if 'success_prob' in data.batch:
            verifier_score = data.batch['success_prob'].detach().to(dtype=torch.float32).cpu().numpy().tolist()
        elif 'acc' in data.batch:
            # Reuse verifier scores computed in verify().
            verifier_score = data.batch['acc'].cpu().numpy().tolist()
        else:
            verifier_score, verifier_metrics, format_metrics, reward_format_metrics = self.verify(data)
            reward_metrics.update(verifier_metrics)
        score_tensor = torch.tensor(verifier_score, dtype=verifier_reward.dtype, device=verifier_reward.device)
        row_idx = torch.arange(verifier_reward.shape[0], device=verifier_reward.device)
        col_idx = safe_valid_response_length.to(dtype=torch.long) - 1
        verifier_reward[row_idx, col_idx] += score_tensor
            
        reward_tensor_dict['gt_scores'] = verifier_reward

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        # if 'rm_scores' in data.batch.keys():
        #     raise  ValueError
        #     reward_tensor_dict['rm_scores'] = data.batch['rm_scores']
        #     reward_metrics['reward_model']=data.batch['rm_scores'].sum(dim=1).mean().item()
        #     if self.config.reward_model.rm_coef!=0:
        #         reward_tensor += self.config.reward_model.rm_coef * reward_tensor_dict['rm_scores']

        # Use continuous success probability as reward when available: reward = 5 * p.
        if 'success_prob' in data.batch:
            verifier_coef = 5.0
        else:
            verifier_coef = self.config.verifier.reward_coef

        if verifier_coef != 0:
            reward_metrics['verifier'] = (verifier_coef * reward_tensor_dict['gt_scores']).sum(dim=1).mean().item()
            reward_tensor += verifier_coef * reward_tensor_dict['gt_scores']

        reward_tensor_dict['all'] = reward_tensor
        reward_metrics['reward_all'] = reward_tensor.sum(dim=-1).mean(dim=0).item()

        return reward_tensor_dict, reward_metrics

import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    OmegaConf.resolve(config)
    _validate_action_alignment_config(config)

    if not ray.is_initialized():
        # this is for local ray cluster
        if os.path.isfile(str(config.trainer.runtime_env)):
            with open(str(config.trainer.runtime_env), 'r') as f:
                runtime_env = json.load(f)
            ray.init(runtime_env=runtime_env)
        else:
            ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker, RobActorRolloutRefWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker, RobActorRolloutRefWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
    if config.actor_rollout_ref.wm.enable:
        from verl.workers.fsdp_workers import RobWMActorRolloutRefWorker
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(RobWMActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
            Role.RefPolicy: ray.remote(RobWMActorRolloutRefWorker)
        }
    else:
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(RobActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
            Role.RefPolicy: ray.remote(RobActorRolloutRefWorker)
        }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable and config.reward_model.rm_coef!=0.:
        if config.reward_model.rm_type == 'normal':
            if config.reward_model.strategy == 'fsdp':
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == 'megatron':
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        elif config.reward_model.rm_type == 'prime':
            from verl.workers.fsdp_workers import PRIMERewardModelWorker
            role_worker_mapping[Role.RewardModel] = ray.remote(PRIMERewardModelWorker)
        else:
            raise NotImplementedError
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RobRewardManager( num_examine=0, config=config) # note: verifier is called both inside reward_fn and outside.

    # Note that we always use function-based RM for validation
    val_reward_fn = RobRewardManager( num_examine=1,config=config)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    import time
    start_time = time.time()
    trainer.init_workers()
    print(f"init_workers time: {time.time() - start_time}")
    trainer.fit()


if __name__ == '__main__':
    main()
