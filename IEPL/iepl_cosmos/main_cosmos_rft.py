"""
IEPL (RoboStereo) - Main entry for Cosmos-RFT training with MimicGen data.
Adapted from VLA-RFT/OEPL.

Usage:
    python -m iepl_cosmos.main_cosmos_rft <hydra overrides>

Uses local verl worker for action generation and CosmosWorldModelWorker for video prediction.
"""

import os
import json
import ray
import hydra
from pathlib import Path


@hydra.main(config_path='config', config_name='cosmos_rft_grpo_trainer', version_base=None)
def main(config):
    # Shutdown any stale Ray session from previous crashed runs (avoids register_center conflicts)
    if ray.is_initialized():
        ray.shutdown()
    if not ray.is_initialized():
        runtime_env_path = str(config.trainer.get('runtime_env', 'none'))
        # Force local IEPL paths for all Ray workers.
        repo_root = Path(__file__).resolve().parents[1]
        local_verl = repo_root / "train" / "verl"
        local_openvla = repo_root / "train" / "verl" / "vla-adapter" / "openvla-oft"
        local_pythonpath = f"{local_verl}:{local_openvla}:{repo_root}:{os.environ.get('PYTHONPATH','')}"
        if os.path.isfile(runtime_env_path):
            with open(runtime_env_path, 'r') as f:
                runtime_env = json.load(f)
            runtime_env.setdefault("env_vars", {})
            runtime_env["env_vars"]["PYTHONPATH"] = local_pythonpath
            ray.init(runtime_env=runtime_env)
        else:
            ray.init(runtime_env={
                'env_vars': {
                    'TOKENIZERS_PARALLELISM': 'true',
                    'NCCL_DEBUG': 'WARN',
                    'PYTHONPATH': local_pythonpath,
                }
            })

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    # Prefer local IEPL verl over external paths.
    import sys
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[1]
    local_verl = repo_root / "train" / "verl"
    local_openvla = repo_root / "train" / "verl" / "vla-adapter" / "openvla-oft"
    # Ensure local paths are first to avoid OEPL verl overrides.
    sys.path = [p for p in sys.path if "/workspace/OEPL" not in p]
    sys.path.insert(0, str(local_openvla))
    sys.path.insert(0, str(local_verl))

    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)
    # Ensure required model fields exist for local verl worker.
    try:
        from omegaconf import open_dict
        with open_dict(config.actor_rollout_ref.model):
            if 'ckpt_path' not in config.actor_rollout_ref.model:
                config.actor_rollout_ref.model.ckpt_path = config.actor_rollout_ref.model.path
            if 'cfg_path' not in config.actor_rollout_ref.model:
                config.actor_rollout_ref.model.cfg_path = config.actor_rollout_ref.model.path
        with open_dict(config.actor_rollout_ref.actor):
            if 'num_patches' not in config.actor_rollout_ref.actor:
                config.actor_rollout_ref.actor.num_patches = 256
            if 'num_tokens' not in config.actor_rollout_ref.actor:
                config.actor_rollout_ref.actor.num_tokens = 64
        with open_dict(config.actor_rollout_ref.rollout):
            if 'num_patches' not in config.actor_rollout_ref.rollout:
                config.actor_rollout_ref.rollout.num_patches = 256
            if 'num_tokens' not in config.actor_rollout_ref.rollout:
                config.actor_rollout_ref.rollout.num_tokens = 64
    except Exception:
        pass

    from verl.utils.fs import copy_local_path_from_hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from verl.single_controller.ray import RayWorkerGroup
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
    from iepl_cosmos.cosmos_wm_worker import CosmosWorldModelWorker

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
        Role.WorldModelRollout: ray.remote(CosmosWorldModelWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.RefPolicy: global_pool_id,
        Role.WorldModelRollout: global_pool_id,
    }

    from iepl_cosmos.cosmos_rft_trainer import MimicGenRewardManager, RayCosmosRFTGRPOTrainer

    reward_fn = MimicGenRewardManager(config=config, num_examine=0)
    val_reward_fn = MimicGenRewardManager(config=config, num_examine=1)

    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec,
        mapping=mapping,
    )

    trainer = RayCosmosRFTGRPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
        ray_worker_group_cls=RayWorkerGroup,
    )

    import time
    start_time = time.time()
    trainer.init_workers()
    print(f'[CosmosRFT] init_workers time: {time.time() - start_time:.1f}s')
    trainer.fit()


if __name__ == '__main__':
    main()
