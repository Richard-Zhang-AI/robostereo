#!/bin/bash
# IEPL (RoboStereo) - Minimal debug run for Cosmos-RFT (fast sanity check)
# Uses MSE reward and tiny batch sizes to get quick signal.

set -x
set -o pipefail

REPO_ROOT="${OEPL_ROOT:-/workspace/OEPL}"
VLA_RFT_ROOT="${VLA_RFT_ROOT:-/workspace/IEPL}"  # IEPL project root
COSMOS_ROOT="${COSMOS_ROOT:-/workspace}"

export NCCL_DEBUG=WARN
export TF_CPP_MIN_LOG_LEVEL=3
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

export RAY_TMPDIR="${RAY_TMPDIR:-/workspace/ray_tmp}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"
export WANDB_DIR="${WANDB_DIR:-/workspace/wandb}"
mkdir -p "$RAY_TMPDIR" "$TMPDIR" "$WANDB_DIR"

# Rollout/worker env flags (legacy WMPO_* names)
export WMPO_ROLLOUT_PROGRESS=1
export WMPO_WM_PROGRESS_BAR=1
export WMPO_PRINT_METRICS=1
export WMPO_PRINT_METRICS_EVERY=1
export WMPO_KEEP_MODELS_ON_GPU=1
export WMPO_KEEP_VLA_FULL_PARAMS=1
export WMPO_WM_TIMING=1

export PYTHONPATH="$VLA_RFT_ROOT/train/verl:$VLA_RFT_ROOT/train/verl/vla-adapter/openvla-oft:$VLA_RFT_ROOT:$COSMOS_ROOT${PYTHONPATH:+:$PYTHONPATH}"

TASK_NAME="${TASK_NAME:-coffee}"
UNNORM_KEY="${TASK_NAME}_d0_300_demos"
# Dataset: /workspace/datasets/train_openvla/{task}_128 (RoboStereo)
VIDEO_ROOT="${VIDEO_ROOT:-/workspace/datasets/train_openvla/${TASK_NAME}_128}"
TRAIN_MODE="${TRAIN_MODE:-rl}"

PROJECT_NAME="cosmos-rft-mimicgen"
EXPERIMENT_NAME="${TASK_NAME}_cosmos_rft_min_debug"

OEPL_CHECKPOINT_FILES="${OEPL_CHECKPOINT_FILES:-$(cd "$VLA_RFT_ROOT/.." && pwd)/OEPL/checkpoint_files}"
SFT_MODEL_PATH="${SFT_MODEL_PATH:-$OEPL_CHECKPOINT_FILES/SFT_models/$TASK_NAME}"
REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-$OEPL_CHECKPOINT_FILES/reward_models/videomae_${TASK_NAME}.pth}"

COSMOS_EXPERIMENT="${COSMOS_EXPERIMENT:-ac_reason_embeddings_rectified_flow_2b_256_320}"
COSMOS_CHECKPOINT_PATH="${COSMOS_CHECKPOINT_PATH:-$(cd "$VLA_RFT_ROOT/.." && pwd)/checkpoints_files/coffee_action_conditioned/checkpoints-RGB/model_ema_fp32.pt}"
COSMOS_CONFIG_FILE="${COSMOS_CONFIG_FILE:-cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py}"
COSMOS_NEGATIVE_PROMPT_FILE="${COSMOS_NEGATIVE_PROMPT_FILE:-/workspace/assets/action_conditioned/openvla-${TASK_NAME}/inference_params.json}"

CKPT_PATH="$OEPL_CHECKPOINT_FILES"
NUM_NODES="${NUM_NODES:-1}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-1}"
ALIGN_PATH="$REPO_ROOT/align.json"

LOG_DIR="$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME"
mkdir -p "$LOG_DIR"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$LOG_DIR/runs/$RUN_ID"
mkdir -p "$RUN_DIR"
export WMPO_ACTION_LOG_PATH="${WMPO_ACTION_LOG_PATH:-$RUN_DIR/action_stats.jsonl}"
export VLA_RFT_DEBUG_LOG="${VLA_RFT_DEBUG_LOG:-$RUN_DIR/debug.log}"
echo "Run directory: $RUN_DIR"

cd "$VLA_RFT_ROOT"

HYDRA_FULL_ERROR=1 python -m iepl_cosmos.main_cosmos_rft \
    data.task_name=$TASK_NAME \
    data.n_samples=1 \
    data.state_path=/workspace/OEPL/verl/utils/dataset/${TASK_NAME}_d0_states.pkl \
    data.video_root=$VIDEO_ROOT \
    data.filter_accuracy=False \
    data.train_batch_size=1 \
    data.val_batch_size=1 \
    +data.val_num_workers=0 \
    +data.reward_scale=1.0 \
    data.reward_mse_weight=1.0 \
    data.reward_lpips_weight=0.0 \
    data.lpips_every=1000000 \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    data.rollout_batch_size=1 \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.model.vla=openvla-oft \
    actor_rollout_ref.model.action_token_len=7 \
    actor_rollout_ref.model.action_chunks_len=8 \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.grad_clip=1 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.num_images_in_input=1 \
    actor_rollout_ref.actor.num_patches=256 \
    actor_rollout_ref.actor.num_tokens=64 \
    actor_rollout_ref.actor.traj_mini_batch_size=1 \
    actor_rollout_ref.actor.use_mse_loss=False \
    +actor_rollout_ref.actor.mse_loss_coef=0.01 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.entropy_coeff=0. \
    actor_rollout_ref.rollout.num_images_in_input=1 \
    actor_rollout_ref.rollout.val_micro_batch_size=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.num_patches=256 \
    actor_rollout_ref.rollout.num_tokens=64 \
    actor_rollout_ref.rollout.experiment_name=$EXPERIMENT_NAME \
    actor_rollout_ref.rollout.micro_batch_size=1 \
    actor_rollout_ref.rollout.model_family=openvla \
    actor_rollout_ref.rollout.num_steps_wait=10 \
    actor_rollout_ref.rollout.pretrained_checkpoint=$SFT_MODEL_PATH \
    actor_rollout_ref.rollout.center_crop=True \
    actor_rollout_ref.rollout.max_prompt_length=512 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.95 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.wm.backend=cosmos \
    actor_rollout_ref.wm.cosmos_root=$COSMOS_ROOT \
    actor_rollout_ref.wm.cosmos_experiment=$COSMOS_EXPERIMENT \
    actor_rollout_ref.wm.cosmos_checkpoint_path=$COSMOS_CHECKPOINT_PATH \
    actor_rollout_ref.wm.cosmos_config_file=$COSMOS_CONFIG_FILE \
    actor_rollout_ref.wm.cosmos_action_chunk_size=12 \
    actor_rollout_ref.wm.cosmos_action_scale=20.0 \
    actor_rollout_ref.wm.cosmos_gripper_scale=1.0 \
    "actor_rollout_ref.wm.cosmos_resolution='256,320'" \
    actor_rollout_ref.wm.cosmos_guidance=0.0 \
    actor_rollout_ref.wm.cosmos_seed=0 \
    actor_rollout_ref.wm.cosmos_negative_prompt_file=$COSMOS_NEGATIVE_PROMPT_FILE \
    actor_rollout_ref.wm.use_normalized_actions=False \
    actor_rollout_ref.wm.update_wm=False \
    actor_rollout_ref.wm.reward_model_path=$REWARD_MODEL_PATH \
    actor_rollout_ref.wm.rm_threshold=0.1 \
    actor_rollout_ref.wm.rm_img_size=224 \
    actor_rollout_ref.wm.rm_lr=1e-4 \
    actor_rollout_ref.wm.rm_batch_size=1 \
    actor_rollout_ref.wm.rm_val_batch_size=1 \
    actor_rollout_ref.wm.rm_num_workers=0 \
    actor_rollout_ref.wm.wm_training_steps_per_epoch=1 \
    actor_rollout_ref.wm.rm_training_steps_per_epoch=1 \
    actor_rollout_ref.wm.enable=True \
    algorithm.kl_ctrl.kl_coef=0.00 \
    trainer.rollout_before_train=False \
    trainer.logger="['console']" \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    +trainer.train_mode=$TRAIN_MODE \
    trainer.n_gpus_per_node=$NUM_GPUS_PER_NODE \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.val_only=False \
    trainer.val_save_rollouts=False \
    trainer.generate_rpc_chunk_size=1 \
    trainer.train_generate_rpc_chunk_size=1 \
    trainer.val_generate_rpc_chunk_size=1 \
    trainer.rollout_generate_rpc_chunk_size=1 \
    trainer.generate_rpc_min_samples_per_rank=1 \
    trainer.sim_rollout_epoch=1 \
    algorithm.adv_estimator=grpo \
    algorithm.adv_params.verifier_gamma=1.0 \
    algorithm.adv_params.reward_model_gamma=1.0 \
    trainer.runtime_env=$ALIGN_PATH \
    trainer.wandb_mode=offline \
    trainer.val_before_train=False
