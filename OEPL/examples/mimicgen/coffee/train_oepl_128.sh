set -x
set -o pipefail
set -euo pipefail

REPO_ROOT="/workspace/OEPL"

export NCCL_DEBUG=WARN 
export TF_CPP_MIN_LOG_LEVEL=3
export WANDB_API_KEY="${WANDB_API_KEY:-}" # set in env or align.json for ray workers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WMPO_ROLLOUT_PROGRESS=1
export WMPO_WM_PROGRESS_BAR=1
export WMPO_PRINT_METRICS=1
export WMPO_PRINT_METRICS_EVERY=1
export WMPO_KEEP_MODELS_ON_GPU=1
export WMPO_KEEP_VLA_FULL_PARAMS=1
export WMPO_WM_TIMING=1

# 显存追踪监控（排障时可改为 1）
export WMPO_MEM_DEBUG="${WMPO_MEM_DEBUG:-0}"
export WMPO_MEM_DEBUG_EVERY="${WMPO_MEM_DEBUG_EVERY:-10}"

# export RAY_METRICS_EXPORT_PORT=0
# export RAY_ENABLE_METRICS=0
# # Ray 的监控指标导出器（metrics exporter）启动失败，我关了

PROJECT_NAME='OEPL-mimicgen'
EXPERIMENT_NAME='coffee_wmpo_128'

TASK_NAME="coffee"
UNNORM_KEY="$TASK_NAME"_d0_300_demos

SFT_MODEL_PATH="$REPO_ROOT/checkpoint_files/SFT_models/coffee"
REWARD_MODEL_PATH="$REPO_ROOT/checkpoint_files/reward_models/videomae_coffee.pth"
PRISMATIC_SRC_DIR="$REPO_ROOT/verl/utils/vla_utils/openvla_oft"

require_path() {
    local path="$1"
    local desc="$2"
    if [[ ! -e "$path" ]]; then
        echo "[ERROR] Missing $desc: $path" >&2
        echo "[ERROR] This repo expects pretrained artifacts under $REPO_ROOT/checkpoint_files." >&2
        echo "[ERROR] Please run 'python download_hf.py' in $REPO_ROOT, or mount/sync the needed checkpoint files before training." >&2
        exit 1
    fi
}

echo "Clearing transformers module cache to prevent stale code issues..."
rm -rf ~/.cache/huggingface/modules/transformers_modules/

require_path "$SFT_MODEL_PATH" "SFT model directory"
require_path "$REWARD_MODEL_PATH" "reward model checkpoint"
require_path "$PRISMATIC_SRC_DIR/modeling_prismatic.py" "OpenVLA-OFT modeling shim"
require_path "$PRISMATIC_SRC_DIR/processing_prismatic.py" "OpenVLA-OFT processor shim"
require_path "$PRISMATIC_SRC_DIR/configuration_prismatic.py" "OpenVLA-OFT config shim"
require_path "$PRISMATIC_SRC_DIR/constants.py" "OpenVLA-OFT constants shim"
require_path "$PRISMATIC_SRC_DIR/train_utils.py" "OpenVLA-OFT train utils shim"

rm -f "$SFT_MODEL_PATH"/config.json.back*
rm -f "$SFT_MODEL_PATH"/modeling_prismatic.py.back*

cp "$PRISMATIC_SRC_DIR/"*.py "$SFT_MODEL_PATH/"

if [[ -f "$REPO_ROOT/checkpoint_files/preprocessor_config.json" ]]; then
    cp "$REPO_ROOT/checkpoint_files/preprocessor_config.json" "$SFT_MODEL_PATH/"
elif [[ ! -f "$SFT_MODEL_PATH/preprocessor_config.json" ]]; then
    echo "[ERROR] Missing preprocessor config in both:" >&2
    echo "  - $REPO_ROOT/checkpoint_files/preprocessor_config.json" >&2
    echo "  - $SFT_MODEL_PATH/preprocessor_config.json" >&2
    echo "[ERROR] The SFT checkpoint is incomplete. Re-download or re-sync checkpoint_files." >&2
    exit 1
fi


CKPT_PATH="$REPO_ROOT/checkpoint_files"
VLA_NAME="openvla-oft"
# If you want to use 2*8 GPU to RL. Set NUM_NODES=2
NUM_NODES=1
NUM_GPUS_PER_NODE=8
NUM_GPUS=$((NUM_NODES * NUM_GPUS_PER_NODE))
ALIGN_PATH="$REPO_ROOT/align.json"

LOG_DIR="$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME"
mkdir -p "$LOG_DIR"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$LOG_DIR/runs/$RUN_ID"
mkdir -p "$RUN_DIR"
TERMINAL_LOG="$RUN_DIR/terminal.log"

# Mirror full terminal output to this run directory in real time.
exec > >(tee -a "$TERMINAL_LOG") 2>&1
echo "Run directory: $RUN_DIR"

# 在训练主循环里，每次更新（计算奖励 + 反向传播）需要凑满：data.train_batch_size * data.n_samples
# 关键约束：train_batch_size * n_samples 要不小于 PPO 的 ppo_mini_batch_size
COSMOS_ROOT="/workspace"
COSMOS_EXPERIMENT="ac_reason_embeddings_rectified_flow_2b_256_320"
COSMOS_CHECKPOINT_PATH="/workspace/checkpoints/coffee_action_condtitioned/checkpoints-RGB/model_ema_fp32.pt"
COSMOS_CONFIG_FILE="cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py"
COSMOS_NEGATIVE_PROMPT_FILE="/workspace/assets/action_conditioned/openvla-coffee/inference_params.json"
COSMOS_PREVIEW_SAVE_DIR="$REPO_ROOT/checkpoint_files/OEPL-mimicgen/coffee_oepl_128/cosmos_previews"

HYDRA_FULL_ERROR=1 python -m verl.trainer.main_ppo \
    +data.task_name=$TASK_NAME \
    data.n_samples=4 \
    data.filter_accuracy=True \
    data.accuracy_lower_bound=0.0 \
    data.accuracy_upper_bound=1.0 \
    data.oversample_factor=1 \
    data.train_batch_size=16 \
    data.val_batch_size=16 \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    +data.rollout_batch_size=32 \
    +actor_rollout_ref.rollout_base_dir=$REPO_ROOT/tmp_files/rollout_$EXPERIMENT_NAME \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.model.vla=$VLA_NAME \
    actor_rollout_ref.model.action_token_len=7 \
    actor_rollout_ref.model.action_chunks_len=8 \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=$NUM_GPUS \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.grad_clip=1 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.num_images_in_input=1 \
    actor_rollout_ref.actor.traj_mini_batch_size=11 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.entropy_coeff=0. \
    actor_rollout_ref.rollout.num_images_in_input=1 \
    actor_rollout_ref.rollout.val_micro_batch_size=8 \
    actor_rollout_ref.rollout.temperature=1.6 \
    actor_rollout_ref.rollout.experiment_name=$EXPERIMENT_NAME \
    actor_rollout_ref.rollout.micro_batch_size=2 \
    actor_rollout_ref.rollout.unnorm_key=$UNNORM_KEY \
    actor_rollout_ref.rollout.model_family=openvla \
    actor_rollout_ref.rollout.num_steps_wait=10 \
    actor_rollout_ref.rollout.pretrained_checkpoint=$SFT_MODEL_PATH \
    actor_rollout_ref.rollout.center_crop=True \
    actor_rollout_ref.rollout.max_prompt_length=512 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.wm.backend=cosmos \
    +actor_rollout_ref.wm.cosmos_root=$COSMOS_ROOT \
    +actor_rollout_ref.wm.cosmos_experiment=$COSMOS_EXPERIMENT \
    +actor_rollout_ref.wm.cosmos_checkpoint_path=$COSMOS_CHECKPOINT_PATH \
    +actor_rollout_ref.wm.cosmos_config_file=$COSMOS_CONFIG_FILE \
    +actor_rollout_ref.wm.cosmos_action_chunk_size=12 \
    +actor_rollout_ref.wm.cosmos_action_scale=20.0 \
    +actor_rollout_ref.wm.cosmos_gripper_scale=1.0 \
    "+actor_rollout_ref.wm.cosmos_resolution='256,320'" \
    +actor_rollout_ref.wm.cosmos_guidance=0.0 \
    +actor_rollout_ref.wm.cosmos_seed=0 \
    +actor_rollout_ref.wm.cosmos_max_parallel_workers=1 \
    +actor_rollout_ref.wm.cosmos_gpu_allocation_mode=auto \
    +actor_rollout_ref.wm.cosmos_negative_prompt_file=$COSMOS_NEGATIVE_PROMPT_FILE \
    +actor_rollout_ref.wm.preview_save_dir=$COSMOS_PREVIEW_SAVE_DIR \
    +actor_rollout_ref.wm.preview_save_fps=6 \
    +actor_rollout_ref.wm.use_normalized_actions=False \
    +actor_rollout_ref.wm.update_wm=False \
    actor_rollout_ref.wm.reward_model_path=$REWARD_MODEL_PATH \
    actor_rollout_ref.wm.rm_threshold=0.1 \
    actor_rollout_ref.wm.rm_img_size=224 \
    +actor_rollout_ref.wm.rm_lr=1e-4 \
    +actor_rollout_ref.wm.rm_batch_size=64 \
    +actor_rollout_ref.wm.rm_val_batch_size=512 \
    +actor_rollout_ref.wm.rm_num_workers=8 \
    +actor_rollout_ref.wm.wm_training_steps_per_epoch=10000 \
    +actor_rollout_ref.wm.rm_training_steps_per_epoch=10000 \
    actor_rollout_ref.wm.enable=True \
    algorithm.kl_ctrl.kl_coef=0.00 \
    +trainer.rollout_before_train=False \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$NUM_GPUS_PER_NODE \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.total_epochs=100 \
    trainer.val_only=False \
    +trainer.val_save_rollouts=False \
    +trainer.generate_rpc_chunk_size=8 \
    +trainer.train_generate_rpc_chunk_size=8 \
    +trainer.val_generate_rpc_chunk_size=8 \
    +trainer.rollout_generate_rpc_chunk_size=8 \
    +trainer.generate_rpc_min_samples_per_rank=1 \
    +trainer.ray_generate_max_retries=4 \
    +trainer.ray_generate_retry_backoff_s=15 \
    +trainer.sim_rollout_epoch=1 \
    algorithm.adv_estimator=grpo \
    algorithm.adv_params.verifier_gamma=1.0 \
    algorithm.adv_params.reward_model_gamma=1.0 \
    trainer.runtime_env=$ALIGN_PATH \
    trainer.wandb_mode=online \
    trainer.val_before_train=False

# rollout.micro_batch_size 控制的是：一次 rollout 调用里，每个 worker 每小批实际并行处理多少个 prompt。


# val_before_train（初始验证）到底在干什么：
#   - 在正式训练前先跑一遍 _validate(global_steps=0)（verl/trainer/ppo/
#     ray_trainer.py:431）。
#   - 这一步会做生成 + 奖励评估，只记录指标，不做反向传播、不更新权重。
#   - 作用是：
#       1. 先确认数据/推理/奖励链路没坏；
#       2. 给一个“训练前基线”方便后面对比。
#   是否必要：
#   - 不是收敛必须项。
#   - 在你这种 WM rollout 很重的场景，它更像“启动前体检”，不是核心训练步骤。
#   - 生产跑长任务时可以关（trainer.val_before_train=False），然后把验证放到训练后周期性做
#     （test_freq）。


#   分片操作：把一次超长 generate_sequences RPC 切成多个短 RPC，再按原顺序拼回
#  分片数计算公式：
#   - num_chunks = ceil(total_size / chunk_size)
#   当前脚本里是：
#   - train: chunk_size=8
#   - val: chunk_size=4
#   - rollout_save: chunk_size=4
#   例如训练时 total_size=16，chunk_size=8，就会分 2 片。
