#!/bin/bash
# VLA Simulator Rollout - 便捷运行脚本

# 设置环境变量
export MUJOCO_GL=osmesa
export CUDA_VISIBLE_DEVICES=0

# 默认参数
TASK="coffee"
INITIAL_IMAGE="/nfs/rczhang/code/WMPO/data_files/first_images/coffee/0.png"
CKPT="/nfs/rczhang/code/WMPO/checkpoint_files/SFT_models/coffee"
ENV_CONFIG="/nfs/rczhang/code/WMPO/data_files/core_train_configs/bc_rnn_image_ds_coffee_D0_seed_101.json"
RESET_STATES="/nfs/rczhang/code/WMPO/verl/utils/dataset/coffee_d0_states.pkl"
OUTPUT_DIR="/nfs/rczhang/code/WMPO/debug/vla_sim_output"
NUM_CHUNKS=25
CHUNK_SIZE=8
MAX_STEPS=256

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --task)
      TASK="$2"
      shift 2
      ;;
    --initial-image)
      INITIAL_IMAGE="$2"
      shift 2
      ;;
    --ckpt)
      CKPT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --num-chunks)
      NUM_CHUNKS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# 打印配置
echo "=================================="
echo "VLA Simulator Rollout"
echo "=================================="
echo "Task:          $TASK"
echo "Initial Image: $INITIAL_IMAGE"
echo "Checkpoint:    $CKPT"
echo "Output Dir:    $OUTPUT_DIR"
echo "Num Chunks:    $NUM_CHUNKS"
echo "=================================="
echo ""

# 运行脚本
python /nfs/rczhang/code/WMPO/scripts/vla_sim_rollout.py \
  --initial-image "$INITIAL_IMAGE" \
  --ckpt "$CKPT" \
  --task "$TASK" \
  --env-config "$ENV_CONFIG" \
  --reset-states "$RESET_STATES" \
  --output-dir "$OUTPUT_DIR" \
  --num-chunks "$NUM_CHUNKS" \
  --chunk-size "$CHUNK_SIZE" \
  --max-steps "$MAX_STEPS" \
  --save-fps 20 \
  --seed 0

echo ""
echo "Done! Check output at: $OUTPUT_DIR"

