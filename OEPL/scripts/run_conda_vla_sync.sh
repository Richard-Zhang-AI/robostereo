#!/bin/bash
# Conda 端：VLA 仿真评估（与 Docker Cosmos 同步）
# 在 Conda 环境中运行，需先启动此脚本，再在 Docker 中启动 cosmos_preview_sync.py
#
# 共享目录需对 Conda 和 Docker 均可访问（如 /nfs/... 或挂载卷内的路径）

set -e

# Conda 在宿主机运行，使用宿主机路径；需与 Docker 挂载对应
WMPO_ROOT="${WMPO_ROOT:-$(dirname "$0")/..}"
SYNC_DIR="${SYNC_DIR:-$WMPO_ROOT/debug/vla_cosmos_sync}"
OUTPUT_DIR="${OUTPUT_DIR:-$WMPO_ROOT/debug/vla_sim_sync_out}"

cd "$(dirname "$0")"

# Conda 环境通常需要 EGL 做 Robosuite 渲染
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

python vla_sim_eval_sync.py \
  --sync-dir "$SYNC_DIR" \
  --vla-path "$WMPO_ROOT/checkpoint_files/SFT_models/coffee" \
  --unnorm-key coffee_d0_300_demos \
  --task coffee \
  --env-config "$WMPO_ROOT/data_files/core_train_configs/bc_rnn_image_ds_coffee_D0_seed_101.json" \
  --reset-states "$WMPO_ROOT/verl/utils/dataset/coffee_d0_states.pkl" \
  --output-dir "$OUTPUT_DIR" \
  --num-chunks 25 \
  --chunk-size 8 \
  --runs 10 \
  --no-video \
  "$@"
