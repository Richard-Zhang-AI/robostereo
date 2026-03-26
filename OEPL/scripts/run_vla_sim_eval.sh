#!/bin/bash
# 终端 1：VLA 仿真评估（无 Cosmos）
# 仅加载 VLA + Robosuite，适用于需要 EGL/渲染的环境
#
# 用法: bash run_vla_sim_eval.sh

set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

cd "$(dirname "$0")"

python vla_sim_eval.py \
  --vla-path /workspace/WMPO/checkpoint_files/SFT_models/coffee \
  --unnorm-key coffee_d0_300_demos \
  --task coffee \
  --env-config /workspace/WMPO/data_files/core_train_configs/bc_rnn_image_ds_coffee_D0_seed_101.json \
  --reset-states /workspace/WMPO/verl/utils/dataset/coffee_d0_states.pkl \
  --output-dir /workspace/WMPO/debug/vla_sim_eval_out \
  --num-chunks 25 \
  --chunk-size 8 \
  --runs 10 \
  --no-video \
  "$@"
