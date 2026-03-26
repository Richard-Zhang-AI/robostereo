#!/bin/bash
# 终端 2：VLA + Cosmos 交互（无仿真）
# 仅加载 VLA + Cosmos，适用于需要大量 GPU 显存的 Cosmos 推理
#
# 用法: bash run_vla_cosmos_interact.sh
# 需要先准备一张初始图像，或从 vla_sim_eval 输出的 frame_inputs 中选取

set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"

cd "$(dirname "$0")"

# 若 vla_sim_eval 已运行，可从其输出取第一帧作为输入
# 否则需设置 INITIAL_IMAGE=/path/to/your/image.png
INITIAL_IMAGE="${INITIAL_IMAGE:-/workspace/WMPO/debug/vla_sim_eval_out/episode_00/frame_inputs/chunk_000_input.png}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/WMPO/debug/vla_cosmos_interact_out}"

if [[ ! -f "$INITIAL_IMAGE" ]]; then
  echo "[WARN] Initial image not found: $INITIAL_IMAGE"
  echo "       Please set INITIAL_IMAGE=/path/to/image.png or run vla_sim_eval first."
  exit 1
fi

python vla_cosmos_bridge.py \
  --vla-path /workspace/WMPO/checkpoint_files/SFT_models/coffee \
  --unnorm-key coffee_d0_300_demos \
  --task coffee \
  --initial-image "$INITIAL_IMAGE" \
  --output-dir "$OUTPUT_DIR" \
  --num-chunks 25 \
  --cosmos-root /workspace \
  --cosmos-experiment ac_reason_embeddings_rectified_flow_2b_256_320 \
  --cosmos-checkpoint-path /workspace/checkpoints/coffee_1280_run1_iter_000028000/model_ema_bf16.pt \
  --cosmos-config-file cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
  --negative-prompt-file /workspace/assets/action_conditioned/openvla-coffee/inference_params.json \
  "$@"
