#!/bin/bash
# Docker 端：Cosmos 预览（与 Conda VLA 同步）
# 在 Docker 容器内运行，需先启动 Conda 端的 run_conda_vla_sync.sh
#
# 共享目录需与 Conda 端一致（通过卷挂载实现）

set -e

SYNC_DIR="${SYNC_DIR:-/workspace/WMPO/debug/vla_cosmos_sync}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/WMPO/debug/cosmos_sync_out}"

cd "$(dirname "$0")"

python cosmos_preview_sync.py \
  --sync-dir "$SYNC_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --cosmos-root /workspace \
  --cosmos-experiment ac_reason_embeddings_rectified_flow_2b_256_320 \
  --cosmos-checkpoint-path /workspace/checkpoints/coffee_1280_run1_iter_000028000/model_ema_bf16.pt \
  --cosmos-config-file cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
  --negative-prompt-file /workspace/assets/action_conditioned/openvla-coffee/inference_params.json \
  "$@"
