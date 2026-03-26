#!/bin/bash
# RoboStereo TTPA - Docker side: Cosmos preview (sync with Conda VLA)
# Adapted from WMPO. Run inside Docker container. Start Conda run_conda_vla_sync.sh first.
#
# Shared dir must match Conda side (via volume mount)

set -e

# RoboStereo layout: OEPL has checkpoint_files, data_files, verl
# Use relative paths (script runs from TTPA dir)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OEPL_ROOT="${OEPL_ROOT:-$SCRIPT_DIR/../OEPL}"
SYNC_DIR="${SYNC_DIR:-$OEPL_ROOT/debug/vla_cosmos_sync}"
OUTPUT_DIR="${OUTPUT_DIR:-$OEPL_ROOT/debug/cosmos_sync_out}"

cd "$SCRIPT_DIR"

# DASHSCOPE_API_KEY required for Qwen scoring
python cosmos_preview_sync.py \
  --sync-dir "$SYNC_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --cosmos-root /workspace \
  --cosmos-experiment ac_reason_embeddings_rectified_flow_2b_256_320 \
  --cosmos-checkpoint-path "${COSMOS_CHECKPOINT_PATH:-/workspace/checkpoints_files/coffee_action_conditioned/checkpoints-RGB/model_ema_fp32.pt}" \
  --cosmos-config-file cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
  --negative-prompt-file /workspace/assets/action_conditioned/openvla-coffee/inference_params.json \
  --qwen-score-threshold "${QWEN_SCORE_THRESHOLD:-6.0}" \
  "$@"
