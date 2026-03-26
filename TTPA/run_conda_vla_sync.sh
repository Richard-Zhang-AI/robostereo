#!/bin/bash
# RoboStereo TTPA - Conda side: VLA sim eval (sync with Docker Cosmos)
# Adapted from WMPO. Run in Conda env. Start this first, then run cosmos_preview_sync.py in Docker.
#
# Shared dir must be accessible by both Conda and Docker (e.g. /nfs/... or mounted volume)

set -e

# Conda runs on host; paths must match Docker mount
# BASE_ROOT: RoboStereo layout uses OEPL for checkpoint_files, data_files, verl (legacy var name WMPO_ROOT)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WMPO_ROOT="${WMPO_ROOT:-$REPO_ROOT/OEPL}"
OEPL_CHECKPOINT_FILES="${OEPL_CHECKPOINT_FILES:-$REPO_ROOT/OEPL/checkpoint_files}"
SYNC_DIR="${SYNC_DIR:-$WMPO_ROOT/debug/vla_cosmos_sync}"
OUTPUT_DIR="${OUTPUT_DIR:-$WMPO_ROOT/debug/vla_sim_sync_out}"

cd "$SCRIPT_DIR"

# Conda typically needs EGL for Robosuite rendering
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

python vla_sim_eval_sync.py \
  --sync-dir "$SYNC_DIR" \
  --vla-path "$OEPL_CHECKPOINT_FILES/SFT_models/coffee" \
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
