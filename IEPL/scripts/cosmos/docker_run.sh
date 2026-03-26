#!/bin/bash
# ============================================================
# IEPL (RoboStereo) - Docker launcher for Cosmos-RFT + MimicGen
# Adapted from VLA-RFT/OEPL.
#
# Modes:
#   train   - Run GRPO training with Cosmos WM + MimicGen data
#   shell   - Interactive shell for debugging
#
# Usage:
#   bash scripts/cosmos/docker_run.sh train [TASK_NAME]
#   bash scripts/cosmos/docker_run.sh shell
#
# Environment variables (override defaults):
#   TASK_NAME          - MimicGen task (default: coffee)
#   DOCKER_IMAGE       - Docker image (default: wm-vla)
#   NUM_GPUS_PER_NODE  - Number of GPUs (default: 8)
#   COSMOS_CHECKPOINT_PATH - Path to Cosmos checkpoint
# ============================================================

set -e

MODE="${1:-shell}"
TASK_NAME="${TASK_NAME:-${2:-coffee}}"

# ---- Host paths (relative to RoboStereo root) ----
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
HOST_COSMOS="${HOST_COSMOS:-$ROOT/../cosmos-predict2.5}"
HOST_VLA_RFT="$ROOT/IEPL"
HOST_OEPL="$ROOT/OEPL"
HOST_HF_CACHE="$HOME/.cache/huggingface"

# ---- Container paths ----
CTN_COSMOS="/workspace"
CTN_VLA_RFT="/workspace/IEPL"   # IEPL project
CTN_OEPL="/workspace/OEPL"
CTN_HF_CACHE="/root/.cache/huggingface"

DOCKER_IMAGE="${DOCKER_IMAGE:-wm-vla}"

# ---- PYTHONPATH inside container ----
# OEPL first (checkpoints, datasets, verl)
# Then openvla-oft, IEPL (for iepl_cosmos module), Cosmos root
PYTHONPATH_CTN="${CTN_OEPL}:${CTN_OEPL}/dependencies/openvla-oft:${CTN_VLA_RFT}:${CTN_COSMOS}"

# ---- Docker command ----
DOCKER_CMD="sudo docker run -it \
    --runtime=nvidia \
    --ipc=host \
    --shm-size=64g \
    --rm \
    -v ${HOST_COSMOS}:${CTN_COSMOS} \
    -v ${HOST_VLA_RFT}:${CTN_VLA_RFT} \
    -v ${HOST_OEPL}:${CTN_OEPL} \
    -v ${HOST_HF_CACHE}:${CTN_HF_CACHE} \
    -e HF_TOKEN=\"${HF_TOKEN}\" \
    -e WANDB_API_KEY=\"${WANDB_API_KEY}\" \
    -e PYTHONPATH=${PYTHONPATH_CTN} \
    -e COSMOS_ROOT=${CTN_COSMOS} \
    -e OEPL_ROOT=${CTN_OEPL} \
    -e VLA_RFT_ROOT=${CTN_VLA_RFT} \
    -e TASK_NAME=${TASK_NAME} \
    -w ${CTN_VLA_RFT}"

case "$MODE" in
    train)
        echo "=== IEPL Cosmos-RFT MimicGen Training ==="
        echo "Task: $TASK_NAME"
        echo "Docker image: $DOCKER_IMAGE"
        echo "Working dir: $CTN_VLA_RFT (IEPL)"
        echo ""

        # Pass through optional env vars
        EXTRA_ENV="-e OEPL_CHECKPOINT_FILES=${OEPL_CHECKPOINT_FILES:-${CTN_OEPL}/checkpoint_files}"
        [ -n "$NUM_GPUS_PER_NODE" ] && EXTRA_ENV="$EXTRA_ENV -e NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE"
        [ -n "$COSMOS_CHECKPOINT_PATH" ] && EXTRA_ENV="$EXTRA_ENV -e COSMOS_CHECKPOINT_PATH=$COSMOS_CHECKPOINT_PATH"
        [ -n "$SFT_MODEL_PATH" ] && EXTRA_ENV="$EXTRA_ENV -e SFT_MODEL_PATH=$SFT_MODEL_PATH"
        [ -n "$REWARD_MODEL_PATH" ] && EXTRA_ENV="$EXTRA_ENV -e REWARD_MODEL_PATH=$REWARD_MODEL_PATH"
        [ -n "$COSMOS_NEGATIVE_PROMPT_FILE" ] && EXTRA_ENV="$EXTRA_ENV -e COSMOS_NEGATIVE_PROMPT_FILE=$COSMOS_NEGATIVE_PROMPT_FILE"

        eval $DOCKER_CMD $EXTRA_ENV $DOCKER_IMAGE \
            bash scripts/cosmos/post_train_cosmos_rft.sh
        ;;

    shell)
        echo "=== Interactive Shell ==="
        echo "PYTHONPATH=$PYTHONPATH_CTN"
        echo ""
        eval $DOCKER_CMD $DOCKER_IMAGE bash
        ;;

    *)
        echo "Usage: $0 {train|shell} [TASK_NAME]"
        echo ""
        echo "Examples:"
        echo "  $0 train coffee     # Train on coffee task"
        echo "  $0 train square     # Train on square task"
        echo "  $0 shell            # Interactive shell"
        exit 1
        ;;
esac
