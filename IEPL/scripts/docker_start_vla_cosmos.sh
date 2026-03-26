#!/bin/bash
# IEPL (RoboStereo) - Launch Cosmos training Docker environment
# Usage: ./scripts/docker_start_vla_cosmos.sh

sudo docker run -it \
  --runtime=nvidia \
  --ipc=host \
  --rm \
  --name vla-rft-cosmos-train-bc \
  --entrypoint /bin/bash \
  -v "$(cd "$(dirname "$0")/../.." && pwd)":/workspace \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN="$HF_TOKEN" \
  -e OEPL_ROOT=/workspace/OEPL \
  -e VLA_RFT_ROOT=/workspace/IEPL \
  -e PYTHONPATH=/workspace/OEPL/dependencies/openvla-oft:/workspace/OEPL:/workspace/IEPL:/workspace:$PYTHONPATH \
  wm-vla:with-deps
