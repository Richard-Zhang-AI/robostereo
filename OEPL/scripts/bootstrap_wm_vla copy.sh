#!/usr/bin/env bash

# bash /nfs/rczhang/code/WMPO/scripts/bootstrap_wm_vla.sh
# 如果你想改镜像名或容器名，可以这样：
# IMAGE=wm-vla NAME=wm-vla-bootstrap bash /nfs/rczhang/code/WMPO/scripts/bootstrap_wm_vla.sh

# 删除：
# sudo docker rm -f wm-vla-run
# 启动：
# sudo docker run -it \
#     --runtime=nvidia \
#     --ipc=host \
#     --rm \
#     --name wm-vla-run \
#     -v /nfs/rczhang/code/cosmos-predict2.5:/workspace \
#     -v /nfs/rczhang/code/WMPO:/workspace/WMPO \
#     -v $HOME/.cache/huggingface:/root/.cache/huggingface \
#     -e HF_TOKEN="$HF_TOKEN" \
#     -e PYTHONPATH=/workspace/WMPO/dependencies/openvla-oft:$PYTHONPATH \
#     wm-vla

set -euo pipefail

IMAGE="${IMAGE:-wm-vla}"
NAME="${NAME:-wm-vla-bootstrap}"

sudo docker run -it \
  --runtime=nvidia \
  --ipc=host \
  --rm \
  --name "$NAME" \
  --entrypoint /bin/bash \
  -v /nfs/rczhang/code/cosmos-predict2.5:/workspace \
  -v /nfs/rczhang/code/WMPO:/workspace/WMPO \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e PYTHONPATH="/workspace/WMPO/dependencies/openvla-oft:${PYTHONPATH:-}" \
  "$IMAGE" -lc '
    if [ -d /workspace/.venv ]; then
      source /workspace/.venv/bin/activate
    fi

    python -m ensurepip --upgrade
    python -m pip install --upgrade pip

    # Base packages from your change list (reinstall all listed items).
    python -m pip install --upgrade \
      absl-py==2.4.0 \
      astunparse==1.6.3 \
      bcrypt==5.0.0 \
      beautifulsoup4==4.14.3 \
      bitsandbytes==0.49.1 \
      cfgv==3.5.0 \
      colossalai==0.5.0 \
      contexttimer==0.3.3 \
      decord==0.6.0 \
      deprecated==1.3.1 \
      diffusers==0.29.0 \
      fabric==3.2.2 \
      flatbuffers==25.12.19 \
      fsspec==2025.10.0 \
      galore-torch==1.0 \
      gast==0.7.0 \
      google==3.0.0 \
      google-auth-oauthlib==1.2.4 \
      google-pasta==0.2.0 \
      grpcio==1.76.0 \
      identify==2.6.16 \
      invoke==2.2.1 \
      keras==2.15.0 \
      libclang==18.1.1 \
      markdown==3.10.1 \
      markupsafe==3.0.3 \
      ml-dtypes==0.2.0 \
      ninja==1.13.0 \
      nodeenv==1.10.0 \
      numpy==1.26.4 \
      oauthlib==3.3.1 \
      opt-einsum==3.4.0 \
      paramiko==4.0.0 \
      peft==0.13.2 \
      pillow==11.3.0 \
      pip==26.0 \
      plumbum==1.10.0 \
      pre-commit==4.5.1 \
      protobuf==4.25.3 \
      pynacl==1.6.2 \
      requests-oauthlib==2.0.0 \
      rpyc==6.0.0 \
      scikit-learn==1.7.2 \
      soupsieve==2.8.3 \
      sympy==1.14.0 \
      tensorboard==2.15.2 \
      tensorboard-data-server==0.7.2 \
      tensorflow==2.15.0 \
      tensorflow-estimator==2.15.0 \
      tensorflow-io-gcs-filesystem==0.37.1 \
      threadpoolctl==3.6.0 \
      uvicorn==0.29.0 \
      werkzeug==3.1.5 \
      wheel==0.46.3 \
      wrapt==1.14.2

    # PyTorch + torchvision (cu128).
    python -m pip install --upgrade \
      torch==2.7.0+cu128 \
      torchvision==0.22.0+cu128 \
      triton==3.3.0 \
      --index-url https://download.pytorch.org/whl/cu128

    # NVIDIA CUDA libs (cu12) and Transformer Engine.
    python -m pip install --upgrade \
      nvidia-cublas-cu12==12.8.3.14 \
      nvidia-cuda-cupti-cu12==12.8.57 \
      nvidia-cuda-nvrtc-cu12==12.8.61 \
      nvidia-cuda-runtime-cu12==12.8.57 \
      nvidia-cudnn-cu12==9.7.1.26 \
      nvidia-cufft-cu12==11.3.3.41 \
      nvidia-curand-cu12==10.3.9.55 \
      nvidia-cusolver-cu12==11.7.2.55 \
      nvidia-cusparse-cu12==12.5.7.53 \
      nvidia-nccl-cu12==2.26.2 \
      nvidia-nvjitlink-cu12==12.8.61 \
      nvidia-nvtx-cu12==12.8.55 \
      transformer-engine==2.2.0 \
      transformer-engine-cu12==2.2.0 \
      --extra-index-url https://pypi.ngc.nvidia.com

    # flash-attn (requires matching wheel).
    python -m pip install --upgrade flash-attn==2.7.3+cu128.torch27 || true

    python -m pip install --upgrade "transformers==4.51.3" "tokenizers==0.21.4"

    python -m pip install tensorflow
  '

# python -m pip install vllm
# pip install "numpy<2" --force-reinstall

# pip install --force-reinstall "jsonschema>=4.0,<4.18"
# pip install "antlr4-python3-runtime==4.9.3"