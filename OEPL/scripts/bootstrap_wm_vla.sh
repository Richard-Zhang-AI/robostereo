#!/usr/bin/env bash

set -euo pipefail

# 配置部分
IMAGE="${IMAGE:-wm-vla}"
NAME="${NAME:-wm-vla-bootstrap}"
MOUNT_DIR="/nfs/rczhang/code/cosmos-predict2.5" # 你的 workspace 挂载路径

# 启动 Docker
sudo docker run -it \
  --runtime=nvidia \
  --ipc=host \
  --rm \
  --name "$NAME" \
  --entrypoint /bin/bash \
  -v "$MOUNT_DIR":/workspace \
  -v /nfs/rczhang/code/WMPO:/workspace/WMPO \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e PYTHONPATH="/workspace/WMPO/dependencies/openvla-oft:${PYTHONPATH:-}" \
  "$IMAGE" -lc '
    echo "=== 开始环境构建 ==="

    # 1. 自动创建虚拟环境（如果不存在）
    if [ ! -d "/workspace/.venv" ]; then
        echo "创建新的虚拟环境..."
        python3 -m venv /workspace/.venv
    fi
    
    # 2. 激活环境
    source /workspace/.venv/bin/activate
    
    # 3. 升级基础工具
    echo "升级 pip..."
    python -m ensurepip --upgrade
    python -m pip install --upgrade pip setuptools wheel

    # 4. 【关键修复】优先安装之前报错的底层库，锁定稳定版本
    echo "安装关键依赖修复..."
    # 修复 omegaconf/hydra 依赖
    python -m pip install "antlr4-python3-runtime==4.9.3"
    # 修复 ray/jsonschema 依赖
    python -m pip install "jsonschema>=4.0,<4.18" "referencing"
    # 修复 numpy 版本（TF 2.15 需要 numpy<2）
    python -m pip install "numpy<2.0"

    # 5. 安装 PyTorch (按照你的配置)
    echo "安装 PyTorch..."
    python -m pip install --upgrade \
      torch==2.7.0+cu128 \
      torchvision==0.22.0+cu128 \
      triton==3.3.0 \
      --index-url https://download.pytorch.org/whl/cu128

    # 6. 安装 NVIDIA CUDA 库
    echo "安装 NVIDIA Libs..."
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

    # 7. 安装 flash-attn
    echo "安装 Flash Attention..."
    python -m pip install --upgrade flash-attn==2.7.3+cu128.torch27 || echo "Flash Attn 安装失败，但继续执行..."

    # 8. 安装业务依赖（已根据你的列表整理）
    # 注意：移除了 numpy，因为它已经在上面被锁定为 <2
    echo "安装业务依赖..."
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
      oauthlib==3.3.1 \
      opt-einsum==3.4.0 \
      paramiko==4.0.0 \
      peft==0.13.2 \
      pillow==11.3.0 \
      plumbum==1.10.0 \
      pre-commit==4.5.1 \
      protobuf==4.25.3 \
      pynacl==1.6.2 \
      pyarrow \
      pandas \
      scipy \
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
      wrapt==1.14.2 \
      "transformers==4.51.3" "tokenizers==0.21.4" \
      tqdm huggingface_hub

    echo "=== 环境构建完成 ==="
  '