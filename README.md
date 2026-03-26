
<div align="center">

# 🤖 RoboStereo
## **Dual-Tower 4D Embodied World Models for Unified Policy Optimization**

<br>

<!-- Authors -->
**Ruicheng Zhang**<sup>1*</sup> &nbsp; **Guangyu Chen**<sup>1*</sup> &nbsp; **Zunnan Xu**<sup>1,2†</sup> &nbsp; **Zihao Liu**<sup>1</sup> &nbsp; <br> **Zhizhou Zhong**<sup>3</sup> &nbsp; **Mingyang Zhang**<sup>1</sup> &nbsp; **Jun Zhou**<sup>1</sup> &nbsp; **Xiu Li**<sup>1‡</sup>

<!-- Affiliations -->
<sup>1</sup>Tsinghua University &nbsp;&nbsp;&nbsp; <sup>2</sup>X Square Robot &nbsp;&nbsp;&nbsp; <sup>3</sup>HKUST

<!-- Footnotes -->
<small>
<sup>*</sup> Equal Contribution &nbsp;&nbsp; 
<sup>†</sup> Project Lead &nbsp;&nbsp; 
<sup>‡</sup> Corresponding Author
</small>

<br>

<!-- Badges/Icons -->
[![arXiv](https://img.shields.io/badge/arXiv-2603.12639-b31b1b.svg?logo=arxiv&style=for-the-badge)](https://arxiv.org/abs/2603.12639)
[![Hugging Face Model](https://img.shields.io/badge/🤗_Models-RoboStereo-FFD21E.svg?style=for-the-badge)](https://huggingface.co/Richard-ZZZZZ/RoboStereo)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗_Dataset-Bridge-10b981.svg?style=for-the-badge)](https://huggingface.co/datasets/Richard-ZZZZZ/bridge)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

</div>

<hr>

## 📖 Abstract

Scalable Embodied AI faces fundamental constraints due to prohibitive costs and safety risks of real-world interaction. While Embodied World Models (EWMs) offer promise through imagined rollouts, existing approaches suffer from geometric hallucinations and lack unified optimization frameworks for practical policy improvement. We introduce RoboStereo, a symmetric dual-tower 4D world model that employs bidirectional cross-modal enhancement to ensure spatiotemporal geometric consistency and alleviate physics hallucinations. Building upon this high-fidelity 4D simulator, we present the first unified framework for world-model-based policy optimization: (1) Test-Time Policy Augmentation (TTPA) for pre-execution verification, (2) Imitative-Evolutionary Policy Learning (IEPL) leveraging visual perceptual rewards to learn from expert demonstrations, and (3) Open-Exploration Policy Learning (OEPL) enabling autonomous skill discovery and self-correction. Comprehensive experiments demonstrate RoboStereo achieves state-of-the-art generation quality, with our unified framework delivering >97% average relative improvement on fine-grained manipulation tasks.

---

## 📢 News
* **[2026.03.13]** 📷 Paper released on arXiv.
* **[2026.03.25]** 🎉 Training and inference code released on GitHub.
* **[2026.03.26]** 🚀 Pre-trained models and datasets are now available on Hugging Face!

## 🔭 Repository Scope

| Module | Role | Main Entry |
| --- | --- | --- |
| `cosmos_predict2/` | Base model implementation of the RoboStereo world model | `scripts/train.py`, `examples/action_conditioned.py` |
| `config/tasks/` | Training configurations | `ac_*_{rgb,xyz,dual}/train.json` |
| `checkpoints_files/` | Main repository world-model checkpoints | RGB / XYZ / Dual checkpoints |
| `OEPL/` | OpenVLA-based policy optimization branch | `OEPL/examples/mimicgen/*` |
| `IEPL/` | Cosmos-RFT post-training branch | `IEPL/scripts/cosmos/post_train_cosmos_rft.sh` |
| `TTPA/` | Test-time policy augmentation branch | `TTPA/run_conda_vla_sync.sh`, `TTPA/run_docker_cosmos_sync.sh` |
| `gs_head/` | RGB+XYZ to 3DGS reconstruction utility | `gs_head/scripts/*.py` |


## 📦 Environment Setup

### 1. Host Prerequisites

The current RoboStereo workflow is primarily optimized for Linux + NVIDIA GPU environments. The following prerequisites are recommended:

- NVIDIA Driver `>= 550`
- Docker + NVIDIA Container Toolkit
- Python `>= 3.10` (Required only for local installation or debugging)
- CUDA-capable GPU; dual-tower training and policy branches typically require larger VRAM.

You can quickly verify your host GPU and Docker runtime using the following commands:

```bash
nvidia-smi
sudo docker run --rm --runtime=nvidia --gpus all \
  nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### 2. Clone Repository

```bash
git clone https://github.com/Richard-Zhang-AI/RoboStereo.git
cd RoboStereo
```

### 3. Recommended: Use the Released Docker Image

```bash
sudo docker pull ghcr.io/richard-zhang-ai/robostereo:latest
sudo docker tag ghcr.io/richard-zhang-ai/robostereo:latest robostereo
sudo docker tag ghcr.io/richard-zhang-ai/robostereo:latest wm-vla:with-deps
```

The additional tag `wm-vla:with-deps` is provided to ensure compatibility with legacy launch scripts, such as `IEPL/scripts/docker_start_vla_cosmos.sh`.

### 4. Optional: Build Locally from `Dockerfile`

```bash
sudo docker build -t robostereo --build-arg CUDA_NAME=cu128 .
sudo docker tag robostereo wm-vla:with-deps
```

### 5. Start the Container

```bash
export HF_TOKEN="hf_your_token"

sudo docker run -it \
  --runtime=nvidia \
  --ipc=host \
  --rm \
  --name robostereo \
  --entrypoint /bin/bash \
  -v "$(pwd)":/workspace \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -e HF_TOKEN="$HF_TOKEN" \
  -e PYTHONPATH=/workspace/OEPL/dependencies/openvla-oft:/workspace/OEPL:/workspace/IEPL:/workspace:$PYTHONPATH \
  robostereo
```

## 🧱 Weights and Data

### World-Model Checkpoints

The main repository provides a unified download script that automatically restores two directory structures: `checkpoints_files/` and `OEPL/checkpoint_files/`.

```bash
huggingface-cli login
python download_hf_weights.py 
```

### Datasets

The public datasets utilized in our current workflow can be downloaded from Hugging Face and should be placed within the `datasets/` directory of the repository. We recommend adhering to the following absolute paths:

- `bridge` -> `/nfs/rczhang/code/RoboStereo/datasets/bridge`
- `train_openvla` -> `/nfs/rczhang/code/RoboStereo/datasets/train_openvla`

Please log in to Hugging Face prior to downloading:

```bash
huggingface-cli login
```

#### 1. Download `bridge`

Source: `https://huggingface.co/datasets/Richard-ZZZZZ/bridge`

```bash
mkdir -p /nfs/rczhang/code/RoboStereo/datasets/bridge
huggingface-cli download \
  --repo-type dataset \
  Richard-ZZZZZ/bridge \
  --local-dir /nfs/rczhang/code/RoboStereo/datasets/bridge
```

Upon successful download, the directory structure should be as follows:

```text
/nfs/rczhang/code/RoboStereo/datasets/bridge/
├── annotation/
│   ├── train/
│   ├── val/
│   └── test/
├── videos/
│   ├── train/
│   ├── val/
│   └── test/
├── latent_videos/
│   ├── train/
│   ├── val/
│   └── test/
└── .cache/huggingface/
```

#### 2. Download `train_openvla`

Source: `https://huggingface.co/datasets/Richard-ZZZZZ/RoboStereo-dataset`

```bash
mkdir -p /nfs/rczhang/code/RoboStereo/datasets/train_openvla
huggingface-cli download \
  --repo-type dataset \
  Richard-ZZZZZ/RoboStereo-dataset \
  --local-dir /nfs/rczhang/code/RoboStereo/datasets/train_openvla
```

The local directory structure should be:

```text
/nfs/rczhang/code/RoboStereo/datasets/train_openvla/
├── coffee_128/
│   ├── annotation/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── videos/
│       ├── train/
│       ├── val/
│       └── test/
├── square_128/
│   ├── annotation/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── videos/
│       ├── train/
│       ├── val/
│       └── test/
└── stack_three_128/
    ├── annotation/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── videos/
        ├── train/
        ├── val/
        └── test/
```

## 🚀 Inference

### Single-Tower RGB Inference

```bash
CUDA_VISIBLE_DEVICES=0 python examples/action_conditioned.py \
  -i assets/action_conditioned/basic/coffee/inference_params.json \
  -o outputs/infer_rgb_coffee \
  --checkpoint-path checkpoints_files/coffee_action_conditioned/checkpoints-RGB/model_ema_fp32.pt \
  --experiment ac_reason_embeddings_rectified_flow_2b_256_320
```

To run the Bridge demo, simply replace `coffee` with `bridge` and update the corresponding checkpoint path.

### Single-Tower Pointmap / XYZ Inference

```bash
CUDA_VISIBLE_DEVICES=0 python examples/action_conditioned.py \
  -i assets/action_conditioned/geometry/coffee/inference_params.json \
  -o outputs/infer_xyz_coffee \
  --visual-condition-source geometry \
  --checkpoint-path checkpoints_files/coffee_action_conditioned/checkpoints-XYZ/model_ema_converted.pt \
  --experiment ac_reason_embeddings_rectified_flow_2b_256_320
```

### Dual-Tower RoboStereo Inference

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python cosmos_predict2/_src/predict2/action/inference/inference_dual.py \
  --experiment ac_reason_embeddings_rectified_flow_2b_256_320_dual \
  --load_mode full \
  --ckpt_path checkpoints_files/bridge_action_conditioned/checkpoints-dual/model_ema_converted.pt \
  --sampling_mode dual \
  --basic_root assets/action_conditioned/basic/bridge \
  --geometry_root assets/action_conditioned/geometry/bridge \
  --split test \
  --guidance 0 \
  --num_latent_conditional_frames 1 \
  --save_dir outputs/dual_infer_bridge
```

## 🏋️ Training

### Released Task Configs

| Mode | Bridge | Coffee |
| --- | --- | --- |
| RGB | `config/tasks/ac_bridge_rgb/train.json` | `config/tasks/ac_coffee_rgb/train.json` |
| XYZ | `config/tasks/ac_bridge_xyz/train.json` | `config/tasks/ac_coffee_xyz/train.json` |
| Dual | `config/tasks/ac_bridge_dual/train.json` | `config/tasks/ac_coffee_dual/train.json` |

### Launch Training

The unified entry point for training is:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12341 \
  scripts/train.py \
  --config cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
  --task-config <task-config-json>
```

For instance, to train the Coffee dual-tower model:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12341 \
  scripts/train.py \
  --config cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
  --task-config config/tasks/ac_coffee_dual/train.json
```

## 🧩 Additional Modules

### `gs_head/`

The `gs_head/` directory provides auxiliary scripts for constructing 3D Gaussian Splatting from RGB and XYZ depth maps. By default, it relies on the model weights located in `checkpoints_files/DA3NESTED-GIANT-LARGE/`.

```bash
cd gs_head
python scripts/gs_from_xyz_depth.py --dataset-dir ../datasets/bridge/videos/train/0 --device cuda
python scripts/gs_from_xyz_video.py --dataset-dir ../datasets/bridge/videos/train/0 --device cuda
```


## OEPL

Open-Exploration Policy Learning covers the OpenVLA-based policy optimization branch used for autonomous exploration and self-correction. See [OEPL/README.md](OEPL/README.md) for details.

## IEPL

Imitative-Evolutionary Policy Learning covers the Cosmos-RFT post-training branch for learning from expert demonstrations with perceptual rewards. See [IEPL/README.md](IEPL/README.md) for details.

## TTPA

Test-Time Policy Augmentation covers the pre-execution verification and synchronized inference workflow. See [TTPA/README.md](TTPA/README.md) for details.


## 📚 Citation

If you find RoboStereo useful in your research, please cite the repository as:

```bibtex
@misc{zhang2026robostereo,
  title        = {RoboStereo: Dual-Tower 4D Embodied World Models for Unified Policy Optimization},
  author       = {Ruicheng Zhang and Guangyu Chen and Zunnan Xu and Zihao Liu and Zhizhou Zhong and Mingyang Zhang and Jun Zhou and Xiu Li},
  howpublished = {\url{https://github.com/Richard-Zhang-AI/RoboStereo}},
  note         = {GitHub repository},
  year         = {2026}
}
```

## 🙏 Acknowledgements

RoboStereo builds upon several excellent open-source projects and research codebases. In particular, we thank the authors and maintainers of:

- [NVIDIA Cosmos / Cosmos-Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5), which provides the world-model foundation used in this repository.
- [OpenVLA](https://github.com/openvla/openvla) and [OpenVLA-OFT](https://github.com/moojink/openvla-oft), which provide the vision-language-action policy backbone used by the policy branches.
- [VERL](https://github.com/volcengine/verl), [Robomimic](https://github.com/ARISE-Initiative/robomimic), [VLA-RFT](https://github.com/OpenHelix-Team/VLA-RFT), [WMPO](https://github.com/WM-PO/WMPO), and [MimicGen](https://github.com/NVlabs/mimicgen), which support training, rollout, and simulator-based evaluation in `OEPL/` and `IEPL/`.
- [Depth-Anything-3](https://github.com/ByteDance-Seed/depth-anything-3) and related 3D reconstruction tooling used by `gs_head/`.
