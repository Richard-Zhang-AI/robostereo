
# OEPL
> Open-Exploration Policy Learning

## 🧭 Overview

The primary components of the Open-Exploration Policy Learning (OEPL) module are outlined below:

| Component | Role |
| --- | --- |
| `examples/mimicgen/*` | Training and evaluation entry points across various tasks and resolutions. |
| `checkpoint_files/` | Checkpoints for SFT policies, reward models, world models, and shim files. |
| `data_files/` | Configurations for MimicGen and the simulator. |
| `reward_model/` | VideoMAE-based reward inference. |
| `scripts/vla_*` | Bridge, rollout, and evaluation scripts connecting OpenVLA with the simulator and Cosmos. |
| `dependencies/` | External dependencies including `openvla-oft`, `robosuite`, `robomimic`, and `mimicgen`. |
| `launch_head.sh`, `launch_worker.sh` | Launch scripts for deploying multi-node Ray clusters. |

## 🚀 Main Training and Evaluation Entry Points

### 1. MimicGen Training Scripts


For example, to launch the `coffee` `P_128` training from the `OEPL/` directory:

```bash
cd OEPL
bash examples/mimicgen/coffee/train_oepl_128.sh
```

Similarly, to run the `coffee` evaluation from the `OEPL/` directory:

```bash
cd OEPL
bash examples/mimicgen/coffee/evaluate.sh
```

### 2. OpenVLA to Cosmos Bridge

This pipeline operates entirely without executing the physical simulator. Instead, it iteratively performs the following autoregressive loop: `current_frame -> VLA action chunk -> Cosmos video chunk -> next frame`.

```bash
cd /workspace/OEPL
CUDA_VISIBLE_DEVICES=0 python scripts/vla_cosmos_bridge.py \
  --vla-path ./checkpoint_files/SFT_models/coffee \
  --unnorm-key coffee_d0_300_demos \
  --task coffee \
  --initial-image ./debug/vla_sim_eval_out/episode_00/frame_inputs/chunk_000_input.png \
  --output-dir ./debug/vla_cosmos_interact_out \
  --num-chunks 25 \
  --cosmos-root /workspace \
  --cosmos-experiment ac_reason_embeddings_rectified_flow_2b_256_320 \
  --cosmos-checkpoint-path /workspace/checkpoints_files/coffee_action_conditioned/checkpoints-RGB/model_ema_fp32.pt \
  --cosmos-config-file cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
  --negative-prompt-file /workspace/assets/action_conditioned/basic/coffee/inference_params.json
```

## 📌 Notes

This repository builds upon and inherits from the [WMPO framework](https://github.com/WM-PO/WMPO).
