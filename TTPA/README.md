# TTPA

> Test-Time Policy Augmentation

TTPA is the test-time policy enhancement branch in RoboStereo. Prior to executing simulator actions, it utilizes the RoboStereo/Cosmos world model to generate a short video preview for candidate action chunks. The generated video clip is then evaluated using the Qwen-VL API; if the score is insufficient, resampling is triggered before the actual execution in the simulator.

## 🧭 Pipeline

TTPA employs a dual-process synchronization mechanism that operates separately across a host Conda environment and a Docker container:

1. On the Conda side, the system reads the current observation from the simulator and samples an action chunk via the VLA policy.
2. The Conda side writes the `frame + actions + meta` data to a shared directory, then suspends execution and waits.
3. On the Docker side, the system reads this chunk and generates a short preview video using the Cosmos world model.
4. The Docker side invokes the Qwen-VL API to assign a score ranging from `0 to 10` to the preview video.
5. If the score is `>= threshold`, it sends a `continue` signal to the Conda side; otherwise, it sends a `resample` signal, prompting the VLA to resample the action chunk.

## 📦 Requirements

### Conda Side

The host machine's Conda environment must be capable of running:

- OpenVLA / `openvla-oft`
- `robosuite`, `robomimic`, `mimicgen`
- `TTPA/vla_sim_eval_sync.py`

Typically, the dependency environment from `OEPL` can be directly reused.

### Docker Side

The Docker side must be capable of running RoboStereo/Cosmos inference. It is recommended to directly launch the image provided in the repository root:

```bash
bash IEPL/scripts/docker_start_vla_cosmos.sh
```

If only the main image tag is retained, please execute the following command in the repository root first:

```bash
sudo docker tag robostereo wm-vla:with-deps
```

The Qwen scoring dependency also requires:

```bash
pip install openai
export DASHSCOPE_API_KEY="your-api-key"
```

## 🚀 Quick Start

### 1. Start the Conda Side First

Execute from the repository root:

```bash
cd TTPA
WMPO_ROOT=../OEPL \
SYNC_DIR=../OEPL/debug/vla_cosmos_sync \
OUTPUT_DIR=../OEPL/debug/vla_sim_sync_out \
bash run_conda_vla_sync.sh
```

By default, this script invokes:

- VLA checkpoint: `../OEPL/checkpoint_files/SFT_models/coffee`
- task config: `../OEPL/data_files/core_train_configs/bc_rnn_image_ds_coffee_D0_seed_101.json`
- reset states: `../OEPL/verl/utils/dataset/coffee_d0_states.pkl`

### 2. Start the Docker Side Inside the Container

```bash
cd /workspace/TTPA
export DASHSCOPE_API_KEY="your-api-key"
export QWEN_SCORE_THRESHOLD=6.0
SYNC_DIR=/workspace/OEPL/debug/vla_cosmos_sync \
OUTPUT_DIR=/workspace/OEPL/debug/cosmos_sync_out \
bash run_docker_cosmos_sync.sh \
  --negative-prompt-file /workspace/assets/action_conditioned/basic/coffee/inference_params.json
```

## ⚙️ Key Variables

| Variable | Side | Meaning | Default |
| --- | --- | --- | --- |
| `WMPO_ROOT` | Conda | The legacy environment variable name pointing to the `OEPL/` root directory. | `../OEPL` |
| `OEPL_CHECKPOINT_FILES` | Conda | The root directory for policy checkpoints. | `../OEPL/checkpoint_files` |
| `SYNC_DIR` | Both | The shared directory between Conda and Docker. | `OEPL/debug/vla_cosmos_sync` |
| `OUTPUT_DIR` | Both | Output directories for each respective side. | Conda: `OEPL/debug/vla_sim_sync_out`; Docker: `OEPL/debug/cosmos_sync_out` |
| `COSMOS_CHECKPOINT_PATH` | Docker | RoboStereo/Cosmos checkpoint path. | `/workspace/checkpoints_files/coffee_action_conditioned/checkpoints-RGB/model_ema_fp32.pt` |
| `QWEN_SCORE_THRESHOLD` | Docker | Passing score threshold. | `6.0` |
| `DASHSCOPE_API_KEY` | Docker | Qwen-VL API key. | required |
| `--max-resamples` | Conda | Maximum number of allowed resamples per action chunk. | `5` |

## 🧪 Behavior Notes

- The Qwen scoring logic is located in `qwen_scorer.py`, with the default model name set to `qwen-vl-max-latest`.
- The Docker side sends a short preview video corresponding to each chunk to Qwen, rather than the full rollout video.
- The Conda side physically executes the action chunk in the simulator only after receiving the `continue` signal.
- If `resample` signals are continuously received and the `--max-resamples` limit is reached, the Conda side will abandon further resampling and forcefully execute the current action.

## 📁 Output Structure

By default, TTPA-related outputs are written to two directories under `OEPL/debug/`:

```text
OEPL/debug/vla_sim_sync_out/   # Conda side simulator rollouts
OEPL/debug/cosmos_sync_out/    # Docker side Cosmos previews
```

Shared signal files are located at:

```text
OEPL/debug/vla_cosmos_sync/
```

## 📌 Notes

- Conda and Docker must have access to the identical physical `SYNC_DIR`. If the host and container paths differ, ensure that the volume mount points both sides to the exact same directory.
- A dedicated `scripts/docker_start_ttpa.sh` is not provided in the current repository; it is recommended to reuse `IEPL/scripts/docker_start_vla_cosmos.sh` or a custom equivalent Docker launch command.
- The default negative-prompt path in `run_docker_cosmos_sync.sh` retains the legacy `openvla-coffee` naming convention. For the current repository, it is advised to explicitly pass `--negative-prompt-file /workspace/assets/action_conditioned/basic/coffee/inference_params.json`.
- TTPA is built upon the policy checkpoints and simulator configs of `OEPL`. Before running, please ensure that `OEPL/checkpoint_files/` and `OEPL/data_files/` are fully prepared.
