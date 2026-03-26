#!/usr/bin/env python3
"""
VLA Simulator Rollout Script

给定一张初始图像，使用VLA预测动作，在Robosuite仿真器中执行并生成视频。

用法:
python vla_sim_rollout.py \
  --initial-image /path/to/initial_image.png \
  --ckpt /path/to/vla/checkpoint \
  --task coffee \
  --env-config /path/to/env_config.json \
  --output-dir ./output \
  --num-chunks 25 \
  --chunk-size 8
"""

import argparse
import json
import os
import pickle
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import imageio
import numpy as np
import torch
from PIL import Image

# 设置环境变量
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# 导入 robomimic
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.config import config_factory
import mimicgen.envs.robosuite  # noqa: F401

# 导入 OpenVLA 工具
openvla_path = Path(__file__).resolve().parent.parent / "dependencies" / "openvla-oft"
if str(openvla_path) not in sys.path:
    sys.path.insert(0, str(openvla_path))

from experiments.robot.openvla_utils import (
    get_processor,
    get_vla,
    get_vla_action,
)


def create_env_from_config(env_config_path: str):
    """从配置文件创建仿真环境"""
    ext_cfg = json.load(open(env_config_path, "r"))
    cfg = config_factory(ext_cfg["algo_name"])
    
    with cfg.values_unlocked():
        cfg.update(ext_cfg)
        # 尝试解析数据集路径
        dataset_path = Path(cfg.train.data)
        if not dataset_path.is_absolute():
            candidates = [
                (Path(env_config_path).parent / dataset_path).resolve(),
                (Path(env_config_path).parent.parent / dataset_path).resolve(),
            ]
            for candidate in candidates:
                if candidate.exists():
                    cfg.train.data = str(candidate)
                    break
    cfg.lock()
    
    ObsUtils.initialize_obs_utils_with_config(cfg)
    
    # 创建环境
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=cfg.train.data)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=cfg.train.data,
        all_obs_keys=cfg.all_obs_keys,
        verbose=False,
    )
    
    if cfg.experiment.env is not None:
        env_meta["env_name"] = cfg.experiment.env
        
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=env_meta["env_name"],
        render=False,
        render_offscreen=True,
        use_image_obs=shape_meta["use_images"],
        use_depth_obs=shape_meta["use_depths"],
    )
    
    return EnvUtils.wrap_env_from_config(env, config=cfg), cfg


def extract_image_from_obs(obs: dict) -> np.ndarray:
    """从观察中提取图像"""
    img = obs["agentview_image"]
    
    # [C,H,W] -> [H,W,C]
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    
    # 归一化到 uint8
    if img.dtype != np.uint8:
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
    
    # 单通道扩展到三通道
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    
    return img


def reset_env_to_match_image(env, initial_image_path: str, reset_states_path: str = None):
    """
    重置环境使其匹配给定的初始图像
    
    如果提供了 reset_states_path，从中加载状态；
    否则使用默认的 env.reset()
    """
    if reset_states_path and os.path.exists(reset_states_path):
        print(f"[INFO] Loading reset states from: {reset_states_path}")
        with open(reset_states_path, 'rb') as f:
            reset_states = pickle.load(f)
        # 使用第一个状态
        obs = env.reset_to(reset_states[0])
        print(f"[INFO] Reset to state 0 from pickle file")
    else:
        print(f"[INFO] Using default env.reset()")
        obs = env.reset()
    
    # 执行几步让环境稳定
    for _ in range(10):
        obs, _, _, _ = env.step(np.zeros(7))
    
    return obs


def main():
    parser = argparse.ArgumentParser(description="VLA Simulator Rollout")
    parser.add_argument("--initial-image", type=str, required=True,
                        help="Path to initial image")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to VLA checkpoint")
    parser.add_argument("--task", type=str, required=True,
                        help="Task name (e.g., coffee, square)")
    parser.add_argument("--env-config", type=str, required=True,
                        help="Path to environment config JSON")
    parser.add_argument("--reset-states", type=str, default=None,
                        help="Path to reset states pickle file")
    parser.add_argument("--output-dir", type=str, default="./vla_sim_output",
                        help="Output directory for videos and logs")
    parser.add_argument("--num-chunks", type=int, default=25,
                        help="Number of action chunks to generate")
    parser.add_argument("--chunk-size", type=int, default=8,
                        help="Number of actions per chunk")
    parser.add_argument("--max-steps", type=int, default=256,
                        help="Maximum steps per episode")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--save-fps", type=int, default=20,
                        help="FPS for saved video")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*60}")
    print(f"VLA Simulator Rollout")
    print(f"{'='*60}")
    print(f"Task: {args.task}")
    print(f"Initial Image: {args.initial_image}")
    print(f"VLA Checkpoint: {args.ckpt}")
    print(f"Output Directory: {args.output_dir}")
    print(f"{'='*60}\n")
    
    print(f"[1/5] Loading VLA model...")
    
    # 加载 VLA 模型
    cfg_ns = SimpleNamespace(
        pretrained_checkpoint=args.ckpt,
        center_crop=True,
        num_images_in_input=1,
        use_proprio=False,
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        load_in_8bit=False,
        load_in_4bit=False,
        num_open_loop_steps=args.chunk_size,
        unnorm_key=f"{args.task}_d0_300_demos",
    )
    
    vla = get_vla(cfg_ns).to(device).eval()
    processor = get_processor(cfg_ns)
    print(f"[✓] VLA model loaded successfully")
    
    print(f"\n[2/5] Creating simulation environment...")
    
    # 创建仿真环境
    env, env_cfg = create_env_from_config(args.env_config)
    print(f"[✓] Environment created successfully")
    
    print(f"\n[3/5] Resetting environment...")
    
    # 重置环境
    obs = reset_env_to_match_image(env, args.initial_image, args.reset_states)
    print(f"[✓] Environment reset successfully")
    
    # 准备任务描述
    task_description = args.task
    
    # 记录视频帧
    frames = []
    frames.append(extract_image_from_obs(obs))
    
    # 日志文件
    log_file = output_dir / f"{args.task}_rollout_log.jsonl"
    log_fp = open(log_file, "w", encoding="utf-8")
    
    print(f"\n[4/5] Starting rollout...")
    print(f"Target: {args.num_chunks} chunks, max {args.max_steps} steps")
    print(f"{'-'*60}")
    
    total_steps = 0
    success = False
    
    for chunk_idx in range(args.num_chunks):
        if total_steps >= args.max_steps:
            print(f"\n[INFO] Reached max steps ({args.max_steps})")
            break
        
        # 从当前观察提取图像
        current_image = extract_image_from_obs(obs)
        
        # VLA 预测动作
        policy_obs = {
            "full_image": current_image,
            "task_description": task_description
        }
        
        print(f"\n[Chunk {chunk_idx+1:02d}/{args.num_chunks}] Predicting actions...", end=" ")
        
        actions = get_vla_action(
            cfg_ns, vla, processor, policy_obs, task_description, None, None
        )
        actions = np.asarray(actions)
        
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        
        # 确保动作维度正确
        if actions.shape[1] != 7:
            raise RuntimeError(f"Expected action dim 7, got {actions.shape}")
        
        # Pad/trim 到 chunk_size
        if actions.shape[0] > args.chunk_size:
            actions = actions[:args.chunk_size]
        elif actions.shape[0] < args.chunk_size:
            pad = np.zeros((args.chunk_size - actions.shape[0], 7), dtype=actions.dtype)
            actions = np.concatenate([actions, pad], axis=0)
        
        print(f"✓")
        print(f"         Executing in simulator...", end=" ")
        
        # 在仿真器中执行动作
        chunk_start_step = total_steps
        for step_idx, action in enumerate(actions):
            if total_steps >= args.max_steps:
                break
            
            # 记录当前状态
            log_entry = {
                "chunk": chunk_idx,
                "step": total_steps,
                "action": action.tolist(),
            }
            log_fp.write(json.dumps(log_entry) + "\n")
            log_fp.flush()
            
            # 执行动作
            obs, reward, done, info = env.step(action)
            
            # 记录帧
            frames.append(extract_image_from_obs(obs))
            
            total_steps += 1
            
            # 检查是否成功
            if reward > 0.0 or done:
                if done:
                    success = True
                    print(f"✓ [SUCCESS at step {total_steps}!]")
                break
        
        if not (success or done):
            print(f"✓ (steps {chunk_start_step}-{total_steps})")
        
        if success or done:
            break
    
    log_fp.close()
    
    print(f"\n{'-'*60}")
    print(f"[5/5] Saving results...")
    
    # 保存视频
    video_path = output_dir / f"{args.task}_vla_sim_rollout.mp4"
    print(f"Saving video ({len(frames)} frames)...", end=" ")
    imageio.mimsave(str(video_path), frames, fps=args.save_fps)
    print(f"✓")
    
    # 保存摘要
    summary = {
        "task": args.task,
        "total_steps": total_steps,
        "num_chunks_executed": chunk_idx + 1,
        "success": success,
        "initial_image": args.initial_image,
        "checkpoint": args.ckpt,
        "video_path": str(video_path),
        "log_path": str(log_file),
    }
    
    summary_path = output_dir / f"{args.task}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Rollout Completed!")
    print(f"{'='*60}")
    print(f"Total Steps:    {total_steps}")
    print(f"Chunks Used:    {chunk_idx + 1}/{args.num_chunks}")
    print(f"Success:        {success}")
    print(f"Video:          {video_path}")
    print(f"Log:            {log_file}")
    print(f"Summary:        {summary_path}")
    print(f"{'='*60}\n")
    
    # 清理环境
    try:
        if hasattr(env, 'close'):
            env.close()
        if hasattr(env, 'env') and hasattr(env.env, 'close'):
            env.env.close()
    except Exception as e:
        print(f"[WARN] Error closing environment: {e}")


if __name__ == "__main__":
    main()

