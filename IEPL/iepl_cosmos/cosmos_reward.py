"""
Reward functions for Cosmos-based world model rollouts in VLA-RFT.

Provides two reward modes:
  1. Image reconstruction reward (MAE + LPIPS) — same spirit as VLA-RFT
  2. VideoMAE success classification reward — same spirit as WMPO
"""

import torch
import torch.nn.functional as F
import numpy as np
from verl import DataProto


def compute_image_recon_reward(
    predicted_frames: torch.Tensor,
    gt_frames: torch.Tensor,
    lpips_fn=None,
    reward_fn_type: str = 'mae',
    loss_weight: dict = None,
):
    """
    Compute image reconstruction reward between predicted and ground-truth frames.
    
    Args:
        predicted_frames: (B, T, H, W, C) uint8 tensor [0,255]
        gt_frames: (B, T, H, W, C) uint8 tensor [0,255]
        lpips_fn: callable that takes (real, pred) in [-1,1] and returns per-sample loss
        reward_fn_type: 'mae' or 'mse'
        loss_weight: dict with 'mae'/'mse' and 'lpips' weights
        
    Returns:
        reward: (B,) tensor, higher is better
        metrics: dict of logged metrics
    """
    if loss_weight is None:
        loss_weight = {'mae': 1.0, 'lpips': 1.0}

    pred = predicted_frames.float() / 255.0
    real = gt_frames.float() / 255.0

    if pred.dim() == 5 and pred.shape[-1] == 3:
        pred = pred.permute(0, 1, 4, 2, 3)
        real = real.permute(0, 1, 4, 2, 3)

    pred_last = pred[:, -1].clamp(0, 1)
    real_last = real[:, -1].clamp(0, 1)

    if reward_fn_type == 'mae':
        recon_loss = torch.mean(torch.abs(real_last - pred_last), dim=(1, 2, 3))
    elif reward_fn_type == 'mse':
        recon_loss = torch.mean((real_last - pred_last) ** 2, dim=(1, 2, 3))
    else:
        raise ValueError(f"Unknown reward_fn_type: {reward_fn_type}")

    perceptual_loss = torch.zeros_like(recon_loss)
    if lpips_fn is not None and loss_weight.get('lpips', 0) > 0:
        perceptual_loss = lpips_fn(
            real_last * 2 - 1.0,
            pred_last * 2 - 1.0,
        ).mean(dim=(1, 2, 3))

    recon_w = loss_weight.get(reward_fn_type, 1.0)
    lpips_w = loss_weight.get('lpips', 1.0)
    total_loss = recon_w * recon_loss + lpips_w * perceptual_loss
    reward = -total_loss

    metrics = {
        'critic/recon_loss/mean': recon_loss.mean().item(),
        'critic/perceptual_loss/mean': perceptual_loss.mean().item(),
        'critic/total_loss/mean': total_loss.mean().item(),
    }
    return reward, metrics


def compute_trajectory_recon_reward(
    predicted_frames: torch.Tensor,
    gt_frames: torch.Tensor,
    gt_lengths: torch.Tensor,
    lpips_fn=None,
    reward_fn_type: str = 'mae',
    loss_weight: dict | None = None,
    resize_hw: tuple[int, int] = (224, 224),
    compute_lpips: bool = True,
):
    """
    Compute trajectory-level MAE/LPIPS reward between predicted and GT frames.

    Args:
        predicted_frames: (B, T, H, W, C) uint8 tensor [0,255]
        gt_frames: (B, T, H, W, C) uint8 tensor [0,255]
        gt_lengths: (B,) lengths of valid GT frames
        lpips_fn: callable returning LPIPS loss
        reward_fn_type: 'mae' or 'mse'
        loss_weight: dict with 'mae'/'mse' and 'lpips' weights
        resize_hw: target (H, W) for both MAE/LPIPS
        compute_lpips: whether to compute LPIPS this step

    Returns:
        reward: (B,) tensor, higher is better
        metrics: dict of logged metrics
    """
    if loss_weight is None:
        loss_weight = {'mae': 1.0, 'lpips': 1.0}

    pred = predicted_frames.float() / 255.0
    real = gt_frames.float() / 255.0

    # Debug: raw diff stats before alignment
    try:
        diff = (predicted_frames.float() - gt_frames.float()).abs()
        print(f"[DEBUG][Diff] raw diff mean={diff.mean().item():.6f} max={diff.max().item():.6f}")
    except Exception as e:
        print(f"[DEBUG][Diff] failed to compute raw diff: {e}")

    if pred.dim() == 5 and pred.shape[-1] == 3:
        pred = pred.permute(0, 1, 4, 2, 3)
        real = real.permute(0, 1, 4, 2, 3)

    # Align lengths and drop the init frame (frame 0).
    max_len = min(pred.shape[1], real.shape[1], int(gt_lengths.max().item()))
    if max_len <= 1:
        reward = torch.zeros(pred.shape[0], device=pred.device)
        metrics = {
            'critic/recon_loss/mean': 0.0,
            'critic/perceptual_loss/mean': 0.0,
            'critic/total_loss/mean': 0.0,
        }
        return reward, metrics

    pred = pred[:, 1:max_len]
    real = real[:, 1:max_len]

    b, t, c, h_pred, w_pred = pred.shape
    _, _, _, h_real, w_real = real.shape

    pred = pred.reshape(b * t, c, h_pred, w_pred)
    real = real.reshape(b * t, c, h_real, w_real)

    pred = F.interpolate(pred, size=resize_hw, mode='bilinear', align_corners=False)
    real = F.interpolate(real, size=resize_hw, mode='bilinear', align_corners=False)

    pred = pred.reshape(b, t, c, resize_hw[0], resize_hw[1])
    real = real.reshape(b, t, c, resize_hw[0], resize_hw[1])

    valid_len = torch.clamp(gt_lengths, max=max_len).to(pred.device)
    valid_counts = torch.clamp(valid_len - 1, min=1).to(pred.device)
    frame_idx = torch.arange(t, device=pred.device).unsqueeze(0)
    mask = frame_idx < valid_counts.unsqueeze(1)
    mask_f = mask.float()

    if reward_fn_type == 'mae':
        per_frame = torch.mean(torch.abs(real - pred), dim=(2, 3, 4))
    elif reward_fn_type == 'mse':
        per_frame = torch.mean((real - pred) ** 2, dim=(2, 3, 4))
    else:
        raise ValueError(f"Unknown reward_fn_type: {reward_fn_type}")
    recon_loss = (per_frame * mask_f).sum(dim=1) / valid_counts

    perceptual_loss = torch.zeros_like(recon_loss)
    if lpips_fn is not None and loss_weight.get('lpips', 0) > 0 and compute_lpips:
        pred_lp = pred * 2.0 - 1.0
        real_lp = real * 2.0 - 1.0
        pred_lp = pred_lp.reshape(b * t, c, resize_hw[0], resize_hw[1])
        real_lp = real_lp.reshape(b * t, c, resize_hw[0], resize_hw[1])
        with torch.no_grad():
            lp = lpips_fn(real_lp, pred_lp)
        lp = lp.reshape(b, t)
        perceptual_loss = (lp * mask_f).sum(dim=1) / valid_counts

    recon_w = loss_weight.get(reward_fn_type, 1.0)
    lpips_w = loss_weight.get('lpips', 1.0)
    total_loss = recon_w * recon_loss + lpips_w * perceptual_loss
    reward = -total_loss

    metrics = {
        'critic/recon_loss/mean': recon_loss.mean().item(),
        'critic/perceptual_loss/mean': perceptual_loss.mean().item(),
        'critic/total_loss/mean': total_loss.mean().item(),
    }
    return reward, metrics


def compute_videomae_reward(
    success_prob: torch.Tensor,
    complete: torch.Tensor,
):
    """
    Compute reward from VideoMAE success classification.
    
    Args:
        success_prob: (B,) tensor of success probabilities
        complete: (B,) binary tensor (1=success, 0=failure)
        
    Returns:
        reward: (B,) tensor
        metrics: dict
    """
    reward = complete.float()
    metrics = {
        'critic/success_prob/mean': success_prob.mean().item(),
        'critic/complete_rate': complete.float().mean().item(),
    }
    return reward, metrics
