# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Depth Anything 3 Loss Functions

This module implements the loss functions for training Depth Anything 3 (DA3).
According to the DA3 paper, the total loss is:

    L = LD(Ď, D) + LM(Ŕ, M) + LP(Ď ⊙ d + t, P) + β·LC(ĉ, v) + α·Lgrad(Ď, D)

where:
    - LD: Depth loss with confidence weighting
    - LM: Ray loss
    - LP: 3D point cloud reconstruction loss
    - LC: Camera parameter loss (optional, not implemented here)
    - Lgrad: Gradient loss for edge-aware smoothness

Reference: https://arxiv.org/abs/2511.10647
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F
from einops import rearrange


def compute_depth_gradient_loss(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute gradient loss for depth prediction.

    This loss preserves sharp edges while ensuring smoothness in planar regions.

    Args:
        pred_depth: Predicted depth map, shape (..., H, W)
        gt_depth: Ground truth depth map, shape (..., H, W)
        mask: Optional validity mask, shape (..., H, W)

    Returns:
        Gradient loss value (scalar)

    Formula:
        Lgrad(Ď, D) = ||∇xĎ - ∇xD||1 + ||∇yĎ - ∇yD||1
    """
    # Compute horizontal gradient using finite difference
    # Shift right and subtract
    pred_grad_x = pred_depth[..., :, 1:] - pred_depth[..., :, :-1]
    gt_grad_x = gt_depth[..., :, 1:] - gt_depth[..., :, :-1]

    # Compute vertical gradient using finite difference
    # Shift down and subtract
    pred_grad_y = pred_depth[..., 1:, :] - pred_depth[..., :-1, :]
    gt_grad_y = gt_depth[..., 1:, :] - gt_depth[..., :-1, :]

    # Compute L1 difference
    grad_loss_x = torch.abs(pred_grad_x - gt_grad_x)
    grad_loss_y = torch.abs(pred_grad_y - gt_grad_y)

    # Apply mask if provided
    if mask is not None:
        # Mask needs to be cropped to match gradient size
        mask_x = mask[..., :, 1:]
        mask_y = mask[..., 1:, :]
        grad_loss_x = grad_loss_x * mask_x
        grad_loss_y = grad_loss_y * mask_y
        num_valid_x = mask_x.sum() + 1e-8
        num_valid_y = mask_y.sum() + 1e-8
    else:
        num_valid_x = grad_loss_x.numel() + 1e-8
        num_valid_y = grad_loss_y.numel() + 1e-8

    # Average over all valid pixels
    grad_loss = grad_loss_x.sum() / num_valid_x + grad_loss_y.sum() / num_valid_y

    return grad_loss


def compute_depth_loss(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    pred_conf: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    lambda_c: float = 0.1,
) -> torch.Tensor:
    """
    Compute depth loss with confidence weighting.

    Args:
        pred_depth: Predicted depth map, shape (..., H, W)
        gt_depth: Ground truth depth map, shape (..., H, W)
        pred_conf: Predicted depth confidence, shape (..., H, W)
        mask: Optional validity mask, shape (..., H, W)
        lambda_c: Confidence weight parameter (default: 0.1)

    Returns:
        Depth loss value (scalar)

    Formula:
        LD(Ď, D; Dc) = (1/ZΩ) Σp∈Ω mp · (Dc,p · |Ďp - Dp| - λc · log Dc,p)

    where:
        - Dc,p: confidence at pixel p
        - mp: validity mask at pixel p
        - Ďp: predicted depth at pixel p
        - Dp: ground truth depth at pixel p
        - ZΩ: normalization factor (number of valid pixels)
    """
    # Compute L1 difference
    depth_diff = torch.abs(pred_depth - gt_depth)

    # Apply confidence weighting: Dc,p · |Ďp - Dp|
    weighted_diff = pred_conf * depth_diff

    # Add regularization term: -λc · log Dc,p
    # This prevents confidence from becoming too large
    conf_reg = -lambda_c * torch.log(pred_conf + 1e-8)

    # Combined loss per pixel
    loss_per_pixel = weighted_diff + conf_reg

    # Apply validity mask
    if mask is not None:
        loss_per_pixel = loss_per_pixel * mask
        num_valid = mask.sum() + 1e-8
    else:
        num_valid = loss_per_pixel.numel() + 1e-8

    # Average over valid pixels
    depth_loss = loss_per_pixel.sum() / num_valid

    return depth_loss


def compute_ray_loss(
    pred_ray: torch.Tensor,
    gt_ray: torch.Tensor,
    conf: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute ray loss.

    Args:
        pred_ray: Predicted ray map, shape (..., H, W, 6)
                   Last 3 dims are ray origin (t), next 3 are direction (d)
        gt_ray: Ground truth ray map, shape (..., H, W, 6)
        conf: Optional confidence for weighting, shape (..., H, W)
        mask: Optional validity mask, shape (..., H, W)

    Returns:
        Ray loss value (scalar)

    Formula:
        LM(Ŕ, M) = ||Ŕ - M||1

    Note:
        If confidence is provided, the loss is weighted by it.
    """
    # Compute L1 difference
    ray_diff = torch.abs(pred_ray - gt_ray)

    # Apply confidence weighting if provided
    if conf is not None:
        # conf shape: (..., H, W), ray_diff shape: (..., H, W, 6)
        ray_diff = ray_diff * conf[..., None]

    # Average over spatial dimensions and ray channels
    if mask is not None:
        ray_diff = ray_diff * mask[..., None]
        num_valid = mask.sum() * 6 + 1e-8
    else:
        num_valid = ray_diff.numel() + 1e-8

    ray_loss = ray_diff.sum() / num_valid

    return ray_loss


def reconstruct_3d_points_from_depth_ray(
    depth: torch.Tensor,
    ray: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct 3D points from depth and ray maps.

    According to DA3 paper formulation:
        P = t + D · d

    where:
        - t: ray origin (first 3 channels of ray)
        - d: ray direction (last 3 channels of ray)
        - D: depth value

    Args:
        depth: Depth map, shape (..., H, W)
        ray: Ray map, shape (..., H, W, 6)
              ray[..., :3] is origin (t)
              ray[..., 3:] is direction (d)

    Returns:
        3D points in world space, shape (..., H, W, 3)
    """
    # Extract ray origin (t) and direction (d)
    ray_origin = ray[..., :3]  # (..., H, W, 3)
    ray_direction = ray[..., 3:]  # (..., H, W, 3)

    # Reconstruct 3D points: P = t + D * d
    # depth shape: (..., H, W), need to add channel dimension
    points_3d = ray_origin + depth[..., None] * ray_direction

    return points_3d


def compute_pointcloud_loss(
    pred_depth: torch.Tensor,
    pred_ray: torch.Tensor,
    gt_depth: Optional[torch.Tensor] = None,
    gt_ray: Optional[torch.Tensor] = None,
    gt_points: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute 3D point cloud reconstruction loss.

    Args:
        pred_depth: Predicted depth map, shape (..., H, W)
        pred_ray: Predicted ray map, shape (..., H, W, 6)
        gt_depth: Ground truth depth map, shape (..., H, W) [optional, used if gt_points is None]
        gt_ray: Ground truth ray map, shape (..., H, W, 6) [optional, used if gt_points is None]
        gt_points: Ground truth 3D points, shape (..., H, W, 3) [optional]
        mask: Optional validity mask, shape (..., H, W)

    Returns:
        Point cloud loss value (scalar)

    Formula:
        LP(Ď ⊙ d + t, P) = ||(Ď · d + t) - P||1

    where:
        - Ď: predicted depth
        - d: ray direction (last 3 channels of ray)
        - t: ray origin (first 3 channels of ray)
        - P: ground truth 3D points

    The reconstructed points are computed as:
        P_reconstructed = t + Ď · d

    Note:
        If gt_points is not provided, it will be reconstructed from gt_depth and gt_ray
        using the same formula: P_gt = t_gt + D_gt · d_gt
    """
    # Reconstruct 3D points from predicted depth and ray
    pred_points = reconstruct_3d_points_from_depth_ray(pred_depth, pred_ray)

    # If gt_points is not provided, reconstruct from gt_depth and gt_ray
    if gt_points is None:
        if gt_depth is None or gt_ray is None:
            raise ValueError(
                "Either gt_points or both gt_depth and gt_ray must be provided"
            )
        gt_points = reconstruct_3d_points_from_depth_ray(gt_depth, gt_ray)

    # Compute L1 difference between predicted and ground truth points
    point_diff = torch.abs(pred_points - gt_points)

    # Apply mask if provided
    if mask is not None:
        point_diff = point_diff * mask[..., None]
        num_valid = mask.sum() * 3 + 1e-8
    else:
        num_valid = point_diff.numel() + 1e-8

    # Average over all valid points
    point_loss = point_diff.sum() / num_valid

    return point_loss


def compute_da3_loss(
    pred: Dict[str, torch.Tensor],
    gt: Dict[str, torch.Tensor],
    alpha: float = 1.0,
    beta: float = 1.0,
    lambda_c: float = 0.1,
    use_pointcloud_loss: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute the complete DA3 loss.

    This is the main loss function for training Depth Anything 3.

    Args:
        pred: Dictionary containing predictions:
            - 'depth': Predicted depth map, shape (B, N, T, H, W) or (B, N, H, W)
            - 'depth_conf': Predicted depth confidence, shape (..., H, W)
            - 'ray': Predicted ray map, shape (..., H, W, 6)
            - 'ray_conf': Predicted ray confidence, shape (..., H, W)
        gt: Dictionary containing ground truth:
            - 'depth': Ground truth depth map, shape (..., H, W)
            - 'ray': Ground truth ray map, shape (..., H, W, 6)
            - 'points': Ground truth 3D points, shape (..., H, W, 3) [optional]
            - 'mask': Validity mask, shape (..., H, W) [optional]
        alpha: Weight for gradient loss (default: 1.0)
        beta: Weight for camera parameter loss (not used here, default: 1.0)
        lambda_c: Confidence weight for depth loss (default: 0.1)
        use_pointcloud_loss: Whether to compute point cloud loss (default: True)

    Returns:
        Tuple containing:
            - Total loss (scalar)
            - Dictionary of individual loss components

    Formula:
        L = LD(Ď, D) + LM(Ŕ, M) + LP(Ď ⊙ d + t, P) + α·Lgrad(Ď, D)

    where:
        - LD: Depth loss with confidence weighting
        - LM: Ray loss
        - LP: 3D point cloud reconstruction loss
        - Lgrad: Gradient loss for edge-aware smoothness
    """
    # Extract predictions
    pred_depth = pred['depth']
    pred_depth_conf = pred['depth_conf']
    pred_ray = pred['ray']
    pred_ray_conf = pred.get('ray_conf', None)  # Optional

    # Extract ground truth
    gt_depth = gt['depth']
    gt_ray = gt['ray']
    gt_points = gt.get('points', None)
    mask = gt.get('mask', None)

    # Ensure shapes are compatible
    # Handle different input shapes by squeezing batch dimension if needed
    if pred_depth.ndim == 5:  # (B, N, T, H, W)
        # Flatten batch and view dimensions
        B, N, T = pred_depth.shape[:3]
        pred_depth = pred_depth.reshape(B * N * T, *pred_depth.shape[3:])
        pred_depth_conf = pred_depth_conf.reshape(B * N * T, *pred_depth_conf.shape[3:])
        pred_ray = pred_ray.reshape(B * N * T, *pred_ray.shape[3:])
        if pred_ray_conf is not None:
            pred_ray_conf = pred_ray_conf.reshape(B * N * T, *pred_ray_conf.shape[3:])

        gt_depth = gt_depth.reshape(B * N * T, *gt_depth.shape[3:])
        gt_ray = gt_ray.reshape(B * N * T, *gt_ray.shape[3:])
        if gt_points is not None:
            gt_points = gt_points.reshape(B * N * T, *gt_points.shape[3:])
        if mask is not None:
            mask = mask.reshape(B * N * T, *mask.shape[3:])

    # Compute individual loss components
    losses = {}

    # 1. Depth loss LD
    loss_depth = compute_depth_loss(
        pred_depth=pred_depth,
        gt_depth=gt_depth,
        pred_conf=pred_depth_conf,
        mask=mask,
        lambda_c=lambda_c,
    )
    losses['depth'] = loss_depth

    # 2. Ray loss LM
    loss_ray = compute_ray_loss(
        pred_ray=pred_ray,
        gt_ray=gt_ray,
        conf=pred_ray_conf,
        mask=mask,
    )
    losses['ray'] = loss_ray

    # 3. Point cloud loss LP
    # Use gt_points if provided, otherwise reconstruct from gt_depth and gt_ray
    if use_pointcloud_loss:
        loss_pointcloud = compute_pointcloud_loss(
            pred_depth=pred_depth,
            pred_ray=pred_ray,
            gt_depth=gt_depth,
            gt_ray=gt_ray,
            gt_points=gt_points,
            mask=mask,
        )
        losses['pointcloud'] = loss_pointcloud
    else:
        losses['pointcloud'] = torch.tensor(0.0, device=pred_depth.device)

    # 4. Gradient loss Lgrad
    loss_grad = compute_depth_gradient_loss(
        pred_depth=pred_depth,
        gt_depth=gt_depth,
        mask=mask,
    )
    losses['gradient'] = loss_grad

    # Total loss
    total_loss = (
        losses['depth'] +
        losses['ray'] +
        losses['pointcloud'] +
        alpha * losses['gradient']
    )

    return total_loss, losses


# Convenience function for backward compatibility
def da3_loss(
    pred_depth: torch.Tensor,
    pred_depth_conf: torch.Tensor,
    pred_ray: torch.Tensor,
    gt_depth: torch.Tensor,
    gt_ray: torch.Tensor,
    gt_points: Optional[torch.Tensor] = None,
    pred_ray_conf: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    lambda_c: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Convenience function for computing DA3 loss with tensor inputs.

    This is a simplified interface that takes tensors directly instead of dictionaries.

    Args:
        pred_depth: Predicted depth map, shape (..., H, W)
        pred_depth_conf: Predicted depth confidence, shape (..., H, W)
        pred_ray: Predicted ray map, shape (..., H, W, 6)
        gt_depth: Ground truth depth map, shape (..., H, W)
        gt_ray: Ground truth ray map, shape (..., H, W, 6)
        gt_points: Ground truth 3D points, shape (..., H, W, 3) [optional]
        pred_ray_conf: Predicted ray confidence, shape (..., H, W) [optional]
        mask: Validity mask, shape (..., H, W) [optional]
        alpha: Weight for gradient loss (default: 1.0)
        lambda_c: Confidence weight for depth loss (default: 0.1)

    Returns:
        Tuple containing:
            - Total loss (scalar)
            - Dictionary of individual loss components
    """
    pred = {
        'depth': pred_depth,
        'depth_conf': pred_depth_conf,
        'ray': pred_ray,
    }
    if pred_ray_conf is not None:
        pred['ray_conf'] = pred_ray_conf

    gt = {
        'depth': gt_depth,
        'ray': gt_ray,
    }
    if gt_points is not None:
        gt['points'] = gt_points
    if mask is not None:
        gt['mask'] = mask

    return compute_da3_loss(pred, gt, alpha=alpha, lambda_c=lambda_c)
