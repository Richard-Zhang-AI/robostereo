#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Example script to load and visualize extracted features.

Usage:
    python examples/visualize_features.py --feature-path outputs/robot_multiview-agibot/.../video_00_features.pt
"""

import argparse
import torch
import matplotlib.pyplot as plt
from pathlib import Path


def load_features(feature_path):
    """Load features from the saved file."""
    data = torch.load(feature_path)
    return data['features'], data['features_spatial'], data['step_info']


def visualize_feature_map(feature_spatial, layer_idx=0, channel_idx=0, frame_idx=0):
    """
    Visualize a single feature map.

    Args:
        feature_spatial: Tensor with shape [B, D, T, H, W]
        layer_idx: Which layer's feature to visualize
        channel_idx: Which channel to visualize
        frame_idx: Which frame to visualize
    """
    # Take the first batch element
    feat = feature_spatial[0]  # [D, T, H, W]

    # Select specific channel and frame
    feat_map = feat[channel_idx, frame_idx].numpy()  # [H, W]

    plt.figure(figsize=(10, 8))
    plt.imshow(feat_map, cmap='viridis')
    plt.colorbar()
    plt.title(f'Feature Map - Layer {layer_idx}, Channel {channel_idx}, Frame {frame_idx}')
    plt.xlabel('Width')
    plt.ylabel('Height')
    return plt.gcf()


def visualize_all_channels_for_frame(feature_spatial, layer_idx=0, frame_idx=0, max_channels=64):
    """
    Visualize multiple channels of a feature map in a grid.

    Args:
        feature_spatial: Tensor with shape [B, D, T, H, W]
        layer_idx: Which layer's feature to visualize
        frame_idx: Which frame to visualize
        max_channels: Maximum number of channels to display
    """
    # Take the first batch element
    feat = feature_spatial[0]  # [D, T, H, W]

    D = feat.shape[0]
    num_channels = min(D, max_channels)

    # Calculate grid size
    grid_size = int(num_channels ** 0.5) + 1

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()

    for i in range(num_channels):
        feat_map = feat[i, frame_idx].numpy()
        axes[i].imshow(feat_map, cmap='viridis')
        axes[i].set_title(f'Ch {i}')
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(num_channels, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'Feature Maps - Layer {layer_idx}, Frame {frame_idx}')
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize extracted features')
    parser.add_argument('--feature-path', type=str, required=True,
                        help='Path to the saved features file')
    parser.add_argument('--output-dir', type=str, default='feature_visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--timestep', type=int, default=None,
                        help='Which timestep to visualize (default: all)')
    args = parser.parse_args()

    # Load features
    print(f"Loading features from {args.feature_path}")
    features, features_spatial, step_info = load_features(args.feature_path)

    print(f"\nLoaded {len(features)} timesteps")
    for i, info in enumerate(step_info):
        print(f"  Timestep {i}: step_idx={info['step_idx']}, timestep={info['timestep']}")
        if 'feature_shape' in info:
            T, H, W = info['feature_shape']
            print(f"    Feature shape: T={T}, H={H}, W={W}")
            print(f"    Number of layers: {len(features_spatial[i])}")
            for j, feat in enumerate(features_spatial[i]):
                print(f"      Layer {info['layer_ids'][j]}: {feat.shape}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which timesteps to visualize
    if args.timestep is not None:
        timesteps_to_vis = [args.timestep]
    else:
        timesteps_to_vis = range(len(features))

    # Visualize each timestep
    for ts_idx in timesteps_to_vis:
        if ts_idx >= len(features_spatial):
            print(f"Warning: timestep {ts_idx} out of range, skipping")
            continue

        info = step_info[ts_idx]
        print(f"\nVisualizing timestep {ts_idx} (step_idx={info['step_idx']})...")

        # Visualize each layer
        for layer_i, feat_spatial in enumerate(features_spatial[ts_idx]):
            layer_id = info['layer_ids'][layer_i]

            # Visualize first frame, first channel
            fig = visualize_feature_map(feat_spatial, layer_idx=layer_id, channel_idx=0, frame_idx=0)
            save_path = output_dir / f"timestep_{ts_idx}_layer_{layer_id}_ch0_frame0.png"
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: {save_path}")

            # Visualize all channels for first frame (up to 64 channels)
            if feat_spatial.shape[1] <= 64:
                fig = visualize_all_channels_for_frame(feat_spatial, layer_idx=layer_id, frame_idx=0)
                save_path = output_dir / f"timestep_{ts_idx}_layer_{layer_id}_allch_frame0.png"
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  Saved: {save_path}")

    print(f"\nDone! Visualizations saved to {output_dir}")


if __name__ == '__main__':
    main()
