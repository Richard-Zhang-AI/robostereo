#!/usr/bin/env python3
"""Simple visualizer for a single depth .npy file."""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a depth .npy file.")
    parser.add_argument(
        "npy_path",
        nargs="?",
        default=("/nfs/rczhang/code/Depth_2_XYZ/image/0_head_da3/depth.npy"
        ),
        help="Path to the depth .npy file.",
    )
    parser.add_argument("--out", help="Output image path (png). Default: <npy>.png")
    parser.add_argument("--cmap", default="magma", help="Matplotlib colormap.")
    parser.add_argument("--show", action="store_true", help="Display the image window.")
    args = parser.parse_args()

    npy_path = Path(args.npy_path)
    depth = np.load(npy_path)

    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]

    depth = depth.astype(np.float32)

    out_path = Path(args.out) if args.out else npy_path.with_suffix(".png")
    plt.imsave(out_path, depth, cmap=args.cmap)

    if args.show:
        plt.imshow(depth, cmap=args.cmap)
        plt.axis("off")
        plt.show()

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
