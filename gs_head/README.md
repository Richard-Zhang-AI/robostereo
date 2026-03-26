# GS_Head

Builds 3DGS from RGB and XYZ depth. Based on Depth-Anything-3.

## Model

Place `DA3NESTED-GIANT-LARGE` at `cosmos-predict2.5/checkpoints/DA3NESTED-GIANT-LARGE/` or use `--model`.

## Run

No PYTHONPATH needed. Scripts add `GS_Head/src` to path automatically.

**1. Single frame: xyz → 3DGS**

```bash
python scripts/gs_from_xyz_depth.py --dataset-dir ../datasets/bridge/videos/train/0 --device cuda
```

**2. Multi-frame: xyz → video**

```bash
python scripts/gs_from_xyz_video.py --dataset-dir ../datasets/bridge/videos/train/0 --device cuda
```

Run from `GS_Head/` so relative paths resolve. Defaults: model `../checkpoints/DA3NESTED-GIANT-LARGE`, dataset `../datasets/bridge/videos/train/0`, outputs under `GS_Head/`.
