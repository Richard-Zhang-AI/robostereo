#!/usr/bin/env python3
"""
Compare Cosmos action-conditioned state distribution with coffee rollout state,
and estimate a rigid alignment (R, t) that maps coffee positions to Cosmos positions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _load_cosmos_states(json_dir: Path) -> np.ndarray:
    all_states: List[np.ndarray] = []
    for path in sorted(json_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        states = np.asarray(data.get("state", []), dtype=np.float32)
        if states.ndim == 2 and states.shape[1] >= 3:
            all_states.append(states)
    if not all_states:
        raise RuntimeError(f"No state arrays found in {json_dir}")
    return np.concatenate(all_states, axis=0)


def _load_coffee_states(jsonl_path: Path) -> np.ndarray:
    states: List[np.ndarray] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            state = np.asarray(row.get("state", []), dtype=np.float32)
            if state.shape[0] >= 3:
                states.append(state)
    if not states:
        raise RuntimeError(f"No state entries found in {jsonl_path}")
    return np.stack(states, axis=0)


def _stats(name: str, xyz: np.ndarray) -> None:
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    mean = xyz.mean(axis=0)
    std = xyz.std(axis=0)
    print(f"{name} xyz min: {mins}")
    print(f"{name} xyz max: {maxs}")
    print(f"{name} xyz mean: {mean}")
    print(f"{name} xyz std: {std}")


def _fit_rigid(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fit rigid transform from a->b using Kabsch: find R,t minimizing ||Ra + t - b||.
    Returns R (3x3), t (3,), and RMSE.
    """
    if a.shape[0] != b.shape[0]:
        n = min(a.shape[0], b.shape[0])
        a = a[:n]
        b = b[:n]
    a_mean = a.mean(axis=0)
    b_mean = b.mean(axis=0)
    a_center = a - a_mean
    b_center = b - b_mean
    h = a_center.T @ b_center
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = b_mean - r @ a_mean
    aligned = (r @ a.T).T + t
    rmse = float(np.sqrt(np.mean(np.sum((aligned - b) ** 2, axis=1))))
    return r, t, rmse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cosmos-json-dir", type=Path, required=True)
    parser.add_argument("--coffee-jsonl", type=Path, required=True)
    args = parser.parse_args()

    cosmos_states = _load_cosmos_states(args.cosmos_json_dir)
    coffee_states = _load_coffee_states(args.coffee_jsonl)

    cosmos_xyz = cosmos_states[:, :3]
    coffee_xyz = coffee_states[:, :3]

    print("== Distribution comparison ==")
    _stats("cosmos", cosmos_xyz)
    _stats("coffee", coffee_xyz)

    # Use equal-length prefix for alignment
    r, t, rmse = _fit_rigid(coffee_xyz, cosmos_xyz)
    print("\n== Rigid alignment (coffee -> cosmos) ==")
    print("R:\n", r)
    print("t:", t)
    print("RMSE:", rmse)


if __name__ == "__main__":
    main()
