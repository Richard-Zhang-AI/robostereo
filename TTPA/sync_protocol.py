"""
RoboStereo TTPA - Conda-Docker Sync Protocol
Adapted from WMPO/OEPL.

Alternating execution between Conda (VLA simulation) and Docker (Cosmos) via shared directory:

  Conda: generate actions -> write to shared dir -> create conda_ready.flag -> wait for continue/resample
  Docker: detect conda_ready.flag -> read actions -> run Cosmos -> Qwen score ->
          if score >= threshold: create continue.flag
          else: create resample.flag (Conda re-samples VLA)
  Conda: if continue.flag -> execute actions; if resample.flag -> re-run VLA, write again
"""

from __future__ import annotations

import json
import time
from pathlib import Path

# Filenames in shared directory
FRAME_FILE = "frame.png"
ACTIONS_FILE = "actions.json"
META_FILE = "meta.json"
RESULT_FILE = "result.json"  # {"action": "continue"|"resample", "score": float}
CONDA_READY_FLAG = "conda_ready.flag"
CONTINUE_FLAG = "continue.flag"
RESAMPLE_FLAG = "resample.flag"


def wait_for_flag(flag_path: Path, timeout_sec: float = 3600, poll_interval: float = 0.5) -> bool:
    """Poll until flag file appears"""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if flag_path.exists():
            return True
        time.sleep(poll_interval)
    return False


def create_flag(flag_path: Path) -> None:
    flag_path.touch()


def remove_flag(flag_path: Path) -> None:
    if flag_path.exists():
        flag_path.unlink()


def conda_write_and_signal(sync_dir: Path, frame: "np.ndarray", actions: list, meta: dict) -> None:
    """Conda side: write data and signal ready"""
    import imageio.v2 as imageio_v2

    sync_dir.mkdir(parents=True, exist_ok=True)
    imageio_v2.imwrite(str(sync_dir / FRAME_FILE), frame)
    (sync_dir / ACTIONS_FILE).write_text(json.dumps(actions, ensure_ascii=True), encoding="utf-8")
    (sync_dir / META_FILE).write_text(json.dumps(meta, ensure_ascii=True, indent=2), encoding="utf-8")
    create_flag(sync_dir / CONDA_READY_FLAG)


def conda_wait_for_continue(sync_dir: Path, timeout_sec: float = 3600) -> bool:
    """Conda side: wait for Docker continue signal (legacy, no Qwen scoring)"""
    flag_path = sync_dir / CONTINUE_FLAG
    ok = wait_for_flag(flag_path, timeout_sec=timeout_sec)
    if ok:
        remove_flag(flag_path)
    return ok


def conda_wait_for_signal(
    sync_dir: Path, timeout_sec: float = 3600
) -> str | None:
    """
    Conda side: wait for Docker signal (continue or resample).

    Returns:
        "continue" - execute actions in sim
        "resample" - re-run VLA, get new actions, write again
        None - timeout
    """
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if (sync_dir / CONTINUE_FLAG).exists():
            remove_flag(sync_dir / CONTINUE_FLAG)
            return "continue"
        if (sync_dir / RESAMPLE_FLAG).exists():
            remove_flag(sync_dir / RESAMPLE_FLAG)
            return "resample"
        time.sleep(0.5)
    return None


def docker_wait_for_conda(sync_dir: Path, timeout_sec: float = 3600) -> bool:
    """Docker side: wait for Conda ready"""
    return wait_for_flag(sync_dir / CONDA_READY_FLAG, timeout_sec=timeout_sec)


def docker_read_and_clear(sync_dir: Path) -> tuple["np.ndarray", list, dict]:
    """Docker side: read data and clear conda_ready signal"""
    import numpy as np
    from PIL import Image

    frame_path = sync_dir / FRAME_FILE
    actions_path = sync_dir / ACTIONS_FILE
    meta_path = sync_dir / META_FILE

    frame = np.array(Image.open(frame_path).convert("RGB"))
    actions = json.loads(actions_path.read_text(encoding="utf-8"))
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    remove_flag(sync_dir / CONDA_READY_FLAG)
    return frame, actions, meta


def docker_signal_continue(sync_dir: Path) -> None:
    """Docker side: signal continue (execute actions)"""
    create_flag(sync_dir / CONTINUE_FLAG)


def docker_signal_resample(sync_dir: Path, score: float) -> None:
    """Docker side: signal resample (reject, Conda re-samples VLA)"""
    (sync_dir / RESULT_FILE).write_text(
        json.dumps({"action": "resample", "score": score}, ensure_ascii=True),
        encoding="utf-8",
    )
    create_flag(sync_dir / RESAMPLE_FLAG)
