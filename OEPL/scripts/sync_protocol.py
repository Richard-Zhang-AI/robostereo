"""
Conda-Docker 同步协议

通过共享目录实现 Conda（VLA 仿真）与 Docker（Cosmos）的交替执行：

  Conda: 生成动作 → 写入共享目录 → 创建 conda_ready.flag → 等待 continue.flag
  Docker: 检测 conda_ready.flag → 读取动作 → 运行 Cosmos → 创建 continue.flag → 删除 conda_ready.flag
  Conda: 检测 continue.flag → 删除 → 在仿真中执行动作 → 获取下一帧 → 重复
"""

from __future__ import annotations

import json
import time
from pathlib import Path

# 共享目录中的文件名
FRAME_FILE = "frame.png"
ACTIONS_FILE = "actions.json"
META_FILE = "meta.json"
CONDA_READY_FLAG = "conda_ready.flag"
CONTINUE_FLAG = "continue.flag"


def wait_for_flag(flag_path: Path, timeout_sec: float = 3600, poll_interval: float = 0.5) -> bool:
    """轮询等待 flag 文件出现"""
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
    """Conda 端：写入数据并发出就绪信号"""
    import imageio.v2 as imageio_v2

    sync_dir.mkdir(parents=True, exist_ok=True)
    imageio_v2.imwrite(str(sync_dir / FRAME_FILE), frame)
    (sync_dir / ACTIONS_FILE).write_text(json.dumps(actions, ensure_ascii=True), encoding="utf-8")
    (sync_dir / META_FILE).write_text(json.dumps(meta, ensure_ascii=True, indent=2), encoding="utf-8")
    create_flag(sync_dir / CONDA_READY_FLAG)


def conda_wait_for_continue(sync_dir: Path, timeout_sec: float = 3600) -> bool:
    """Conda 端：等待 Docker 的继续信号"""
    flag_path = sync_dir / CONTINUE_FLAG
    ok = wait_for_flag(flag_path, timeout_sec=timeout_sec)
    if ok:
        remove_flag(flag_path)
    return ok


def docker_wait_for_conda(sync_dir: Path, timeout_sec: float = 3600) -> bool:
    """Docker 端：等待 Conda 就绪"""
    return wait_for_flag(sync_dir / CONDA_READY_FLAG, timeout_sec=timeout_sec)


def docker_read_and_clear(sync_dir: Path) -> tuple["np.ndarray", list, dict]:
    """Docker 端：读取数据并清除 conda_ready 信号"""
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
    """Docker 端：发出继续信号"""
    create_flag(sync_dir / CONTINUE_FLAG)
