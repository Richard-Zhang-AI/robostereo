#!/usr/bin/env python3
"""
RoboStereo TTPA - Qwen VL API scorer for Cosmos-predicted video segments.
Adapted from WMPO/OEPL.

Calls Qwen API to judge if a video segment shows success trend and returns a score.
Used for Test-Time Policy Augmentation: if score >= threshold, continue; else resample VLA.
"""

from __future__ import annotations

import base64
import io
import re
from typing import Optional

import numpy as np


DEFAULT_PROMPT = """Please watch this video segment (composed of multiple frames) and judge whether the robot actions in it show a trend toward task success.

Score from 0 to 10:
- 0-3: Unlikely to succeed, actions are clearly wrong or off-target
- 4-6: Some possibility, but obvious problems exist
- 7-10: Clear success trend, actions are reasonable and goal-oriented

Reply with only a number (integer or decimal 0-10), e.g.: 7.5"""


def _frames_to_base64_list(
    frames: np.ndarray,
    max_frames: int = 8,
) -> list[dict]:
    """Convert video frames to base64 content for Qwen VL API."""
    n_frames = frames.shape[0]
    if n_frames <= max_frames:
        indices = list(range(n_frames))
    else:
        indices = [
            int(i * (n_frames - 1) / (max_frames - 1))
            for i in range(max_frames)
        ]
    content = []
    for i in indices:
        frame = frames[i]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        try:
            import imageio.v2 as imageio_v2
            imageio_v2.imwrite(buf, frame, format="PNG")
        except ImportError:
            from PIL import Image
            img = Image.fromarray(frame)
            img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
    return content


def _parse_score(response: str) -> Optional[float]:
    """Extract numeric score from Qwen response."""
    patterns = [
        r"[sS]core[:\s]+(\d+(?:\.\d+)?)",
        r"^(\d+(?:\.\d+)?)\s*$",
        r"\b(\d+(?:\.\d+)?)\b",
    ]
    for pat in patterns:
        m = re.search(pat, response.strip())
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def score_video_segment(
    frames: np.ndarray,
    *,
    prompt: str = DEFAULT_PROMPT,
    api_key: Optional[str] = None,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    model: str = "qwen-vl-max-latest",
    max_frames: int = 8,
) -> tuple[float, str]:
    """
    Call Qwen VL API to score a video segment.

    Args:
        frames: Video frames (T, H, W, C) uint8 numpy array
        prompt: Prompt for scoring
        api_key: DashScope API key (default: DASHSCOPE_API_KEY env)
        base_url: API base URL
        model: Model name
        max_frames: Max frames to send (sampled evenly)

    Returns:
        (score, raw_response): score in [0, 10], or 0.0 if parse failed
    """
    import os

    key = api_key or os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise ValueError(
            "DASHSCOPE_API_KEY not set. Set env var or pass api_key."
        )

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai package required for Qwen API. Run: pip install openai"
        )

    content = _frames_to_base64_list(frames, max_frames=max_frames)
    content.append({"type": "text", "text": prompt})

    client = OpenAI(api_key=key, base_url=base_url)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": content},
        ],
    )
    raw = completion.choices[0].message.content or ""

    score = _parse_score(raw)
    if score is None:
        return 0.0, raw
    return float(np.clip(score, 0.0, 10.0)), raw
