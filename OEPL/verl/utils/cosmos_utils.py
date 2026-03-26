import json
import logging
import sys
import io
import contextlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image


class _SuppressCosmosFp8ExtraStateFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if "introduced by TransformerEngine for FP8 in the checkpoint" in message and "_extra_state" in message:
            return False
        return True


class _SuppressCosmosInferenceNoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        noisy_substrings = (
            "DEBUG _get_data_batch_input: extrinsics is None",
            "DEBUG _get_data_batch_input: intrinsics is None",
            "GPU memory usage after getting data_batch",
            "[Memory Optimization] Starting latent sample generation",
        )
        return not any(sub in message for sub in noisy_substrings)


class _FilteredTextStream(io.TextIOBase):
    def __init__(self, wrapped, noisy_substrings):
        self.wrapped = wrapped
        self.noisy_substrings = tuple(noisy_substrings)
        self._buffer = ""

    def write(self, s):
        if not isinstance(s, str):
            s = str(s)
        self._buffer += s
        written = len(s)
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if not any(sub in line for sub in self.noisy_substrings):
                self.wrapped.write(line + "\n")
        return written

    def flush(self):
        if self._buffer:
            if not any(sub in self._buffer for sub in self.noisy_substrings):
                self.wrapped.write(self._buffer)
            self._buffer = ""
        self.wrapped.flush()

    def isatty(self):
        return getattr(self.wrapped, "isatty", lambda: False)()


@contextlib.contextmanager
def suppress_cosmos_inference_output():
    noisy_substrings = (
        "DEBUG _get_data_batch_input: extrinsics is None",
        "DEBUG _get_data_batch_input: intrinsics is None",
        "GPU memory usage after getting data_batch",
        "[Memory Optimization] Starting latent sample generation",
    )
    stdout_stream = _FilteredTextStream(sys.stdout, noisy_substrings)
    stderr_stream = _FilteredTextStream(sys.stderr, noisy_substrings)
    with contextlib.redirect_stdout(stdout_stream), contextlib.redirect_stderr(stderr_stream):
        try:
            yield
        finally:
            stdout_stream.flush()
            stderr_stream.flush()


def suppress_cosmos_noise_logs() -> None:
    logger_filters = {
        "cosmos_predict2._src.imaginaire.utils.checkpointer": _SuppressCosmosFp8ExtraStateFilter,
        "imaginaire.utils.checkpointer": _SuppressCosmosFp8ExtraStateFilter,
        "cosmos_predict2._src.predict2.inference.video2world": _SuppressCosmosInferenceNoiseFilter,
        "predict2.inference.video2world": _SuppressCosmosInferenceNoiseFilter,
    }
    for logger_name, filter_cls in logger_filters.items():
        logger = logging.getLogger(logger_name)
        already_added = any(isinstance(existing, filter_cls) for existing in logger.filters)
        if not already_added:
            logger.addFilter(filter_cls())

    # Some Cosmos code emits these lines at INFO level through module loggers; raise those loggers
    # to WARNING and stop propagation so the worker root logger does not re-emit them.
    noisy_info_loggers = [
        "cosmos_predict2._src.predict2.inference.video2world",
        "predict2.inference.video2world",
    ]
    for logger_name in noisy_info_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.WARNING)
        logger.propagate = False


def add_cosmos_to_path(cosmos_root: str | Path) -> Path:
    cosmos_root = Path(cosmos_root).resolve()
    if str(cosmos_root) not in sys.path:
        sys.path.insert(0, str(cosmos_root))
    return cosmos_root


def read_negative_prompt(negative_prompt_file: str | Path | None, default: str = "") -> str:
    if not negative_prompt_file:
        return default
    path = Path(negative_prompt_file)
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    if isinstance(data, dict):
        return str(data.get("negative_prompt", default) or default)
    return default


def load_cosmos_infer(
    cosmos_root: str | Path,
    experiment: str,
    checkpoint_path: str | None,
    model_key: str | None,
    config_file: str,
):
    suppress_cosmos_noise_logs()
    cosmos_root = add_cosmos_to_path(cosmos_root)
    from cosmos_predict2.config import MODEL_CHECKPOINTS
    from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference

    config_path = Path(config_file)
    if not config_path.is_absolute():
        config_path = cosmos_root / config_path
    if config_path.suffix == ".py":
        if not config_path.exists():
            raise FileNotFoundError(f"Cosmos config file not found: {config_path}")
        config_arg = str(config_path.relative_to(cosmos_root))
    else:
        config_arg = config_file

    if checkpoint_path is None:
        if model_key is None:
            raise ValueError("Either cosmos_checkpoint_path or cosmos_model_key must be provided.")
        checkpoint = MODEL_CHECKPOINTS.get(model_key)
        if checkpoint is None:
            for key, value in MODEL_CHECKPOINTS.items():
                if str(key) == str(model_key):
                    checkpoint = value
                    break
        if checkpoint is None:
            raise KeyError(f"Unknown Cosmos model key: {model_key}")
        checkpoint_path = checkpoint.s3.uri

    return Video2WorldInference(
        experiment_name=experiment,
        ckpt_path=checkpoint_path,
        s3_credential_path="",
        context_parallel_size=1,
        config_file=config_arg,
    )


def pad_actions(actions: np.ndarray, target_len: int) -> np.ndarray:
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    if actions.shape[0] >= target_len:
        return actions[:target_len]
    pad_len = target_len - actions.shape[0]
    pad = np.zeros((pad_len, actions.shape[1]), dtype=actions.dtype)
    return np.concatenate([actions, pad], axis=0)


def adapt_actions_to_cosmos(
    actions_chunk: torch.Tensor | np.ndarray,
    target_chunk_size: int,
    action_scale: float,
    gripper_scale: float,
) -> np.ndarray:
    if isinstance(actions_chunk, torch.Tensor):
        actions_chunk = actions_chunk.detach().cpu().numpy()
    actions_chunk = actions_chunk.astype(np.float32, copy=True)
    actions_chunk = pad_actions(actions_chunk, target_chunk_size)

    if actions_chunk.shape[1] >= 7:
        actions_chunk[:, :6] *= action_scale
        actions_chunk[:, 6] *= gripper_scale
    return actions_chunk


def resize_frame_to_resolution(frame: np.ndarray, resolution: str | None) -> np.ndarray:
    if not resolution or str(resolution).lower() == "none":
        return frame
    height, width = map(int, str(resolution).split(","))
    return np.array(Image.fromarray(frame).resize((width, height), Image.BICUBIC))


def prepare_video_input(current_frame: np.ndarray, num_video_frames: int) -> torch.Tensor:
    frame = torch.from_numpy(current_frame).permute(2, 0, 1).unsqueeze(0).contiguous()
    zeros = torch.zeros_like(frame).repeat(num_video_frames - 1, 1, 1, 1)
    vid_input = torch.cat([frame, zeros], dim=0).to(torch.uint8)
    return vid_input.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()


def postprocess_cosmos_video(video: torch.Tensor) -> np.ndarray:
    video_normalized = (video - (-1)) / 2.0
    video_normalized = torch.clamp(video_normalized, 0, 1)
    return (video_normalized[0] * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
