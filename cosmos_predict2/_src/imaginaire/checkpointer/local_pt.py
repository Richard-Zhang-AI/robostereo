from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from cosmos_predict2._src.imaginaire.checkpointer.base import AbstractCheckpointer
from cosmos_predict2._src.imaginaire.model import ImaginaireModel
from cosmos_predict2._src.imaginaire.utils import distributed, log, misc


class Checkpointer(AbstractCheckpointer):
    """Save weight-only local `.pt` files under `job.path_local`.

    This intentionally does not save DCP shards or trainer/optimizer state.
    """

    def _resolve_save_path(self, path: str) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = Path(self._local_dirname).parent / candidate

        if candidate.suffix == ".pt":
            return candidate
        return candidate / "model.pt"

    def save(
        self,
        model: ImaginaireModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        iteration: int,
    ) -> None:
        del optimizer, scheduler, grad_scaler
        if not distributed.is_rank0():
            return

        if not self.config_checkpoint.save_path:
            log.warning("checkpoint.save_path is empty; skipping local pt save.")
            return

        self.callbacks.on_save_checkpoint_start(model, iteration)

        base_model = model.module if hasattr(model, "module") else model
        state_dict = {"model": base_model.state_dict(), "iteration": iteration}
        state_dict = misc.to(state_dict, device="cpu")
        output_path = self._resolve_save_path(self.config_checkpoint.save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, output_path)
        log.success(f"Saved local weight checkpoint: {output_path}")

        self.callbacks.on_save_checkpoint(model, state_dict=state_dict)
        self.callbacks.on_save_checkpoint_end(model=None, iteration=iteration)
        self.callbacks.on_save_checkpoint_success(iteration=iteration)

    def load(
        self,
        model: ImaginaireModel,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        grad_scaler: Optional[torch.amp.GradScaler] = None,
    ) -> int:
        del optimizer, scheduler, grad_scaler
        if not self.resume_from_checkpoint:
            return 0

        checkpoint_path = self.load_path or None
        if checkpoint_path is None:
            return 0

        self.callbacks.on_load_checkpoint_start(model)
        resolved_path = Path(checkpoint_path)
        if not resolved_path.is_absolute():
            resolved_path = Path(self._local_dirname).parent / resolved_path
        if not resolved_path.exists():
            raise FileNotFoundError(f"File not found (local): {resolved_path}")

        payload = torch.load(resolved_path, map_location="cpu")
        self.callbacks.on_load_checkpoint(model, state_dict=payload)
        state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        if not isinstance(state_dict, dict):
            raise ValueError(f"Unexpected checkpoint format for local pt checkpointer: {type(payload)}")

        if self.load_training_state or self.only_load_scheduler_state:
            log.warning("local_pt checkpointer only restores model weights; training state restore is ignored.")

        log.info(f"Loading model weights from local pt checkpoint: {resolved_path}")
        model.load_state_dict(state_dict, strict=self.strict_resume)
        iteration = payload.get("iteration", 0) if isinstance(payload, dict) else 0
        self.callbacks.on_load_checkpoint_end(model, iteration=iteration, checkpoint_path=str(resolved_path))
        return iteration
