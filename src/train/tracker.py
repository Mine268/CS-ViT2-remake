from __future__ import annotations

"""Thin experiment-tracking wrapper that keeps logging concerns out of the training loop."""

from typing import Any, Dict, Optional

import numpy as np
from omegaconf import OmegaConf


class Tracker:
    """Thin SwanLab wrapper that keeps logging side effects on the main process only."""

    def __init__(self, cfg, accelerator, experiment_name: Optional[str] = None):
        """
        Initialize the tracker on the main process only.

        Args:
            cfg: Hydra config containing `TRACKER.*` settings.
            accelerator: Accelerate runtime used to determine whether this is the main process.
            experiment_name: Optional explicit run name shared with checkpoint/tmux naming.
        """
        self.accelerator = accelerator
        self.enabled = bool(cfg.TRACKER.enabled) and accelerator.is_main_process
        self.run = None
        self._swanlab = None
        self._image_cls = None
        self._print_to_console = bool(cfg.TRACKER.get("print_to_console", True))

        if not self.enabled:
            return

        try:
            import swanlab  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "TRACKER.enabled=true but swanlab is not installed. Run `uv sync` first."
            ) from exc

        self._swanlab = swanlab
        self._image_cls = getattr(swanlab, "Image", None)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        # Always persist the fully resolved Hydra config so remote runs are reproducible.
        init_kwargs = {
            "project": cfg.TRACKER.project,
            "config": cfg_dict,
        }
        if experiment_name is not None:
            init_kwargs["experiment_name"] = experiment_name
        # Only forward optional keys when the config sets them, keeping SwanLab defaults intact.
        if cfg.TRACKER.get("workspace") is not None:
            init_kwargs["workspace"] = cfg.TRACKER.workspace
        if cfg.TRACKER.get("mode") is not None:
            init_kwargs["mode"] = cfg.TRACKER.mode
        self.run = swanlab.init(**init_kwargs)

    def _to_scalar(self, value: Any) -> float:
        """Convert tensors/scalar-like values into plain Python floats for tracker backends."""
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().item()
        elif hasattr(value, "item"):
            value = value.item()
        return float(value)

    def log_scalars(self, metrics: Dict[str, Any], step: int, split: str):
        """Log scalar metrics to SwanLab and optionally mirror them to local stdout."""
        if not self.enabled:
            return
        payload = {f"{split}/{key}": self._to_scalar(value) for key, value in metrics.items()}
        self._swanlab.log(
            payload,
            step=int(step),
            print_to_console=self._print_to_console,
        )

    def log_image(self, name: str, image, step: int, split: str):
        """Log one visualization image, using SwanLab's wrapper when available."""
        if not self.enabled:
            return
        image_np = np.asarray(image)
        # Prefer SwanLab's image wrapper when available, but keep a plain-array fallback.
        if self._image_cls is not None:
            value = self._image_cls(image_np)
        else:
            value = image_np
        self._swanlab.log({f"{split}/{name}": value}, step=int(step))

    def finish(self):
        """Close the tracker run cleanly if the backend exposes a finish hook."""
        if not self.enabled:
            return
        finish_fn = getattr(self.run, "finish", None)
        if callable(finish_fn):
            finish_fn()
