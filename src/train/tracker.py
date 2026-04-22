from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from omegaconf import OmegaConf


class Tracker:
    def __init__(self, cfg, accelerator):
        self.accelerator = accelerator
        self.enabled = bool(cfg.TRACKER.enabled) and accelerator.is_main_process
        self.run = None
        self._swanlab = None
        self._image_cls = None

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
        init_kwargs = {
            "project": cfg.TRACKER.project,
            "config": cfg_dict,
        }
        if cfg.TRACKER.get("workspace") is not None:
            init_kwargs["workspace"] = cfg.TRACKER.workspace
        if cfg.TRACKER.get("mode") is not None:
            init_kwargs["mode"] = cfg.TRACKER.mode
        self.run = swanlab.init(**init_kwargs)

    def _to_scalar(self, value: Any) -> float:
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().item()
        elif hasattr(value, "item"):
            value = value.item()
        return float(value)

    def log_scalars(self, metrics: Dict[str, Any], step: int, split: str):
        if not self.enabled:
            return
        payload = {f"{split}/{key}": self._to_scalar(value) for key, value in metrics.items()}
        self._swanlab.log(payload, step=int(step))

    def log_image(self, name: str, image, step: int, split: str):
        if not self.enabled:
            return
        image_np = np.asarray(image)
        if self._image_cls is not None:
            value = self._image_cls(image_np)
        else:
            value = image_np
        self._swanlab.log({f"{split}/{name}": value}, step=int(step))

    def finish(self):
        if not self.enabled:
            return
        finish_fn = getattr(self.run, "finish", None)
        if callable(finish_fn):
            finish_fn()
