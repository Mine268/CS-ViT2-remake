from __future__ import annotations

"""Hydra entrypoint for training."""

import hydra
from omegaconf import DictConfig

from src.train.engine import train


@hydra.main(version_base=None, config_path="../config", config_name="stage1")
def main(cfg: DictConfig):
    """Run the configured training job."""
    train(cfg)


if __name__ == "__main__":
    main()
