"""Tests for tracker/checkpoint helpers that should work without the real SwanLab backend."""

import sys
from types import SimpleNamespace

from omegaconf import OmegaConf

from src.train.checkpoint import build_run_dir
from src.train.tracker import Tracker


class _FakeSwanLab:
    """Minimal fake backend used to inspect how Tracker initializes SwanLab."""

    def __init__(self):
        self.Image = object
        self.init_kwargs = None
        self.logged = []

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return SimpleNamespace(finish=lambda: None)

    def log(self, *_args, **_kwargs):
        self.logged.append((_args, _kwargs))
        return None


def test_build_run_dir_uses_env_override(monkeypatch):
    """tmux launch scripts should be able to force the exact checkpoint directory."""
    monkeypatch.setenv("CSVIT2_RUN_DIR", "/tmp/csvit2-custom-run")
    assert build_run_dir("ignored description") == "/tmp/csvit2-custom-run"


def test_tracker_passes_experiment_name_to_swanlab(monkeypatch):
    """Tracker should forward the externally chosen run name to SwanLab."""
    fake_swanlab = _FakeSwanLab()
    monkeypatch.setitem(sys.modules, "swanlab", fake_swanlab)

    cfg = OmegaConf.create(
        {
            "TRACKER": {
                "enabled": True,
                "project": "cs-vit2-remake",
                "workspace": "mine268",
                "mode": "cloud",
                "print_to_console": True,
            }
        }
    )
    accelerator = SimpleNamespace(is_main_process=True)

    Tracker(cfg, accelerator, experiment_name="custom-run-name")

    assert fake_swanlab.init_kwargs is not None
    assert fake_swanlab.init_kwargs["experiment_name"] == "custom-run-name"


def test_tracker_logs_scalars_to_swanlab_and_stdout(monkeypatch):
    """Scalar logs should be uploaded and mirrored to console when enabled in config."""
    fake_swanlab = _FakeSwanLab()
    monkeypatch.setitem(sys.modules, "swanlab", fake_swanlab)

    cfg = OmegaConf.create(
        {
            "TRACKER": {
                "enabled": True,
                "project": "cs-vit2-remake",
                "workspace": "mine268",
                "mode": "cloud",
                "print_to_console": True,
            }
        }
    )
    accelerator = SimpleNamespace(is_main_process=True)

    tracker = Tracker(cfg, accelerator, experiment_name="custom-run-name")
    tracker.log_scalars({"loss_total": 1.25}, step=12, split="train")

    assert len(fake_swanlab.logged) == 1
    args, kwargs = fake_swanlab.logged[0]
    assert args[0] == {"train/loss_total": 1.25}
    assert kwargs["step"] == 12
    assert kwargs["print_to_console"] is True
