from pathlib import Path

from hydra import compose, initialize_config_dir


def test_stage1_config_parses():
    config_dir = str((Path(__file__).resolve().parents[1] / "config").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="stage1")
    assert cfg.MODEL.handec.cam_head_type == "patch_uv_rho_multibin"
    assert cfg.MODEL.norm_by_hand is False
    assert "HOT3D" in cfg.DATA.ego_abs_datasets
    assert "MTC" in cfg.DATA.aux_local_datasets


def test_stage2_config_extends_stage1():
    config_dir = str((Path(__file__).resolve().parents[1] / "config").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="stage2")
    assert cfg.MODEL.stage == "stage2"
    assert cfg.MODEL.num_frame == 7
