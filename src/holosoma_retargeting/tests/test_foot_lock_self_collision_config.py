"""Unit tests for FootLockConfig, SelfCollisionConfig, and OmniRetargeterConfig fields."""
import pytest
from holosoma_retargeting.config_types.retargeters.omniretarget import (
    FootLockConfig,
    OmniRetargeterConfig,
    SelfCollisionConfig,
)


def test_foot_lock_config_defaults():
    cfg = FootLockConfig()
    assert cfg.enable is False
    assert cfg.windows is None
    assert cfg.z_floor == 0.0
    assert cfg.tolerance == pytest.approx(5e-3)


def test_foot_lock_config_custom():
    cfg = FootLockConfig(enable=True, windows={"L_Toe": [(0, 10)]}, z_floor=0.1, tolerance=1e-2)
    assert cfg.enable is True
    assert cfg.windows == {"L_Toe": [(0, 10)]}
    assert cfg.z_floor == pytest.approx(0.1)
    assert cfg.tolerance == pytest.approx(1e-2)


def test_self_collision_config_defaults():
    cfg = SelfCollisionConfig()
    assert cfg.enable is False
    assert cfg.pairs == []
    assert cfg.windows is None
    assert cfg.tolerance == pytest.approx(0.02)


def test_self_collision_config_custom():
    pairs = [("left_elbow_link", "left_knee_link")]
    cfg = SelfCollisionConfig(enable=True, pairs=pairs, windows=[(0, 50)], tolerance=0.05)
    assert cfg.enable is True
    assert cfg.pairs == pairs
    assert cfg.windows == [(0, 50)]
    assert cfg.tolerance == pytest.approx(0.05)


def test_omni_retargeter_config_has_foot_lock_field():
    cfg = OmniRetargeterConfig()
    assert hasattr(cfg, "foot_lock")
    assert isinstance(cfg.foot_lock, FootLockConfig)
    assert cfg.foot_lock.enable is False


def test_omni_retargeter_config_has_self_collision_field():
    cfg = OmniRetargeterConfig()
    assert hasattr(cfg, "self_collision")
    assert isinstance(cfg.self_collision, SelfCollisionConfig)
    assert cfg.self_collision.enable is False


def test_omni_retargeter_config_custom_foot_lock():
    fl = FootLockConfig(enable=True, windows={"L_Toe": [(5, 20)]})
    cfg = OmniRetargeterConfig(foot_lock=fl)
    assert cfg.foot_lock.enable is True
    assert cfg.foot_lock.windows == {"L_Toe": [(5, 20)]}


def test_omni_retargeter_config_custom_self_collision():
    sc = SelfCollisionConfig(enable=True, pairs=[("left_elbow_link", "left_knee_link")], tolerance=0.05)
    cfg = OmniRetargeterConfig(self_collision=sc)
    assert cfg.self_collision.enable is True
    assert cfg.self_collision.pairs == [("left_elbow_link", "left_knee_link")]
    assert cfg.self_collision.tolerance == pytest.approx(0.05)
