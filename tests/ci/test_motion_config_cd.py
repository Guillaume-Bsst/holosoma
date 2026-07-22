from holosoma.config_types.command import MotionConfig


def test_motion_config_cd_defaults():
    cfg = MotionConfig(
        motion_file="x.npz",
        body_name_ref=["torso_link"],
        body_names_to_track=["pelvis"],
    )
    assert cfg.beta_scale == 0.1
    assert cfg.hand_body_names == ["left_wrist_yaw_link", "right_wrist_yaw_link"]
