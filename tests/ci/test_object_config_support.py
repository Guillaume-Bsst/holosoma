from holosoma.config_types.robot import ObjectConfig


def test_defaut_support_absent():
    c = ObjectConfig()
    assert c.support_urdf_path is None
    assert c.support_pos == (0.0, 0.0, 0.0)
    assert c.support_rot == (1.0, 0.0, 0.0, 0.0)


def test_support_renseigne():
    c = ObjectConfig(object_urdf_path="a.urdf",
                     support_urdf_path="s.urdf",
                     support_pos=(-1.583, 0.417, 0.373),
                     support_rot=(0.608, 0.0, 0.0, -0.794))
    assert c.support_urdf_path == "s.urdf"
    assert c.support_pos == (-1.583, 0.417, 0.373)
    assert c.support_rot == (0.608, 0.0, 0.0, -0.794)
