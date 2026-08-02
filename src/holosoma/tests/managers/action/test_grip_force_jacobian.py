"""Regression test for the analytic wrist Jacobian used by the grip-force controller
(JointPositionActionTerm._wrist_jacobian / _compute_grip_force_bias).

Pure-math check, no simulator required: builds a synthetic copy of the G1 rubber-hand wrist chain
(elbow -> wrist_roll -> wrist_pitch -> wrist_yaw -> fixed offset to the hand contact point, per
main_mesh_collision_rubberhand.urdf) and validates the cross-product Jacobian-transpose formula
against finite differences on the forward-kinematics chain.
"""

import torch

from holosoma.utils.rotations import quat_apply, quat_from_angle_axis, quat_mul

_DTYPE = torch.float64  # central differences need double precision to be numerically tight

OFF_ROLL = torch.tensor([0.100, 0.00188791, -0.010], dtype=_DTYPE)
OFF_PITCH = torch.tensor([0.038, 0.0, 0.0], dtype=_DTYPE)
OFF_YAW = torch.tensor([0.046, 0.0, 0.0], dtype=_DTYPE)
OFF_HAND = torch.tensor([0.0415, 0.003, 0.0], dtype=_DTYPE)

AX_ROLL = torch.tensor([1.0, 0.0, 0.0], dtype=_DTYPE)
AX_PITCH = torch.tensor([0.0, 1.0, 0.0], dtype=_DTYPE)
AX_YAW = torch.tensor([0.0, 0.0, 1.0], dtype=_DTYPE)


def _forward_kinematics(q: torch.Tensor, q_elbow0: torch.Tensor, p_elbow0: torch.Tensor) -> dict:
    """q = (..., 3) [roll, pitch, yaw] -> world (pos, quat) per link + hand contact point."""
    roll, pitch, yaw = q.unbind(-1)

    q_roll_rel = quat_from_angle_axis(roll, AX_ROLL.expand(*q.shape[:-1], 3), w_last=True)
    q_wr = quat_mul(q_elbow0, q_roll_rel, w_last=True)
    p_wr = p_elbow0 + quat_apply(q_elbow0, OFF_ROLL.expand(*q.shape[:-1], 3), w_last=True)

    q_pitch_rel = quat_from_angle_axis(pitch, AX_PITCH.expand(*q.shape[:-1], 3), w_last=True)
    q_wp = quat_mul(q_wr, q_pitch_rel, w_last=True)
    p_wp = p_wr + quat_apply(q_wr, OFF_PITCH.expand(*q.shape[:-1], 3), w_last=True)

    q_yaw_rel = quat_from_angle_axis(yaw, AX_YAW.expand(*q.shape[:-1], 3), w_last=True)
    q_wy = quat_mul(q_wp, q_yaw_rel, w_last=True)
    p_wy = p_wp + quat_apply(q_wp, OFF_YAW.expand(*q.shape[:-1], 3), w_last=True)

    p_hand = p_wy + quat_apply(q_wy, OFF_HAND.expand(*q.shape[:-1], 3), w_last=True)

    return {
        "elbow": (p_elbow0, q_elbow0),
        "wrist_roll": (p_wr, q_wr),
        "wrist_pitch": (p_wp, q_wp),
        "wrist_yaw": (p_wy, q_wy),
        "hand": p_hand,
    }


def _analytic_jacobian(fk: dict) -> torch.Tensor:
    """Same formula as JointPositionActionTerm._wrist_jacobian: J[..., :, k] = axis_k x (p_hand - p_k)."""
    p_hand = fk["hand"]
    p_elbow, q_elbow = fk["elbow"]
    p_roll, q_roll = fk["wrist_roll"]
    p_pitch, q_pitch = fk["wrist_pitch"]
    p_yaw, _q_yaw = fk["wrist_yaw"]

    n = p_hand.shape[:-1]
    a_roll = quat_apply(q_elbow, AX_ROLL.expand(*n, 3), w_last=True)
    a_pitch = quat_apply(q_roll, AX_PITCH.expand(*n, 3), w_last=True)
    a_yaw = quat_apply(q_pitch, AX_YAW.expand(*n, 3), w_last=True)

    jacobian = torch.zeros(*n, 3, 3, dtype=p_hand.dtype)
    jacobian[..., :, 0] = torch.cross(a_roll, p_hand - p_roll, dim=-1)
    jacobian[..., :, 1] = torch.cross(a_pitch, p_hand - p_pitch, dim=-1)
    jacobian[..., :, 2] = torch.cross(a_yaw, p_hand - p_yaw, dim=-1)
    return jacobian


def test_analytic_wrist_jacobian_matches_finite_differences():
    torch.manual_seed(0)
    num_cases = 64
    q_elbow0 = torch.nn.functional.normalize(torch.randn(num_cases, 4, dtype=_DTYPE), dim=-1)
    p_elbow0 = torch.randn(num_cases, 3, dtype=_DTYPE)
    q = (torch.rand(num_cases, 3, dtype=_DTYPE) - 0.5) * 2 * 1.6  # within the wrist joint limits

    fk = _forward_kinematics(q, q_elbow0, p_elbow0)
    jacobian = _analytic_jacobian(fk)

    eps = 1e-6
    jacobian_fd = torch.zeros(num_cases, 3, 3, dtype=_DTYPE)
    for i in range(3):
        dq = torch.zeros_like(q)
        dq[:, i] = eps
        p_plus = _forward_kinematics(q + dq, q_elbow0, p_elbow0)["hand"]
        p_minus = _forward_kinematics(q - dq, q_elbow0, p_elbow0)["hand"]
        jacobian_fd[:, :, i] = (p_plus - p_minus) / (2 * eps)

    max_err = (jacobian - jacobian_fd).abs().max().item()
    assert max_err < 1e-6, f"analytic wrist Jacobian diverges from finite differences: max_err={max_err}"
