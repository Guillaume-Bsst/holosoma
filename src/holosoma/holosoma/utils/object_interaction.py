"""Pure math backing the optional object-interaction reward terms.

Deliberately free of any env/simulator handle: these are plain tensor functions, so they can be
unit-tested without a simulator, the same way ``utils/contact_targets.py`` is (see
``tests/ci/test_contact_targets.py``). The thin wrappers that gather the sim/reference tensors live
in ``managers/reward/terms/wbt.py``.
"""

from __future__ import annotations

import torch


def velocity_tracking_reward(vel_ref: torch.Tensor, vel_sim: torch.Tensor, sigma: float) -> torch.Tensor:
    """``exp(-||v_ref - v_sim||^2 / sigma^2)`` over the last dim. Shape (N, 3) -> (N,).

    Same squared-error-in-the-exponent convention as ``object_global_ref_position_error_exp``, so the
    object velocity terms read on the same scale as the object pose terms they sit next to.
    """
    error = torch.sum(torch.square(vel_ref - vel_sim), dim=-1)
    return torch.exp(-error / sigma**2)


def hdmi_contact_reward(
    distance: torch.Tensor,
    force: torch.Tensor,
    gate: torch.Tensor,
    *,
    sigma_pos: float,
    sigma_force: float,
    force_threshold: float,
    max_force_bonus: float,
) -> torch.Tensor:
    """HDMI's interaction reward for a single end effector. All inputs shape (N,) -> (N,).

    Reproduces eq. (contact reward) of HDMI (arXiv:2509.16757, Table I weight 5.0)::

        R_contact = exp(-||p_eef - p_target|| / sigma_pos)
                    * max(exp((||F_contact|| - F_thres) / sigma_frc), 1)

    gated by the binary reference contact indicator ``c_t``: the reward is paid only on frames where
    the reference prescribes contact, and is 0 elsewhere. Note the force factor is FLAT (== 1) below
    the threshold and grows above it -- it is a "press harder" bonus, not a penalty for weak contact;
    the binary character of the term comes entirely from the gate.

    Deviation from the paper: ``max_force_bonus`` caps the force factor. As written the factor
    diverges exponentially with ``||F||``, so a single collision spike (contact forces reach ~2400 N
    on this robot) would swamp every other reward term for that step.

    Off-gate returns 0 rather than 1: this is an additive term, so paying a constant on non-contact
    frames would bonus the (easy) pre-contact phase against the (hard) carry phase -- the same
    attractor-removal argument spelled out in ``object_flat_contact_quality_exp``.

    ``gate`` may be boolean (HDMI's c_t) or a weight in [0, 1]; it multiplies the reward, so a bool
    gate is exactly the on/off behaviour and a ramped activation
    (``utils.contact_schedule.ramp_activation``) fades the term in with the contact.
    """
    proximity = torch.exp(-distance / sigma_pos)
    force_bonus = torch.exp((force - force_threshold) / sigma_force).clamp(min=1.0, max=max_force_bonus)
    reward = proximity * force_bonus
    return reward * gate.to(reward.dtype)
