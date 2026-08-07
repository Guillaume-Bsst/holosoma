"""Configuration types for reward manager."""

from __future__ import annotations

from dataclasses import field
from typing import Any

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class RewardTermCfg:
    """Configuration for a single reward term."""

    func: str
    """Import path to the reward function or class"""
    """(e.g. ``holosoma.managers.reward.terms.locomotion:tracking_lin_vel``)."""

    params: dict[str, Any] = field(default_factory=dict)
    """Additional parameters forwarded to the reward term."""

    weight: float = 1.0
    """Weight applied to the reward term (manager multiplies by ``dt``)."""

    tags: list[str] = field(default_factory=list)
    """Tags for categorizing reward terms (e.g., ["penalty", "tracking"])."""

    achievable_gate: str | None = None
    """Optional ``"module:function"`` returning a per-env float mask (1 = this term can pay at this
    frame, 0 = it is structurally zero here), used ONLY for the ``reward/achievable`` diagnostic --
    it never changes the reward itself.

    Terms that are 0 unless the reference is in contact make the reward budget vary along the clip,
    so the same scalar reward means different things in different phases. Declaring the gate lets
    the manager log the ceiling that was actually reachable alongside what was earned. ``None``
    means "always reachable", which is right for every unconditional tracking term."""


@dataclass(frozen=True)
class RewardManagerCfg:
    """Configuration for the reward manager."""

    terms: dict[str, RewardTermCfg] = field(default_factory=dict)
    """Mapping of reward term name to configuration."""

    only_positive_rewards: bool = False
    """If ``True``, clip the total reward to be non-negative."""
