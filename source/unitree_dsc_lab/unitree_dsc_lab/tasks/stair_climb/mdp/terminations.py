"""Episode termination terms for the stair-climbing task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def fell_over(env: "ManagerBasedRLEnv", roll_pitch_threshold: float = 0.7) -> torch.Tensor:
    """Terminate when |roll| or |pitch| exceeds threshold (rad)."""
    raise NotImplementedError
