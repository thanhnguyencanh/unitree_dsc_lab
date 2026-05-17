"""Custom reward terms for stair-climbing (paper §III-C).

Stack the IsaacLab `rough-locomotion` defaults with two stair-specific shaping
terms that proved critical in reproduction:

* `swing_clearance_bonus` — reward foot apex height >= h_step + 5 cm during
  stair-up phase.
* `step_alignment_bonus`  — reward landing inside the d_step window after the
  stair edge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def swing_clearance_bonus(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    clearance_margin: float = 0.05,
) -> torch.Tensor:
    """Reward foot apex height >= h_step + clearance_margin during stair-up."""
    raise NotImplementedError


def step_alignment_bonus(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    window_margin: float = 0.04,
) -> torch.Tensor:
    """Reward foot landing inside the d_step window past the previous stair edge."""
    raise NotImplementedError
