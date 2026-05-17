"""Actor-critic that consumes proprioception ⊕ z_t (paper Eq. 1, Fig. 2).

Action: target joint positions offset from default joint config, wrapped by a
PD controller at 50 Hz (Kp ~ 100-200, Kd ~ 2-5, joint-specific from Unitree
spec).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StairClimbActorCritic(nn.Module):
    def __init__(self, num_proprio: int, num_z: int = 4, num_actions: int = 23) -> None:
        super().__init__()
        self.num_proprio = num_proprio
        self.num_z = num_z
        self.num_actions = num_actions
        # TODO: build MLP actor + critic heads consistent with rsl_rl's
        # ActorCritic interface so RslRlOnPolicyRunnerCfg can wrap it.

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
