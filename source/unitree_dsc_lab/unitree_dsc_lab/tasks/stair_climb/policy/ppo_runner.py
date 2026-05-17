"""Three-stage trainer (paper §III-D).

Stage 1: PPO with teacher z_t                  -> logs/stage1/
Stage 2: freeze policy, train BEV student      -> logs/stage2/
         loss = 0.6*CE(class) + L1(h) + L1(d)
Stage 3: unfreeze both, joint fine-tune        -> logs/stage3/
         loss = L_PPO + 1.0 * L_terrain;  LR x 0.3

Wraps `rsl_rl.runners.OnPolicyRunner` and adds the perception side-loss in
stages 2/3.
"""

from __future__ import annotations


class ThreeStagePPORunner:
    """Stage-aware wrapper around rsl_rl's OnPolicyRunner."""

    def __init__(self, env, runner_cfg, stage: int) -> None:
        assert stage in (1, 2, 3), stage
        self.env = env
        self.runner_cfg = runner_cfg
        self.stage = stage

    def learn(self, num_iters: int) -> None:
        raise NotImplementedError
