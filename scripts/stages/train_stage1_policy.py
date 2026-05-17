"""Stage 1 — Pre-train policy with ground-truth z_t (privileged teacher).

Target: `success_rate > 0.85` on training stair heights (paper §III-D, Table II).
"""

from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Unitree-G1-23dof-StairClimb-v0")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=6000)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--logdir", type=str, default="logs/stage1")
    args = parser.parse_args()

    raise NotImplementedError(
        "Build env with privileged teacher observation, train with PPO. "
        "Delegate to unitree_dsc_lab.tasks.stair_climb.policy.ppo_runner.ThreeStagePPORunner(stage=1)."
    )


if __name__ == "__main__":
    main()
