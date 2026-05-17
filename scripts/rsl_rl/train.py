"""Stage-1 PPO training entry point.

Equivalent to:
    isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
        --task Unitree-G1-23dof-StairClimb-v0 --num_envs 4096 --headless

For stage 2/3 see `scripts/stages/`.
"""

from __future__ import annotations

import argparse

# PYTHON_ARGCOMPLETE_OK
import argcomplete

from cli_args import add_rsl_rl_args


def main():
    parser = argparse.ArgumentParser(description="Train PPO on stair-climbing task.")
    add_rsl_rl_args(parser)
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    # Late imports — Isaac Sim must be started after argparse.
    from isaaclab.app import AppLauncher  # type: ignore

    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    import gymnasium as gym  # noqa: F401
    import unitree_dsc_lab  # noqa: F401  registers the task

    raise NotImplementedError(
        "Wire `RslRlOnPolicyRunnerCfg` + `OnPolicyRunner` (see unitree_rl_lab/scripts/rsl_rl/train.py)."
    )

    simulation_app.close()


if __name__ == "__main__":
    main()
