"""Play / visualize a trained checkpoint in Isaac Sim."""

from __future__ import annotations

import argparse

# PYTHON_ARGCOMPLETE_OK
import argcomplete

from cli_args import add_rsl_rl_args


def main():
    parser = argparse.ArgumentParser(description="Play a trained checkpoint.")
    add_rsl_rl_args(parser)
    parser.add_argument("--num_play_envs", type=int, default=16)
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    from isaaclab.app import AppLauncher  # type: ignore

    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    import unitree_dsc_lab  # noqa: F401  registers the task

    raise NotImplementedError(
        "Load checkpoint via `RslRlOnPolicyRunnerCfg.load_run` and step the env."
    )

    simulation_app.close()


if __name__ == "__main__":
    main()
