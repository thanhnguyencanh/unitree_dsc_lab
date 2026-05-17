"""Stage 1 — Pre-train policy with ground-truth z_t (privileged teacher).

Target: success_rate > 0.85 on training stair heights (paper §III-D, Table II).

Usage:
    ./unitree_dsc_lab.sh -p python scripts/stages/train_stage1_policy.py \\
        --task Unitree-G1-23dof-StairClimb-v0 \\
        --num_envs 4096 --max_iterations 6000 --headless

Isaac Sim is launched by AppLauncher before any isaaclab imports.
"""

from __future__ import annotations

import argparse
import sys

# ---- 1. Parse args + launch Isaac Sim (MUST happen before isaaclab imports) ----
parser = argparse.ArgumentParser(description="Stage 1: PPO with privileged teacher z_t.")
parser.add_argument("--task", type=str, default="Unitree-G1-23dof-StairClimb-v0")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=6000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--logdir", type=str, default="logs/stage1")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--checkpoint", type=str, default=None)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- 2. Post-launch imports ----
import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import unitree_dsc_lab  # noqa: E402, F401 — registers gym task

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from unitree_dsc_lab.tasks.stair_climb.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg  # noqa: E402
from unitree_dsc_lab.tasks.stair_climb.perception.encoder import BEVStudentEncoder  # noqa: E402
from unitree_dsc_lab.tasks.stair_climb.policy.ppo_runner import ThreeStagePPORunner  # noqa: E402


def main() -> None:
    device = args_cli.device if hasattr(args_cli, "device") else "cuda:0"

    # Environment
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=BasePPORunnerCfg.clip_actions)

    # Log directory
    log_dir = os.path.join(args_cli.logdir, args_cli.task)
    os.makedirs(log_dir, exist_ok=True)

    # Runner
    encoder = BEVStudentEncoder()
    runner = ThreeStagePPORunner(
        env,
        BasePPORunnerCfg().to_dict(),
        encoder,
        log_dir=log_dir,
        device=device,
    )

    if args_cli.resume and args_cli.checkpoint:
        runner.load(args_cli.checkpoint)

    runner.learn_stage1(args_cli.max_iterations)

    # --- evaluate Stage-1 success rate (target > 0.85) ---
    print("[Stage 1] Evaluating success rate …")
    success_rate = runner.evaluate_success_rate(num_episodes=200)
    print(f"[Stage 1] success_rate = {success_rate:.3f}  (target > 0.85)")
    runner.save(os.path.join(log_dir, "model_final.pt"))
    print(f"[Stage 1] Final checkpoint saved to {log_dir}/model_final.pt")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
