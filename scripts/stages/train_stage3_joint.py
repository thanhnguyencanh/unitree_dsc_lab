"""Stage 3 — Joint fine-tuning of policy + encoder.

Unfreeze both. Loss = L_PPO + alpha * L_terrain (alpha=1). LR × 0.3.

Usage:
    ./unitree_dsc_lab.sh -p python scripts/stages/train_stage3_joint.py \\
        --resume_policy logs/stage1/Unitree-G1-23dof-StairClimb-v0/model_6000.pt \\
        --resume_encoder logs/stage2/Unitree-G1-23dof-StairClimb-v0/encoder_best.pt \\
        --num_envs 2048 --max_iterations 3000

Note: The env must expose ``get_bev()`` for the perception loss to be active.
Without it, Stage 3 silently falls back to vanilla PPO (a warning is printed).
"""

from __future__ import annotations

import argparse
import sys

# ---- 1. Parse args + launch Isaac Sim ----
parser = argparse.ArgumentParser(description="Stage 3: joint PPO + encoder fine-tuning.")
parser.add_argument("--task", type=str, default="Unitree-G1-23dof-StairClimb-v0")
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--resume_policy", type=str, required=True, help="Stage 1 policy checkpoint.")
parser.add_argument("--resume_encoder", type=str, required=True, help="Stage 2 encoder checkpoint.")
parser.add_argument("--max_iterations", type=int, default=3000)
parser.add_argument("--lr_scale", type=float, default=0.3, help="LR multiplier (paper ×0.3).")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--logdir", type=str, default="logs/stage3")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- 2. Post-launch imports ----
import os  # noqa: E402

import gymnasium as gym  # noqa: E402

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

    log_dir = os.path.join(args_cli.logdir, args_cli.task)
    os.makedirs(log_dir, exist_ok=True)

    # Runner — load Stage 1 policy + Stage 2 encoder
    encoder = BEVStudentEncoder()
    runner = ThreeStagePPORunner(
        env,
        BasePPORunnerCfg().to_dict(),
        encoder,
        log_dir=log_dir,
        device=device,
    )
    # Policy weights from Stage 1
    runner.load(args_cli.resume_policy, load_cfg={"actor": True, "critic": True, "optimizer": True})
    # Encoder weights from Stage 2 (load_cfg=None also loads encoder_state_dict if present)
    runner.load(args_cli.resume_encoder, load_cfg={"actor": False, "critic": False, "optimizer": False})

    runner.learn_stage3(
        num_iterations=args_cli.max_iterations,
        lr_scale=args_cli.lr_scale,
    )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
