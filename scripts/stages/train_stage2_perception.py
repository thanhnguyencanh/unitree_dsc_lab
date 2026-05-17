"""Stage 2 — Train BEV student encoder under teacher supervision.

Freezes the policy from Stage 1, rolls out, stores (BEV, z_t_gt) pairs, trains
the CNN until: MAE(h) < 1 cm, MAE(d) < 1 cm, class_acc > 99 % (Table II).

Usage:
    ./unitree_dsc_lab.sh -p python scripts/stages/train_stage2_perception.py \\
        --policy_ckpt logs/stage1/Unitree-G1-23dof-StairClimb-v0/model_6000.pt \\
        --num_envs 1024 --epochs 50 --batch_size 256

Note: The env must expose ``get_bev() -> Tensor(num_envs, 6, 60, 60)`` for
the BEV rollout collection to work.  Implement a simulated depth sensor or
override RslRlVecEnvWrapper to provide this method.
"""

from __future__ import annotations

import argparse
import sys

# ---- 1. Parse args + launch Isaac Sim ----
parser = argparse.ArgumentParser(description="Stage 2: supervised BEV encoder training.")
parser.add_argument("--task", type=str, default="Unitree-G1-23dof-StairClimb-v0")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--policy_ckpt", type=str, required=True, help="Path to Stage 1 checkpoint.")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--sl_epochs_per_rollout", type=int, default=5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--logdir", type=str, default="logs/stage2")

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

    # Environment (same task, fewer envs for rollout collection)
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=BasePPORunnerCfg.clip_actions)

    log_dir = os.path.join(args_cli.logdir, args_cli.task)
    os.makedirs(log_dir, exist_ok=True)

    # Runner — load Stage 1 policy checkpoint, fresh encoder
    encoder = BEVStudentEncoder()
    runner = ThreeStagePPORunner(
        env,
        BasePPORunnerCfg().to_dict(),
        encoder,
        log_dir=log_dir,
        device=device,
    )
    # Load policy weights only (encoder is freshly initialised)
    runner.load(args_cli.policy_ckpt, load_cfg={"actor": True, "critic": True, "optimizer": False})

    runner.learn_stage2(
        num_epochs=args_cli.epochs,
        batch_size=args_cli.batch_size,
        sl_epochs_per_rollout=args_cli.sl_epochs_per_rollout,
    )

    # Save encoder-only checkpoint for Stage 3
    encoder_path = os.path.join(log_dir, "encoder_best.pt")
    runner.save(encoder_path)
    print(f"[Stage 2] Encoder checkpoint saved to {encoder_path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
