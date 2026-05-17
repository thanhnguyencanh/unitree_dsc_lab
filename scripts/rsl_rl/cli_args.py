"""Shared rsl_rl CLI arguments for train.py / play.py."""

from __future__ import annotations

import argparse


def add_rsl_rl_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--task", type=str, default="Unitree-G1-23dof-StairClimb-v0")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iterations", type=int, default=6000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--logdir", type=str, default="logs/stage1")
    parser.add_argument("--headless", action="store_true")
    return parser
