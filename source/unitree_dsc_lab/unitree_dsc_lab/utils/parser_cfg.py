"""CLI argument helpers shared by train/play/export scripts."""

from __future__ import annotations

import argparse


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--task", type=str, default="Unitree-G1-23dof-StairClimb-v0")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iterations", type=int, default=6000)
    parser.add_argument("--logdir", type=str, default="logs/stage1")
    return parser
