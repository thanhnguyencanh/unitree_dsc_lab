"""Stage 3 — Joint fine-tuning of policy + encoder.

Unfreeze both. Loss = L_PPO + 1.0 * L_terrain. Reduce learning rate by x0.3.
"""

from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_policy", type=str, required=True)
    parser.add_argument("--resume_encoder", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=2048)
    parser.add_argument("--max_iterations", type=int, default=3000)
    parser.add_argument("--logdir", type=str, default="logs/stage3")
    args = parser.parse_args()

    raise NotImplementedError(
        "Wrap with ThreeStagePPORunner(stage=3); inject L_terrain into the PPO update."
    )


if __name__ == "__main__":
    main()
