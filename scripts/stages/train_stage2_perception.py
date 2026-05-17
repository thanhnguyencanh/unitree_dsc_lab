"""Stage 2 — Train BEV student encoder under teacher supervision.

Freezes the policy from stage 1, rolls out, stores (BEV, z_t_gt) pairs, trains
the CNN until: MAE(h) < 1 cm, MAE(d) < 1 cm, class_acc > 99 % (Table II).
"""

from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_ckpt", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--logdir", type=str, default="logs/stage2")
    args = parser.parse_args()

    raise NotImplementedError(
        "Roll out frozen policy, collect BEV + z_t_gt, train BEVStudentEncoder "
        "with L_terrain = 0.6*CE + L1(h) + L1(d)."
    )


if __name__ == "__main__":
    main()
