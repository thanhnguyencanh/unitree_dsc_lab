"""Export trained policy + encoder to ONNX (sim-to-real bridge, §14).

Two separate ONNX files — on-board node runs encoder at 10 Hz, policy at 50 Hz.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export to ONNX.")
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--encoder", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="deploy")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raise NotImplementedError(
        "torch.onnx.export(policy, dummy_proprio+z, ...); same for encoder(BEV)."
    )


if __name__ == "__main__":
    main()
