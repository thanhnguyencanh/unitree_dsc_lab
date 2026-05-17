"""Export the runtime-side YAML consumed by the on-board ROS 2 node.

Mirrors the structure of `unitree_rl_lab/deploy/robots/g1_29dof/config/`.
Writes joint order, default qpos, PD gains, action scale, observation order,
and ONNX paths so the C++/Python runtime can stay in sync with training cfg.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def export(cfg: dict, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
