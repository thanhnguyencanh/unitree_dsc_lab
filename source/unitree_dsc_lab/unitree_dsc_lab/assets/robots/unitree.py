"""Unitree robot asset configuration for the stair-climbing task.

Point either `UNITREE_ROS_DIR` (URDF, recommended on Isaac Sim >= 5.0) or
`UNITREE_MODEL_DIR` (USD bundle) to a local checkout. The env config picks the
matching `spawn` field from `G1_23DOF_CFG`.
"""

from __future__ import annotations

import os

from isaaclab_assets.robots.unitree import G1_CFG  # type: ignore  # noqa: F401

# --- Local override paths -----------------------------------------------------
# git clone https://github.com/unitreerobotics/unitree_ros.git  -> unitree_ros/unitree_ros
UNITREE_ROS_DIR: str | None = os.environ.get("UNITREE_ROS_DIR")
# git clone https://huggingface.co/datasets/unitreerobotics/unitree_model
UNITREE_MODEL_DIR: str | None = os.environ.get("UNITREE_MODEL_DIR")


# Re-export the stock G1 config so env cfgs can import a stable name. The
# 23-DoF variant freezes the wrists at the env-cfg level (see paper §III-A).
G1_23DOF_CFG = G1_CFG
