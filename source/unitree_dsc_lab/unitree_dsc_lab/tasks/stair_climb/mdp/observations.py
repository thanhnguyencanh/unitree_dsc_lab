"""Observation terms for the stair-climbing task.

Paper Eq. (1): o_t = concat(o_prop, z_t)
  o_prop = [base_lin_vel, base_ang_vel, projected_gravity,
            joint_pos - default, joint_vel, last_action,
            foot_contact_flags, command_vel_xy_yaw]
  z_t    = [s_t, h_step, d_step, theta_yaw_current]   (4-D terrain token)

Stage 1: z_t comes from the privileged teacher (terrain-manager ground truth).
Stage 2/3: z_t comes from the BEV student encoder.

The teacher state lives on the env as `env.privileged_teacher`
(`PrivilegedTeacher` instance). See `tasks/stair_climb/perception/teacher.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _base_yaw(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """World-frame yaw of the robot base (rad), shape (num_envs,)."""
    asset = env.scene["robot"]
    quat_w = asset.data.root_quat_w  # (num_envs, 4) wxyz
    _, _, yaw = euler_xyz_from_quat(quat_w)
    return yaw


def terrain_token_privileged(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Per-env ground-truth 4-D terrain token from the privileged teacher.

    Returns: (num_envs, 4) — [class_id, h_step, d_step, theta_yaw_current].
    """
    teacher = getattr(env, "privileged_teacher", None)
    if teacher is None:
        raise RuntimeError(
            "PrivilegedTeacher not attached to env. Construct one during env build:\n"
            "    env.privileged_teacher = PrivilegedTeacher(num_envs=env.num_envs, device=env.device)\n"
            "and call `teacher.refresh_from_cfgs(...)` on every env reset."
        )
    return teacher.token(_base_yaw(env))


def foot_contact_flags(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Binary foot-contact flags (1=in contact, 0=in air), shape (num_envs, 2).

    Channels follow the contact-sensor body resolution order (L then R for G1).
    The ObservationManager resolves ``sensor_cfg.body_ids`` before the first call.
    """
    sensor = env.scene[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # (num_envs, 2, 3)
    return (torch.linalg.norm(forces, dim=-1) > threshold).float()


__all__ = ["terrain_token_privileged", "foot_contact_flags"]
