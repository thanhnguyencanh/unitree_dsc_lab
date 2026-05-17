"""Stair-specific shaping rewards (paper §III-C).

Two stateful :class:`ManagerTermBase` terms that complement IsaacLab's stock
rough-locomotion stack. Both read per-env terrain ground truth from
``env.privileged_teacher`` (attached lazily by ``mdp.reset_privileged_teacher``)
so the reward only fires on the stair classes where it is meaningful.

* :class:`swing_clearance_bonus` — dense per-step bonus while a foot is in the
  air, equal to ``clamp(foot_z - lift_off_z - (h_step + margin), 0)``. Active
  only when ``class_id == 1`` (stairs-up). Encourages the swing foot to clear
  the *next* step.
* :class:`step_alignment_bonus`  — sparse Gaussian bonus on touchdown events
  (using :meth:`ContactSensor.compute_first_contact`). Measures forward
  distance from the previous touchdown projected onto the stair-forward axis
  ``[cos(theta_yaw_terrain), sin(theta_yaw_terrain)]`` and rewards a landing
  ``d_step`` ahead with tolerance ``window_margin``. Active on
  ``class_id ∈ {1, 2}``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


CLASS_STAIRS_UP = 1
CLASS_STAIRS_DOWN = 2


class swing_clearance_bonus(ManagerTermBase):
    """Reward foot apex height ``>= h_step + margin`` during stair-up.

    Stateful: tracks ``lift_off_z`` per foot (the last in-contact height) and
    on every step reads the current foot height. While the foot is airborne,
    the bonus equals ``clamp(foot_z - lift_off_z - h_step - margin, 0)``.

    Active only when ``env.privileged_teacher.gt.class_id == 1``. Returns
    zero for flat and stairs-down tiles.
    """

    def __init__(self, cfg: "RewardTermCfg", env: "ManagerBasedRLEnv") -> None:
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        # body_ids are resolved by the manager before __init__ runs.
        self._foot_body_ids: list[int] = list(asset_cfg.body_ids)  # type: ignore[arg-type]
        self._foot_sensor_ids: list[int] = list(sensor_cfg.body_ids)  # type: ignore[arg-type]
        n_feet = len(self._foot_body_ids)
        if n_feet == 0 or n_feet != len(self._foot_sensor_ids):
            raise ValueError(
                "swing_clearance_bonus: asset_cfg.body_ids and sensor_cfg.body_ids "
                f"must be the same non-empty length, got {n_feet} and "
                f"{len(self._foot_sensor_ids)}."
            )
        self._lift_off_z = torch.zeros(self.num_envs, n_feet, device=self.device)
        self._init_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self._lift_off_z.zero_()
            self._init_done.zero_()
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            self._lift_off_z[ids] = 0.0
            self._init_done[ids] = False

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        clearance_margin: float = 0.05,
    ) -> torch.Tensor:
        teacher = getattr(env, "privileged_teacher", None)
        if teacher is None:
            return torch.zeros(self.num_envs, device=self.device)

        asset: "Articulation" = env.scene[asset_cfg.name]
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

        foot_z = asset.data.body_pos_w[:, self._foot_body_ids, 2]  # (B, F)
        in_contact = sensor.data.current_contact_time[:, self._foot_sensor_ids] > 0.0

        # Seed lift_off_z with current foot_z on first call so we don't reward
        # the spawn drop.
        first_seen = ~self._init_done
        update_lift = in_contact | first_seen[:, None]
        self._lift_off_z = torch.where(update_lift, foot_z, self._lift_off_z)
        self._init_done = self._init_done | in_contact.any(dim=1)

        h_step = teacher.gt.h_step                  # (B,)
        class_id = teacher.gt.class_id              # (B,)
        target_clear = h_step + clearance_margin    # (B,)

        clearance = (foot_z - self._lift_off_z - target_clear[:, None]).clamp(min=0.0)
        in_air = ~in_contact
        bonus = (clearance * in_air).sum(dim=1)

        return bonus * (class_id == CLASS_STAIRS_UP).to(bonus.dtype)


class step_alignment_bonus(ManagerTermBase):
    """Reward foot touchdowns that land ``d_step`` ahead of the previous one.

    Stateful: tracks the world-frame xy position of each foot at its last
    touchdown. On each new touchdown event (detected by
    :meth:`ContactSensor.compute_first_contact`) the forward distance from the
    previous touchdown — projected onto the stair-forward axis
    ``[cos(theta_yaw_terrain), sin(theta_yaw_terrain)]`` — is compared against
    ``d_step`` and rewarded with a Gaussian of width ``window_margin``.

    Active only when ``env.privileged_teacher.gt.class_id`` is stairs-up or
    stairs-down (``1`` or ``2``). The first touchdown after a reset is used to
    seed state and emits no reward.
    """

    def __init__(self, cfg: "RewardTermCfg", env: "ManagerBasedRLEnv") -> None:
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        self._foot_body_ids: list[int] = list(asset_cfg.body_ids)  # type: ignore[arg-type]
        self._foot_sensor_ids: list[int] = list(sensor_cfg.body_ids)  # type: ignore[arg-type]
        n_feet = len(self._foot_body_ids)
        if n_feet == 0 or n_feet != len(self._foot_sensor_ids):
            raise ValueError(
                "step_alignment_bonus: asset_cfg.body_ids and sensor_cfg.body_ids "
                f"must be the same non-empty length, got {n_feet} and "
                f"{len(self._foot_sensor_ids)}."
            )
        self._last_landing_xy = torch.zeros(self.num_envs, n_feet, 2, device=self.device)
        self._has_landed = torch.zeros(
            self.num_envs, n_feet, dtype=torch.bool, device=self.device
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self._last_landing_xy.zero_()
            self._has_landed.zero_()
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            self._last_landing_xy[ids] = 0.0
            self._has_landed[ids] = False

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        window_margin: float = 0.04,
    ) -> torch.Tensor:
        teacher = getattr(env, "privileged_teacher", None)
        if teacher is None:
            return torch.zeros(self.num_envs, device=self.device)

        asset: "Articulation" = env.scene[asset_cfg.name]
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

        foot_xy = asset.data.body_pos_w[:, self._foot_body_ids, :2]      # (B, F, 2)
        first_contact = sensor.compute_first_contact(env.step_dt)[
            :, self._foot_sensor_ids
        ]                                                                # (B, F)

        theta = teacher.gt.theta_yaw_terrain                             # (B,)
        fwd_axis = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)  # (B, 2)
        delta = foot_xy - self._last_landing_xy                          # (B, F, 2)
        fwd_delta = (delta * fwd_axis[:, None, :]).sum(dim=-1)           # (B, F)

        d_step = teacher.gt.d_step                                       # (B,)
        err = fwd_delta - d_step[:, None]                                # (B, F)
        bonus = torch.exp(-(err * err) / (window_margin * window_margin))  # (B, F)

        # Only emit on touchdown transitions where this foot has landed before
        # (so we have a valid `_last_landing_xy` to subtract from).
        active = first_contact & self._has_landed
        reward = (bonus * active.to(bonus.dtype)).sum(dim=1)             # (B,)

        class_id = teacher.gt.class_id
        on_stairs = (class_id == CLASS_STAIRS_UP) | (class_id == CLASS_STAIRS_DOWN)
        reward = reward * on_stairs.to(reward.dtype)

        # Update buffers AFTER computing the reward.
        update = first_contact
        self._last_landing_xy = torch.where(
            update.unsqueeze(-1), foot_xy, self._last_landing_xy
        )
        self._has_landed = self._has_landed | first_contact

        return reward


__all__ = ["swing_clearance_bonus", "step_alignment_bonus"]
