"""Privileged teacher = direct ground-truth read from the terrain manager.

This is NOT a network. It publishes (s_t, h_step, d_step, theta_yaw_current)
sampled by `terrains/stair_generator.py::stairs_terrain` for each env so:

  * Stage 1 — feed teacher z_t directly into the policy observation
              (no perception in the loop yet).
  * Stage 2 — pair teacher z_t with BEV observations to supervise the student
              (Eq. 8 of the paper):

                  L_terrain = 0.6 * CE(class)
                            + 1.0 * SmoothL1(h)
                            + 1.0 * SmoothL1(d)

`theta_yaw_current` is `wrap(robot_yaw_world - terrain_yaw_world)` — recomputed
every step from the base orientation. The other three are tile-static and read
from `StairGTRegistry` at episode start.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from unitree_dsc_lab.tasks.stair_climb.terrains.stair_generator import (
    StairGT,
    StairGTRegistry,
    StairsTerrainCfg,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.terrains import TerrainImporter


@dataclass
class TerrainGroundTruth:
    """Per-env terrain GT. All tensors are shape (num_envs,) on the env device."""

    class_id: torch.Tensor      # long, in {0: flat, 1: stairs-up, 2: stairs-down}
    h_step: torch.Tensor        # float — 0 when class=flat
    d_step: torch.Tensor        # float — 0 when class=flat
    theta_yaw_terrain: torch.Tensor  # float — yaw of stair forward axis in world


def _wrap_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


class PrivilegedTeacher:
    """Reads per-env terrain ground-truth from `StairGTRegistry`.

    Build once per env; call `refresh()` whenever envs reset, and `token()`
    once per simulation step.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str = "cuda:0",
        default_class: int = 0,
    ) -> None:
        self.num_envs = num_envs
        self.device = torch.device(device)
        self._gt = TerrainGroundTruth(
            class_id=torch.full((num_envs,), default_class, dtype=torch.long, device=self.device),
            h_step=torch.zeros(num_envs, device=self.device),
            d_step=torch.zeros(num_envs, device=self.device),
            theta_yaw_terrain=torch.zeros(num_envs, device=self.device),
        )

    # ----- accessors ------------------------------------------------------

    @property
    def gt(self) -> TerrainGroundTruth:
        return self._gt

    def token(self, robot_yaw_world: torch.Tensor) -> torch.Tensor:
        """4-D terrain token `z_t = [class_id, h_step, d_step, theta_yaw_current]`.

        Args:
            robot_yaw_world: (num_envs,) — base yaw in world frame (rad).
        """
        theta_yaw_current = _wrap_pi(robot_yaw_world - self._gt.theta_yaw_terrain)
        return torch.stack(
            (
                self._gt.class_id.to(torch.float32),
                self._gt.h_step,
                self._gt.d_step,
                theta_yaw_current,
            ),
            dim=-1,
        )

    # ----- update from terrain manager -----------------------------------

    def refresh_from_keys(self, env_ids: torch.Tensor, tile_keys: list[str]) -> None:
        """Pull GT for `env_ids` whose tiles map to `tile_keys` (same length)."""
        assert len(env_ids) == len(tile_keys)
        for env_idx, key in zip(env_ids.tolist(), tile_keys):
            gt: StairGT | None = StairGTRegistry.get(key)
            if gt is None:
                # Tile not found — leave defaults (class=0, zeros).
                continue
            self._gt.class_id[env_idx] = gt.class_id
            self._gt.h_step[env_idx] = gt.h_step
            self._gt.d_step[env_idx] = gt.d_step
            self._gt.theta_yaw_terrain[env_idx] = gt.theta_yaw_terrain

    def refresh_from_cfgs(
        self,
        env_ids: torch.Tensor,
        per_env_cfg: list[tuple[StairsTerrainCfg, float, int | None]],
    ) -> None:
        """Same as `refresh_from_keys` but resolves the hash here.

        `per_env_cfg[i] = (sub_terrain_cfg, difficulty, seed)` for `env_ids[i]`.
        """
        keys: list[str] = []
        for cfg, difficulty, seed in per_env_cfg:
            tmp = cfg.copy()
            tmp.difficulty = float(difficulty)  # type: ignore[attr-defined]
            tmp.seed = seed
            keys.append(StairGTRegistry.hash_for_cfg(tmp))
        self.refresh_from_keys(env_ids, keys)

    def refresh_from_terrain(
        self,
        env_ids: torch.Tensor,
        terrain: "TerrainImporter",
    ) -> None:
        """Populate per-env GT from the live ``TerrainImporter``.

        Reads ``terrain.terrain_levels[env_ids]`` (row) and
        ``terrain.terrain_types[env_ids]`` (col), then looks up
        ``StairGTRegistry._by_row_col`` populated by
        :class:`StairTerrainGenerator` during terrain build.

        This is the recommended hook for the env reset event — it does not
        require re-deriving the per-tile difficulty hash.
        """
        if getattr(terrain, "terrain_levels", None) is None:
            # plane/usd terrain — no per-tile GT to read; leave defaults.
            return

        rows = terrain.terrain_levels[env_ids].detach().cpu().tolist()
        cols = terrain.terrain_types[env_ids].detach().cpu().tolist()

        for env_idx, row, col in zip(env_ids.tolist(), rows, cols):
            gt = StairGTRegistry.get_by_row_col(int(row), int(col))
            if gt is None:
                continue
            self._gt.class_id[env_idx] = gt.class_id
            self._gt.h_step[env_idx] = gt.h_step
            self._gt.d_step[env_idx] = gt.d_step
            self._gt.theta_yaw_terrain[env_idx] = gt.theta_yaw_terrain


__all__ = ["TerrainGroundTruth", "PrivilegedTeacher"]
