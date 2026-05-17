"""Procedural staircase terrain for stair-climbing (paper §III-B).

Provides three sub-terrain classes drawn per tile:

  s_t = 0  flat
  s_t = 1  stairs-up   (lead-in flat -> climbing steps -> top platform)
  s_t = 2  stairs-down (lead-in flat -> descending steps -> bottom platform)

Each tile is rotated about its centre by a random yaw `theta_yaw_terrain` so the
robot does not always face directly into the staircase. The robot spawns on the
lead-in flat patch (origin = patch centre).

Per-tile ground truth ``(class_id, h_step, d_step, theta_yaw_terrain)`` is
written to a process-local registry (`StairGTRegistry`) keyed by the same
``dict_to_md5_hash(cfg.to_dict())`` value IsaacLab uses for terrain caching.
At runtime the env-side teacher term recomputes that key per env (from the
tile's ``(row, col)``, the cfg and the difficulty) and looks up the GT.

Train-time randomization ranges follow paper Table II:

    h_step          0.12 - 0.16 m
    d_step          0.26 - 0.32 m
    yaw offset      -25 deg .. +25 deg
    flat lead-in    1.0 - 2.0 m
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import trimesh

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.terrains.terrain_generator import TerrainGenerator
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.dict import dict_to_md5_hash


# ---------------------------------------------------------------------------
# Per-tile ground-truth registry
# ---------------------------------------------------------------------------


@dataclass
class StairGT:
    class_id: int          # 0=flat, 1=stairs-up, 2=stairs-down
    h_step: float          # m  (0.0 when flat)
    d_step: float          # m  (0.0 when flat)
    theta_yaw_terrain: float  # rad — yaw of the stair "forward" axis in the world


class StairGTRegistry:
    """Process-local maps for per-tile ground truth.

    Two indexes are kept in sync:

    * ``_store[tile_hash] -> StairGT`` — populated by :func:`stairs_terrain` on
      every build, keyed by the same ``dict_to_md5_hash(cfg.to_dict())`` value
      IsaacLab uses for terrain caching.
    * ``_by_row_col[(row, col)] -> StairGT`` — populated by
      :class:`StairTerrainGenerator` so the env-side teacher can look up GT by
      the env's assigned terrain row/col instead of having to recompute the
      per-tile difficulty hash.
    """

    _store: ClassVar[dict[str, StairGT]] = {}
    _by_row_col: ClassVar[dict[tuple[int, int], StairGT]] = {}

    @classmethod
    def put(cls, key: str, gt: StairGT) -> None:
        cls._store[key] = gt

    @classmethod
    def get(cls, key: str) -> StairGT | None:
        return cls._store.get(key)

    @classmethod
    def put_by_row_col(cls, row: int, col: int, gt: StairGT) -> None:
        cls._by_row_col[(int(row), int(col))] = gt

    @classmethod
    def get_by_row_col(cls, row: int, col: int) -> StairGT | None:
        return cls._by_row_col.get((int(row), int(col)))

    @classmethod
    def clear(cls) -> None:
        cls._store.clear()
        cls._by_row_col.clear()

    @classmethod
    def hash_for_cfg(cls, cfg: "StairsTerrainCfg") -> str:
        """Stable per-tile key — must match `TerrainGenerator._get_terrain_mesh`."""
        return dict_to_md5_hash(cfg.to_dict())


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _box(size: tuple[float, float, float], pos: tuple[float, float, float]) -> trimesh.Trimesh:
    return trimesh.creation.box(size, trimesh.transformations.translation_matrix(pos))


def _yaw_transform(theta: float, pivot: tuple[float, float, float]) -> np.ndarray:
    """Homogeneous rotation about +Z passing through ``pivot``."""
    c, s = math.cos(theta), math.sin(theta)
    T = np.eye(4)
    T[0, 0], T[0, 1] = c, -s
    T[1, 0], T[1, 1] = s, c
    px, py, pz = pivot
    T[0, 3] = px - c * px + s * py
    T[1, 3] = py - s * px - c * py
    T[2, 3] = pz - pz
    return T


def _flat_tile(size_xy: tuple[float, float], slab_thickness: float = 0.05) -> list[trimesh.Trimesh]:
    sx, sy = size_xy
    return [_box((sx, sy, slab_thickness), (sx / 2.0, sy / 2.0, -slab_thickness / 2.0))]


def _stair_band(
    size_xy: tuple[float, float],
    h_step: float,
    d_step: float,
    num_steps: int,
    lead_in: float,
    sign: int,
) -> tuple[list[trimesh.Trimesh], float, float]:
    """Build a one-way staircase along +X.

    Layout (sign=+1, stairs-up):

        |<- lead_in ->|<- num_steps * d_step ->|<- top platform ->|
        flat (z=0)     ascending boxes           top flat (z=top_h)

    For sign=-1 (stairs-down) the top flat is at z=0 and the lead-in is
    elevated, so the robot starts on top and descends.

    Returns: (meshes, lead_in_centre_x, lead_in_z) where the lead-in centre is
    where the robot should spawn (origin).
    """
    sx, sy = size_xy
    slab_t = 0.05

    stair_run = num_steps * d_step
    top_run = max(0.0, sx - lead_in - stair_run)
    top_h = num_steps * h_step

    meshes: list[trimesh.Trimesh] = []

    if sign > 0:
        # 1) lead-in flat at z=0, top surface at z=0
        meshes.append(_box((lead_in, sy, slab_t), (lead_in / 2.0, sy / 2.0, -slab_t / 2.0)))
        # 2) ascending steps — step k top surface at z = (k+1) * h_step
        for k in range(num_steps):
            top_z = (k + 1) * h_step
            cx = lead_in + (k + 0.5) * d_step
            meshes.append(_box((d_step, sy, top_z + slab_t), (cx, sy / 2.0, top_z / 2.0 - slab_t / 2.0)))
        # 3) top platform at z = top_h
        if top_run > 0:
            cx = lead_in + stair_run + top_run / 2.0
            meshes.append(_box((top_run, sy, top_h + slab_t), (cx, sy / 2.0, top_h / 2.0 - slab_t / 2.0)))
        spawn_x = lead_in / 2.0
        spawn_z = 0.0
    else:
        # 1) lead-in flat elevated at z = top_h
        meshes.append(_box((lead_in, sy, top_h + slab_t), (lead_in / 2.0, sy / 2.0, top_h / 2.0 - slab_t / 2.0)))
        # 2) descending steps — step k top surface at z = (num_steps - k - 1) * h_step
        for k in range(num_steps):
            top_z = (num_steps - k - 1) * h_step
            cx = lead_in + (k + 0.5) * d_step
            meshes.append(_box((d_step, sy, top_z + slab_t), (cx, sy / 2.0, top_z / 2.0 - slab_t / 2.0)))
        # 3) bottom platform at z = 0
        if top_run > 0:
            cx = lead_in + stair_run + top_run / 2.0
            meshes.append(_box((top_run, sy, slab_t), (cx, sy / 2.0, -slab_t / 2.0)))
        spawn_x = lead_in / 2.0
        spawn_z = top_h

    return meshes, spawn_x, spawn_z


# ---------------------------------------------------------------------------
# Sub-terrain config + function
# ---------------------------------------------------------------------------


def stairs_terrain(
    difficulty: float, cfg: "StairsTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Build one stair-climb sub-terrain tile.

    Per-tile sampling is seeded from ``(cfg.seed, dict_to_md5_hash(cfg))`` so
    the same tile re-built later produces identical geometry — matching what
    the IsaacLab cache layer already assumes.
    """
    # ---- per-tile RNG -----------------------------------------------------
    key = StairGTRegistry.hash_for_cfg(cfg)
    base_seed = 0 if cfg.seed is None else int(cfg.seed)
    tile_seed = (base_seed + int(key[:8], 16)) & 0xFFFFFFFF
    rng = np.random.default_rng(tile_seed)

    # ---- sample tile params from difficulty + ranges ----------------------
    h_lo, h_hi = cfg.h_step_range
    d_lo, d_hi = cfg.d_step_range
    yaw_lo, yaw_hi = cfg.yaw_range_rad
    lead_lo, lead_hi = cfg.flat_lead_range

    # difficulty interpolates h_step within range, then ±10% jitter
    h_step = h_lo + difficulty * (h_hi - h_lo)
    h_step *= 1.0 + 0.1 * rng.uniform(-1.0, 1.0)
    h_step = float(np.clip(h_step, h_lo, h_hi))

    d_step = float(rng.uniform(d_lo, d_hi))
    yaw = float(rng.uniform(yaw_lo, yaw_hi))
    lead_in = float(rng.uniform(lead_lo, lead_hi))

    class_probs = np.asarray(cfg.class_probs, dtype=np.float64)
    class_probs /= class_probs.sum()
    class_id = int(rng.choice(3, p=class_probs))

    sx, sy = cfg.size

    # ---- build meshes -----------------------------------------------------
    if class_id == 0:  # flat
        meshes = _flat_tile(cfg.size)
        spawn_x, spawn_z = sx / 2.0, 0.0
        h_step_gt, d_step_gt = 0.0, 0.0
    else:
        sign = +1 if class_id == 1 else -1
        # cap num_steps so the staircase fits within the tile
        usable_x = max(0.5, sx - lead_in - cfg.platform_min)
        num_steps = min(cfg.num_steps, max(1, int(usable_x // d_step)))
        meshes, spawn_x, spawn_z = _stair_band(
            cfg.size, h_step, d_step, num_steps, lead_in, sign
        )
        h_step_gt, d_step_gt = h_step, d_step

    # ---- apply yaw rotation about tile centre -----------------------------
    # spawn is on the centreline (y = sy/2) before rotation
    spawn_y = sy / 2.0
    if abs(yaw) > 1e-6:
        cx, cy = sx / 2.0, sy / 2.0
        T = _yaw_transform(yaw, (cx, cy, 0.0))
        for m in meshes:
            m.apply_transform(T)
        # rotate spawn point with the mesh (dy = 0 by construction)
        dx = spawn_x - cx
        c, s = math.cos(yaw), math.sin(yaw)
        spawn_x = cx + c * dx
        spawn_y = cy + s * dx

    origin = np.array([spawn_x, spawn_y, spawn_z], dtype=np.float64)

    # ---- record GT --------------------------------------------------------
    StairGTRegistry.put(
        key,
        StairGT(
            class_id=class_id,
            h_step=float(h_step_gt),
            d_step=float(d_step_gt),
            theta_yaw_terrain=float(yaw),
        ),
    )

    return meshes, origin


@configclass
class StairsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for the stair-climbing sub-terrain (paper §III-B)."""

    function = stairs_terrain

    h_step_range: tuple[float, float] = (0.12, 0.16)
    """Train range for step height (m)."""

    d_step_range: tuple[float, float] = (0.26, 0.32)
    """Train range for step depth / tread (m)."""

    yaw_range_rad: tuple[float, float] = (-25.0 * math.pi / 180.0, 25.0 * math.pi / 180.0)
    """Yaw rotation of the staircase about the tile centre (rad). Train ± 25 deg."""

    flat_lead_range: tuple[float, float] = (1.0, 2.0)
    """Lead-in / lead-out flat patch length (m)."""

    num_steps: int = 12
    """Maximum number of steps in the staircase. Auto-clipped to fit the tile."""

    platform_min: float = 0.5
    """Minimum top/bottom platform run after the staircase (m)."""

    class_probs: tuple[float, float, float] = (0.2, 0.4, 0.4)
    """Sampling probabilities for (flat, stairs-up, stairs-down)."""


# ---------------------------------------------------------------------------
# Test-range preset
# ---------------------------------------------------------------------------


@configclass
class StairsTerrainTestCfg(StairsTerrainCfg):
    """Out-of-distribution test ranges (paper §IV-D)."""

    h_step_range: tuple[float, float] = (0.18, 0.22)
    yaw_range_rad: tuple[float, float] = (-35.0 * math.pi / 180.0, 35.0 * math.pi / 180.0)


# ---------------------------------------------------------------------------
# Class-locked presets for use with TerrainGenerator proportions
# ---------------------------------------------------------------------------


@configclass
class FlatLeadInCfg(StairsTerrainCfg):
    """Flat-only tile (class_id = 0)."""

    class_probs: tuple[float, float, float] = (1.0, 0.0, 0.0)


@configclass
class StairsUpCfg(StairsTerrainCfg):
    """Stairs-up-only tile (class_id = 1)."""

    class_probs: tuple[float, float, float] = (0.0, 1.0, 0.0)


@configclass
class StairsDownCfg(StairsTerrainCfg):
    """Stairs-down-only tile (class_id = 2)."""

    class_probs: tuple[float, float, float] = (0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# TerrainGenerator subclass that records the (row, col) -> tile GT mapping
# ---------------------------------------------------------------------------


class StairTerrainGenerator(TerrainGenerator):
    """A :class:`TerrainGenerator` that records per-tile GT keyed by (row, col).

    IsaacLab's stock ``TerrainGenerator`` calls ``_get_terrain_mesh(difficulty, cfg)``
    to build a tile, then ``_add_sub_terrain(mesh, origin, row, col, cfg)`` to
    place it in the grid. We hook both:

    * ``_get_terrain_mesh`` — stash the same ``dict_to_md5_hash(cfg)`` key the
      function used to register ``StairGT``.
    * ``_add_sub_terrain`` — pair the stashed key with ``(row, col)`` and copy
      the ``StairGT`` into ``StairGTRegistry._by_row_col``.

    With this in place the env-side teacher only needs
    ``terrain.terrain_levels[env_id]`` and ``terrain.terrain_types[env_id]`` to
    look up the GT, which removes any need to recompute / re-seed the
    per-tile RNG.
    """

    def __init__(self, cfg: TerrainGeneratorCfg, device: str = "cpu") -> None:
        self._pending_hash: str | None = None
        super().__init__(cfg, device=device)

    def _get_terrain_mesh(self, difficulty: float, cfg: SubTerrainBaseCfg):
        # Compute the same hash IsaacLab will use, so we can pair it with the
        # row/col reported in _add_sub_terrain.
        cfg_for_hash = cfg.copy()
        cfg_for_hash.difficulty = float(difficulty)  # type: ignore[attr-defined]
        cfg_for_hash.seed = self.cfg.seed
        self._pending_hash = dict_to_md5_hash(cfg_for_hash.to_dict())
        return super()._get_terrain_mesh(difficulty, cfg)

    def _add_sub_terrain(self, mesh, origin, row: int, col: int, sub_terrain_cfg):
        super()._add_sub_terrain(mesh, origin, row, col, sub_terrain_cfg)
        if self._pending_hash is not None:
            gt = StairGTRegistry.get(self._pending_hash)
            if gt is not None:
                StairGTRegistry.put_by_row_col(row, col, gt)
            self._pending_hash = None


@configclass
class StairTerrainGeneratorCfg(TerrainGeneratorCfg):
    """``TerrainGeneratorCfg`` that uses :class:`StairTerrainGenerator`.

    Drop this in wherever you'd use ``terrain_gen.TerrainGeneratorCfg`` so the
    per-(row, col) GT registry is populated alongside the meshes.
    """

    class_type: type = StairTerrainGenerator


# ---------------------------------------------------------------------------
# Convenience: lookup helper for the env-side teacher
# ---------------------------------------------------------------------------


def lookup_tile_gt(cfg: StairsTerrainCfg, difficulty: float, seed: int | None) -> StairGT | None:
    """Reproduce the same hash key IsaacLab uses, then read from the registry.

    Used by the privileged teacher term so it can recover per-env (class, h, d, yaw)
    given each env's assigned ``(row, col)`` and the corresponding difficulty.
    """
    cfg = cfg.copy()
    cfg.difficulty = float(difficulty)  # type: ignore[attr-defined]
    cfg.seed = seed
    key = StairGTRegistry.hash_for_cfg(cfg)
    return StairGTRegistry.get(key)
