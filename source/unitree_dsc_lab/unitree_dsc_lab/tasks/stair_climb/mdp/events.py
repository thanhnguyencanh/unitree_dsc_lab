"""Event terms for the stair-climbing task.

The key term here is :class:`reset_privileged_teacher`, which:

* lazily attaches a :class:`PrivilegedTeacher` to ``env.privileged_teacher`` on
  the first call (so ``observations.terrain_token_privileged`` can find it);
* on every env reset, refreshes the per-env terrain ground truth by reading
  ``env.scene.terrain.terrain_levels`` / ``terrain_types`` and looking up
  ``StairGTRegistry._by_row_col`` (populated at terrain build by
  :class:`StairTerrainGenerator`).

Wire it into the env cfg as a reset-mode event:

    @configclass
    class EventCfg:
        reset_privileged_teacher = EventTerm(
            func=mdp.reset_privileged_teacher,
            mode="reset",
        )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import EventTermCfg, ManagerTermBase

from unitree_dsc_lab.tasks.stair_climb.perception.teacher import PrivilegedTeacher

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class reset_privileged_teacher(ManagerTermBase):
    """Attach + refresh ``env.privileged_teacher`` on reset.

    On first construction this term creates ``env.privileged_teacher`` (a
    :class:`PrivilegedTeacher` sized to ``env.num_envs``). On every reset it
    refreshes the per-env terrain GT from the live ``TerrainImporter``.
    """

    def __init__(self, cfg: EventTermCfg, env: "ManagerBasedEnv") -> None:
        super().__init__(cfg, env)
        if not hasattr(env, "privileged_teacher"):
            env.privileged_teacher = PrivilegedTeacher(
                num_envs=env.num_envs,
                device=env.device,
            )

    def __call__(self, env: "ManagerBasedEnv", env_ids: torch.Tensor | None) -> None:
        teacher: PrivilegedTeacher = env.privileged_teacher  # type: ignore[attr-defined]
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        terrain = env.scene.terrain
        teacher.refresh_from_terrain(env_ids.to(env.device), terrain)


__all__ = ["reset_privileged_teacher"]
