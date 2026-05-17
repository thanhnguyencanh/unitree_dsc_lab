"""Curriculum terms for the stair-climbing task.

We reuse IsaacLab's stock ``terrain_levels_vel`` (paper-equivalent of standard
locomotion curriculum: tiles get harder when the robot walks far enough on its
commanded velocity, easier otherwise). Live in :mod:`isaaclab_tasks` rather
than core ``isaaclab``, so import from there explicitly.
"""

from __future__ import annotations

from isaaclab_tasks.manager_based.locomotion.velocity.mdp.curriculums import (  # noqa: F401
    terrain_levels_vel,
)

__all__ = ["terrain_levels_vel"]
