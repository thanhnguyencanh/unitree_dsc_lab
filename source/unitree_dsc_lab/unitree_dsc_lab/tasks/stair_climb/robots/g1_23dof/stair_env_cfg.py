"""Isaac Lab env config for stair-climbing on Unitree G1 (23-DoF).

If you are running the 29-DoF G1, the wrist joints must be frozen in this env
config so the action dim stays at 23 (paper §III-A).
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnvCfg

from unitree_dsc_lab.assets.robots.unitree import G1_23DOF_CFG  # noqa: F401


@configclass
class G1StairClimbEnvCfg(ManagerBasedRLEnvCfg):
    """Training-mode env config.

    Plug into:
      * scene:        G1 articulation + StairsTerrainGenerator + depth camera
      * observations: o_prop + privileged z_t (stage 1) OR student z_t (stage 2/3)
      * rewards:      rough-locomotion defaults + swing_clearance_bonus + step_alignment_bonus
      * commands:     forward velocity 0.0 - 0.7 m/s
    """

    # TODO: populate scene / observations / rewards / events / terminations
    # cfgs once the matching IsaacLab-2.3.2 manager APIs are wired through the
    # mdp/ stubs.


@configclass
class G1StairClimbPlayEnvCfg(G1StairClimbEnvCfg):
    """Play-mode override: fewer envs, deterministic curriculum, no domain rand."""
