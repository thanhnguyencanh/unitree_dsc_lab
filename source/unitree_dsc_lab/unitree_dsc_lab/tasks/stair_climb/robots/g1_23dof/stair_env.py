"""Custom ManagerBasedRLEnv for G1 stair-climbing that exposes get_bev().

``G1StairClimbEnv`` subclasses ``ManagerBasedRLEnv`` and adds a single method:

    get_bev() -> Tensor(num_envs, 6, 60, 60)

The BEV is built from ``height_scanner`` (a RayCaster attached to the torso,
scanning 3 m × 3 m ahead of the robot at 0.05 m resolution).  Ray hits are
transformed into robot-forward/left coordinates and fed into
``perception.bev.points_to_bev`` to produce the 6-channel BEV expected by
``BEVStudentEncoder``.

This class is registered as the entry-point for the gym task so that
``ThreeStagePPORunner`` can call ``env.unwrapped.get_bev()`` transparently.
"""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import euler_xyz_from_quat

from unitree_dsc_lab.tasks.stair_climb.perception.bev import points_to_bev


class G1StairClimbEnv(ManagerBasedRLEnv):
    """G1 stair-climbing env with simulated BEV scan for Stage 2/3 training."""

    def get_bev(self) -> torch.Tensor:
        """Return ``(num_envs, 6, 60, 60)`` BEV from the on-robot height scanner.

        Pipeline:
          1. Read world-frame hit positions from the ``height_scanner`` RayCaster.
          2. Translate by robot root position (world → robot-centred).
          3. Rotate by ``−yaw`` (world-aligned → robot-forward/left frame).
          4. Pass to ``points_to_bev()`` with default x ∈ [0, 3] m, y ∈ [−1.5, 1.5] m.

        NaN hits (rays that miss the mesh entirely) are masked out automatically
        by ``points_to_bev``'s empty-cell zero-fill logic.
        """
        scanner = self.scene["height_scanner"]
        hits_w = scanner.data.ray_hits_w  # (num_envs, num_rays, 3)

        robot = self.scene["robot"]
        root_pos = robot.data.root_pos_w    # (num_envs, 3)
        root_quat = robot.data.root_quat_w  # (num_envs, 4) wxyz

        # --- translate to robot-centred frame ---
        local = hits_w - root_pos.unsqueeze(1)  # (num_envs, num_rays, 3)

        # --- rotate by -yaw (world → robot forward/left) ---
        # R(-θ): [x_b, y_b] = [cos(θ)*x + sin(θ)*y, -sin(θ)*x + cos(θ)*y]
        _, _, yaw = euler_xyz_from_quat(root_quat)   # (num_envs,)
        c = torch.cos(yaw).unsqueeze(1)              # (num_envs, 1)
        s = torch.sin(yaw).unsqueeze(1)
        lx = local[..., 0] * c + local[..., 1] * s
        ly = -local[..., 0] * s + local[..., 1] * c
        local = torch.stack([lx, ly, local[..., 2]], dim=-1)

        # mask out rays that didn't hit the mesh (NaN / ±inf)
        valid = torch.isfinite(local).all(dim=-1)

        return points_to_bev(local, valid_mask=valid)
