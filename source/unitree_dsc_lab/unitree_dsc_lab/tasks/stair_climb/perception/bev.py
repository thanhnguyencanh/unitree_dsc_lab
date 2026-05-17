"""On-GPU point-cloud -> 6-channel BEV (paper §III-A.2).

Region:     3 m x 3 m, robot-centric, +X forward, +Y left
Resolution: 0.05 m  ->  60 x 60 grid
Channels:   max(z), min(z), mean(z), max-min, std(z), normalized density
Empty:      zero-filled
"""

from __future__ import annotations

import torch


BEV_GRID = 60
BEV_RES_M = 0.05
BEV_REGION_M = BEV_GRID * BEV_RES_M  # 3.0 m


def points_to_bev(points: torch.Tensor) -> torch.Tensor:
    """Project a (N, 3) point cloud to a (6, 60, 60) BEV tensor on the same device.

    Args:
        points: (N, 3) in robot frame, +X forward, +Y left, +Z up.

    Returns:
        (6, 60, 60) float tensor.
    """
    raise NotImplementedError("Implement vectorized GPU scatter to BEV bins.")
