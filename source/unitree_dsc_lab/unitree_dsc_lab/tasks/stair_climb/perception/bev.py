"""On-GPU point-cloud -> 6-channel BEV (paper §III-A.2).

Region:     3 m x 3 m, robot-centric, +X forward (rows), +Y left (cols)
Resolution: 0.05 m  ->  60 x 60 grid
Channels:   [max(z), min(z), mean(z), max - min, std(z), normalized density]
Empty:      zero-filled

The whole transform is a single set of ``scatter_add_`` / ``scatter_reduce_``
calls on flattened cell indices, so it stays vectorized on GPU and supports
batched (B, N, 3) inputs without a Python loop over envs.
"""

from __future__ import annotations

import torch


BEV_GRID = 60
BEV_RES_M = 0.05
BEV_REGION_M = BEV_GRID * BEV_RES_M  # 3.0 m

# Default footprint: forward-only along +X (stairs ahead), centred laterally.
DEFAULT_X_RANGE = (0.0, BEV_REGION_M)
DEFAULT_Y_RANGE = (-BEV_REGION_M / 2.0, BEV_REGION_M / 2.0)


def points_to_bev(
    points: torch.Tensor,
    *,
    x_range: tuple[float, float] = DEFAULT_X_RANGE,
    y_range: tuple[float, float] = DEFAULT_Y_RANGE,
    resolution: float = BEV_RES_M,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Vectorized point-cloud -> 6-channel BEV.

    Args:
        points: ``(N, 3)`` or ``(B, N, 3)`` in robot frame
            (+X forward, +Y left, +Z up).
        x_range: BEV span along +X. Default ``(0, 3)`` m (stair-ahead footprint).
        y_range: BEV span along +Y. Default ``(-1.5, +1.5)`` m.
        resolution: Cell size in metres. Default ``0.05`` m.
        valid_mask: Optional ``(N,)`` / ``(B, N)`` boolean mask. Useful for
            padded ragged clouds where some points are dummies.

    Returns:
        ``(6, H, W)`` for an unbatched input, or ``(B, 6, H, W)`` for a batched
        one. Channel order is ``[max, min, mean, max - min, std, density]``.
        Density is per-batch normalized so the busiest cell is 1.0; empty cells
        are zero across every channel.
    """
    assert points.shape[-1] == 3, f"points must have last dim 3, got {tuple(points.shape)}"

    batched = points.dim() == 3
    if not batched:
        points = points.unsqueeze(0)
        if valid_mask is not None:
            valid_mask = valid_mask.unsqueeze(0)

    B, N, _ = points.shape
    device = points.device
    dtype = points.dtype

    # Grid extents derived from ranges + resolution (handles non-square BEVs too).
    W = int(round((x_range[1] - x_range[0]) / resolution))  # cells along +X
    H = int(round((y_range[1] - y_range[0]) / resolution))  # cells along +Y
    n_cells = B * H * W

    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]

    in_x = (x >= x_range[0]) & (x < x_range[1])
    in_y = (y >= y_range[0]) & (y < y_range[1])
    valid = in_x & in_y
    if valid_mask is not None:
        valid = valid & valid_mask.bool()

    # cell indices, clamped (invalid rows are masked out below)
    ix = torch.clamp(((x - x_range[0]) / resolution).long(), 0, W - 1)  # X bin
    iy = torch.clamp(((y - y_range[0]) / resolution).long(), 0, H - 1)  # Y bin

    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
    flat_idx = batch_idx * (H * W) + iy * W + ix  # (B, N) in [0, B*H*W)
    flat_idx = flat_idx.reshape(-1)

    valid_f = valid.to(dtype).reshape(-1)
    z_flat = z.reshape(-1)
    z_valid = torch.where(valid.reshape(-1), z_flat, torch.zeros_like(z_flat))

    # --- per-cell reductions ----------------------------------------------------
    count = torch.zeros(n_cells, device=device, dtype=dtype)
    count.scatter_add_(0, flat_idx, valid_f)

    sum_z = torch.zeros(n_cells, device=device, dtype=dtype)
    sum_z.scatter_add_(0, flat_idx, z_valid)

    sum_z2 = torch.zeros(n_cells, device=device, dtype=dtype)
    sum_z2.scatter_add_(0, flat_idx, z_valid * z_valid)

    finfo = torch.finfo(dtype)
    neg_inf = finfo.min
    pos_inf = finfo.max
    z_for_max = torch.where(valid.reshape(-1), z_flat, torch.full_like(z_flat, neg_inf))
    z_for_min = torch.where(valid.reshape(-1), z_flat, torch.full_like(z_flat, pos_inf))

    max_z = torch.full((n_cells,), neg_inf, device=device, dtype=dtype)
    max_z.scatter_reduce_(0, flat_idx, z_for_max, reduce="amax", include_self=True)

    min_z = torch.full((n_cells,), pos_inf, device=device, dtype=dtype)
    min_z.scatter_reduce_(0, flat_idx, z_for_min, reduce="amin", include_self=True)

    # --- derived statistics, with empty-cell zero-fill -------------------------
    nonempty = count > 0
    safe_count = torch.where(nonempty, count, torch.ones_like(count))
    mean_z = torch.where(nonempty, sum_z / safe_count, torch.zeros_like(count))
    var = (sum_z2 / safe_count) - mean_z * mean_z
    std_z = torch.where(nonempty, torch.sqrt(var.clamp_min(0.0)), torch.zeros_like(count))
    range_z = torch.where(nonempty, max_z - min_z, torch.zeros_like(count))
    max_z = torch.where(nonempty, max_z, torch.zeros_like(count))
    min_z = torch.where(nonempty, min_z, torch.zeros_like(count))

    # density: per-batch normalize so the busiest cell == 1.0
    count_bhw = count.view(B, H * W)
    max_count = count_bhw.amax(dim=1, keepdim=True).clamp_min(1.0)
    density = (count_bhw / max_count).view(n_cells)

    out = torch.stack(
        [max_z, min_z, mean_z, range_z, std_z, density],
        dim=0,
    )  # (6, B*H*W)
    out = out.view(6, B, H, W).permute(1, 0, 2, 3).contiguous()  # (B, 6, H, W)

    if not batched:
        out = out.squeeze(0)
    return out


__all__ = ["BEV_GRID", "BEV_RES_M", "BEV_REGION_M", "points_to_bev"]
