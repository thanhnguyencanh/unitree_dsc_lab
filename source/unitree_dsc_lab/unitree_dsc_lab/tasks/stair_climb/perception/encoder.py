"""CNN student encoder: BEV (6 x 60 x 60) -> z_t = (class_logits[3], h, d, yaw).

Architecture (paper §III-A.3):
  Conv blocks (Conv2d + BatchNorm + ReLU), progressive downsampling
  -> F_enc in R^{128 x 8 x 8}
  -> MLP heads:  (logits_class_3,  h_pred,  d_pred,  yaw_pred)

Yaw is regressed as a sanity head only — at runtime yaw comes directly from the
proprioception observation (see `mdp/observations.py`).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BEVStudentEncoder(nn.Module):
    def __init__(self, in_channels: int = 6, feat_dim: int = 128) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.feat_dim = feat_dim
        # TODO: build Conv2d stack -> (B, 128, 8, 8) and the four MLP heads.

    def forward(self, bev: torch.Tensor) -> dict[str, torch.Tensor]:
        """Args: bev (B, 6, 60, 60).

        Returns dict with keys: logits_class (B, 3), h (B,), d (B,), yaw (B,).
        """
        raise NotImplementedError
