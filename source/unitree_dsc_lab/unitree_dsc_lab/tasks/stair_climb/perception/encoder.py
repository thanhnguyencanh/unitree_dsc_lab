"""CNN student encoder: BEV (6 x 60 x 60) -> z_t = (class_logits[3], h, d, yaw).

Architecture (paper §III-A.3):

  Conv2d(6,  32, k=3, s=2, p=1) + BN + ReLU   -> (32, 30, 30)
  Conv2d(32, 64, k=3, s=2, p=1) + BN + ReLU   -> (64, 15, 15)
  Conv2d(64, 128, k=3, s=2, p=1) + BN + ReLU  -> (128, 8, 8)   = F_enc
  Flatten -> Linear(128*8*8, 256) + ReLU -> Linear(256, 128) + ReLU
  4 heads: class (3), h_step (1), d_step (1), theta_yaw (1)

The yaw head is regressed as a sanity check; the **policy** receives yaw from
the proprioception observation (IMU). Stage 2 supervision uses only class + h
+ d (Eq. 8): ``L_terrain = 0.6 * CE + 1.0 * L1(h) + 1.0 * L1(d)``.

At inference time, :meth:`predict_token` returns a ``(B, 4)`` tensor with the
same channel layout as :meth:`PrivilegedTeacher.token`, so swapping the
student for the teacher in the env observation is a one-line change.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BEVPrediction:
    """Encoder forward output.

    Each tensor has leading batch dim ``B``.
    """

    logits_class: torch.Tensor   # (B, 3)
    h_step: torch.Tensor          # (B,)
    d_step: torch.Tensor          # (B,)
    theta_yaw: torch.Tensor       # (B,)
    feat: torch.Tensor            # (B, 128, 8, 8) — F_enc, exposed for joint training


class BEVStudentEncoder(nn.Module):
    """6-channel BEV -> 4-D terrain token."""

    def __init__(self, in_channels: int = 6, mlp_hidden: int = 256, head_hidden: int = 128) -> None:
        super().__init__()
        self.in_channels = in_channels

        # ---- Conv trunk: 60 -> 30 -> 15 -> 8 ---------------------------------
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # ---- Shared MLP backbone --------------------------------------------
        flat_dim = 128 * 8 * 8
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, head_hidden),
            nn.ReLU(inplace=True),
        )

        # ---- Heads ----------------------------------------------------------
        self.head_class = nn.Linear(head_hidden, 3)
        self.head_h = nn.Linear(head_hidden, 1)
        self.head_d = nn.Linear(head_hidden, 1)
        self.head_yaw = nn.Linear(head_hidden, 1)

    def forward(self, bev: torch.Tensor) -> BEVPrediction:
        """Args:
            bev: ``(B, 6, 60, 60)`` BEV from :func:`points_to_bev`.

        Returns:
            :class:`BEVPrediction` with class logits, scalar h/d/yaw, and F_enc.
        """
        feat = self.trunk(bev)            # (B, 128, 8, 8)
        h = self.mlp(feat)                # (B, head_hidden)
        return BEVPrediction(
            logits_class=self.head_class(h),
            h_step=self.head_h(h).squeeze(-1),
            d_step=self.head_d(h).squeeze(-1),
            theta_yaw=self.head_yaw(h).squeeze(-1),
            feat=feat,
        )

    # ------------------------------------------------------------------
    # Convenience: drop-in replacement for ``PrivilegedTeacher.token``
    # ------------------------------------------------------------------

    def predict_token(self, bev: torch.Tensor) -> torch.Tensor:
        """Return ``(B, 4)`` z_t = ``[class_id, h, d, theta_yaw]``.

        ``class_id`` is the argmax of the class logits cast to float, matching
        the layout used by :class:`PrivilegedTeacher` so the env-side
        observation term can swap teacher for student without further glue.
        """
        pred = self.forward(bev)
        class_id = pred.logits_class.argmax(dim=-1).to(pred.h_step.dtype)
        return torch.stack((class_id, pred.h_step, pred.d_step, pred.theta_yaw), dim=-1)


def terrain_loss(
    pred: BEVPrediction,
    target: torch.Tensor,
    *,
    lambda_cls: float = 0.6,
    lambda_h: float = 1.0,
    lambda_d: float = 1.0,
    lambda_yaw: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Paper Eq. 8 + yaw supervision — terrain perception loss.

    ``L_terrain = λ_cls*CE(class) + λ_h*SmoothL1(h) + λ_d*SmoothL1(d) + λ_yaw*SmoothL1(yaw)``.

    Paper Eq. (8) lists only cls/h/d, but Table II reports MAE(θ_yaw)=1.8° in sim,
    which requires explicit yaw supervision.  λ_yaw=1.0 matches the h/d weights.

    Args:
        pred: Encoder forward output.
        target: ``(B, 4)`` ground-truth ``[class_id, h, d, yaw]`` — same layout
            as :meth:`PrivilegedTeacher.token`.

    Returns:
        Dict with ``"total"``, ``"cls"``, ``"h"``, ``"d"``, ``"yaw"`` losses.
    """
    if target.shape[-1] != 4:
        raise ValueError(f"target must be (B, 4); got {tuple(target.shape)}")

    class_gt = target[..., 0].to(torch.long)
    h_gt = target[..., 1]
    d_gt = target[..., 2]
    yaw_gt = target[..., 3]

    loss_cls = F.cross_entropy(pred.logits_class, class_gt)
    loss_h = F.smooth_l1_loss(pred.h_step, h_gt)
    loss_d = F.smooth_l1_loss(pred.d_step, d_gt)
    loss_yaw = F.smooth_l1_loss(pred.theta_yaw, yaw_gt)

    total = lambda_cls * loss_cls + lambda_h * loss_h + lambda_d * loss_d + lambda_yaw * loss_yaw
    return {"total": total, "cls": loss_cls, "h": loss_h, "d": loss_d, "yaw": loss_yaw}


__all__ = ["BEVPrediction", "BEVStudentEncoder", "terrain_loss"]
