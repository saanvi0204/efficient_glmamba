from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class CELoss(nn.Module):
    """
    Contrastive Edge Loss (paper Eq. 14) using 3 fixed 3x3 Laplacian-like kernels.

    L_CELoss = (1/3) * sum_i || E_i ⊙ SR - E_i ⊙ HR ||^2
    Where ⊙ is convolution with kernel E_i.
    """

    def __init__(self) -> None:
        super().__init__()
        k1 = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32)
        k2 = torch.tensor([[-1, 0, -1], [0, 4, 0], [-1, 0, 1]], dtype=torch.float32)
        k3 = torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=torch.float32)
        kernels = torch.stack([k1, k2, k3], dim=0)  # (3,3,3)
        self.register_buffer("kernels", kernels, persistent=False)

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        assert sr.shape == hr.shape
        B, C, H, W = sr.shape

        # apply each kernel depthwise per channel
        k = self.kernels.to(sr.dtype).to(sr.device)  # (3,3,3)
        k = k[:, None, :, :]  # (3,1,3,3)
        k = k.repeat(C, 1, 1, 1)  # (3*C,1,3,3)

        # group conv: apply 3 filters per channel by reshaping
        # reshape input to (B*C,1,H,W) then conv with (3*C,1,3,3) groups=C? simpler:
        # We'll do grouped conv with groups=C using weight shaped (C*3,1,3,3) and
        # input shaped (B,C,H,W) after repeating channels 3x via unfold trick.
        # Instead: compute per-kernel conv with groups=C and average.
        losses = []
        for i in range(3):
            wi = self.kernels[i].to(sr.dtype).to(sr.device)[None, None, :, :]  # (1,1,3,3)
            wi = wi.repeat(C, 1, 1, 1)  # (C,1,3,3)
            sr_e = F.conv2d(sr, wi, padding=1, groups=C)
            hr_e = F.conv2d(hr, wi, padding=1, groups=C)
            losses.append(F.mse_loss(sr_e, hr_e, reduction="mean"))
        return sum(losses) / 3.0


@dataclass(frozen=True)
class GLMambaLossConfig:
    alpha: float = 0.7
    beta: float = 0.3
    gamma: float = 0.1


class GLMambaLoss(nn.Module):
    """
    Paper Eq.15: alpha*L1(sr,hr) + beta*L1(rec_ref,ref) + gamma*CELoss(sr,hr)
    """

    def __init__(self, cfg: GLMambaLossConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or GLMambaLossConfig()
        self.ce = CELoss()

    def forward(
        self,
        sr: torch.Tensor,
        hr: torch.Tensor,
        rec_ref: torch.Tensor,
        ref: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        l1_sr = F.l1_loss(sr, hr)
        l1_ref = F.l1_loss(rec_ref, ref)
        l_ce = self.ce(sr, hr)
        total = self.cfg.alpha * l1_sr + self.cfg.beta * l1_ref + self.cfg.gamma * l_ce
        return {
            "loss": total,
            "l1_sr": l1_sr.detach(),
            "l1_ref": l1_ref.detach(),
            "celoss": l_ce.detach(),
        }

