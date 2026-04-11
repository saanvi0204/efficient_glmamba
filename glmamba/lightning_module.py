from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

try:
    import pytorch_lightning as pl
except Exception as e:  # pragma: no cover
    raise ImportError(
        "pytorch-lightning is required for glmamba.lightning_module. "
        "Install it with `pip install pytorch-lightning`."
    ) from e

from glmamba.losses import GLMambaLoss, GLMambaLossConfig
from glmamba.metrics import nmse, psnr, ssim
from glmamba.models import GLMamba, GLMambaConfig


@dataclass(frozen=True)
class GLMambaLightningConfig:
    lr: float = 2e-4
    weight_decay: float = 0.0
    model: GLMambaConfig = GLMambaConfig()
    loss: GLMambaLossConfig = GLMambaLossConfig()


class GLMambaLightningModule(pl.LightningModule):
    """
    Lightning wrapper for GLMamba that mirrors glmamba/train.py behavior:
    - loss: alpha*L1(sr,hr) + beta*L1(rec_ref,ref) + gamma*CELoss(sr,hr)
    - val metrics: PSNR/SSIM on clamped [0,1], NMSE on raw tensors
    """

    def __init__(self, cfg: GLMambaLightningConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or GLMambaLightningConfig()
        self.model = GLMamba(self.cfg.model)
        self.loss_fn = GLMambaLoss(self.cfg.loss)

        # Lightning will store this in checkpoints for reproducibility.
        self.save_hyperparameters(
            {
                "lr": self.cfg.lr,
                "weight_decay": self.cfg.weight_decay,
                "model": self.cfg.model.__dict__,
                "loss": self.cfg.loss.__dict__,
            }
        )

    def forward(self, lr: torch.Tensor, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(lr, ref)

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        lr = batch["lr"]
        hr = batch["hr"]
        ref = batch["ref"]

        sr, rec_ref = self(lr, ref)
        losses = self.loss_fn(sr, hr, rec_ref, ref)
        loss = losses["loss"]

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=lr.shape[0])
        return loss

    @torch.no_grad()
    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        lr = batch["lr"]
        hr = batch["hr"]
        ref = batch["ref"]

        sr, _ = self(lr, ref)

        # match glmamba/train.py: clamp for PSNR/SSIM, raw for NMSE
        sr01 = sr.clamp(0, 1)
        hr01 = hr.clamp(0, 1)

        psnr_v = psnr(sr01, hr01, data_range=1.0).to(torch.float32)
        ssim_v = ssim(sr01, hr01, data_range=1.0).to(torch.float32)
        nmse_v = nmse(sr, hr).to(torch.float32)

        self.log("val/psnr", psnr_v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/ssim", ssim_v, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val/nmse", nmse_v, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

