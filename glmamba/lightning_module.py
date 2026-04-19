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

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics import Metric, MeanSquaredError

from glmamba.losses import GLMambaLoss, GLMambaLossConfig
from glmamba.models import GLMamba, GLMambaConfig


class NormalizedMeanSquaredError(Metric):
    """
    Normalized Mean Squared Error metric for TorchMetrics.
    NMSE = MSE(pred, target) / mean(target^2)
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_squared_target", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update state with predictions and targets."""
        assert preds.shape == target.shape
        
        self.sum_squared_error += torch.sum((preds - target) ** 2)
        self.sum_squared_target += torch.sum(target ** 2)
        self.total += target.numel()
        
    def compute(self):
        """Compute NMSE from accumulated statistics."""
        return self.sum_squared_error / (self.sum_squared_target + 1e-8)


@dataclass(frozen=True)
class GLMambaLightningConfig:
    lr: float = 2e-4
    weight_decay: float = 0.0
    model: GLMambaConfig = GLMambaConfig()
    loss: GLMambaLossConfig = GLMambaLossConfig()


class GLMambaLightningModule(pl.LightningModule):
    """
    Lightning wrapper:
    - loss: alpha*L1(sr,hr) + beta*L1(rec_ref,ref) + gamma*CELoss(sr,hr)
    - val metrics: PSNR/SSIM on clamped [0,1], NMSE on raw tensors
    """

    def __init__(self, cfg: GLMambaLightningConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or GLMambaLightningConfig()
        self.model = GLMamba(self.cfg.model)
        self.loss_fn = GLMambaLoss(self.cfg.loss)

        # Initialize torchmetrics as stateful metrics (proper Lightning way)
        # These automatically handle DDP synchronization and state accumulation
        self.val_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.val_nmse = NormalizedMeanSquaredError()

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

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=lr.shape[0], sync_dist=True)
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        lr = batch["lr"]
        hr = batch["hr"]
        ref = batch["ref"]

        sr, _ = self(lr, ref)

        # Clamp for PSNR/SSIM
        sr_clamped = sr.clamp(0, 1)
        hr_clamped = hr.clamp(0, 1)

        # Update metrics
        self.val_psnr(sr_clamped, hr_clamped)
        self.val_ssim(sr_clamped, hr_clamped)
        self.val_nmse(sr, hr)

        # Log metrics
        self.log("val/psnr", self.val_psnr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/ssim", self.val_ssim, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/nmse", self.val_nmse, on_step=False, on_epoch=True, prog_bar=False)

