from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import DeformBlock, Modulator, MultiModalityFusion, PatchEmbed2x2, PatchUnembed2x2
from .mamba_block import LocalMamba2D, MambaBlock2D


@dataclass(frozen=True)
class GLMambaConfig:
    in_ch: int = 1
    out_ch: int = 1
    channels: int = 96
    n_blocks: int = 4  # deform + mamba blocks count (paper uses 4)


class GLMamba(nn.Module):
    """
    GLMamba implementation (high-level faithful):
      - LR is upsampled to Ref size and embedded into patch features (2x2 stride-2)
      - Global branch: stacked MambaBlock2D + DeformBlock (+ Modulator)
      - Ref branch: stacked LocalMamba2D + DeformBlock (+ Modulator)
      - Fusion: MultiModalityFusion on modulated features
      - Reconstruct: SR and RecRef via convtranspose (unpatch) + conv
    """

    def __init__(self, cfg: GLMambaConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.lr_up = nn.Upsample(scale_factor=1.0, mode="bilinear", align_corners=False)  # scale chosen at runtime

        self.embed_lr = PatchEmbed2x2(cfg.in_ch, cfg.channels)
        self.embed_ref = PatchEmbed2x2(cfg.in_ch, cfg.channels)

        self.g_mamba = nn.ModuleList([MambaBlock2D(cfg.channels) for _ in range(cfg.n_blocks)])
        self.l_mamba = nn.ModuleList([LocalMamba2D(cfg.channels) for _ in range(cfg.n_blocks)])

        self.g_deform = nn.Sequential(*[DeformBlock(cfg.channels) for _ in range(cfg.n_blocks)])
        self.l_deform = nn.Sequential(*[DeformBlock(cfg.channels) for _ in range(cfg.n_blocks)])

        self.mod_g = Modulator(cfg.channels)
        self.mod_l = Modulator(cfg.channels)

        self.fuse = MultiModalityFusion(cfg.channels)

        self.unembed_sr = PatchUnembed2x2(cfg.channels, cfg.channels)
        self.unembed_ref = PatchUnembed2x2(cfg.channels, cfg.channels)

        self.recon_sr = nn.Conv2d(cfg.channels, cfg.out_ch, kernel_size=3, padding=1)
        self.recon_ref = nn.Conv2d(cfg.channels, cfg.out_ch, kernel_size=3, padding=1)

    def forward(self, lr: torch.Tensor, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        lr: (B,1,h,w) low-res target modality
        ref: (B,1,H,W) high-res reference modality
        Returns:
          sr: (B,1,H,W)
          rec_ref: (B,1,H,W)
        """
        assert lr.ndim == 4 and ref.ndim == 4
        _, _, H, W = ref.shape

        # Upsample LR to match Ref size (paper: upsampling layer first)
        lr_up = F.interpolate(lr, size=(H, W), mode="bilinear", align_corners=False)

        # Patch embedding 2x2 stride2 (paper)
        f_lr = self.embed_lr(lr_up)   # (B,C,H/2,W/2)
        f_ref = self.embed_ref(ref)   # (B,C,H/2,W/2)

        # Global branch (LR): mamba + deform
        f_lr_m = f_lr
        for i, blk in enumerate(self.g_mamba):
            f_lr_m = blk(f_lr_m, block_idx=i)
        f_lr_d = self.g_deform(f_lr)
        f_lr_mod = self.mod_g(f_lr_d, f_lr_m)

        # Local branch (Ref): local mamba + deform
        f_ref_m = f_ref
        for i, blk in enumerate(self.l_mamba):
            f_ref_m = blk(f_ref_m, block_idx=i)
        f_ref_d = self.l_deform(f_ref)
        f_ref_mod = self.mod_l(f_ref_d, f_ref_m)

        # Fuse modalities
        f_fuse = self.fuse(f_lr_mod, f_ref_mod)

        # Reconstruct SR and RecRef (paper outputs both)
        sr_feat = self.unembed_sr(f_fuse)
        ref_feat = self.unembed_ref(f_ref_mod)
        sr = self.recon_sr(sr_feat)
        rec_ref = self.recon_ref(ref_feat)
        return sr, rec_ref

