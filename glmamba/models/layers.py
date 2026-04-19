from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


class LayerNorm(nn.LayerNorm):
    """
    LayerNorm for NCHW when ``channel_first=True`` (permute to NHWC, normalize channel dim, permute back).
    Same behavior as the VMamba SS2D helper; use for Mamba blocks and anywhere else feature maps are NCHW.
    """

    def __init__(
        self,
        *args,
        channel_first: bool | None = None,
        in_channel_first: bool = False,
        out_channel_first: bool = False,
        **kwargs,
    ) -> None:
        nn.LayerNorm.__init__(self, *args, **kwargs)
        if channel_first is not None:
            in_channel_first = channel_first
            out_channel_first = channel_first
        self.in_channel_first = in_channel_first
        self.out_channel_first = out_channel_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.in_channel_first:
            x = x.permute(0, 2, 3, 1)
        x = nn.LayerNorm.forward(self, x)
        if self.out_channel_first:
            x = x.permute(0, 3, 1, 2)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)
        w = F.adaptive_avg_pool2d(x, 1).contiguous()
        w = F.silu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


class PatchEmbed2x2(nn.Module):
    """
    Non-overlapping 2x2 patches, stride=2.
    Returns a feature map at H/2, W/2 with channels=embed_dim.
    """

    def __init__(self, in_ch: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=2, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class PatchUnembed2x2(nn.Module):
    """
    Inverse of PatchEmbed2x2 (upsample by 2).
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.proj = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)          
    
class DeformBlock(nn.Module):
    """Bottleneck: C→C//4→deformConv(C//4)→C//4→C + residual. ~7× cheaper per block."""
    def __init__(self, channels: int, kernel_size: int = 3, bottleneck_ratio: int = 4) -> None:
        super().__init__()
        bn = max(16, channels // bottleneck_ratio)     # 96//4 = 24
        self.padding = kernel_size // 2
        self.down = nn.Conv2d(channels, bn, kernel_size=1, bias=False)
        self.offset_conv = nn.Conv2d(bn, 2*kernel_size*kernel_size, kernel_size=3, padding=1)
        self.mask_conv = nn.Conv2d(bn, kernel_size*kernel_size,   kernel_size=3, padding=1)
        self.deform_conv = DeformConv2d(
            in_channels=bn, out_channels=bn,
            kernel_size=kernel_size, stride=1, padding=self.padding, bias=True)
        self.up = nn.Conv2d(bn, channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(channels)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.down(x)                # C → C//4
        offset = self.offset_conv(z)    # deform on C/4
        mask = torch.sigmoid(self.mask_conv(z))
        z = self.deform_conv(z, offset, mask)
        z = self.up(z)                  # C//4 → C
        return F.silu(self.norm(x + z)) # residual + norm


class Modulator(nn.Module):
    """
    Paper Fig.2(C): Mamba features are selectively scaled by sigmoid(deform_feat), then summed
    with the deformable features (no extra mixing layers).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

    def forward(self, deform_feat: torch.Tensor, mamba_feat: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(deform_feat)
        modulated_mamba = mamba_feat * gate
        return modulated_mamba + deform_feat


class MultiModalityFusion(nn.Module):
    """
    Multi-modality feature fusion block (paper §3.3, Fig. 4).

    Expects branch features aligned with the paper's ``F_LR↑`` and ``F_Ref`` (here: modulated
    LR / Ref maps at the same resolution).

    - Eq. (6)  difference: ``F_fuse^di = F_LR↑ − F_Ref``
    - Eq. (7)  similarity: ``F_fuse^sim = F_LR↑ ⊙ F_Ref`` (element-wise product)
    - Eqs. (8–9) complementarity: ``w_LR↑, w_Ref = softmax(conv(cat(F_LR↑, F_Ref)))`` (softmax
      over the ``2C`` conv channels at each spatial location), then
      ``F_fuse^com = F_LR↑ ⊙ w_LR↑ + F_Ref ⊙ w_Ref``
    - Eqs. (10–11) dynamic fusion: ``cat(F_fuse^di, F_fuse^sim, F_fuse^com)`` → global max
      pooling per channel → FC → softmax → weighted sum of the three feature maps.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.comp_conv = nn.Conv2d(2 * channels, 2 * channels, kernel_size=1, bias=True)
        self.w_fc = nn.Linear(3 * channels, 3, bias=True)

    def forward(self, f_lr: torch.Tensor, f_ref: torch.Tensor) -> torch.Tensor:
        # Eqs. (6–7)
        f_fuse_di = f_lr - f_ref
        f_fuse_sim = f_lr * f_ref

        # Eqs. (8–9): complementary weights then weighted sum of modalities
        # Softmax across the 2-modality dimension per channel so w_lr[c] + w_ref[c] = 1
        B = f_lr.shape[0]
        c = f_lr.shape[1]
        z = self.comp_conv(torch.cat([f_lr, f_ref], dim=1))  # (B, 2C, H, W)
        z = z.view(B, 2, c, z.shape[2], z.shape[3])
        w_pair = torch.softmax(z, dim=1)  # softmax over modality pair
        w_lr = w_pair[:, 0]   # (B, C, H, W)
        w_ref = w_pair[:, 1]  # (B, C, H, W)
        f_fuse_com = f_lr * w_lr + f_ref * w_ref

        # Eqs. (10–11): GMP over spatial dims per channel, then FC + softmax → three scalars per batch
        fused_cat = torch.cat([f_fuse_di, f_fuse_sim, f_fuse_com], dim=1)
        pooled = F.adaptive_max_pool2d(fused_cat, 1).flatten(1)
        w = torch.softmax(self.w_fc(pooled), dim=1)
        w_di = w[:, 0:1, None, None]
        w_sim = w[:, 1:2, None, None]
        w_com = w[:, 2:3, None, None]
        return f_fuse_di * w_di + f_fuse_sim * w_sim + f_fuse_com * w_com

