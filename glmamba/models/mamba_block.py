from __future__ import annotations

import torch
import torch.nn as nn

from .layers import ChannelAttention, LayerNorm
from .ss2d import SS2D


class MambaBlock2D(nn.Module):
    """
    Paper Fig. 2(A): layer norm → two paths —
      path1: linear (1×1 conv) + activation;
      path2: linear, depthwise separable (here: 1×1 + depthwise 3×3), activation, SS2D, layer norm;
      multiply paths; channel attention; residual add.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

        self.ln = LayerNorm(channels, channel_first=True)

        self.p1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.SiLU(),
        )

        self.p2_in = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.p2_dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=True)
        self.p2_act = nn.SiLU()
        self.ss2d = SS2D(channels)
        self.p2_norm = LayerNorm(channels, channel_first=True)

        self.post_gate = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.ca = ChannelAttention(channels)

    @staticmethod
    def _evs_transform(x: torch.Tensor, block_idx: int) -> torch.Tensor:
        """EVSSM scan schedule: alternate transpose and flip on the full block input.
        The transform is NOT inverted — the next block's opposite transform implicitly
        re-orients the feature map (see EVSSM CVPR 2025, EVS block)."""
        if block_idx % 2 == 0:
            return x.transpose(2, 3).contiguous()
        return torch.flip(x, dims=(2, 3)).contiguous()

    def forward(self, x: torch.Tensor, *, block_idx: int = 0) -> torch.Tensor:
        # Apply EVSSM flip/transpose on the full input (not inverted)
        x = self._evs_transform(x, block_idx)

        h = self.ln(x)
        a = self.p1(h)
        b = self.p2_in(h)
        b = self.p2_dw(b)
        b = self.p2_act(b)
        b = self.ss2d(b)
        b = self.p2_norm(b)
        y = a * b
        y = self.post_gate(y)
        y = self.ca(y)
        return x + y


class LocalMamba2D(nn.Module):
    """
    Local Mamba (paper Fig. 3(B)): partition the feature map into four spatial quadrants
    (top-left, top-right, bottom-left, bottom-right), run a Mamba block in each, then merge.
    Odd H/W use floor/ceil halves via ``split`` so every pixel belongs to exactly one quadrant.
    Four blocks give independent parameters per quadrant (“independently learn” in the paper).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(MambaBlock2D(channels) for _ in range(4))

    def forward(self, x: torch.Tensor, *, block_idx: int = 0) -> torch.Tensor:
        _b, _c, h, w = x.shape
        h2, w2 = h // 2, w // 2
        if h2 < 1 or w2 < 1:
            return self.blocks[0](x, block_idx=block_idx)

        top, bottom = x.split((h2, h - h2), dim=2)
        q1, q2 = top.split((w2, w - w2), dim=3)
        q3, q4 = bottom.split((w2, w - w2), dim=3)

        q1 = self.blocks[0](q1, block_idx=block_idx)
        q2 = self.blocks[1](q2, block_idx=block_idx)
        q3 = self.blocks[2](q3, block_idx=block_idx)
        q4 = self.blocks[3](q4, block_idx=block_idx)

        top_m = torch.cat([q1, q2], dim=3)
        bottom_m = torch.cat([q3, q4], dim=3)
        return torch.cat([top_m, bottom_m], dim=2)

