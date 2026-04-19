from __future__ import annotations

import torch


def _center_crop2d(x: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    h, w = x.shape[-2], x.shape[-1]
    if out_h > h or out_w > w:
        raise ValueError(f"Requested crop {(out_h,out_w)} larger than {(h,w)}")
    top = (h - out_h) // 2
    left = (w - out_w) // 2
    return x[..., top : top + out_h, left : left + out_w]


@torch.no_grad()
def degrade_frequency_domain(hr: torch.Tensor, scale: int) -> torch.Tensor:
    """
    Frequency-domain degradation to generate LR samples

    Args:
      hr: (H,W) or (...,H,W) real tensor
      scale: 2 or 4
    Returns:
      lr: (H/scale, W/scale) or (...,H/scale, W/scale)
    """
    if scale <= 1:
        raise ValueError("scale must be >=2")
    if hr.dtype not in (torch.float16, torch.float32, torch.float64):
        hr = hr.float()

    H, W = hr.shape[-2], hr.shape[-1]
    H2 = (H // scale)
    W2 = (W // scale)
    if H2 < 2 or W2 < 2:
        raise ValueError(f"Too small after scaling: {(H,W)} / {scale}")

    # FFT -> shift -> center crop in k-space -> IFFT to smaller spatial resolution
    k = torch.fft.fft2(hr, dim=(-2, -1))
    k = torch.fft.fftshift(k, dim=(-2, -1))
    k_crop = _center_crop2d(k, H2, W2)
    k_crop = torch.fft.ifftshift(k_crop, dim=(-2, -1))
    lr_c = torch.fft.ifft2(k_crop, dim=(-2, -1))
    # Compensate for energy change after k-space crop: IFFT divides by H'*W' instead
    # of H*W, inflating pixel values by scale². Rescale to preserve intensity range.
    lr = lr_c.real / (scale * scale)
    return lr

