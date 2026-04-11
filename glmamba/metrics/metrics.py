from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def nmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalized Mean Squared Error (tensor)."""
    num = torch.mean((pred - target) ** 2)
    den = torch.mean(target**2) + eps
    return num / den


def _psnr_update(preds: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, int]:
    sum_squared_error = torch.sum((preds - target) ** 2)
    n_obs = int(target.numel())
    return sum_squared_error, n_obs


def _psnr_compute(
    sum_squared_error: torch.Tensor,
    n_obs: int,
    data_range: torch.Tensor,
    base: float = 10.0,
) -> torch.Tensor:
    # Equivalent to: 10 * log10(data_range^2 / (SSE / n_obs))
    psnr_base_e = 2 * torch.log(data_range) - torch.log(sum_squared_error / float(n_obs))
    return psnr_base_e * (10.0 / torch.log(torch.tensor(base, device=sum_squared_error.device)))


def psnr(
    preds: torch.Tensor,
    target: torch.Tensor,
    data_range: float | None = 1.0,
    base: float = 10.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """PSNR in dB (tensor). Assumes inputs are scaled to [0, data_range] if provided."""
    if data_range is None:
        data_range_t = (target.max() - target.min()).clamp_min(eps)
    else:
        data_range_t = torch.tensor(float(data_range), device=target.device, dtype=target.dtype).clamp_min(eps)
    sse, n_obs = _psnr_update(preds, target)
    sse = sse.clamp_min(eps)
    return _psnr_compute(sse, n_obs, data_range_t, base=base)


def _gaussian_kernel(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel2d = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)  # (1,1,W,W)
    return kernel2d


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    SSIM (single-scale) for NCHW tensors. Returns mean SSIM over batch and channels.
    """
    assert pred.shape == target.shape and pred.ndim == 4
    B, C, H, W = pred.shape

    # ensure odd window and not larger than image
    window_size = int(window_size)
    if window_size % 2 == 0:
        window_size += 1
    window_size = min(window_size, H if H % 2 == 1 else H - 1, W if W % 2 == 1 else W - 1)
    if window_size < 3:
        # too small; fall back to a simple similarity proxy
        return (1.0 - torch.mean((pred - target).abs())).clamp(0, 1)

    kernel = _gaussian_kernel(window_size, sigma, pred.device, pred.dtype)
    kernel = kernel.repeat(C, 1, 1, 1)  # (C,1,ws,ws)

    pad = window_size // 2
    mu1 = F.conv2d(pred, kernel, padding=pad, groups=C)
    mu2 = F.conv2d(target, kernel, padding=pad, groups=C)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, kernel, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel, padding=pad, groups=C) - mu12

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + eps)
    return ssim_map.mean()

