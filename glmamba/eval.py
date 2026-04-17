from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm

from glmamba.data import BraTS2021SliceDataset, BraTS2021SliceDatasetConfig
from glmamba.lightning_module import NormalizedMeanSquaredError
from glmamba.models import GLMamba, GLMambaConfig
from glmamba.utils.checkpoint import load_checkpoint
from glmamba.utils.device import get_device


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("glmamba-eval")
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--test-subjects", type=str, required=True, help="Text file with subject IDs (one per line).")
    p.add_argument("--scale", type=int, default=2, choices=(2, 4))
    p.add_argument("--normalize", type=str, default="minmax", choices=("minmax", "zscore_nonzero", "none"))
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default="auto")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    device = get_device(args.device)

    ds = BraTS2021SliceDataset(
        BraTS2021SliceDatasetConfig(
            root_dir=str(Path(args.data_root)),
            split="test",
            subjects_list=args.test_subjects,
            scale=args.scale,
            normalize=args.normalize,
        )
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = GLMamba(GLMambaConfig()).to(device)
    
    # Handle both Lightning (.ckpt) and regular (.pt) checkpoints
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    
    if "state_dict" in ckpt:
        # Lightning checkpoint - extract model weights
        state_dict = {}
        for key, value in ckpt["state_dict"].items():
            if key.startswith("model."):
                # Remove "model." prefix
                state_dict[key[6:]] = value
        model.load_state_dict(state_dict)
        print(f"Loaded Lightning checkpoint from epoch {ckpt.get('epoch', 'unknown')}")
    elif "model" in ckpt:
        # Regular checkpoint
        model.load_state_dict(ckpt["model"])
        print(f"Loaded regular checkpoint from epoch {ckpt.get('epoch', 'unknown')}")
    else:
        raise ValueError("Checkpoint format not recognized. Expected 'state_dict' or 'model' key.")
    
    model.eval()

    # Stateful TorchMetrics — exact match with Lightning validation_step
    metric_psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    metric_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    metric_nmse = NormalizedMeanSquaredError().to(device)

    n_images = 0
    for batch in tqdm(dl, desc="eval"):
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)
        ref = batch["ref"].to(device)
        with torch.no_grad():
            sr, _ = model(lr, ref)
        sr_clamped = sr.clamp(0, 1)
        hr_clamped = hr.clamp(0, 1)
        metric_psnr.update(sr_clamped, hr_clamped)
        metric_ssim.update(sr_clamped, hr_clamped)
        metric_nmse.update(sr, hr)
        n_images += lr.shape[0]

    final_psnr = float(metric_psnr.compute().item())
    final_ssim = float(metric_ssim.compute().item())
    final_nmse = float(metric_nmse.compute().item())

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Number of images: {n_images}")
    print(f"PSNR:  {final_psnr:.4f} dB")
    print(f"SSIM:  {final_ssim:.6f}")
    print(f"NMSE:  {final_nmse:.8f}")
    print("=" * 60)

    results = {"psnr": final_psnr, "ssim": final_ssim, "nmse": final_nmse, "num_images": n_images}
    print(f"\nJSON: {results}")


if __name__ == "__main__":
    main()

