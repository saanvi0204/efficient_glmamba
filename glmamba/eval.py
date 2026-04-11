from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from glmamba.data import BraTS2021SliceDataset, BraTS2021SliceDatasetConfig
from glmamba.metrics import nmse, psnr, ssim
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
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    agg = {"psnr": 0.0, "ssim": 0.0, "nmse": 0.0, "n": 0.0}
    for batch in tqdm(dl, desc="eval"):
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)
        ref = batch["ref"].to(device)
        with torch.no_grad():
            sr, _ = model(lr, ref)
        agg["psnr"] += float(psnr(sr.clamp(0, 1), hr.clamp(0, 1), data_range=1.0).detach().cpu().item())
        agg["ssim"] += float(ssim(sr.clamp(0, 1), hr.clamp(0, 1), data_range=1.0).detach().cpu().item())
        agg["nmse"] += float(nmse(sr, hr).detach().cpu().item())
        agg["n"] += 1.0
    n = max(1.0, agg["n"])
    print(
        {
            "psnr": agg["psnr"] / n,
            "ssim": agg["ssim"] / n,
            "nmse": agg["nmse"] / n,
            "num_samples": int(n),
        }
    )


if __name__ == "__main__":
    main()

