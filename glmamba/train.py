from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from glmamba.data import BraTS2021SliceDataset, BraTS2021SliceDatasetConfig
from glmamba.losses import GLMambaLoss, GLMambaLossConfig
from glmamba.metrics import nmse, psnr, ssim
from glmamba.models import GLMamba, GLMambaConfig
from glmamba.utils.checkpoint import load_checkpoint, save_checkpoint
from glmamba.utils.device import get_device
from glmamba.utils.seed import SeedConfig, seed_everything

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("glmamba-train")
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--train-subjects", type=str, default=None, help="Text file with subject IDs (one per line).")
    p.add_argument("--val-subjects", type=str, default=None, help="Text file with subject IDs (one per line).")
    p.add_argument("--scale", type=int, default=2, choices=(2, 4))
    p.add_argument("--normalize", type=str, default="minmax", choices=("minmax", "zscore_nonzero", "none"))

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--amp", action="store_true", help="Use mixed precision when CUDA is available.")

    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")

    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")
    return p


def _list_subject_dirs(root: Path) -> list[str]:
    return [p.name for p in sorted(root.iterdir()) if p.is_dir()]


def _make_split(root: Path, seed: int) -> tuple[list[str], list[str]]:
    # If no explicit split is provided: stable random split (approx. paper ratio 419/575 ~ 0.7287)
    ids = _list_subject_dirs(root)
    if len(ids) < 2:
        raise RuntimeError("Not enough subjects to split.")
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(ids), generator=g).tolist()
    ids = [ids[i] for i in perm]
    n_train = int(round(len(ids) * 0.7287))
    n_train = max(1, min(n_train, len(ids) - 1))
    return ids[:n_train], ids[n_train:]


@torch.no_grad()
def _run_val(
    model: GLMamba,
    loader: DataLoader,
    device: torch.device,
    scale: int,
) -> dict[str, float]:
    model.eval()
    agg = {"psnr": 0.0, "ssim": 0.0, "nmse": 0.0, "n": 0.0}
    for batch in loader:
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)
        ref = batch["ref"].to(device)
        sr, _ = model(lr, ref)
        agg["psnr"] += float(psnr(sr.clamp(0, 1), hr.clamp(0, 1), data_range=1.0).detach().cpu().item())
        agg["ssim"] += float(ssim(sr.clamp(0, 1), hr.clamp(0, 1), data_range=1.0).detach().cpu().item())
        agg["nmse"] += float(nmse(sr, hr).detach().cpu().item())
        agg["n"] += 1.0
    n = max(1.0, agg["n"])
    return {"psnr": agg["psnr"] / n, "ssim": agg["ssim"] / n, "nmse": agg["nmse"] / n}


def main() -> None:
    args = build_argparser().parse_args()

    seed_everything(SeedConfig(seed=args.seed, deterministic=args.deterministic))
    device = get_device(args.device)

    root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.train_subjects and args.val_subjects:
        train_list = args.train_subjects
        val_list = args.val_subjects
    else:
        train_ids, val_ids = _make_split(root, args.seed)
        train_list = out_dir / "train_subjects.txt"
        val_list = out_dir / "val_subjects.txt"
        train_list.write_text("\n".join(train_ids) + "\n")
        val_list.write_text("\n".join(val_ids) + "\n")
        train_list = str(train_list)
        val_list = str(val_list)

    ds_train = BraTS2021SliceDataset(
        BraTS2021SliceDatasetConfig(
            root_dir=str(root),
            split="train",
            subjects_list=train_list,
            scale=args.scale,
            normalize=args.normalize,
        )
    )
    ds_val = BraTS2021SliceDataset(
        BraTS2021SliceDatasetConfig(
            root_dir=str(root),
            split="test",
            subjects_list=val_list,
            scale=args.scale,
            normalize=args.normalize,
        )
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, args.num_workers // 2),
        pin_memory=(device.type == "cuda"),
    )

    model = GLMamba(GLMambaConfig()).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=(args.amp and device.type == "cuda"))
    loss_fn = GLMambaLoss(GLMambaLossConfig()).to(device)

    start_epoch = 0
    best_psnr = float("-inf")

    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optim"])
        if "scaler" in ckpt and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_psnr = float(ckpt.get("best_psnr", best_psnr))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        pbar = tqdm(dl_train, desc=f"epoch {epoch}/{args.epochs-1}", leave=False)
        for batch in pbar:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            ref = batch["ref"].to(device)

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()):
                sr, rec_ref = model(lr, ref)
                losses = loss_fn(sr, hr, rec_ref, ref)
                loss = losses["loss"]

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            pbar.set_postfix({"loss": f"{float(loss.detach().cpu()):.4f}"})

        # validation
        if (epoch % args.val_every) == 0:
            val_metrics = _run_val(model, dl_val, device, args.scale)
            # save best
            if val_metrics["psnr"] > best_psnr:
                best_psnr = val_metrics["psnr"]
                save_checkpoint(
                    out_dir / "best_psnr.pt",
                    {
                        "model": model.state_dict(),
                        "optim": opt.state_dict(),
                        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                        "epoch": epoch,
                        "best_psnr": best_psnr,
                        "val": val_metrics,
                        "args": vars(args),
                    },
                )

        # periodic save
        if (epoch % args.save_every) == 0:
            save_checkpoint(
                out_dir / "last.pt",
                {
                    "model": model.state_dict(),
                    "optim": opt.state_dict(),
                    "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                    "epoch": epoch,
                    "best_psnr": best_psnr,
                    "args": vars(args),
                },
            )


if __name__ == "__main__":
    main()

