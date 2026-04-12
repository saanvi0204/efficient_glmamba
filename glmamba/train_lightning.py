from __future__ import annotations

import argparse
from pathlib import Path

import torch

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
except Exception as e:  # pragma: no cover
    raise ImportError(
        "pytorch-lightning is required for glmamba/train_lightning.py. "
        "Install it with `pip install pytorch-lightning`."
    ) from e

from glmamba.lightning_datamodule import BraTS2021SliceDataModule, BraTSLightningDataConfig
from glmamba.lightning_module import GLMambaLightningConfig, GLMambaLightningModule
from glmamba.utils.seed import SeedConfig, seed_everything


def _run_diagnostics(device: torch.device) -> None:
    """Smoke-test DeformBlock and selective_scan_cuda before training."""
    print("=" * 50)
    print("DIAGNOSTICS: verifying critical components")
    print("=" * 50)

    # --- 0. GPU info ---
    print("\n[0] GPU status:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Device count:   {torch.cuda.device_count()}")
        print(f"  Current device: {torch.cuda.current_device()}")
        print(f"  GPU name:       {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU memory:     {mem:.1f} GB")
    else:
        print("  WARNING: No GPU detected — training will run on CPU!")

    # --- 1. DeformConv2d / DeformBlock ---
    print("\n[1/2] DeformBlock (torchvision.ops.DeformConv2d) ...")
    try:
        from glmamba.models.layers import DeformBlock

        blk = DeformBlock(channels=16, kernel_size=3).to(device)
        x = torch.randn(1, 16, 8, 8, device=device)
        y = blk(x)
        print(f"  input  shape: {tuple(x.shape)}")
        print(f"  output shape: {tuple(y.shape)}")
        assert y.shape == x.shape, f"shape mismatch: expected {x.shape}, got {y.shape}"
        assert torch.isfinite(y).all(), "output contains NaN/Inf"
        print("  => DeformBlock OK")
    except Exception as exc:
        print(f"  => DeformBlock FAILED: {exc}")

    # --- 2. selective_scan_cuda / SS2D ---
    print("\n[2/2] SS2D (selective_scan_cuda) ...")
    try:
        import selective_scan_cuda

        print(f"  selective_scan_cuda module loaded: {selective_scan_cuda}")
    except ImportError as exc:
        print(f"  => selective_scan_cuda import FAILED: {exc}")
        print("=" * 50)
        return

    try:
        from glmamba.models.ss2d import SS2D

        ss = SS2D(dim=16).to(device)
        x = torch.randn(1, 16, 8, 8, device=device)
        y = ss(x)
        print(f"  input  shape: {tuple(x.shape)}")
        print(f"  output shape: {tuple(y.shape)}")
        assert y.shape == x.shape, f"shape mismatch: expected {x.shape}, got {y.shape}"
        assert torch.isfinite(y).all(), "output contains NaN/Inf"
        print("  => SS2D OK")
    except Exception as exc:
        print(f"  => SS2D FAILED: {exc}")

    print("=" * 50)
    print("DIAGNOSTICS complete")
    print("=" * 50, flush=True)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("glmamba-train-lightning")
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
    p.add_argument("--amp", action="store_true")

    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--resume", type=str, default=None, help="Path to Lightning checkpoint to resume from.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")
    return p


def _list_subject_dirs(root: Path) -> list[str]:
    return [p.name for p in sorted(root.iterdir()) if p.is_dir()]


def _make_split(root: Path, seed: int) -> tuple[list[str], list[str]]:
    ids = _list_subject_dirs(root)
    if len(ids) < 2:
        raise RuntimeError("Not enough subjects to split.")
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(ids), generator=g).tolist()
    ids = [ids[i] for i in perm]
    n_train = int(round(len(ids) * 0.7287))
    n_train = max(1, min(n_train, len(ids) - 1))
    return ids[:n_train], ids[n_train:]


def main() -> None:
    args = build_argparser().parse_args()

    seed_everything(SeedConfig(seed=args.seed, deterministic=args.deterministic))

    diag_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _run_diagnostics(diag_device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.train_subjects and args.val_subjects:
        train_list = args.train_subjects
        val_list = args.val_subjects
    else:
        root = Path(args.data_root)
        train_ids, val_ids = _make_split(root, args.seed)
        train_path = out_dir / "train_subjects.txt"
        val_path = out_dir / "val_subjects.txt"
        train_path.write_text("\n".join(train_ids) + "\n")
        val_path.write_text("\n".join(val_ids) + "\n")
        train_list = str(train_path)
        val_list = str(val_path)

    dm = BraTS2021SliceDataModule(
        BraTSLightningDataConfig(
            data_root=args.data_root,
            scale=args.scale,
            normalize=args.normalize,
            train_subjects=train_list,
            val_subjects=val_list,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    )

    module = GLMambaLightningModule(
        GLMambaLightningConfig(lr=args.lr, weight_decay=args.weight_decay)
    )

    ckpt_best = ModelCheckpoint(
        dirpath=str(out_dir),
        filename="best_psnr",
        monitor="val/psnr",
        mode="max",
        save_top_k=1,
        save_last=False,
    )
    ckpt_last = ModelCheckpoint(
        dirpath=str(out_dir),
        filename="last",
        save_last=True,
        monitor=None,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1
    precision = "16-mixed" if (args.amp and torch.cuda.is_available()) else "32"

    trainer = pl.Trainer(
        default_root_dir=str(out_dir),
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        deterministic=args.deterministic,
        callbacks=[ckpt_best, ckpt_last],
        log_every_n_steps=10,
        enable_checkpointing=True,
    )

    trainer.fit(module, datamodule=dm, ckpt_path=args.resume)


if __name__ == "__main__":
    main()

