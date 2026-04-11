from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

try:
    import pytorch_lightning as pl
except Exception as e:  # pragma: no cover
    raise ImportError(
        "pytorch-lightning is required for glmamba.lightning_datamodule. "
        "Install it with `pip install pytorch-lightning`."
    ) from e

from glmamba.data import BraTS2021SliceDataset, BraTS2021SliceDatasetConfig


@dataclass(frozen=True)
class BraTSLightningDataConfig:
    data_root: str
    scale: int = 2
    normalize: str = "minmax"
    train_subjects: Optional[str] = None
    val_subjects: Optional[str] = None
    batch_size: int = 2
    num_workers: int = 4


class BraTS2021SliceDataModule(pl.LightningDataModule):
    def __init__(self, cfg: BraTSLightningDataConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.ds_train: BraTS2021SliceDataset | None = None
        self.ds_val: BraTS2021SliceDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        root = Path(self.cfg.data_root)

        if self.cfg.train_subjects is None or self.cfg.val_subjects is None:
            raise ValueError("train_subjects and val_subjects must be provided (Lightning entrypoint writes them).")

        self.ds_train = BraTS2021SliceDataset(
            BraTS2021SliceDatasetConfig(
                root_dir=str(root),
                split="train",
                subjects_list=self.cfg.train_subjects,
                scale=self.cfg.scale,
                normalize=self.cfg.normalize,
            )
        )
        self.ds_val = BraTS2021SliceDataset(
            BraTS2021SliceDatasetConfig(
                root_dir=str(root),
                split="test",
                subjects_list=self.cfg.val_subjects,
                scale=self.cfg.scale,
                normalize=self.cfg.normalize,
            )
        )

    def train_dataloader(self) -> DataLoader:
        assert self.ds_train is not None
        return DataLoader(
            self.ds_train,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.ds_val is not None
        return DataLoader(
            self.ds_val,
            batch_size=1,
            shuffle=False,
            num_workers=max(0, self.cfg.num_workers // 2),
            pin_memory=torch.cuda.is_available(),
        )

