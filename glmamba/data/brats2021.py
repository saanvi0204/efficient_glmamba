from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from .degrade import degrade_frequency_domain


@dataclass(frozen=True)
class BraTS2021SliceDatasetConfig:
    root_dir: str
    split: str = "train"  
    subjects_list: str | None = None  
    scale: int = 2  
    target_modality: str = "t2"
    ref_modality: str = "t1"
    normalize: str = "minmax"
    eps: float = 1e-8


def _find_modality_file(subject_dir: Path, modality: str) -> Path:
    """
    Tries to find a modality file for a subject directory.
    Common BraTS naming patterns include:
      BraTS2021_XXXX_flair.nii.gz, *_t1.nii.gz, *_t1ce.nii.gz, *_t2.nii.gz
    """
    modality = modality.lower()
    candidates = sorted(subject_dir.glob(f"*_{modality}.nii.gz"))
    if not candidates:
        candidates = sorted(subject_dir.glob(f"*_{modality}.nii"))
    if not candidates:
        raise FileNotFoundError(f"Could not find modality '{modality}' under {subject_dir}")
    return candidates[0]


def _normalize_slice(x: torch.Tensor, mode: str, eps: float) -> torch.Tensor:
    if mode == "none":
        return x
    if mode == "minmax":
        mn = torch.min(x)
        mx = torch.max(x)
        return (x - mn) / (mx - mn + eps)
    if mode == "zscore_nonzero":
        mask = x != 0
        if torch.count_nonzero(mask) < 10:
            # fall back
            mu = torch.mean(x)
            std = torch.std(x)
        else:
            vals = x[mask]
            mu = vals.mean()
            std = vals.std()
        return (x - mu) / (std + eps)
    raise ValueError(f"Unknown normalize mode: {mode}")


class BraTS2021SliceDataset(Dataset[dict[str, torch.Tensor]]):
    """
    Slice dataset for BraTS2021 (2D), returning:
      - lr: (1,h,w) low-res target slice (generated in frequency domain)
      - hr: (1,H,W) target high-res slice (ground truth)
      - ref: (1,H,W) high-res reference slice

    Expected directory structure:
      root_dir/
        BraTS2021_00000/
          BraTS2021_00000_t1.nii.gz
          BraTS2021_00000_t2.nii.gz
          ...
        BraTS2021_00001/
          ...
    """

    def __init__(self, cfg: BraTS2021SliceDatasetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.root = Path(cfg.root_dir)
        if not self.root.exists():
            raise FileNotFoundError(f"BraTS root_dir not found: {self.root}")
        if cfg.scale not in (2, 4):
            raise ValueError("scale must be 2 or 4")

        self.subject_dirs = self._resolve_subject_dirs()
        self.samples: list[tuple[int, int]] = []  
        self._build_index()

    def _resolve_subject_dirs(self) -> list[Path]:
        if self.cfg.subjects_list is not None:
            ids = [ln.strip() for ln in Path(self.cfg.subjects_list).read_text().splitlines() if ln.strip()]
            dirs = []
            for sid in ids:
                p = self.root / sid
                if not p.exists():
                    raise FileNotFoundError(f"Subject dir not found: {p}")
                dirs.append(p)
            return dirs

        dirs = [p for p in sorted(self.root.iterdir()) if p.is_dir()]
        if not dirs:
            raise FileNotFoundError(f"No subject directories found under {self.root}")
        return dirs

    def _build_index(self) -> None:
        for si, sdir in enumerate(self.subject_dirs):
            t_path = _find_modality_file(sdir, self.cfg.target_modality)
            img = nib.load(str(t_path))
            shape = img.shape
            if len(shape) != 3:
                raise ValueError(f"Expected 3D volume for {t_path}, got shape {shape}")
            depth = shape[2]
            for z in range(depth):
                self.samples.append((si, z))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        si, z = self.samples[idx]
        sdir = self.subject_dirs[si]
        t_path = _find_modality_file(sdir, self.cfg.target_modality)
        r_path = _find_modality_file(sdir, self.cfg.ref_modality)

        t_img = nib.load(str(t_path))
        r_img = nib.load(str(r_path))

        hr_np = np.asarray(t_img.dataobj[:, :, z], dtype=np.float32).copy()
        ref_np = np.asarray(r_img.dataobj[:, :, z], dtype=np.float32).copy()

        hr = torch.from_numpy(hr_np)
        ref = torch.from_numpy(ref_np)

        hr = _normalize_slice(hr, self.cfg.normalize, self.cfg.eps)
        ref = _normalize_slice(ref, self.cfg.normalize, self.cfg.eps)

        # Ensure H,W divisible by scale by center-cropping
        H, W = hr.shape
        Hc = (H // self.cfg.scale) * self.cfg.scale
        Wc = (W // self.cfg.scale) * self.cfg.scale
        if (Hc != H) or (Wc != W):
            top = (H - Hc) // 2
            left = (W - Wc) // 2
            hr = hr[top : top + Hc, left : left + Wc]
            ref = ref[top : top + Hc, left : left + Wc]

        lr = degrade_frequency_domain(hr, self.cfg.scale)

        # add channel dimension
        return {
            "lr": lr.unsqueeze(0),   # (1,H/scale,W/scale)
            "hr": hr.unsqueeze(0),   # (1,H,W)
            "ref": ref.unsqueeze(0),  # (1,H,W)
            "subject_idx": torch.tensor(si, dtype=torch.int64),
            "slice_idx": torch.tensor(z, dtype=torch.int64),
        }

