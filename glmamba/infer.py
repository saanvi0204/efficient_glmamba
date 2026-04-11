from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from glmamba.data.degrade import degrade_frequency_domain
from glmamba.models import GLMamba, GLMambaConfig
from glmamba.utils.checkpoint import load_checkpoint
from glmamba.utils.device import get_device


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("glmamba-infer")
    p.add_argument("--t2-nifti", type=str, required=True, help="Target modality volume (T2) NIfTI path.")
    p.add_argument("--t1-nifti", type=str, required=True, help="Reference modality volume (T1) NIfTI path.")
    p.add_argument("--slice-idx", type=int, required=True)
    p.add_argument("--scale", type=int, default=2, choices=(2, 4))
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out-npy", type=str, required=True)
    p.add_argument("--device", type=str, default="auto")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    device = get_device(args.device)

    t2 = nib.load(args.t2_nifti)
    t1 = nib.load(args.t1_nifti)
    z = int(args.slice_idx)
    hr_np = np.asarray(t2.dataobj[:, :, z], dtype=np.float32)
    ref_np = np.asarray(t1.dataobj[:, :, z], dtype=np.float32)

    hr = torch.from_numpy(hr_np)
    ref = torch.from_numpy(ref_np)
    # min-max to [0,1] for inference
    hr = (hr - hr.min()) / (hr.max() - hr.min() + 1e-8)
    ref = (ref - ref.min()) / (ref.max() - ref.min() + 1e-8)

    H, W = hr.shape
    Hc = (H // args.scale) * args.scale
    Wc = (W // args.scale) * args.scale
    if (Hc != H) or (Wc != W):
        top = (H - Hc) // 2
        left = (W - Wc) // 2
        hr = hr[top : top + Hc, left : left + Wc]
        ref = ref[top : top + Hc, left : left + Wc]

    lr = degrade_frequency_domain(hr, args.scale)
    lr = lr.unsqueeze(0).unsqueeze(0).to(device)  # (1,1,h,w)
    ref = ref.unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)

    model = GLMamba(GLMambaConfig()).to(device)
    ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    with torch.no_grad():
        sr, _ = model(lr, ref)
    sr_np = sr.squeeze().detach().cpu().numpy()
    Path(args.out_npy).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_npy, sr_np)


if __name__ == "__main__":
    main()

