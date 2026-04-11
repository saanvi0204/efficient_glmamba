from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SeedConfig:
    seed: int = 42
    deterministic: bool = False


def seed_everything(cfg: SeedConfig) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    if cfg.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

