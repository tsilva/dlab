from __future__ import annotations

import os

import pytorch_lightning as pl
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    pl.seed_everything(seed, workers=True)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

