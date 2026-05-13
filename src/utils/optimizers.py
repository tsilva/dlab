from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

OPTIMIZER_REGISTRY = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}


def build_optimizer(parameters: Iterable[torch.nn.Parameter], cfg: DictConfig):
    name = cfg.name
    if name not in OPTIMIZER_REGISTRY:
        raise KeyError(f"Unknown optimizer '{name}'. Available: {sorted(OPTIMIZER_REGISTRY)}")
    params: dict[str, Any] = dict(OmegaConf.to_container(cfg.get("params", {}), resolve=True))
    params.setdefault("lr", cfg.get("lr", 1e-3))
    params.setdefault("weight_decay", cfg.get("weight_decay", 0.0))
    return OPTIMIZER_REGISTRY[name](parameters, **params)

