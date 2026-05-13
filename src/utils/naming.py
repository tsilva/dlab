from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig


@dataclass(frozen=True)
class RunIdentity:
    name: str
    group: str


def resolve_run_identity(cfg: DictConfig) -> RunIdentity:
    explicit_name = _optional_str(cfg.get("experiment_name"))
    explicit_group = _optional_str(cfg.run.get("group"))
    study = _optional_str(cfg.run.get("study")) or _optional_str(cfg.get("task")) or "experiment"

    group = explicit_group or _slug_join(cfg.dataset.name, cfg.model.name, study)
    name = explicit_name or _build_run_name(cfg, study)
    return RunIdentity(name=name, group=group)


def _build_run_name(cfg: DictConfig, study: str) -> str:
    parts = [
        cfg.dataset.name,
        cfg.model.name,
        study,
        cfg.optimizer.name,
        f"lr{_format_number(cfg.optimizer.lr)}",
        f"bs{cfg.dataset.batch_size}",
    ]

    model_params = cfg.model.get("params", {})
    if "hidden_dim" in model_params:
        parts.append(f"w{model_params.hidden_dim}")
    if "num_layers" in model_params:
        parts.append(f"d{model_params.num_layers}")
    if "dropout" in model_params:
        parts.append(f"do{_format_number(model_params.dropout)}")
    if "latent_dim" in model_params:
        parts.append(f"z{model_params.latent_dim}")
    if "channels" in model_params:
        parts.append("ch" + "x".join(str(width) for width in model_params.channels))
    if cfg.loss.get("beta", 1.0) != 1.0:
        parts.append(f"beta{_format_number(cfg.loss.beta)}")

    sweep_name = _optional_str(cfg.run.get("sweep_name"))
    sweep_index = cfg.run.get("sweep_index")
    if sweep_name:
        parts.extend([sweep_name, f"i{int(sweep_index):03d}"])

    parts.append(f"seed{cfg.seed}")
    return _slug_join(*parts)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _format_number(value: Any) -> str:
    if isinstance(value, float):
        text = f"{value:g}"
    else:
        text = str(value)
    return text.replace("-", "m").replace(".", "p")


def _slug_join(*parts: Any) -> str:
    text = "-".join(str(part) for part in parts if part is not None and str(part).strip())
    text = text.replace("_", "-")
    text = re.sub(r"[^A-Za-z0-9.+-]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-").lower()

