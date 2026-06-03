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
    dataset_slug = _slug_join(cfg.dataset.name)
    model_slug = _slug_join(cfg.model.name)
    base_slug = _slug_join(dataset_slug, model_slug)
    study_slug = _relative_slug(study, [base_slug, model_slug, dataset_slug])

    group = explicit_group or _slug_join(base_slug, study_slug)
    name = explicit_name or _build_run_name(cfg, study_slug, base_slug, study)
    return RunIdentity(name=name, group=group)


def _build_run_name(cfg: DictConfig, study: str, base_slug: str, raw_study: str) -> str:
    training_parts = [
        cfg.optimizer.name,
        f"lr{_format_number(cfg.optimizer.lr)}",
        f"bs{cfg.dataset.batch_size}",
    ]
    scheduler = cfg.optimizer.get("scheduler")
    if scheduler is not None:
        scheduler_name = _optional_str(scheduler.get("name"))
        if scheduler_name and scheduler_name not in {"none"}:
            training_parts.append(scheduler_name)

    details_parts = []
    model_params = cfg.model.get("params", {})
    if "hidden_dim" in model_params:
        details_parts.append(f"w{model_params.hidden_dim}")
    if "num_layers" in model_params:
        details_parts.append(f"d{model_params.num_layers}")
    if "dropout" in model_params:
        details_parts.append(f"do{_format_number(model_params.dropout)}")
    if "batch_norm" in model_params and model_params.batch_norm:
        details_parts.append("bn")
    if "residual" in model_params and model_params.residual:
        details_parts.append("res")
    if "convs_per_stage" in model_params:
        convs_per_stage = model_params.convs_per_stage
        if isinstance(convs_per_stage, list):
            convs_text = "x".join(str(depth) for depth in convs_per_stage)
        else:
            convs_text = str(convs_per_stage)
        details_parts.append(f"cps{convs_text}")
    if "latent_dim" in model_params:
        details_parts.append(f"z{model_params.latent_dim}")
    if "channels" in model_params:
        details_parts.append("ch" + "x".join(str(width) for width in model_params.channels))
    if "depth" in model_params:
        details_parts.append(f"d{model_params.depth}")
    if "width_factor" in model_params:
        details_parts.append(f"k{model_params.width_factor}")
    if cfg.loss.get("beta", 1.0) != 1.0:
        details_parts.append(f"beta{_format_number(cfg.loss.beta)}")
    if cfg.loss.get("label_smoothing", 0.0) != 0.0:
        details_parts.append(f"ls{_format_number(cfg.loss.label_smoothing)}")
    mixup = cfg.loss.get("mixup", {})
    if mixup.get("enabled", False):
        details_parts.append(f"mixup{_format_number(mixup.get('alpha', 0.2))}")
    cutmix = cfg.loss.get("cutmix", {})
    if cutmix.get("enabled", False):
        details_parts.append(f"cutmix{_format_number(cutmix.get('alpha', 1.0))}")
    weight_averaging = cfg.get("weight_averaging", {})
    weight_averaging_name = _optional_str(weight_averaging.get("name"))
    if weight_averaging_name and weight_averaging_name not in {"none", "null"}:
        details_parts.append(weight_averaging_name)

    sections = [base_slug, study, _slug_join(*training_parts), _slug_join(*details_parts)]
    sweep_name = _optional_str(cfg.run.get("sweep_name"))
    sweep_index = cfg.run.get("sweep_index")
    if sweep_name:
        sweep_prefixes = [
            _slug_join(base_slug, study),
            base_slug,
            _slug_join(cfg.model.name),
            _slug_join(cfg.dataset.name),
            _slug_join(raw_study),
            study,
        ]
        sweep_parts = [_relative_slug(sweep_name, sweep_prefixes)]
        if sweep_index is not None:
            sweep_parts.append(f"i{int(sweep_index):03d}")
        sections.append(_slug_join(*sweep_parts))

    sections.append(f"seed{cfg.seed}")
    return _section_join(*sections)


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


def _relative_slug(value: Any, prefixes: list[str]) -> str:
    slug = _slug_join(value)
    for prefix in prefixes:
        candidate = _strip_token_prefix(slug, prefix)
        if candidate != slug and candidate not in {"run", "sweep", "experiment"}:
            return candidate
    return slug


def _strip_token_prefix(slug: str, prefix: str) -> str:
    slug_tokens = _slug_tokens(slug)
    prefix_tokens = _slug_tokens(prefix)
    if not prefix_tokens or slug_tokens[: len(prefix_tokens)] != prefix_tokens:
        return slug
    return "-".join(slug_tokens[len(prefix_tokens) :])


def _slug_tokens(value: str) -> list[str]:
    return [token for token in _slug_join(value).split("-") if token]


def _section_join(*sections: Any) -> str:
    return "_".join(
        slug for slug in (_slug_join(section) for section in sections) if slug
    )


def _slug_join(*parts: Any) -> str:
    text = "-".join(str(part) for part in parts if part is not None and str(part).strip())
    text = text.replace("_", "-")
    text = re.sub(r"[^A-Za-z0-9.+-]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-").lower()
