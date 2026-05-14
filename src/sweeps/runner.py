from __future__ import annotations

import itertools
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.execution.launchers import (
    format_override_value,
    launch_train_command,
    launcher_cli_overrides,
)


def expand_grid(parameters: Mapping[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(parameters.keys())
    values = [parameters[key] for key in keys]
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]


def run_sweep(cfg: DictConfig) -> list[int]:
    if cfg.get("backend", "local") == "wandb":
        from src.sweeps.wandb import run_wandb_sweep

        run_wandb_sweep(cfg)
        return [0]

    raw_params = OmegaConf.to_container(cfg.parameters, resolve=True)
    if not isinstance(raw_params, Mapping):
        raise TypeError("Sweep parameters must be a mapping.")
    params = flatten_parameter_grid(raw_params)
    commands = []
    for index, combo in enumerate(expand_grid(params)):
        overrides = [f"{key}={format_override_value(value)}" for key, value in combo.items()]
        base_overrides = []
        if "experiment" not in params:
            base_overrides.append(f"experiment={cfg.experiment}")
        command = [
            sys.executable,
            str(Path(cfg.get("train_script", "train.py"))),
            *base_overrides,
            *launcher_cli_overrides(cfg),
            f"run.sweep_name={cfg.name}",
            f"run.sweep_index={index}",
            *_run_metadata_overrides(cfg),
            *overrides,
        ]
        commands.append(command)

    exit_codes = []
    for command in commands:
        print(" ".join(command), flush=True)
        return_code = launch_train_command(cfg, command)
        exit_codes.append(return_code)
        if return_code != 0 and cfg.get("stop_on_failure", True):
            break
    return exit_codes


def flatten_parameter_grid(parameters: Mapping[str, Any], prefix: str = "") -> dict[str, list[Any]]:
    flat: dict[str, list[Any]] = {}
    for key, value in parameters.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(flatten_parameter_grid(value, full_key))
            continue
        if not isinstance(value, list):
            raise TypeError(f"Sweep parameter '{full_key}' must be a list.")
        flat[full_key] = value
    return flat


def _run_metadata_overrides(cfg: DictConfig) -> list[str]:
    run_cfg = cfg.get("run")
    if run_cfg is None:
        return []
    run_metadata = OmegaConf.to_container(run_cfg, resolve=True)
    if not isinstance(run_metadata, Mapping):
        return []
    return [
        f"run.{key}={format_override_value(value)}"
        for key, value in sorted(run_metadata.items())
        if value is not None
    ]
