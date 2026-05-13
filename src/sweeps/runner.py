from __future__ import annotations

import itertools
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def expand_grid(parameters: Mapping[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(parameters.keys())
    values = [parameters[key] for key in keys]
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]


def run_sweep(cfg: DictConfig) -> list[int]:
    params = OmegaConf.to_container(cfg.parameters, resolve=True)
    commands = []
    for index, combo in enumerate(expand_grid(params)):
        overrides = [f"{key}={_format_override_value(value)}" for key, value in combo.items()]
        command = [
            sys.executable,
            str(Path(cfg.get("train_script", "train.py"))),
            f"experiment={cfg.experiment}",
            f"run.sweep_name={cfg.name}",
            f"run.sweep_index={index}",
            *overrides,
        ]
        commands.append(command)

    exit_codes = []
    for command in commands:
        print(" ".join(command), flush=True)
        completed = subprocess.run(command, check=False)
        exit_codes.append(completed.returncode)
        if completed.returncode != 0 and cfg.get("stop_on_failure", True):
            break
    return exit_codes


def _format_override_value(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ",".join(_format_override_value(item) for item in value) + "]"
    if isinstance(value, tuple):
        return "[" + ",".join(_format_override_value(item) for item in value) + "]"
    return str(value)
