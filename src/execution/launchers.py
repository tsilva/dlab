from __future__ import annotations

import subprocess
from collections.abc import Sequence
from typing import Protocol

from omegaconf import DictConfig, ListConfig, OmegaConf

from src.execution.experiment import RunResult, run_experiment


class ExperimentLauncher(Protocol):
    def launch_experiment(self, cfg: DictConfig) -> RunResult: ...

    def launch_train_command(self, command: Sequence[str]) -> int: ...


class LocalLauncher:
    def launch_experiment(self, cfg: DictConfig) -> RunResult:
        return run_experiment(cfg)

    def launch_train_command(self, command: Sequence[str]) -> int:
        completed = subprocess.run(command, check=False)
        return completed.returncode


def launch_experiment(cfg: DictConfig) -> RunResult:
    return get_launcher(cfg).launch_experiment(cfg)


def launch_train_command(cfg: DictConfig, command: Sequence[str]) -> int:
    return get_launcher(cfg).launch_train_command(command)


def get_launcher(cfg: DictConfig) -> ExperimentLauncher:
    name = get_launcher_name(cfg)
    if name == "local":
        return LocalLauncher()
    if name == "modal":
        from src.execution.modal_launcher import ModalLauncher

        return ModalLauncher()
    if name == "runpod_flash":
        from src.execution.runpod_flash_launcher import RunpodFlashLauncher

        return RunpodFlashLauncher()
    raise ValueError(f"Unknown launcher '{name}'. Expected one of: local, modal, runpod_flash")


def get_launcher_name(cfg: DictConfig) -> str:
    launcher_cfg = cfg.get("launcher")
    if launcher_cfg is None:
        return "local"
    return str(launcher_cfg.get("name", "local"))


def launcher_cli_overrides(cfg: DictConfig) -> list[str]:
    launcher_cfg = cfg.get("launcher")
    if launcher_cfg is None:
        return []

    launcher_name = get_launcher_name(cfg)
    container = OmegaConf.to_container(launcher_cfg, resolve=True)
    if not isinstance(container, dict):
        return [f"launcher={launcher_name}"]

    overrides = [f"launcher={launcher_name}"]
    for key, value in sorted(container.items()):
        if key == "name":
            continue
        overrides.append(f"launcher.{key}={format_override_value(value)}")
    overrides.extend(training_default_cli_overrides(cfg))
    return overrides


def training_default_cli_overrides(cfg: DictConfig) -> list[str]:
    overrides = []

    runtime_cfg = cfg.get("runtime")
    if runtime_cfg is not None:
        precision = runtime_cfg.get("float32_matmul_precision")
        if precision is not None:
            overrides.append(f"runtime.float32_matmul_precision={format_override_value(precision)}")

    dataset_cfg = cfg.get("dataset")
    if dataset_cfg is not None and dataset_cfg.get("num_workers") is not None:
        overrides.append(f"dataset.num_workers={format_override_value(dataset_cfg.num_workers)}")

    return overrides


def format_override_value(value: object) -> str:
    if isinstance(value, list | tuple | ListConfig):
        return "[" + ",".join(format_override_value(item) for item in value) + "]"
    if isinstance(value, dict):
        items = ",".join(f"{key}:{format_override_value(item)}" for key, item in value.items())
        return "{" + items + "}"
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"
    return str(value)
