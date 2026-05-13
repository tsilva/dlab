from __future__ import annotations

import asyncio
import inspect
from collections.abc import Sequence

from omegaconf import DictConfig, OmegaConf

from src.execution.experiment import RunResult
from src.execution.launchers import LocalLauncher


class RunpodFlashLauncher:
    def launch_experiment(self, cfg: DictConfig) -> RunResult:
        try:
            from src.execution.runpod_flash_app import remote_train
        except ImportError as exc:
            raise RuntimeError(
                "RunPod Flash launcher requires the Flash SDK. Install it with "
                "`uv add runpod-flash`, then authenticate with `uv run flash login`."
            ) from exc

        config_yaml = OmegaConf.to_yaml(cfg, resolve=True)

        result = remote_train(config_yaml)
        if inspect.isawaitable(result):
            result = asyncio.run(result)
        return RunResult(**result)

    def launch_train_command(self, command: Sequence[str]) -> int:
        return LocalLauncher().launch_train_command(command)


def _resolve_gpu(name: str, gpu_type, gpu_group):
    if hasattr(gpu_type, name):
        return getattr(gpu_type, name)
    if hasattr(gpu_group, name):
        return getattr(gpu_group, name)
    raise ValueError(f"Unknown RunPod Flash GPU '{name}'. Use a GpuType or GpuGroup enum name.")


def _filter_supported_endpoint_kwargs(endpoint, kwargs: dict[str, object]) -> dict[str, object]:
    try:
        parameters = inspect.signature(endpoint).parameters
    except (TypeError, ValueError):
        return kwargs

    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return kwargs

    return {key: value for key, value in kwargs.items() if key in parameters}
