from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf
from runpod_flash import Endpoint, GpuGroup, GpuType

from src.execution.runpod_flash_launcher import (
    _filter_supported_endpoint_kwargs,
    _resolve_gpu,
)


def _load_launcher_defaults():
    root_dir = Path(__file__).resolve().parents[2]
    return OmegaConf.load(root_dir / "configs" / "launcher" / "runpod_flash.yaml")


def _endpoint_kwargs() -> dict[str, object]:
    launcher = _load_launcher_defaults()
    endpoint_options = {
        "name": str(launcher.get("endpoint_name", "dlab-train")),
        "gpu": _resolve_gpu(launcher.get("gpu", "ANY"), GpuType, GpuGroup),
        "workers": tuple(launcher.get("workers", [0, 1])),
        "dependencies": list(launcher.get("dependencies", [])),
        "python_version": str(launcher.get("python_version", "3.12")),
    }
    for key in ("idle_timeout", "execution_timeout_ms", "flashboot", "max_concurrency"):
        if key in launcher and launcher.get(key) is not None:
            endpoint_options[key] = launcher.get(key)

    return _filter_supported_endpoint_kwargs(
        Endpoint,
        endpoint_options,
    )


@Endpoint(**_endpoint_kwargs())
async def remote_train(config_yaml: str) -> dict:
    from omegaconf import OmegaConf

    from src.execution.experiment import run_experiment

    return run_experiment(OmegaConf.create(config_yaml)).to_dict()
