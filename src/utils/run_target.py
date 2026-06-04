from __future__ import annotations

import os
import platform
import socket
import subprocess
from typing import Any

from omegaconf import DictConfig, OmegaConf


def collect_run_target(cfg: DictConfig) -> dict[str, Any]:
    launcher = cfg.get("launcher", {})
    launcher_name = str(launcher.get("name", "local"))
    provider = _provider_name(launcher_name)
    gpu_requested = launcher.get("gpu")
    hostname = socket.gethostname()

    target: dict[str, Any] = {
        "name": _target_name(provider, hostname, gpu_requested),
        "provider": provider,
        "launcher": launcher_name,
        "is_remote": provider != "local",
        "hostname": hostname,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "cpu": {
            "count_logical": os.cpu_count(),
            "brand": _cpu_brand(),
        },
        "memory": {
            "total_gb": _memory_total_gb(),
        },
        "gpu": _gpu_specs(gpu_requested),
        "container": _container_metadata(provider),
    }
    return target


def run_target_summary(target: dict[str, Any] | DictConfig) -> dict[str, Any]:
    if isinstance(target, DictConfig):
        target = OmegaConf.to_container(target, resolve=True)  # type: ignore[assignment]
    gpu = target.get("gpu", {})
    cpu = target.get("cpu", {})
    memory = target.get("memory", {})
    platform_info = target.get("platform", {})
    container = target.get("container", {})

    summary = {
        "target/name": target.get("name"),
        "target/provider": target.get("provider"),
        "target/launcher": target.get("launcher"),
        "target/is_remote": bool(target.get("is_remote", False)),
        "target/hostname": target.get("hostname"),
        "target/platform": _join_present(
            platform_info.get("system"),
            platform_info.get("release"),
            platform_info.get("machine"),
        ),
        "target/python": platform_info.get("python"),
        "target/cpu_count_logical": cpu.get("count_logical"),
        "target/cpu_brand": cpu.get("brand"),
        "target/memory_total_gb": memory.get("total_gb"),
        "target/gpu_requested": gpu.get("requested"),
        "target/gpu_count": gpu.get("count"),
        "target/gpu_names": ", ".join(gpu.get("names") or []),
        "target/cuda_available": bool(gpu.get("cuda_available", False)),
        "target/cuda_version": gpu.get("cuda_version"),
        "target/mps_available": bool(gpu.get("mps_available", False)),
    }
    for key in ("modal_app_id", "modal_function_call_id", "runpod_pod_id", "runpod_gpu_count"):
        value = container.get(key)
        if value not in {None, ""}:
            summary[f"target/{key}"] = value
    return {key: value for key, value in summary.items() if value not in {None, ""}}


def _provider_name(launcher_name: str) -> str:
    if os.environ.get("MODAL_APP_ID") or os.environ.get("MODAL_TASK_ID"):
        return "modal"
    if os.environ.get("RUNPOD_POD_ID") or os.environ.get("RUNPOD_GPU_COUNT"):
        return "runpod_flash"
    if launcher_name in {"modal", "runpod_flash"}:
        return launcher_name
    return "local"


def _target_name(provider: str, hostname: str, gpu_requested: Any) -> str:
    if provider == "local":
        return f"local:{hostname}"
    if gpu_requested not in {None, "", "none", "null"}:
        return f"{provider}:{gpu_requested}"
    return provider


def _gpu_specs(gpu_requested: Any) -> dict[str, Any]:
    specs: dict[str, Any] = {
        "requested": None if gpu_requested in {None, "", "none", "null"} else str(gpu_requested),
        "cuda_available": False,
        "count": 0,
        "names": [],
        "cuda_version": None,
        "mps_available": False,
    }
    try:
        import torch
    except ImportError:
        return specs

    specs["cuda_available"] = bool(torch.cuda.is_available())
    specs["cuda_version"] = getattr(torch.version, "cuda", None)
    if specs["cuda_available"]:
        specs["count"] = torch.cuda.device_count()
        specs["names"] = [torch.cuda.get_device_name(index) for index in range(specs["count"])]
    mps = getattr(torch.backends, "mps", None)
    if mps is not None:
        specs["mps_available"] = bool(mps.is_available())
    return specs


def _container_metadata(provider: str) -> dict[str, Any]:
    if provider == "modal":
        return {
            "modal_app_id": os.environ.get("MODAL_APP_ID"),
            "modal_function_call_id": os.environ.get("MODAL_FUNCTION_CALL_ID"),
            "modal_task_id": os.environ.get("MODAL_TASK_ID"),
        }
    if provider == "runpod_flash":
        return {
            "runpod_pod_id": os.environ.get("RUNPOD_POD_ID"),
            "runpod_gpu_count": os.environ.get("RUNPOD_GPU_COUNT"),
        }
    return {}


def _cpu_brand() -> str | None:
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        brand = result.stdout.strip()
        return brand or None
    processor = platform.processor()
    return processor or None


def _memory_total_gb() -> float | None:
    if not hasattr(os, "sysconf"):
        return None
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError):
        return None
    return round((pages * page_size) / (1024**3), 2)


def _join_present(*parts: Any) -> str:
    return " ".join(str(part) for part in parts if part not in {None, ""})
