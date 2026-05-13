from __future__ import annotations

import asyncio
import inspect
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path

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
        endpoint_name = str(cfg.launcher.get("endpoint_name", "dlab-train"))
        _print_monitor_urls(result, endpoint_name=endpoint_name)
        result = _poll_queued_result(result, endpoint_name=endpoint_name)
        return RunResult(**_run_result_payload(result))

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


def _run_result_payload(result: object) -> dict:
    if not isinstance(result, dict):
        raise RuntimeError(f"RunPod Flash returned {type(result).__name__}, expected a dict.")

    if "run_dir" in result and "metrics" in result:
        return result

    output = result.get("output")
    if isinstance(output, dict):
        if "error" in output:
            raise RuntimeError(f"RunPod Flash remote execution failed: {output['error']}")
        if "run_dir" in output and "metrics" in output:
            return output

    status = result.get("status")
    if status is not None:
        raise RuntimeError(
            "RunPod Flash did not return experiment output. "
            f"Job status: {status}; response keys: {sorted(result)}"
        )

    raise RuntimeError(f"RunPod Flash returned an unexpected response with keys: {sorted(result)}")


def _poll_queued_result(result: object, endpoint_name: str) -> object:
    if not isinstance(result, dict):
        return result
    if result.get("status") not in {"IN_QUEUE", "IN_PROGRESS"} or "id" not in result:
        return result

    endpoint_id = _deployed_endpoint_id(endpoint_name)
    if endpoint_id is None:
        raise RuntimeError(
            f"RunPod Flash job {result['id']} is {result['status']}, but no endpoint id was "
            f"found for {endpoint_name!r}. Run `uv run flash deploy --app dlab --env production "
            "--python-version 3.12` and try again."
        )

    from runpod_flash.core.credentials import get_api_key
    from runpod_flash.core.utils.http import get_authenticated_requests_session

    timeout = float(os.environ.get("FLASH_SENTINEL_TIMEOUT", "600"))
    deadline = time.monotonic() + timeout
    job_id = str(result["id"])
    poll_interval = float(os.environ.get("RUNPOD_FLASH_POLL_INTERVAL", "5"))
    next_health_check = 0.0

    print(
        f"RunPod Flash job {job_id} is {result['status']} on endpoint {endpoint_id}. "
        f"Polling for up to {timeout:g}s..."
    )

    with get_authenticated_requests_session(api_key_override=get_api_key()) as session:
        while True:
            status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
            response = session.get(status_url, timeout=30)
            response.raise_for_status()
            job_state = response.json()
            status = job_state.get("status")
            if status not in {"IN_QUEUE", "IN_PROGRESS"}:
                print(f"RunPod Flash job {job_id} finished with status {status}.")
                return job_state

            now = time.monotonic()
            if now >= next_health_check:
                health = _endpoint_health(session, endpoint_id)
                health_summary = ""
                if isinstance(health, dict):
                    health_summary = f" health={health}"
                print(f"RunPod Flash job {job_id}: {status}.{health_summary}")
                next_health_check = now + 30

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"RunPod Flash job {job_id} stayed {status} for {timeout:g}s. "
                    "Check endpoint health/logs in RunPod or increase FLASH_SENTINEL_TIMEOUT."
                )
            time.sleep(poll_interval)


def _endpoint_health(session, endpoint_id: str) -> dict | None:
    try:
        response = session.get(f"https://api.runpod.ai/v2/{endpoint_id}/health", timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        print(f"Could not read RunPod endpoint health: {exc}")
        return None


def _deployed_endpoint_id(endpoint_name: str) -> str | None:
    manifest_path = Path(".flash") / "flash_manifest.json"
    if not manifest_path.exists():
        return None

    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return None

    resource = manifest.get("resources", {}).get(endpoint_name, {})
    endpoint_id = resource.get("endpoint_id")
    if isinstance(endpoint_id, str) and endpoint_id:
        return endpoint_id

    endpoint_url = manifest.get("resources_endpoints", {}).get(endpoint_name)
    if isinstance(endpoint_url, str) and endpoint_url.rstrip("/"):
        return endpoint_url.rstrip("/").split("/")[-1]

    return None


def _print_monitor_urls(result: object, endpoint_name: str) -> None:
    if not isinstance(result, dict) or "id" not in result:
        return

    endpoint_id = _deployed_endpoint_id(endpoint_name)
    if endpoint_id is None:
        return

    job_id = result["id"]
    print(f"RunPod endpoint: https://console.runpod.io/serverless/user/endpoint/{endpoint_id}")
    print(f"RunPod job status API: https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}")
    print(f"RunPod endpoint health API: https://api.runpod.ai/v2/{endpoint_id}/health")
