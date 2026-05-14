from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.execution.launchers import format_override_value, get_launcher_name, launcher_cli_overrides
from src.sweeps.runner import flatten_parameter_grid


def build_wandb_sweep_config(cfg: DictConfig) -> dict[str, Any]:
    raw_params = OmegaConf.to_container(cfg.parameters, resolve=True)
    if not isinstance(raw_params, dict):
        raise TypeError("Sweep parameters must be a mapping.")

    parameters = {
        key: {"values": values}
        for key, values in flatten_parameter_grid(raw_params).items()
    }
    if "experiment" not in parameters:
        parameters["experiment"] = {"value": _hydra_constant(cfg.experiment)}
    parameters["run.sweep_name"] = {"value": _hydra_constant(cfg.name)}
    parameters["wandb.enabled"] = {"value": True}
    parameters["wandb.project"] = {"value": _hydra_constant(cfg.wandb.get("project", "dlab"))}
    if cfg.wandb.get("entity") is not None:
        parameters["wandb.entity"] = {"value": _hydra_constant(cfg.wandb.entity)}
    for key, value in _run_metadata(cfg).items():
        if value is not None:
            parameters[f"run.{key}"] = {"value": _hydra_constant(value)}
    for override in launcher_cli_overrides(cfg):
        key, value = override.split("=", 1)
        parameters[key] = {"value": _parse_constant_value(value)}

    sweep_cfg = cfg.get("wandb_sweep", {})
    command_cfg = sweep_cfg.get("command")
    command = []
    if command_cfg is not None:
        command = list(OmegaConf.to_container(command_cfg, resolve=False))
    if not command:
        command = ["uv", "run", "python", "${program}", "${args_no_hyphens}"]

    return {
        "name": cfg.name,
        "program": str(cfg.get("train_script", "train.py")),
        "method": str(sweep_cfg.get("method", "grid")),
        "metric": OmegaConf.to_container(sweep_cfg.get("metric", {}), resolve=True),
        "parameters": parameters,
        "command": command,
    }


def write_wandb_sweep_config(cfg: DictConfig, sweep_config: dict[str, Any]) -> Path:
    output_dir = Path(cfg.wandb_sweep.get("output_dir", "outputs/wandb_sweeps"))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{cfg.name}.yaml"
    OmegaConf.save(config=OmegaConf.create(sweep_config), f=path)
    return path


def run_wandb_sweep(cfg: DictConfig) -> str | None:
    sweep_config = build_wandb_sweep_config(cfg)
    path = write_wandb_sweep_config(cfg, sweep_config)
    print(f"Wrote W&B sweep config: {path}", flush=True)

    sweep_id = cfg.wandb_sweep.get("id")
    if cfg.wandb_sweep.get("create", True):
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("W&B sweep backend requires the wandb package.") from exc
        sweep_id = wandb.sweep(
            sweep=sweep_config,
            project=cfg.wandb.get("project"),
            entity=cfg.wandb.get("entity"),
        )
        print(f"Created W&B sweep: {sweep_id}", flush=True)

    if sweep_id and cfg.wandb_sweep.get("start_agent", False):
        agents = int(cfg.wandb_sweep.get("agents", 1))
        if agents > 1:
            return _run_parallel_agents(cfg, sweep_id, agents)
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("W&B sweep agent requires the wandb package.") from exc
        if not hasattr(wandb, "START_TIME"):
            wandb.START_TIME = time.time()
        wandb.agent(
            sweep_id,
            project=cfg.wandb.get("project"),
            entity=cfg.wandb.get("entity"),
            count=cfg.wandb_sweep.get("count"),
        )
    return sweep_id


def _run_parallel_agents(cfg: DictConfig, sweep_id: str, agents: int) -> str:
    commands = [_agent_command(cfg, sweep_id) for _ in range(agents)]
    processes = []
    for index, command in enumerate(commands, start=1):
        print(f"Starting W&B agent {index}/{agents}: {' '.join(command)}", flush=True)
        processes.append(subprocess.Popen(command))

    exit_codes = [process.wait() for process in processes]
    failures = [code for code in exit_codes if code != 0]
    if failures:
        raise RuntimeError(f"W&B agents failed with exit codes: {failures}")
    return sweep_id


def _agent_command(cfg: DictConfig, sweep_id: str) -> list[str]:
    command = [
        sys.executable,
        "sweep.py",
        f"sweep={_sweep_config_name(cfg)}",
        "backend=wandb",
        "wandb_sweep.create=false",
        "wandb_sweep.start_agent=true",
        "wandb_sweep.agents=1",
        f"wandb_sweep.id={sweep_id}",
    ]
    launcher_name = get_launcher_name(cfg)
    if launcher_name != "local":
        command.append(f"launcher={launcher_name}")
    if cfg.wandb_sweep.get("count") is not None:
        command.append(f"wandb_sweep.count={cfg.wandb_sweep.count}")
    if cfg.wandb.get("project") is not None:
        command.append(f"wandb.project={cfg.wandb.project}")
    if cfg.wandb.get("entity") is not None:
        command.append(f"wandb.entity={cfg.wandb.entity}")
    return command


def _sweep_config_name(cfg: DictConfig) -> str:
    name = str(cfg.get("name"))
    if name.endswith("_sweep"):
        return name[: -len("_sweep")]
    return name


def _parse_constant_value(value: str) -> Any:
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1].replace("\\'", "'").replace("\\\\", "\\")
    if value == "true":
        return True
    if value == "false":
        return False
    if value == "null":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _hydra_constant(value: Any) -> Any:
    if isinstance(value, str):
        return format_override_value(value)
    return value


def _run_metadata(cfg: DictConfig) -> dict[str, Any]:
    run_cfg = cfg.get("run")
    if run_cfg is None:
        return {}
    run_metadata = OmegaConf.to_container(run_cfg, resolve=True)
    if not isinstance(run_metadata, dict):
        return {}
    return run_metadata
