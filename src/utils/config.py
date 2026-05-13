from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def save_resolved_config(cfg: DictConfig, output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "config.yaml"
    OmegaConf.save(config=cfg, f=path)
    return path


def config_to_dict(cfg: DictConfig) -> dict:
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

