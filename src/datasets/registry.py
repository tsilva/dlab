from __future__ import annotations

from omegaconf import DictConfig

from src.datasets.vision import DATASETS, VisionDataModule, datamodule_from_config

DATASET_REGISTRY = {name: VisionDataModule for name in DATASETS}


def build_datamodule(dataset_cfg: DictConfig, seed: int) -> VisionDataModule:
    return datamodule_from_config(dataset_cfg, seed)

