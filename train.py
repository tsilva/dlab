from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    from src.execution import launch_experiment

    launch_experiment(cfg)


if __name__ == "__main__":
    main()
