from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.sweeps import run_sweep


@hydra.main(version_base="1.3", config_path="configs", config_name="sweep")
def main(cfg: DictConfig) -> None:
    exit_codes = run_sweep(cfg)
    failures = [code for code in exit_codes if code != 0]
    if failures:
        raise SystemExit(f"Failed sweep runs: {failures}")


if __name__ == "__main__":
    main()

