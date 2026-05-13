from __future__ import annotations

from omegaconf import OmegaConf

from src.utils.naming import resolve_run_identity


def test_wandb_sweep_run_name_allows_missing_sweep_index() -> None:
    cfg = OmegaConf.create(
        {
            "dataset": {"name": "mnist", "batch_size": 64},
            "model": {
                "name": "mlp",
                "params": {"hidden_dim": 256, "num_layers": 2, "dropout": 0.1},
            },
            "optimizer": {"name": "adam", "lr": 0.003},
            "loss": {"beta": 1.0},
            "seed": 1337,
            "task": "classification",
            "run": {
                "study": "mlp_lr",
                "group": None,
                "sweep_name": "mlp_lr_sweep",
                "sweep_index": None,
            },
        }
    )

    identity = resolve_run_identity(cfg)

    assert identity.name == "mnist-mlp-mlp-lr-adam-lr0p003-bs64-w256-d2-do0p1-mlp-lr-sweep-seed1337"
    assert identity.group == "mnist-mlp-mlp-lr"
