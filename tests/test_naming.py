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

    assert identity.name == "mnist-mlp_lr_adam-lr0p003-bs64_w256-d2-do0p1_lr-sweep_seed1337"
    assert identity.group == "mnist-mlp-lr"


def test_run_name_collapses_redundant_dataset_model_and_study_prefixes() -> None:
    cfg = OmegaConf.create(
        {
            "dataset": {"name": "fashion_mnist", "batch_size": 512},
            "model": {
                "name": "cnn",
                "params": {
                    "channels": [64, 128, 256],
                    "dropout": 0.2,
                    "convs_per_stage": 6,
                    "batch_norm": False,
                },
            },
            "optimizer": {
                "name": "adam",
                "lr": 0.001,
                "scheduler": {"name": "cosine"},
            },
            "loss": {"beta": 1.0, "label_smoothing": 0.02},
            "weight_averaging": {"name": "ema"},
            "seed": 1,
            "task": "classification",
            "run": {
                "study": "fashion_mnist_cnn_gradient_flow",
                "group": None,
                "sweep_name": "fashion_mnist_cnn_gradient_flow_depth_sweep",
                "sweep_index": None,
            },
        }
    )

    identity = resolve_run_identity(cfg)

    assert identity.name == (
        "fashion-mnist-cnn_gradient-flow_adam-lr0p001-bs512-cosine_"
        "do0p2-cps6-ch64x128x256-ls0p02-ema_depth-sweep_seed1"
    )
    assert identity.group == "fashion-mnist-cnn-gradient-flow"
