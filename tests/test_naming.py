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


def test_run_name_includes_weight_decay_and_enabled_augmentation() -> None:
    cfg = OmegaConf.create(
        {
            "dataset": {
                "name": "chestmnist",
                "batch_size": 512,
                "augmentation": {"enabled": True},
            },
            "model": {
                "name": "cnn",
                "params": {
                    "channels": [64, 128, 256],
                    "dropout": 0.1,
                    "convs_per_stage": 2,
                    "batch_norm": True,
                    "residual": True,
                },
            },
            "optimizer": {
                "name": "adamw",
                "lr": 0.001,
                "weight_decay": 0.00003,
                "scheduler": {"name": "cosine"},
            },
            "loss": {"beta": 1.0},
            "seed": 1337,
            "task": "classification",
            "run": {
                "study": "001_compact_cnn_family_search",
                "group": None,
                "sweep_name": "chestmnist_cnn_val_acc_phase1_sweep",
                "sweep_index": None,
            },
        }
    )

    identity = resolve_run_identity(cfg)

    assert "wd3em05" in identity.name
    assert "_do0p1-bn-res-cps2-ch64x128x256-aug_" in identity.name


def test_run_name_includes_non_default_multilabel_threshold() -> None:
    cfg = OmegaConf.create(
        {
            "dataset": {
                "name": "chestmnist",
                "batch_size": 512,
                "augmentation": {"enabled": True},
            },
            "model": {
                "name": "cnn",
                "params": {
                    "channels": [64, 128, 256],
                    "dropout": 0.0,
                    "convs_per_stage": 2,
                    "batch_norm": True,
                    "residual": True,
                },
            },
            "optimizer": {
                "name": "adamw",
                "lr": 0.001,
                "weight_decay": 0.00003,
                "scheduler": {"name": "cosine"},
            },
            "loss": {"beta": 1.0, "threshold": 0.3},
            "seed": 1337,
            "task": "classification",
            "run": {
                "study": "003_incumbent_threshold_search",
                "group": None,
                "sweep_name": "chestmnist_cnn_val_acc_threshold_phase2_sweep",
                "sweep_index": None,
            },
        }
    )

    identity = resolve_run_identity(cfg)

    assert "_do0-bn-res-cps2-ch64x128x256-thr0p3-aug_" in identity.name
