from __future__ import annotations

import torch
from omegaconf import OmegaConf
from torch import nn

from src.trainers import ResearchLitModule


def _classification_cfg(
    *,
    mixup_enabled: bool = False,
    cutmix_enabled: bool = False,
) -> OmegaConf:
    return OmegaConf.create(
        {
            "task": "classification",
            "optimizer": {"name": "adamw", "lr": 0.001},
            "loss": {
                "label_smoothing": 0.05,
                "mixup": {"enabled": mixup_enabled, "alpha": 0.2, "p": 1.0},
                "cutmix": {"enabled": cutmix_enabled, "alpha": 1.0, "p": 1.0},
            },
            "dataset": {"name": "cifar10"},
            "trainer": {"gradient_clip_val": 0.0},
            "gradient_flow": {"enabled": False},
        }
    )


def test_mixup_batch_preserves_shape_and_pairs_labels() -> None:
    module = ResearchLitModule(nn.Linear(4, 3), _classification_cfg(mixup_enabled=True))
    x = torch.randn(8, 4)
    y = torch.arange(8)

    mixed_x, y_a, y_b, lam = module._mixup_batch(x, y)

    assert mixed_x.shape == x.shape
    assert torch.equal(y_a, y)
    assert sorted(y_b.tolist()) == y.tolist()
    assert 0.0 <= float(lam) <= 1.0


def test_mixup_classification_step_logs_weighted_training_metrics() -> None:
    module = ResearchLitModule(nn.Linear(4, 3), _classification_cfg(mixup_enabled=True))
    batch = (torch.randn(8, 4), torch.randint(0, 3, (8,)))

    loss, metrics = module._mixup_classification_step(batch)

    assert loss.ndim == 0
    assert set(metrics) == {"train/loss", "train/acc", "train/mixup_lambda"}
    assert metrics["train/loss"] is loss
    assert 0.0 <= float(metrics["train/acc"]) <= 1.0


def test_cutmix_batch_preserves_shape_and_pairs_labels() -> None:
    module = ResearchLitModule(nn.Linear(4, 3), _classification_cfg(cutmix_enabled=True))
    x = torch.randn(8, 3, 32, 32)
    y = torch.arange(8)

    mixed_x, y_a, y_b, lam = module._cutmix_batch(x, y)

    assert mixed_x.shape == x.shape
    assert torch.equal(y_a, y)
    assert sorted(y_b.tolist()) == y.tolist()
    assert 0.0 <= float(lam) <= 1.0


def test_cutmix_classification_step_logs_weighted_training_metrics() -> None:
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 3))
    module = ResearchLitModule(model, _classification_cfg(cutmix_enabled=True))
    batch = (torch.randn(8, 3, 8, 8), torch.randint(0, 3, (8,)))

    loss, metrics = module._cutmix_classification_step(batch)

    assert loss.ndim == 0
    assert set(metrics) == {"train/loss", "train/acc", "train/cutmix_lambda"}
    assert metrics["train/loss"] is loss
    assert 0.0 <= float(metrics["train/acc"]) <= 1.0
