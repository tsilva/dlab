from __future__ import annotations

import torch
from omegaconf import OmegaConf
from torch import nn

from src.models.classifiers import SequenceClassifier
from src.trainers import ResearchLitModule


def _classification_cfg(
    *,
    mixup_enabled: bool = False,
    cutmix_enabled: bool = False,
    target_type: str = "single_label",
) -> OmegaConf:
    return OmegaConf.create(
        {
            "task": "classification",
            "optimizer": {"name": "adamw", "lr": 0.001},
            "loss": {
                "label_smoothing": 0.05,
                "target_type": target_type,
                "threshold": 0.5,
                "mixup": {"enabled": mixup_enabled, "alpha": 0.2, "p": 1.0},
                "cutmix": {"enabled": cutmix_enabled, "alpha": 1.0, "p": 1.0},
            },
            "dataset": {"name": "cifar10"},
            "trainer": {"gradient_clip_val": 0.0},
            "gradient_flow": {"enabled": False},
            "sequence_diagnostics": {
                "enabled": False,
                "log_every_n_steps": 50,
                "log_validation": True,
            },
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


def test_classification_step_logs_prediction_entropy() -> None:
    module = ResearchLitModule(nn.Linear(4, 3), _classification_cfg())
    batch = (torch.randn(8, 4), torch.randint(0, 3, (8,)))

    loss, metrics = module._shared_step(batch, "train", batch_idx=0)

    assert loss.ndim == 0
    assert "train/pred_entropy" in metrics
    assert "train/pred_entropy_normalized" in metrics
    assert "train/pred_max_prob" in metrics


def test_multilabel_classification_step_uses_bce_and_threshold_accuracy() -> None:
    model = nn.Linear(4, 3)
    module = ResearchLitModule(model, _classification_cfg(target_type="multi_label_binary"))
    batch = (
        torch.randn(8, 4),
        torch.randint(0, 2, (8, 3), dtype=torch.float32),
    )

    loss, metrics = module._shared_step(batch, "val", batch_idx=0)

    assert loss.ndim == 0
    assert "val/acc" in metrics
    assert "val/pred_positive_rate" in metrics
    assert 0.0 <= float(metrics["val/acc"]) <= 1.0


def test_sequence_diagnostics_log_hidden_state_norms_when_enabled() -> None:
    cfg = _classification_cfg()
    cfg.sequence_diagnostics.enabled = True
    cfg.sequence_diagnostics.log_every_n_steps = 1
    model = SequenceClassifier(
        input_size=1,
        hidden_dim=8,
        num_classes=10,
        rnn_type="rnn",
        sequence_axis="pixels",
    )
    module = ResearchLitModule(model, cfg)
    batch = (torch.randn(4, 1, 28, 28), torch.randint(0, 10, (4,)))

    _, metrics = module._shared_step(batch, "train", batch_idx=0)

    assert "train/sequence/hidden_norm_t000" in metrics
    assert "train/sequence/hidden_norm_t50pct" in metrics
    assert "train/sequence/hidden_norm_tlast" in metrics
    assert "train/sequence/hidden_norm_last_to_first_ratio" in metrics
