from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation import evaluate_classification_tta, tta_logits


class MeanPixelClassifier(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = x.mean(dim=(1, 2, 3))
        return torch.stack([-score, score], dim=1)


def test_tta_logits_averages_identity_and_horizontal_flip() -> None:
    model = MeanPixelClassifier()
    x = torch.randn(4, 3, 8, 8)

    logits = tta_logits(model, x, ["identity", "hflip"])

    assert torch.allclose(logits, model(x))


def test_evaluate_classification_tta_reports_loss_accuracy_and_view_count() -> None:
    model = MeanPixelClassifier()
    x = torch.stack([torch.ones(3, 4, 4), -torch.ones(3, 4, 4)])
    y = torch.tensor([1, 0])
    dataloader = DataLoader(TensorDataset(x, y), batch_size=2)

    metrics = evaluate_classification_tta(
        model=model,
        dataloader=dataloader,
        device=torch.device("cpu"),
        transforms=["identity", "hflip"],
    )

    assert metrics["acc"] == 1.0
    assert metrics["num_examples"] == 2.0
    assert metrics["num_views"] == 2.0
    assert metrics["loss"] >= 0.0


def test_tta_logits_accepts_shifted_crop_specs_with_normalization() -> None:
    model = MeanPixelClassifier()
    x = torch.randn(2, 3, 8, 8)

    logits = tta_logits(
        model,
        x,
        ["identity", {"name": "shifted_crop", "padding": 2, "dy": -1, "dx": 1}],
        mean=(0.5, 0.5, 0.5),
        std=(0.2, 0.2, 0.2),
    )

    assert logits.shape == (2, 2)


def test_tta_logits_accepts_image_space_color_jitter_specs() -> None:
    model = MeanPixelClassifier()
    x = torch.randn(2, 3, 8, 8)

    logits = tta_logits(
        model,
        x,
        ["identity", {"name": "color_jitter", "brightness": 0.05, "contrast": -0.05}],
        mean=(0.5, 0.5, 0.5),
        std=(0.2, 0.2, 0.2),
    )

    assert logits.shape == (2, 2)


def test_tta_logits_accepts_seeded_trivial_augment_specs() -> None:
    model = MeanPixelClassifier()
    x = torch.rand(2, 3, 8, 8)

    logits = tta_logits(
        model,
        x,
        ["identity", {"name": "trivial_augment", "seed": 1, "num_magnitude_bins": 8}],
        mean=(0.5, 0.5, 0.5),
        std=(0.2, 0.2, 0.2),
    )

    assert logits.shape == (2, 2)
