from __future__ import annotations

import pytest
import torch

from src.models.classifiers import ConvNet


def test_residual_convnet_preserves_classifier_shape() -> None:
    model = ConvNet(
        in_channels=1,
        num_classes=10,
        channels=[16, 32],
        convs_per_stage=2,
        batch_norm=True,
        residual=True,
    )

    logits = model(torch.randn(4, 1, 28, 28))

    assert logits.shape == (4, 10)


def test_residual_convnet_requires_even_stage_depths() -> None:
    with pytest.raises(ValueError, match="even convs_per_stage"):
        ConvNet(channels=[16], convs_per_stage=3, residual=True)
