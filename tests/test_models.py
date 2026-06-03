from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from src.models.classifiers import ConvNet, ResNetClassifier, TimmClassifier, WideResNet
from src.models.registry import build_model


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


def test_resnet_classifier_uses_default_imagenet_stem_without_override() -> None:
    model = ResNetClassifier(model_name="resnet18", in_channels=3, num_classes=10)

    assert model.net.conv1.kernel_size == (7, 7)
    assert model.net.conv1.stride == (2, 2)
    assert isinstance(model.net.maxpool, nn.MaxPool2d)


def test_resnet_classifier_can_replace_stem_from_config() -> None:
    model = ResNetClassifier(
        model_name="resnet18",
        in_channels=3,
        num_classes=10,
        stem={"kernel_size": 3, "stride": 1, "padding": 1, "max_pool": False},
    )

    assert model.net.conv1.kernel_size == (3, 3)
    assert model.net.conv1.stride == (1, 1)
    assert model.net.conv1.padding == (1, 1)
    assert isinstance(model.net.maxpool, nn.Identity)
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_wide_resnet_preserves_cifar_classifier_shape() -> None:
    model = WideResNet(depth=16, width_factor=2, in_channels=3, num_classes=10)

    logits = model(torch.randn(2, 3, 32, 32))

    assert logits.shape == (2, 10)


def test_wide_resnet_requires_valid_depth() -> None:
    with pytest.raises(ValueError, match="6n \\+ 4"):
        WideResNet(depth=18)


def test_build_model_ignores_resnet_stem_when_inheriting_wide_resnet_recipe() -> None:
    cfg = OmegaConf.create(
        {
            "name": "wide_resnet",
            "params": {
                "depth": 16,
                "width_factor": 2,
                "stem": {"kernel_size": 3, "stride": 1, "max_pool": False},
            },
        }
    )

    model = build_model(cfg, {"input_shape": (3, 32, 32), "num_classes": 10})

    assert isinstance(model, WideResNet)
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_build_model_ignores_wide_resnet_params_when_inheriting_densenet_recipe() -> None:
    cfg = OmegaConf.create(
        {
            "name": "densenet",
            "params": {
                "model_name": "densenet121",
                "pretrained": False,
                "stem": {"kernel_size": 3, "stride": 1, "padding": 1, "max_pool": False},
                "depth": 28,
                "width_factor": 10,
                "dropout": 0.0,
            },
        }
    )

    model = build_model(cfg, {"input_shape": (3, 32, 32), "num_classes": 10})

    assert isinstance(model, TimmClassifier)
    assert model.net.features.conv0.kernel_size == (3, 3)
    assert model.net.features.conv0.stride == (1, 1)
    assert isinstance(model.net.features.pool0, nn.Identity)
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)
