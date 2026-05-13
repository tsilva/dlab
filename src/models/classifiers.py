from __future__ import annotations

import math

import torch
from torch import nn


class MLP(nn.Module):
    task = "classification"

    def __init__(
        self,
        input_dim: int = 784,
        num_classes: int = 10,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers: list[nn.Module] = [nn.Flatten()]
        in_dim = input_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvNet(nn.Module):
    task = "classification"

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        channels: list[int] | tuple[int, ...] = (32, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        prev_channels = in_channels
        for width in channels:
            blocks.extend(
                [
                    nn.Conv2d(prev_channels, width, kernel_size=3, padding=1),
                    nn.BatchNorm2d(width),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(width, width, kernel_size=3, padding=1),
                    nn.BatchNorm2d(width),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ]
            )
            prev_channels = width

        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(prev_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

    def first_layer_filters(self) -> torch.Tensor | None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                return module.weight.detach().cpu()
        return None


class ResNetClassifier(nn.Module):
    task = "classification"

    def __init__(
        self,
        model_name: str = "resnet18",
        num_classes: int = 10,
        in_channels: int = 3,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:  # pragma: no cover - dependency declared in pyproject
            message = "ResNetClassifier requires timm. Install project dependencies."
            raise RuntimeError(message) from exc

        self.net = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def image_dim(input_shape: tuple[int, int, int] | list[int]) -> int:
    return math.prod(input_shape)
