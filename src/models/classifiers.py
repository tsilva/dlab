from __future__ import annotations

import math
from collections.abc import Sequence

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
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers: list[nn.Module] = [nn.Flatten()]
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
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
        convs_per_stage: int | Sequence[int] = 2,
        batch_norm: bool = True,
        residual: bool = False,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        prev_channels = in_channels
        stage_depths = _stage_depths(convs_per_stage, len(channels))
        for width, stage_depth in zip(channels, stage_depths, strict=True):
            if stage_depth < 1:
                raise ValueError("convs_per_stage entries must be >= 1")
            if residual:
                if stage_depth % 2 != 0:
                    raise ValueError(
                        "Residual ConvNet requires even convs_per_stage values "
                        "because each residual block uses two convolutions."
                    )
                for block_index in range(stage_depth // 2):
                    conv_in_channels = prev_channels if block_index == 0 else width
                    blocks.append(_ResidualConvBlock(conv_in_channels, width, batch_norm))
            else:
                for conv_index in range(stage_depth):
                    conv_in_channels = prev_channels if conv_index == 0 else width
                    blocks.append(nn.Conv2d(conv_in_channels, width, kernel_size=3, padding=1))
                    if batch_norm:
                        blocks.append(nn.BatchNorm2d(width))
                    blocks.append(nn.ReLU(inplace=True))
            blocks.append(nn.MaxPool2d(2))
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


class _ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, batch_norm: bool) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not batch_norm)
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=not batch_norm)
        )
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        self.main = nn.Sequential(*layers)
        self.projection: nn.Module
        if in_channels == out_channels:
            self.projection = nn.Identity()
        else:
            projection_layers: list[nn.Module] = [
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=not batch_norm)
            ]
            if batch_norm:
                projection_layers.append(nn.BatchNorm2d(out_channels))
            self.projection = nn.Sequential(*projection_layers)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(x) + self.projection(x))


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


def _stage_depths(convs_per_stage: int | Sequence[int], num_stages: int) -> list[int]:
    if isinstance(convs_per_stage, int):
        return [convs_per_stage] * num_stages
    depths = list(convs_per_stage)
    if len(depths) != num_stages:
        raise ValueError(
            f"convs_per_stage must have {num_stages} entries when provided as a sequence."
        )
    return [int(depth) for depth in depths]
