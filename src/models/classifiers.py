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


class WideResNet(nn.Module):
    task = "classification"

    def __init__(
        self,
        depth: int = 28,
        width_factor: int = 4,
        dropout: float = 0.0,
        in_channels: int = 3,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError("WideResNet depth must satisfy depth = 6n + 4.")
        if width_factor < 1:
            raise ValueError("WideResNet width_factor must be >= 1.")

        blocks_per_stage = (depth - 4) // 6
        widths = [16, 16 * width_factor, 32 * width_factor, 64 * width_factor]

        self.conv1 = nn.Conv2d(in_channels, widths[0], kernel_size=3, padding=1, bias=False)
        self.stage1 = self._make_stage(
            blocks_per_stage,
            in_channels=widths[0],
            out_channels=widths[1],
            stride=1,
            dropout=dropout,
        )
        self.stage2 = self._make_stage(
            blocks_per_stage,
            in_channels=widths[1],
            out_channels=widths[2],
            stride=2,
            dropout=dropout,
        )
        self.stage3 = self._make_stage(
            blocks_per_stage,
            in_channels=widths[2],
            out_channels=widths[3],
            stride=2,
            dropout=dropout,
        )
        self.bn = nn.BatchNorm2d(widths[3])
        self.activation = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(widths[3], num_classes)

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.activation(self.bn(x))
        x = self.pool(x).flatten(1)
        return self.classifier(x)

    def first_layer_filters(self) -> torch.Tensor | None:
        return self.conv1.weight.detach().cpu()

    def _make_stage(
        self,
        blocks_per_stage: int,
        *,
        in_channels: int,
        out_channels: int,
        stride: int,
        dropout: float,
    ) -> nn.Sequential:
        blocks: list[nn.Module] = [
            _WideResNetBlock(in_channels, out_channels, stride=stride, dropout=dropout)
        ]
        blocks.extend(
            _WideResNetBlock(out_channels, out_channels, stride=1, dropout=dropout)
            for _ in range(blocks_per_stage - 1)
        )
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)


class _WideResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.shortcut: nn.Module
        if in_channels == out_channels and stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.bn1(x))
        shortcut = self.shortcut(x)
        out = self.conv1(out)
        out = self.dropout(self.activation(self.bn2(out)))
        out = self.conv2(out)
        return out + shortcut


class TimmClassifier(nn.Module):
    task = "classification"

    def __init__(
        self,
        model_name: str = "resnet18",
        num_classes: int = 10,
        in_channels: int = 3,
        pretrained: bool = False,
        stem: dict[str, object] | None = None,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:  # pragma: no cover - dependency declared in pyproject
            message = "TimmClassifier requires timm. Install project dependencies."
            raise RuntimeError(message) from exc

        self.net = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_channels,
        )
        if stem is not None:
            self._replace_stem(stem, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _replace_stem(self, stem: dict[str, object], in_channels: int) -> None:
        conv_parent = self.net
        conv_name = "conv1"
        pool_parent = self.net
        pool_name = "maxpool"
        if not hasattr(conv_parent, conv_name) and _has_sequential_stem_conv(self.net):
            conv_parent = self.net.stem
            conv_name = "0"
            pool_parent = None
            pool_name = None
        elif not hasattr(conv_parent, conv_name) and hasattr(self.net, "features"):
            conv_parent = self.net.features
            conv_name = "conv0"
            pool_parent = self.net.features
            pool_name = "pool0"
        if not _has_named_module(conv_parent, conv_name) or (
            pool_parent is not None and pool_name is not None and not hasattr(pool_parent, pool_name)
        ):
            raise ValueError(
                "Configured timm stem replacement requires conv1/maxpool, "
                "features.conv0/pool0, or stem[0]."
            )

        current_conv = _get_named_module(conv_parent, conv_name)
        if not isinstance(current_conv, nn.Conv2d):
            raise ValueError("Configured timm stem replacement requires a single stem conv module.")

        out_channels = current_conv.out_channels
        kernel_size = int(stem.get("kernel_size", 7))
        stride = int(stem.get("stride", 2))
        padding = int(stem.get("padding", kernel_size // 2))
        bias = bool(stem.get("bias", current_conv.bias is not None))
        max_pool = bool(stem.get("max_pool", True))

        _set_named_module(
            conv_parent,
            conv_name,
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
        )
        if pool_parent is not None and pool_name is not None:
            setattr(
                pool_parent,
                pool_name,
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if max_pool else nn.Identity(),
            )


class ResNetClassifier(TimmClassifier):
    pass


def image_dim(input_shape: tuple[int, int, int] | list[int]) -> int:
    return math.prod(input_shape)


def _has_sequential_stem_conv(module: nn.Module) -> bool:
    stem = getattr(module, "stem", None)
    return isinstance(stem, nn.Sequential) and len(stem) > 0 and isinstance(stem[0], nn.Conv2d)


def _has_named_module(module: nn.Module, name: str) -> bool:
    if name.isdigit() and isinstance(module, nn.Sequential):
        return int(name) < len(module)
    return hasattr(module, name)


def _get_named_module(module: nn.Module, name: str) -> nn.Module:
    if name.isdigit() and isinstance(module, nn.Sequential):
        return module[int(name)]
    return getattr(module, name)


def _set_named_module(module: nn.Module, name: str, replacement: nn.Module) -> None:
    if name.isdigit() and isinstance(module, nn.Sequential):
        module[int(name)] = replacement
        return
    setattr(module, name, replacement)


def _stage_depths(convs_per_stage: int | Sequence[int], num_stages: int) -> list[int]:
    if isinstance(convs_per_stage, int):
        return [convs_per_stage] * num_stages
    depths = list(convs_per_stage)
    if len(depths) != num_stages:
        raise ValueError(
            f"convs_per_stage must have {num_stages} entries when provided as a sequence."
        )
    return [int(depth) for depth in depths]
