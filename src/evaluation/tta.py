from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import InterpolationMode, TrivialAugmentWide
from torchvision.transforms import functional as TF

TransformSpec = str | Mapping[str, Any]


def tta_logits(
    model: nn.Module,
    x: torch.Tensor,
    transforms: Iterable[TransformSpec],
    *,
    mean: Iterable[float] | None = None,
    std: Iterable[float] | None = None,
) -> torch.Tensor:
    """Average logits across tensor-space and image-space test-time transforms."""
    normalizer = _Normalizer.from_values(mean, std, device=x.device, dtype=x.dtype)
    views = [_apply_tta_transform(x, spec, normalizer) for spec in transforms]
    if not views:
        raise ValueError("TTA requires at least one transform.")
    logits = [model(view) for view in views]
    return torch.stack(logits, dim=0).mean(dim=0)


def evaluate_classification_tta(
    *,
    model: nn.Module,
    dataloader,
    device: torch.device,
    transforms: Iterable[TransformSpec],
    label_smoothing: float = 0.0,
    mean: Iterable[float] | None = None,
    std: Iterable[float] | None = None,
) -> dict[str, float]:
    """Run classification evaluation with TTA on any image model."""
    was_training = model.training
    model.eval()
    model.to(device)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    transform_names = list(transforms)
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits = tta_logits(model, x, transform_names, mean=mean, std=std)
            loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
            batch_size = int(y.numel())
            total_loss += float(loss.item()) * batch_size
            total_correct += int((logits.argmax(dim=1) == y).sum().item())
            total_examples += batch_size

    if was_training:
        model.train()
    if total_examples == 0:
        raise ValueError("TTA evaluation dataloader produced no examples.")
    return {
        "loss": total_loss / total_examples,
        "acc": total_correct / total_examples,
        "num_examples": float(total_examples),
        "num_views": float(len(transform_names)),
    }


def _apply_tta_transform(
    x: torch.Tensor,
    spec: TransformSpec,
    normalizer: _Normalizer | None,
) -> torch.Tensor:
    transform_name, params = _transform_name_and_params(spec)
    if transform_name in {"identity", "none"}:
        return x
    if transform_name in {"hflip", "horizontal_flip"}:
        return torch.flip(x, dims=(-1,))
    if transform_name in {"vflip", "vertical_flip"}:
        return torch.flip(x, dims=(-2,))
    if transform_name in {"pad4_shift_up", "shift_up"}:
        return _shifted_crop(x, padding=4, dy=-4, dx=0, normalizer=normalizer)
    if transform_name in {"pad4_shift_down", "shift_down"}:
        return _shifted_crop(x, padding=4, dy=4, dx=0, normalizer=normalizer)
    if transform_name in {"pad4_shift_left", "shift_left"}:
        return _shifted_crop(x, padding=4, dy=0, dx=-4, normalizer=normalizer)
    if transform_name in {"pad4_shift_right", "shift_right"}:
        return _shifted_crop(x, padding=4, dy=0, dx=4, normalizer=normalizer)
    if transform_name in {"shifted_crop", "crop_shift"}:
        return _shifted_crop(
            x,
            padding=int(params.get("padding", 4)),
            dy=int(params.get("dy", 0)),
            dx=int(params.get("dx", 0)),
            normalizer=normalizer,
        )
    if transform_name in {"color_jitter", "mild_color_jitter"}:
        return _color_jitter_view(x, params, normalizer)
    if transform_name in {"trivial_augment", "trivialaugment"}:
        return _trivial_augment_view(x, params, normalizer)
    raise ValueError(f"Unknown TTA transform '{spec}'.")


def _transform_name_and_params(spec: TransformSpec) -> tuple[str, Mapping[str, Any]]:
    if isinstance(spec, str):
        return spec.strip().lower(), {}
    name = str(spec.get("name", "")).strip().lower()
    if not name:
        raise ValueError("TTA transform mapping requires a non-empty 'name'.")
    return name, spec


def _shifted_crop(
    x: torch.Tensor,
    *,
    padding: int,
    dy: int,
    dx: int,
    normalizer: _Normalizer | None,
) -> torch.Tensor:
    if padding < 0:
        raise ValueError("TTA shifted_crop padding must be non-negative.")
    if padding == 0:
        return x
    source = _to_image_space(x, normalizer) if normalizer is not None else x
    height, width = x.shape[-2:]
    padded = F.pad(source, (padding, padding, padding, padding), mode="constant", value=0.0)
    top = max(0, min(2 * padding, padding + dy))
    left = max(0, min(2 * padding, padding + dx))
    shifted = padded[..., top : top + height, left : left + width]
    return _from_image_space(shifted, normalizer) if normalizer is not None else shifted


def _color_jitter_view(
    x: torch.Tensor,
    params: Mapping[str, Any],
    normalizer: _Normalizer | None,
) -> torch.Tensor:
    image = _to_image_space(x, normalizer)
    seed = params.get("seed")
    with _maybe_torch_seed(None if seed is None else int(seed)):
        brightness = float(params.get("brightness", 0.0))
        contrast = float(params.get("contrast", 0.0))
        saturation = float(params.get("saturation", 0.0))
        hue = float(params.get("hue", 0.0))
        if brightness:
            image = TF.adjust_brightness(image, 1.0 + brightness)
        if contrast:
            image = TF.adjust_contrast(image, 1.0 + contrast)
        if saturation:
            image = TF.adjust_saturation(image, 1.0 + saturation)
        if hue:
            image = TF.adjust_hue(image, hue)
    return _from_image_space(image, normalizer)


def _trivial_augment_view(
    x: torch.Tensor,
    params: Mapping[str, Any],
    normalizer: _Normalizer | None,
) -> torch.Tensor:
    image = _to_image_space(x, normalizer)
    image_uint8 = (image * 255).round().clamp(0, 255).to(torch.uint8).cpu()
    transform = TrivialAugmentWide(
        num_magnitude_bins=int(params.get("num_magnitude_bins", 31)),
        interpolation=InterpolationMode[str(params.get("interpolation", "NEAREST")).upper()],
        fill=params.get("fill", 0),
    )
    seed = params.get("seed")
    with _maybe_torch_seed(None if seed is None else int(seed)):
        views_uint8 = torch.stack([transform(example) for example in image_uint8], dim=0)
    views = views_uint8.to(device=image.device, dtype=image.dtype).div(255.0)
    return _from_image_space(views, normalizer)


def _to_image_space(x: torch.Tensor, normalizer: _Normalizer | None) -> torch.Tensor:
    if normalizer is None:
        return x.clamp(0, 1)
    return normalizer.denormalize(x).clamp(0, 1)


def _from_image_space(x: torch.Tensor, normalizer: _Normalizer | None) -> torch.Tensor:
    if normalizer is None:
        return x.clamp(0, 1)
    return normalizer.normalize(x.clamp(0, 1))


@contextmanager
def _maybe_torch_seed(seed: int | None):
    if seed is None:
        yield
        return
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(state)


class _Normalizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean = mean
        self.std = std

    @classmethod
    def from_values(
        cls,
        mean: Iterable[float] | None,
        std: Iterable[float] | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> _Normalizer | None:
        if mean is None or std is None:
            return None
        mean_tensor = torch.tensor(list(mean), device=device, dtype=dtype).view(1, -1, 1, 1)
        std_tensor = torch.tensor(list(std), device=device, dtype=dtype).view(1, -1, 1, 1)
        return cls(mean_tensor, std_tensor)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean
