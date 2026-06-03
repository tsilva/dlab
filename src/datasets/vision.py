from __future__ import annotations

from dataclasses import dataclass

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DatasetSpec:
    dataset_cls: type
    input_shape: tuple[int, int, int]
    num_classes: int = 10


DATASETS = {
    "mnist": DatasetSpec(datasets.MNIST, (1, 28, 28)),
    "fashion_mnist": DatasetSpec(datasets.FashionMNIST, (1, 28, 28)),
    "cifar10": DatasetSpec(datasets.CIFAR10, (3, 32, 32)),
}


class VisionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        name: str,
        data_dir: str = "datasets",
        batch_size: int = 64,
        num_workers: int = 4,
        val_split: float = 0.1,
        normalize: bool = True,
        download: bool = True,
        pin_memory: bool | None = None,
        augmentation: dict | None = None,
        seed: int = 1337,
    ) -> None:
        super().__init__()
        if name not in DATASETS:
            raise KeyError(f"Unknown dataset '{name}'. Available: {sorted(DATASETS)}")
        self.name = name
        self.spec = DATASETS[name]
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.normalize = normalize
        self.download = download
        self.pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory
        self.augmentation = augmentation or {}
        self.seed = seed
        self.train_data = None
        self.val_data = None
        self.test_data = None

    @property
    def info(self) -> dict[str, object]:
        return {"input_shape": self.spec.input_shape, "num_classes": self.spec.num_classes}

    def _transform(self, train: bool = False):
        items: list[object] = []
        if train and self.augmentation.get("enabled", False):
            crop = self.augmentation.get("random_crop")
            if crop is not None:
                if "size" not in crop:
                    raise ValueError("augmentation.random_crop.size is required.")
                items.append(
                    transforms.RandomCrop(
                        size=_int_or_pair(crop["size"], "augmentation.random_crop.size"),
                        padding=_optional_int_or_tuple(
                            crop.get("padding", None),
                            "augmentation.random_crop.padding",
                        ),
                        pad_if_needed=bool(crop.get("pad_if_needed", False)),
                        fill=crop.get("fill", 0),
                        padding_mode=crop.get("padding_mode", "constant"),
                    )
                )
            horizontal_flip = self.augmentation.get("horizontal_flip")
            if horizontal_flip is not None:
                items.append(
                    transforms.RandomHorizontalFlip(
                        p=_probability(
                            horizontal_flip.get("p", 0.5),
                            "augmentation.horizontal_flip.p",
                        )
                    )
                )
            color_jitter = self.augmentation.get("color_jitter")
            if color_jitter is not None:
                items.append(
                    transforms.ColorJitter(
                        brightness=color_jitter.get("brightness", 0),
                        contrast=color_jitter.get("contrast", 0),
                        saturation=color_jitter.get("saturation", 0),
                        hue=color_jitter.get("hue", 0),
                    )
                )
            rand_augment = self.augmentation.get("rand_augment")
            if rand_augment is not None:
                items.append(
                    transforms.RandAugment(
                        num_ops=int(rand_augment.get("num_ops", 2)),
                        magnitude=int(rand_augment.get("magnitude", 9)),
                        num_magnitude_bins=int(rand_augment.get("num_magnitude_bins", 31)),
                        interpolation=_interpolation_mode(
                            rand_augment.get("interpolation", "nearest"),
                            "augmentation.rand_augment.interpolation",
                        ),
                        fill=rand_augment.get("fill"),
                    )
                )
            trivial_augment = self.augmentation.get("trivial_augment")
            if trivial_augment is not None:
                items.append(
                    transforms.TrivialAugmentWide(
                        num_magnitude_bins=int(trivial_augment.get("num_magnitude_bins", 31)),
                        interpolation=_interpolation_mode(
                            trivial_augment.get("interpolation", "nearest"),
                            "augmentation.trivial_augment.interpolation",
                        ),
                        fill=trivial_augment.get("fill"),
                    )
                )
            affine = self.augmentation.get("random_affine")
            if affine is not None:
                degrees = affine.get("degrees", 0)
                translate = _translate_tuple(affine.get("translate", None))
                scale = _scale_tuple(affine.get("scale", None))
                fill = affine.get("fill", 0)
                items.append(
                    transforms.RandomAffine(
                        degrees=degrees,
                        translate=translate,
                        scale=scale,
                        fill=fill,
                    )
                )
        items.append(transforms.ToTensor())
        if train and self.augmentation.get("enabled", False):
            erasing = self.augmentation.get("random_erasing")
            if erasing is not None:
                items.append(
                    transforms.RandomErasing(
                        p=_probability(erasing.get("p", 0.5), "augmentation.random_erasing.p"),
                        scale=_float_pair(erasing.get("scale", (0.02, 0.33))),
                        ratio=_float_pair(erasing.get("ratio", (0.3, 3.3))),
                        value=erasing.get("value", 0),
                        inplace=bool(erasing.get("inplace", False)),
                    )
                )
        if self.normalize:
            mean, std = normalization_stats(self.name)
            items.append(transforms.Normalize(mean, std))
        return transforms.Compose(items)

    def prepare_data(self) -> None:
        self.spec.dataset_cls(self.data_dir, train=True, download=self.download)
        self.spec.dataset_cls(self.data_dir, train=False, download=self.download)

    def setup(self, stage: str | None = None) -> None:
        if stage in {None, "fit"}:
            full = self.spec.dataset_cls(self.data_dir, train=True, transform=None)
            val_size = int(len(full) * self.val_split)
            train_size = len(full) - val_size
            generator = torch.Generator().manual_seed(self.seed)
            train_indices, val_indices = random_split(
                range(len(full)),
                [train_size, val_size],
                generator,
            )
            train_full = self.spec.dataset_cls(
                self.data_dir,
                train=True,
                transform=self._transform(train=True),
            )
            val_full = self.spec.dataset_cls(
                self.data_dir,
                train=True,
                transform=self._transform(train=False),
            )
            self.train_data = Subset(train_full, train_indices.indices)
            self.val_data = Subset(val_full, val_indices.indices)
        if stage in {None, "test", "predict"}:
            self.test_data = self.spec.dataset_cls(
                self.data_dir,
                train=False,
                transform=self._transform(train=False),
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )


def datamodule_from_config(cfg: DictConfig, seed: int) -> VisionDataModule:
    params = dict(OmegaConf.to_container(cfg, resolve=True))
    params["seed"] = seed
    return VisionDataModule(**params)


def normalization_stats(dataset_name: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if dataset_name == "cifar10":
        return (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
    if dataset_name == "fashion_mnist":
        return (0.2860,), (0.3530,)
    return (0.1307,), (0.3081,)


def _translate_tuple(value: object) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return (float(value), float(value))
    if isinstance(value, list | tuple):
        if len(value) != 2:
            raise ValueError("augmentation.random_affine.translate must have length 2.")
        return (float(value[0]), float(value[1]))
    raise TypeError("augmentation.random_affine.translate must be a number or length-2 list.")


def _scale_tuple(value: object) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, list | tuple):
        if len(value) != 2:
            raise ValueError("augmentation.random_affine.scale must have length 2.")
        return (float(value[0]), float(value[1]))
    raise TypeError("augmentation.random_affine.scale must be a length-2 list.")


def _int_or_pair(value: object, field_name: str) -> int | tuple[int, int]:
    if isinstance(value, int):
        return value
    if isinstance(value, list | tuple):
        if len(value) != 2:
            raise ValueError(f"{field_name} must be an int or length-2 list.")
        return (int(value[0]), int(value[1]))
    raise TypeError(f"{field_name} must be an int or length-2 list.")


def _optional_int_or_tuple(value: object, field_name: str) -> int | tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, list | tuple):
        if not value:
            raise ValueError(f"{field_name} must not be empty.")
        return tuple(int(item) for item in value)
    raise TypeError(f"{field_name} must be an int or list.")


def _probability(value: object, field_name: str) -> float:
    probability = float(value)
    if probability < 0.0 or probability > 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1.")
    return probability


def _float_pair(value: object) -> tuple[float, float]:
    if isinstance(value, list | tuple):
        if len(value) != 2:
            raise ValueError("Expected a length-2 list.")
        return (float(value[0]), float(value[1]))
    raise TypeError("Expected a length-2 list.")


def _interpolation_mode(value: object, field_name: str) -> transforms.InterpolationMode:
    name = str(value).lower()
    modes = {
        "nearest": transforms.InterpolationMode.NEAREST,
        "bilinear": transforms.InterpolationMode.BILINEAR,
        "bicubic": transforms.InterpolationMode.BICUBIC,
    }
    if name not in modes:
        raise ValueError(f"{field_name} must be one of {sorted(modes)}.")
    return modes[name]
