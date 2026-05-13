from __future__ import annotations

from dataclasses import dataclass

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, random_split
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
        self.seed = seed
        self.train_data = None
        self.val_data = None
        self.test_data = None

    @property
    def info(self) -> dict[str, object]:
        return {"input_shape": self.spec.input_shape, "num_classes": self.spec.num_classes}

    def _transform(self):
        items: list[object] = [transforms.ToTensor()]
        if self.normalize:
            if self.name == "cifar10":
                items.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)))
            else:
                items.append(transforms.Normalize((0.1307,), (0.3081,)))
        return transforms.Compose(items)

    def prepare_data(self) -> None:
        self.spec.dataset_cls(self.data_dir, train=True, download=self.download)
        self.spec.dataset_cls(self.data_dir, train=False, download=self.download)

    def setup(self, stage: str | None = None) -> None:
        transform = self._transform()
        if stage in {None, "fit"}:
            full = self.spec.dataset_cls(self.data_dir, train=True, transform=transform)
            val_size = int(len(full) * self.val_split)
            train_size = len(full) - val_size
            generator = torch.Generator().manual_seed(self.seed)
            self.train_data, self.val_data = random_split(full, [train_size, val_size], generator)
        if stage in {None, "test", "predict"}:
            self.test_data = self.spec.dataset_cls(self.data_dir, train=False, transform=transform)

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
