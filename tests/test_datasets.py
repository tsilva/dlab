from __future__ import annotations

import pytest
from torchvision import transforms

from src.datasets.vision import VisionDataModule


def test_training_transform_supports_crop_and_flip_without_affine() -> None:
    datamodule = VisionDataModule(
        name="cifar10",
        normalize=False,
        augmentation={
            "enabled": True,
            "random_crop": {"size": 32, "padding": 4},
            "horizontal_flip": {"p": 0.5},
        },
    )

    transform = datamodule._transform(train=True)

    assert [type(item) for item in transform.transforms] == [
        transforms.RandomCrop,
        transforms.RandomHorizontalFlip,
        transforms.ToTensor,
    ]
    assert transform.transforms[0].size == (32, 32)
    assert transform.transforms[0].padding == 4
    assert transform.transforms[1].p == 0.5


def test_eval_transform_does_not_apply_training_augmentation() -> None:
    datamodule = VisionDataModule(
        name="cifar10",
        normalize=False,
        augmentation={
            "enabled": True,
            "random_crop": {"size": 32, "padding": 4},
            "horizontal_flip": {"p": 0.5},
        },
    )

    transform = datamodule._transform(train=False)

    assert [type(item) for item in transform.transforms] == [transforms.ToTensor]


def test_random_crop_requires_size() -> None:
    datamodule = VisionDataModule(
        name="cifar10",
        augmentation={"enabled": True, "random_crop": {"padding": 4}},
    )

    with pytest.raises(ValueError, match="random_crop.size"):
        datamodule._transform(train=True)


def test_training_transform_applies_random_erasing_after_tensor_conversion() -> None:
    datamodule = VisionDataModule(
        name="cifar10",
        normalize=True,
        augmentation={
            "enabled": True,
            "random_crop": {"size": 32, "padding": 4},
            "horizontal_flip": {"p": 0.5},
            "random_erasing": {"p": 0.25, "scale": [0.02, 0.2], "ratio": [0.3, 3.3]},
        },
    )

    transform = datamodule._transform(train=True)

    assert [type(item) for item in transform.transforms] == [
        transforms.RandomCrop,
        transforms.RandomHorizontalFlip,
        transforms.ToTensor,
        transforms.RandomErasing,
        transforms.Normalize,
    ]
    assert transform.transforms[3].p == 0.25
    assert transform.transforms[3].scale == (0.02, 0.2)
    assert transform.transforms[3].ratio == (0.3, 3.3)


def test_training_transform_applies_color_jitter_before_tensor_conversion() -> None:
    datamodule = VisionDataModule(
        name="cifar10",
        normalize=False,
        augmentation={
            "enabled": True,
            "random_crop": {"size": 32, "padding": 4},
            "horizontal_flip": {"p": 0.5},
            "color_jitter": {
                "brightness": 0.1,
                "contrast": 0.1,
                "saturation": 0.1,
                "hue": 0.02,
            },
        },
    )

    transform = datamodule._transform(train=True)

    assert [type(item) for item in transform.transforms] == [
        transforms.RandomCrop,
        transforms.RandomHorizontalFlip,
        transforms.ColorJitter,
        transforms.ToTensor,
    ]
    assert transform.transforms[2].brightness == (0.9, 1.1)
    assert transform.transforms[2].contrast == (0.9, 1.1)
    assert transform.transforms[2].saturation == (0.9, 1.1)
    assert transform.transforms[2].hue == (-0.02, 0.02)


def test_training_transform_applies_randaugment_before_tensor_conversion() -> None:
    datamodule = VisionDataModule(
        name="cifar10",
        normalize=False,
        augmentation={
            "enabled": True,
            "random_crop": {"size": 32, "padding": 4},
            "horizontal_flip": {"p": 0.5},
            "color_jitter": {
                "brightness": 0.1,
                "contrast": 0.1,
                "saturation": 0.1,
                "hue": 0.02,
            },
            "rand_augment": {
                "num_ops": 2,
                "magnitude": 9,
                "num_magnitude_bins": 31,
                "interpolation": "nearest",
            },
        },
    )

    transform = datamodule._transform(train=True)

    assert [type(item) for item in transform.transforms] == [
        transforms.RandomCrop,
        transforms.RandomHorizontalFlip,
        transforms.ColorJitter,
        transforms.RandAugment,
        transforms.ToTensor,
    ]
    assert transform.transforms[3].num_ops == 2
    assert transform.transforms[3].magnitude == 9
    assert transform.transforms[3].num_magnitude_bins == 31


def test_randaugment_rejects_unknown_interpolation() -> None:
    datamodule = VisionDataModule(
        name="cifar10",
        augmentation={
            "enabled": True,
            "rand_augment": {"interpolation": "lanczos"},
        },
    )

    with pytest.raises(ValueError, match="rand_augment.interpolation"):
        datamodule._transform(train=True)


def test_training_transform_applies_trivialaugment_before_tensor_conversion() -> None:
    datamodule = VisionDataModule(
        name="cifar10",
        normalize=False,
        augmentation={
            "enabled": True,
            "random_crop": {"size": 32, "padding": 4},
            "horizontal_flip": {"p": 0.5},
            "trivial_augment": {
                "num_magnitude_bins": 31,
                "interpolation": "nearest",
            },
        },
    )

    transform = datamodule._transform(train=True)

    assert [type(item) for item in transform.transforms] == [
        transforms.RandomCrop,
        transforms.RandomHorizontalFlip,
        transforms.TrivialAugmentWide,
        transforms.ToTensor,
    ]
    assert transform.transforms[2].num_magnitude_bins == 31


def test_trivialaugment_rejects_unknown_interpolation() -> None:
    datamodule = VisionDataModule(
        name="cifar10",
        augmentation={
            "enabled": True,
            "trivial_augment": {"interpolation": "lanczos"},
        },
    )

    with pytest.raises(ValueError, match="trivial_augment.interpolation"):
        datamodule._transform(train=True)
