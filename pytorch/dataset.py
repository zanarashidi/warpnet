"""
CIFAR-10/100 data pipeline for the PyTorch WarpNet port.

Uses torchvision's built-in CIFAR-10/100 datasets (auto-downloading,
replacing the original's manual pickle-file loader), but reproduces the
original preprocessing exactly:
  - zero-pad `padding_size` pixels on each side, then random-crop back to
    32x32 (instead of torchvision's usual random-crop-with-reflect)
  - random horizontal flip
  - per-image whitening: subtract each image's own mean and divide by
    its own std (not a dataset-wide channel mean/std, which is the more
    common CIFAR normalization) — this matches `whitening_image` in the
    original `cifar10_input.py`
Validation/test data is only whitened, never cropped or flipped, again
matching the original.
"""

import math

import torch
import torchvision
import torchvision.transforms as T

IMG_HEIGHT = 32
IMG_WIDTH = 32
IMG_DEPTH = 3


class PerImageWhitening:
    """Matches `whitening_image` in the original cifar10_input.py."""

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        mean = img.mean()
        std = img.std(unbiased=False).clamp(min=1.0 / math.sqrt(img.numel()))
        return (img - mean) / std


def build_transforms(padding_size: int, train: bool):
    if train:
        return T.Compose(
            [
                T.RandomCrop(IMG_HEIGHT, padding=padding_size, padding_mode="constant", fill=0),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                PerImageWhitening(),
            ]
        )
    return T.Compose([T.ToTensor(), PerImageWhitening()])


def build_datasets(dataset: str, data_dir: str, padding_size: int):
    """
    :param dataset: "cifar10" or "cifar100"
    :param data_dir: directory to download/cache the dataset in
    :param padding_size: zero-padding (pixels per side) before random crop
    :return: (train_dataset, test_dataset, num_classes)
    """
    if dataset == "cifar10":
        cls = torchvision.datasets.CIFAR10
        num_classes = 10
    elif dataset == "cifar100":
        cls = torchvision.datasets.CIFAR100
        num_classes = 100
    else:
        raise ValueError(f"unknown dataset {dataset!r}, expected 'cifar10' or 'cifar100'")

    train_set = cls(
        root=data_dir, train=True, download=True, transform=build_transforms(padding_size, train=True)
    )
    test_set = cls(
        root=data_dir, train=False, download=True, transform=build_transforms(padding_size, train=False)
    )
    return train_set, test_set, num_classes
