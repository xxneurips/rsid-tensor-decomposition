"""
Dataset Loading and Preprocessing
===================================

Loads CIFAR-10, CIFAR-100, and TinyImageNet with appropriate transforms.

Performance optimizations for fast data loading:
- pin_memory=True: Pre-allocates pinned (page-locked) memory for faster GPU transfer
- persistent_workers=True: Keeps worker processes alive between epochs (avoids respawn cost)
- num_workers=4: Parallel data loading (more workers ≠ faster beyond CPU core count)
- prefetch_factor=2: Each worker prefetches 2 batches ahead
"""

import os
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from typing import Tuple, Dict


def get_dataset_info(dataset: str) -> Dict:
    """
    Return dataset metadata.

    Args:
        dataset: One of 'cifar10', 'cifar100', 'tinyimagenet'.

    Returns:
        Dict with 'num_classes', 'image_size', 'mean', 'std'.
    """
    info = {
        "cifar10": {
            "num_classes": 10,
            "image_size": 32,
            "mean": (0.4914, 0.4822, 0.4465),
            "std": (0.2023, 0.1994, 0.2010),
        },
        "cifar100": {
            "num_classes": 100,
            "image_size": 32,
            "mean": (0.5071, 0.4867, 0.4408),
            "std": (0.2675, 0.2565, 0.2761),
        },
        "tinyimagenet": {
            "num_classes": 200,
            "image_size": 64,
            "mean": (0.4802, 0.4481, 0.3975),
            "std": (0.2770, 0.2691, 0.2821),
        },
        "imagenette": {
            "num_classes": 10,
            "image_size": 224,
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
        },
        "imagewoof": {
            "num_classes": 10,
            "image_size": 224,
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
        },
        "imagenet": {
            "num_classes": 1000,
            "image_size": 224,
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
        },
        "imagenet100": {
            "num_classes": 100,
            "image_size": 224,
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
        },
    }
    return info[dataset.lower()]


def get_data_loaders(
    dataset: str = "cifar10",
    batch_size: int = 128,
    num_workers: int = 4,
    data_dir: str = "./data",
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders with appropriate augmentation.

    Training augmentation:
    - RandomCrop with padding: shifts the image randomly for translation invariance
    - RandomHorizontalFlip: 50% chance to mirror (standard for natural images)
    - Normalize: zero-mean, unit-variance per channel

    Validation: only resize + normalize (no augmentation).

    Args:
        dataset: Dataset name ('cifar10', 'cifar100', 'tinyimagenet').
        batch_size: Batch size per GPU. 128 works well for 3090 (24GB).
        num_workers: Number of data loading workers.
        data_dir: Directory to download/store datasets.

    Returns:
        (train_loader, val_loader) tuple.
    """
    info = get_dataset_info(dataset)
    img_size = info["image_size"]
    mean, std = info["mean"], info["std"]

    if dataset.lower() in ("cifar10", "cifar100"):
        return _get_cifar_loaders(dataset, batch_size, num_workers, data_dir, mean, std)
    elif dataset.lower() == "tinyimagenet":
        return _get_tinyimagenet_loaders(batch_size, num_workers, data_dir, mean, std)
    elif dataset.lower() in ("imagenette", "imagewoof"):
        return _get_imagenet_subset_loaders(dataset.lower(), batch_size, num_workers, data_dir, mean, std)
    elif dataset.lower() == "imagenet":
        return _get_imagenet1k_loaders(batch_size, num_workers, data_dir, mean, std)
    elif dataset.lower() == "imagenet100":
        return _get_imagenet100_loaders(batch_size, num_workers, data_dir, mean, std)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def _get_imagenet100_loaders(
    batch_size: int, num_workers: int, data_dir: str,
    mean: tuple, std: tuple,
) -> Tuple[DataLoader, DataLoader]:
    """
    ImageNet-100: a 100-class subset of ImageNet at the same 224x224 resolution.
    The 100 wnids are the alphabetically-first 100 classes from the
    ImageNet-1k train listing, served from local SSD for fast random access.
    The classifier head is replaced with a 100-class linear (handled by the
    model factory). Standard ImageNet preprocessing.
    """
    import os
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"ImageNet-100 not found at {data_dir}. Expected ImageFolder layout under {data_dir}/."
        )

    train_transform = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    val_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
    val_set = torchvision.datasets.ImageFolder(val_dir, transform=val_transform)
    print(f"[ImageNet-100] train: {len(train_set)} images across {len(train_set.classes)} classes; "
          f"val: {len(val_set)} images")

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers,
        pin_memory=True, persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    return train_loader, val_loader


class _CanonicalImageFolder:
    """Module-level (picklable) ImageFolder-like dataset using an external
    wnid_to_idx map for canonical 1000-class labelling. Defined at module
    level so DataLoader workers can pickle it under Python 3.14+."""
    def __init__(self, root, transform, wnid_to_idx):
        import os
        self.root = root
        self.transform = transform
        self.samples = []
        for wnid in sorted(os.listdir(root)):
            wd = os.path.join(root, wnid)
            if not os.path.isdir(wd):
                continue
            lbl = wnid_to_idx.get(wnid)
            if lbl is None:
                continue
            for fn in os.listdir(wd):
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(wd, fn), lbl))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        from PIL import Image
        path, lbl = self.samples[i]
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                if self.transform is not None:
                    im = self.transform(im)
            return im, lbl
        except (OSError, IOError):
            # Skip occasional unreadable files by returning the next sample
            return self.__getitem__((i + 1) % len(self.samples))


def _get_imagenet1k_loaders(
    batch_size: int, num_workers: int, data_dir: str,
    mean: tuple, std: tuple,
) -> Tuple[DataLoader, DataLoader]:
    """
    Load full ImageNet-1k from an ImageFolder layout: <data_dir>/train/<wnid>/*.JPEG
    and <data_dir>/val/<wnid>/*.JPEG. Standard 224x224 preprocessing.

    Some on-disk train/ trees are partial (subset of the 1000 classes populated).
    We canonicalise the wnid -> label mapping to the sorted full ImageNet-1k
    ordering so labels always align with the torchvision pretrained model. Empty
    train classes are simply absent from the train sampler; KD from the teacher's
    soft targets covers their gradient signal.
    """
    import os
    from PIL import Image
    from torch.utils.data import Dataset

    # Parquet path: data_dir contains data/*.parquet (HF imagenet-1k mirror)
    parquet_subdir = os.path.join(data_dir, "data")
    if os.path.isdir(parquet_subdir) and any(
            f.endswith(".parquet") for f in os.listdir(parquet_subdir)):
        from .parquet_dataset import ParquetImageNetDataset
        train_transform_pq = T.Compose([
            T.RandomResizedCrop(224), T.RandomHorizontalFlip(),
            T.ToTensor(), T.Normalize(mean, std)])
        val_transform_pq = T.Compose([
            T.Resize(256), T.CenterCrop(224),
            T.ToTensor(), T.Normalize(mean, std)])
        train_set = ParquetImageNetDataset(data_dir, "train", train_transform_pq, shuffle=True)
        val_set = ParquetImageNetDataset(data_dir, "validation", val_transform_pq, shuffle=False)
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                   num_workers=num_workers, pin_memory=True,
                                   persistent_workers=num_workers > 0)
        val_loader = DataLoader(val_set, batch_size=batch_size,
                                 num_workers=num_workers, pin_memory=True,
                                 persistent_workers=num_workers > 0)
        return train_loader, val_loader

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"ImageNet-1k train/ not found at {train_dir}. "
            f"Expected ImageFolder layout under {data_dir}/."
        )

    # Canonical 1000-class ordering: sorted wnid list under train/ (the val/ tree
    # is the authoritative source of all 1000 wnids if train is partial).
    train_wnids = sorted(d for d in os.listdir(train_dir)
                          if os.path.isdir(os.path.join(train_dir, d)))
    val_wnids = sorted(d for d in os.listdir(val_dir)
                        if os.path.isdir(os.path.join(val_dir, d)))
    full_wnids = sorted(set(train_wnids) | set(val_wnids))
    if len(full_wnids) != 1000:
        # Be loud about this  --  ImageNet must have exactly 1000 classes
        print(f"[WARN] {len(full_wnids)} wnid classes detected (expected 1000)")
    wnid_to_idx = {w: i for i, w in enumerate(full_wnids)}

    train_transform = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    val_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    train_set = _CanonicalImageFolder(train_dir, train_transform, wnid_to_idx)
    val_set = _CanonicalImageFolder(val_dir, val_transform, wnid_to_idx)
    print(f"[ImageNet] train: {len(train_set)} images across {len(train_wnids)} wnids; "
          f"val: {len(val_set)} images across {len(val_wnids)} wnids; "
          f"label space: {len(full_wnids)} canonical classes")

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers,
        pin_memory=True, persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    return train_loader, val_loader


def _get_imagenet_subset_loaders(
    dataset: str, batch_size: int, num_workers: int, data_dir: str,
    mean: tuple, std: tuple,
) -> Tuple[DataLoader, DataLoader]:
    """
    Load Imagenette or Imagewoof. Both are 10-class subsets of ImageNet at
    320×320 source resolution. We resize to 256 then center-crop to 224 for
    validation, and use random-resized-crop to 224 for training, matching the
    standard ImageNet preprocessing pipeline.

    Imagenette directory: imagenette2-320/{train,val}/<wnid>/<image>.JPEG
    Imagewoof  directory: imagewoof2-320/{train,val}/<wnid>/<image>.JPEG
    """
    import os
    folder = "imagenette2-320" if dataset == "imagenette" else "imagewoof2-320"
    root = os.path.join(data_dir, folder)
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"{dataset} not found at {root}. Place the extracted "
            f"'{folder}' directory under {data_dir}/."
        )

    train_transform = T.Compose([
        T.RandomResizedCrop(224, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    val_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
    val_set = torchvision.datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return train_loader, val_loader


def _get_cifar_loaders(
    dataset: str, batch_size: int, num_workers: int, data_dir: str,
    mean: tuple, std: tuple,
) -> Tuple[DataLoader, DataLoader]:
    """Load CIFAR-10 or CIFAR-100 with standard augmentation."""

    # Training: augment + normalize
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    # Validation: normalize only
    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    # Select dataset class
    DatasetClass = torchvision.datasets.CIFAR10 if dataset.lower() == "cifar10" else torchvision.datasets.CIFAR100

    train_set = DatasetClass(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    val_set = DatasetClass(
        root=data_dir, train=False, download=True, transform=val_transform
    )

    # DataLoader with performance optimizations
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,            # Faster CPU→GPU transfer
        persistent_workers=True,     # Don't respawn workers each epoch
        prefetch_factor=2,           # Prefetch 2 batches per worker
        drop_last=True,              # Drop incomplete last batch (better for BN)
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size * 2,   # Larger batch OK for validation (no gradients)
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    return train_loader, val_loader


def _get_tinyimagenet_loaders(
    batch_size: int, num_workers: int, data_dir: str,
    mean: tuple, std: tuple,
) -> Tuple[DataLoader, DataLoader]:
    """
    Load TinyImageNet (200 classes, 64×64 images).

    TinyImageNet is a subset of ImageNet with 200 classes, 500 training
    images per class, and 50 validation images per class.

    Downloads from the official Stanford source if not present.
    Uses torchvision.datasets.ImageFolder for loading.
    """
    tiny_dir = os.path.join(data_dir, "tiny-imagenet-200")

    # Download if not exists
    if not os.path.exists(tiny_dir):
        _download_tinyimagenet(data_dir)

    train_dir = os.path.join(tiny_dir, "train")
    val_dir = os.path.join(tiny_dir, "val")

    # Fix TinyImageNet val directory structure if needed
    # (default structure puts all images in one folder with a text annotation file)
    _fix_tinyimagenet_val(val_dir)

    train_transform = T.Compose([
        T.RandomCrop(64, padding=8),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
    val_set = torchvision.datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    return train_loader, val_loader


def _download_tinyimagenet(data_dir: str):
    """Download and extract TinyImageNet-200."""
    import urllib.request
    import zipfile

    os.makedirs(data_dir, exist_ok=True)
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")

    print(f"Downloading TinyImageNet to {zip_path}...")
    urllib.request.urlretrieve(url, zip_path)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    os.remove(zip_path)
    print("TinyImageNet ready.")


def _fix_tinyimagenet_val(val_dir: str):
    """
    Reorganize TinyImageNet validation directory into class subfolders.

    Default structure: val/images/val_0.JPEG + val/val_annotations.txt
    Required structure: val/n01443537/val_0.JPEG (ImageFolder compatible)
    """
    annotations_file = os.path.join(val_dir, "val_annotations.txt")
    images_dir = os.path.join(val_dir, "images")

    # Skip if already reorganized
    if not os.path.exists(annotations_file) or not os.path.exists(images_dir):
        return

    print("Reorganizing TinyImageNet val directory...")

    # Parse annotations: image_name -> class_id
    with open(annotations_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            img_name, class_id = parts[0], parts[1]

            # Create class directory
            class_dir = os.path.join(val_dir, class_id)
            os.makedirs(class_dir, exist_ok=True)

            # Move image to class directory
            src = os.path.join(images_dir, img_name)
            dst = os.path.join(class_dir, img_name)
            if os.path.exists(src):
                os.rename(src, dst)

    # Clean up
    if os.path.exists(images_dir):
        try:
            os.rmdir(images_dir)
        except OSError:
            pass  # Not empty, some images may have failed to move

    print("Val directory reorganized.")
