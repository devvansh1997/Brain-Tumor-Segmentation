from pathlib import Path
from typing import Sequence, Tuple

from torch.utils.data import DataLoader

from data.dataset import BratsSliceDataset


def build_datasets(
    data_root: str | Path,
    train_case_ids: Sequence[str],
    val_case_ids: Sequence[str],
    ignore_empty_slices: bool = False,
    min_foreground_pixels: int = 1,
):
    """
    Build train and validation datasets.

    Expected directory structure:
        data_root/
            imagesTr/
            labelsTr/
    """
    data_root = Path(data_root)
    images_dir = data_root / "imagesTr"
    labels_dir = data_root / "labelsTr"

    train_dataset = BratsSliceDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        case_ids=train_case_ids,
        ignore_empty_slices=ignore_empty_slices,
        min_foreground_pixels=min_foreground_pixels,
    )

    val_dataset = BratsSliceDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        case_ids=val_case_ids,
        ignore_empty_slices=False,  # keep validation full by default
        min_foreground_pixels=1,
    )

    return train_dataset, val_dataset


def build_dataloaders(
    data_root: str | Path,
    train_case_ids: Sequence[str],
    val_case_ids: Sequence[str],
    batch_size: int = 8,
    num_workers: int = 4,
    ignore_empty_slices: bool = False,
    min_foreground_pixels: int = 1,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation dataloaders.
    """
    train_dataset, val_dataset = build_datasets(
        data_root=data_root,
        train_case_ids=train_case_ids,
        val_case_ids=val_case_ids,
        ignore_empty_slices=ignore_empty_slices,
        min_foreground_pixels=min_foreground_pixels,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader