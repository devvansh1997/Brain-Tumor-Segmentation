from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


def remap_brats_labels(mask: np.ndarray) -> np.ndarray:
    """
    Remap BraTS labels:
        0 -> 0
        1 -> 1
        2 -> 2
        4 -> 3
    """
    remapped = np.zeros_like(mask, dtype=np.uint8)
    remapped[mask == 1] = 1
    remapped[mask == 2] = 2
    remapped[mask == 4] = 3
    return remapped


def zscore_normalize_per_modality(volume: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Z-score normalize each modality independently using non-zero voxels only.

    Expected input shape:
        (H, W, D, C)
    """
    volume = volume.astype(np.float32, copy=False)
    out = np.zeros_like(volume, dtype=np.float32)

    num_modalities = volume.shape[-1]

    for c in range(num_modalities):
        channel = volume[..., c]
        nonzero_mask = channel != 0

        if np.any(nonzero_mask):
            values = channel[nonzero_mask]
            mean = values.mean()
            std = values.std()
            out[..., c] = channel
            out[..., c][nonzero_mask] = (values - mean) / (std + eps)
        else:
            out[..., c] = channel

    return out


class BratsSliceDataset(Dataset):
    """
    Slice-wise 2D BraTS dataset.

    Each item returns:
        image: FloatTensor of shape [C, H, W]
        mask:  LongTensor of shape [H, W]
        meta:  dict with case_id and slice_idx
    """

    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path,
        case_ids: Sequence[str],
        ignore_empty_slices: bool = False,
        min_foreground_pixels: int = 1,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.case_ids = list(case_ids)
        self.ignore_empty_slices = ignore_empty_slices
        self.min_foreground_pixels = min_foreground_pixels
        self._cache = {}

        self.samples = self._build_index()

    def _build_index(self) -> List[Tuple[str, Path, Path, int]]:
        """
        Build an index of (case_id, image_path, label_path, slice_idx).
        """
        samples = []

        for case_id in self.case_ids:
            image_path = self.images_dir / f"{case_id}.nii.gz"
            label_path = self.labels_dir / f"{case_id}.nii.gz"

            if not image_path.exists():
                raise FileNotFoundError(f"Missing image file: {image_path}")
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label file: {label_path}")

            label_nii = nib.load(str(label_path))
            label = label_nii.get_fdata().astype(np.uint8)  # [H, W, D]

            if label.ndim != 3:
                raise ValueError(
                    f"Expected label shape [H, W, D], got {label.shape} for case {case_id}"
                )

            depth = label.shape[2]

            for slice_idx in range(depth):
                if self.ignore_empty_slices:
                    fg_pixels = int((label[:, :, slice_idx] > 0).sum())
                    if fg_pixels < self.min_foreground_pixels:
                        continue

                samples.append((case_id, image_path, label_path, slice_idx))

        if len(samples) == 0:
            raise RuntimeError("No valid slices found. Check paths or slice filtering settings.")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_case_slice(
        self,
        image_path: Path,
        label_path: Path,
        slice_idx: int,
    ):
        """
        Load one 2D slice from a BraTS case.

        Returns:
            image_slice: [H, W, C]
            mask_slice:  [H, W]
        """

        image_nii = nib.load(str(image_path))
        label_nii = nib.load(str(label_path))

        # Load directly as float32 to avoid float64 memory spike
        image = image_nii.get_fdata(dtype=np.float32)

        # nibabel requires float dtype; convert to uint8 after loading
        mask = label_nii.get_fdata(dtype=np.float32).astype(np.uint8)

        if image.ndim != 4:
            raise ValueError(
                f"Expected image shape [H, W, D, C], got {image.shape} for {image_path.name}"
            )

        if mask.ndim != 3:
            raise ValueError(
                f"Expected mask shape [H, W, D], got {mask.shape} for {label_path.name}"
            )

        # Normalize MRI modalities
        image = zscore_normalize_per_modality(image)

        # Remap BraTS labels {0,1,2,4} -> {0,1,2,3}
        mask = remap_brats_labels(mask)

        image_slice = image[:, :, slice_idx, :]   # [H, W, C]
        mask_slice = mask[:, :, slice_idx]        # [H, W]

        return image_slice, mask_slice

    def __getitem__(self, index: int):
        case_id, image_path, label_path, slice_idx = self.samples[index]

        image_slice, mask_slice = self._load_case_slice(
            image_path=image_path,
            label_path=label_path,
            slice_idx=slice_idx,
        )

        image_tensor = torch.from_numpy(image_slice).permute(2, 0, 1).float()  # [C, H, W]
        mask_tensor = torch.from_numpy(mask_slice).long()  # [H, W]

        meta = {
            "case_id": case_id,
            "slice_idx": slice_idx,
        }

        return image_tensor, mask_tensor, meta