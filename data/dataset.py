from pathlib import Path
from typing import List, Sequence, Tuple

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
        preprocessed: bool = False,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.case_ids = list(case_ids)
        self.ignore_empty_slices = ignore_empty_slices
        self.min_foreground_pixels = min_foreground_pixels
        self.preprocessed = preprocessed

        # case_id -> (image_array, mask_array)
        self._case_cache = {}

        self.samples = self._build_index()

    def _get_case_paths(self, case_id: str) -> Tuple[Path, Path]:
        if self.preprocessed:
            image_path = self.images_dir / f"{case_id}.npy"
            label_path = self.labels_dir / f"{case_id}.npy"
        else:
            image_path = self.images_dir / f"{case_id}.nii.gz"
            label_path = self.labels_dir / f"{case_id}.nii.gz"
        return image_path, label_path

    def _load_case_arrays(self, case_id: str):
        """
        Returns:
            preprocessed=True:
                image: [D, C, H, W]
                mask:  [D, H, W]
            preprocessed=False:
                image: [H, W, D, C]
                mask:  [H, W, D]
        """
        if case_id in self._case_cache:
            return self._case_cache[case_id]

        image_path, label_path = self._get_case_paths(case_id)

        if not image_path.exists():
            raise FileNotFoundError(f"Missing image file: {image_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label file: {label_path}")

        if self.preprocessed:
            image = np.load(image_path, mmap_mode="r")   # [D, C, H, W]
            mask = np.load(label_path, mmap_mode="r")    # [D, H, W]
        else:
            image_nii = nib.load(str(image_path))
            label_nii = nib.load(str(label_path))

            image = image_nii.get_fdata(dtype=np.float32)          # [H, W, D, C]
            mask = label_nii.get_fdata(dtype=np.float32).astype(np.uint8)  # [H, W, D]

            if image.ndim != 4:
                raise ValueError(
                    f"Expected image shape [H, W, D, C], got {image.shape} for {image_path.name}"
                )
            if mask.ndim != 3:
                raise ValueError(
                    f"Expected mask shape [H, W, D], got {mask.shape} for {label_path.name}"
                )

            image = zscore_normalize_per_modality(image)
            mask = remap_brats_labels(mask)

        self._case_cache[case_id] = (image, mask)
        return image, mask

    def _build_index(self) -> List[Tuple[str, int]]:
        """
        Build an index of (case_id, slice_idx).
        """
        samples: List[Tuple[str, int]] = []

        for case_id in self.case_ids:
            image, mask = self._load_case_arrays(case_id)

            if self.preprocessed:
                # mask: [D, H, W]
                if mask.ndim != 3:
                    raise ValueError(
                        f"Expected preprocessed mask shape [D, H, W], got {mask.shape} for case {case_id}"
                    )
                depth = mask.shape[0]

                for slice_idx in range(depth):
                    if self.ignore_empty_slices:
                        fg_pixels = int((mask[slice_idx] > 0).sum())
                        if fg_pixels < self.min_foreground_pixels:
                            continue
                    samples.append((case_id, slice_idx))
            else:
                # mask: [H, W, D]
                if mask.ndim != 3:
                    raise ValueError(
                        f"Expected raw mask shape [H, W, D], got {mask.shape} for case {case_id}"
                    )
                depth = mask.shape[2]

                for slice_idx in range(depth):
                    if self.ignore_empty_slices:
                        fg_pixels = int((mask[:, :, slice_idx] > 0).sum())
                        if fg_pixels < self.min_foreground_pixels:
                            continue
                    samples.append((case_id, slice_idx))

        if len(samples) == 0:
            raise RuntimeError("No valid slices found. Check paths or slice filtering settings.")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        case_id, slice_idx = self.samples[index]
        image, mask = self._load_case_arrays(case_id)

        if self.preprocessed:
            # image: [D, C, H, W]
            # mask:  [D, H, W]
            image_slice = np.asarray(image[slice_idx], dtype=np.float32)  # [C, H, W]
            mask_slice = np.asarray(mask[slice_idx], dtype=np.uint8)      # [H, W]
            image_tensor = torch.from_numpy(image_slice).float()
            mask_tensor = torch.from_numpy(mask_slice).long()
        else:
            # image: [H, W, D, C]
            # mask:  [H, W, D]
            image_slice = image[:, :, slice_idx, :]   # [H, W, C]
            mask_slice = mask[:, :, slice_idx]        # [H, W]
            image_tensor = torch.from_numpy(image_slice).permute(2, 0, 1).float()  # [C, H, W]
            mask_tensor = torch.from_numpy(mask_slice).long()

        meta = {
            "case_id": case_id,
            "slice_idx": torch.tensor(slice_idx, dtype=torch.long),
        }

        return image_tensor, mask_tensor, meta