import sys
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess BraTS NIfTI files into .npy files")
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to Task01_BrainTumour root",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npy files",
    )
    return parser.parse_args()


def is_valid_nii_file(path: Path) -> bool:
    name = path.name
    if not name.endswith(".nii.gz"):
        return False
    if name.startswith("._"):
        return False
    if name.startswith("."):
        return False
    return True


def remap_labels(mask: np.ndarray) -> np.ndarray:
    """
    BraTS labels: {0,1,2,4}
    Remap to:     {0,1,2,3}
    """
    mask = mask.astype(np.int16, copy=False)
    mask[mask == 4] = 3
    return mask


def zscore_normalize(volume: np.ndarray) -> np.ndarray:
    """
    volume shape: [H, W, D, C]
    Z-score normalize each modality independently on non-zero voxels.
    """
    volume = volume.astype(np.float32, copy=False)

    for c in range(volume.shape[-1]):
        m = volume[..., c]
        nz = m != 0
        if np.any(nz):
            mean = m[nz].mean()
            std = m[nz].std()
            if std > 0:
                m[nz] = (m[nz] - mean) / std
            else:
                m[nz] = m[nz] - mean
        volume[..., c] = m

    return volume


def main():
    args = parse_args()

    data_root = Path(args.data_root).resolve()

    print(f"[INFO] DATA_ROOT = {data_root}")
    print(f"[INFO] EXISTS = {data_root.exists()}")
    print(f"[INFO] imagesTr exists = {(data_root / 'imagesTr').exists()}")
    print(f"[INFO] labelsTr exists = {(data_root / 'labelsTr').exists()}")

    assert data_root.exists(), f"DATA_ROOT does not exist: {data_root}"
    assert (data_root / "imagesTr").exists(), "imagesTr missing"
    assert (data_root / "labelsTr").exists(), "labelsTr missing"

    preproc_root = data_root / "preprocessed"
    out_img_dir = preproc_root / "imagesTr"
    out_lbl_dir = preproc_root / "labelsTr"

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Output images -> {out_img_dir}")
    print(f"[INFO] Output labels -> {out_lbl_dir}")

    all_image_files = sorted((data_root / "imagesTr").glob("*.nii.gz"))
    all_label_files = sorted((data_root / "labelsTr").glob("*.nii.gz"))

    image_files = [p for p in all_image_files if is_valid_nii_file(p)]
    label_files = [p for p in all_label_files if is_valid_nii_file(p)]

    skipped_images = [p.name for p in all_image_files if not is_valid_nii_file(p)]
    skipped_labels = [p.name for p in all_label_files if not is_valid_nii_file(p)]

    print(f"[INFO] Found {len(all_image_files)} raw image-path matches")
    print(f"[INFO] Found {len(all_label_files)} raw label-path matches")
    print(f"[INFO] Valid image files: {len(image_files)}")
    print(f"[INFO] Valid label files: {len(label_files)}")

    if skipped_images:
        print(f"[INFO] Skipping {len(skipped_images)} invalid image files (examples: {skipped_images[:5]})")
    if skipped_labels:
        print(f"[INFO] Skipping {len(skipped_labels)} invalid label files (examples: {skipped_labels[:5]})")

    assert len(image_files) > 0, "No valid image files found"
    assert len(label_files) > 0, "No valid label files found"

    label_map = {p.name: p for p in label_files}

    written = 0
    skipped_existing = 0
    missing_labels = 0

    for img_path in tqdm(image_files, desc="Preprocessing cases"):
        case_name = img_path.name
        lbl_path = label_map.get(case_name)

        if lbl_path is None:
            print(f"[WARN] Missing label for {case_name}")
            missing_labels += 1
            continue

        out_img_path = out_img_dir / case_name.replace(".nii.gz", ".npy")
        out_lbl_path = out_lbl_dir / case_name.replace(".nii.gz", ".npy")

        if out_img_path.exists() and out_lbl_path.exists() and not args.overwrite:
            skipped_existing += 1
            continue

        try:
            img = nib.load(str(img_path)).get_fdata()
            lbl = nib.load(str(lbl_path)).get_fdata()
        except Exception as e:
            print(f"[ERROR] Failed to load {case_name}: {e}")
            continue

        # Expected:
        # img: [H, W, D, C]
        # lbl: [H, W, D]
        if img.ndim != 4:
            print(f"[ERROR] Unexpected image shape for {case_name}: {img.shape}")
            continue
        if lbl.ndim != 3:
            print(f"[ERROR] Unexpected label shape for {case_name}: {lbl.shape}")
            continue

        img = zscore_normalize(img)
        lbl = remap_labels(lbl)

        # [H, W, D, C] -> [D, C, H, W]
        img = np.transpose(img, (2, 3, 0, 1)).astype(np.float32, copy=False)

        # [H, W, D] -> [D, H, W]
        lbl = np.transpose(lbl, (2, 0, 1)).astype(np.uint8, copy=False)

        np.save(out_img_path, img)
        np.save(out_lbl_path, lbl)

        if not out_img_path.exists() or not out_lbl_path.exists():
            raise RuntimeError(f"Failed writing outputs for {case_name}")

        written += 1

    print("\n[INFO] Preprocessing complete.")
    print(f"[INFO] Cases written: {written}")
    print(f"[INFO] Existing skipped: {skipped_existing}")
    print(f"[INFO] Missing labels: {missing_labels}")
    print(f"[INFO] Output root: {preproc_root}")


if __name__ == "__main__":
    main()