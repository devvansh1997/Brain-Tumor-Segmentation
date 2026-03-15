from pathlib import Path
from typing import List, Tuple

from sklearn.model_selection import KFold


def get_case_ids(data_root: str | Path) -> List[str]:
    """
    Read all valid case IDs from imagesTr directory.

    Expected:
        data_root/
            imagesTr/
                BRATS_001.nii.gz
                BRATS_002.nii.gz
                ...

    Ignores hidden/system files such as:
        ._BRATS_001.nii.gz
    """
    data_root = Path(data_root)
    images_dir = data_root / "imagesTr"

    if not images_dir.exists():
        raise FileNotFoundError(f"imagesTr directory not found: {images_dir}")

    case_ids = []

    for path in images_dir.glob("*.nii.gz"):
        name = path.name

        # Skip hidden/system artifacts like ._BRATS_001.nii.gz
        if name.startswith("."):
            continue
        if name.startswith("._"):
            continue

        case_id = name.replace(".nii.gz", "")
        case_ids.append(case_id)

    case_ids = sorted(case_ids)

    if len(case_ids) == 0:
        raise RuntimeError(f"No valid .nii.gz files found in: {images_dir}")

    return case_ids


def get_fold_case_ids(
    data_root: str | Path,
    fold: int,
    num_folds: int = 5,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Return train/val case IDs for the requested fold.

    fold is 1-based:
        fold=1,2,3,4,5
    """
    if fold < 1 or fold > num_folds:
        raise ValueError(f"fold must be in [1, {num_folds}], got {fold}")

    case_ids = get_case_ids(data_root)

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    splits = list(kf.split(case_ids))
    train_idx, val_idx = splits[fold - 1]

    train_case_ids = [case_ids[i] for i in train_idx]
    val_case_ids = [case_ids[i] for i in val_idx]

    return train_case_ids, val_case_ids