from typing import Dict

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure


def make_et_mask(mask: np.ndarray) -> np.ndarray:
    """
    ET = Enhancing Tumor = label 3 after remapping.
    """
    return mask == 3


def make_tc_mask(mask: np.ndarray) -> np.ndarray:
    """
    TC = Tumor Core = labels 1 and 3 after remapping.
    """
    return np.logical_or(mask == 1, mask == 3)


def make_wt_mask(mask: np.ndarray) -> np.ndarray:
    """
    WT = Whole Tumor = labels 1, 2, and 3 after remapping.
    """
    return mask > 0


def dice_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    """
    Dice score for two binary masks.

    Special case:
    - if both masks are empty, return 1.0
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    pred_sum = pred.sum()
    target_sum = target.sum()

    if pred_sum == 0 and target_sum == 0:
        return 1.0

    intersection = np.logical_and(pred, target).sum()
    return float((2.0 * intersection) / (pred_sum + target_sum + eps))


def _surface_distances(mask_gt: np.ndarray, mask_pred: np.ndarray) -> np.ndarray:
    """
    Compute distances from the surface voxels of one binary object to the other.

    Returns distances for both directions later through hd95().
    """
    mask_gt = mask_gt.astype(bool)
    mask_pred = mask_pred.astype(bool)

    if mask_gt.sum() == 0 or mask_pred.sum() == 0:
        return np.array([], dtype=np.float32)

    footprint = generate_binary_structure(mask_gt.ndim, 1)

    gt_border = np.logical_xor(mask_gt, binary_erosion(mask_gt, structure=footprint, border_value=0))
    pred_border = np.logical_xor(mask_pred, binary_erosion(mask_pred, structure=footprint, border_value=0))

    if not np.any(gt_border) or not np.any(pred_border):
        return np.array([], dtype=np.float32)

    dt = distance_transform_edt(~mask_pred)
    distances = dt[gt_border]

    return distances.astype(np.float32)


def hd95(pred: np.ndarray, target: np.ndarray) -> float:
    """
    95th percentile Hausdorff Distance for binary masks.

    Special cases:
    - if both masks are empty, return 0.0
    - if exactly one mask is empty, return NaN
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    if pred.sum() == 0 and target.sum() == 0:
        return 0.0

    if pred.sum() == 0 or target.sum() == 0:
        return float("nan")

    d1 = _surface_distances(target, pred)
    d2 = _surface_distances(pred, target)

    if len(d1) == 0 or len(d2) == 0:
        return float("nan")

    all_distances = np.concatenate([d1, d2], axis=0)
    return float(np.percentile(all_distances, 95))


def compute_brats_region_metrics(pred_mask: np.ndarray, target_mask: np.ndarray) -> Dict[str, float]:
    """
    Compute Dice and HD95 for ET, TC, WT.

    Inputs:
        pred_mask:   [H, W] or [D, H, W] integer label map
        target_mask: [H, W] or [D, H, W] integer label map
    """
    pred_et = make_et_mask(pred_mask)
    pred_tc = make_tc_mask(pred_mask)
    pred_wt = make_wt_mask(pred_mask)

    target_et = make_et_mask(target_mask)
    target_tc = make_tc_mask(target_mask)
    target_wt = make_wt_mask(target_mask)

    metrics = {
        "dice_et": dice_score(pred_et, target_et),
        "dice_tc": dice_score(pred_tc, target_tc),
        "dice_wt": dice_score(pred_wt, target_wt),
        "hd95_et": hd95(pred_et, target_et),
        "hd95_tc": hd95(pred_tc, target_tc),
        "hd95_wt": hd95(pred_wt, target_wt),
    }

    return metrics