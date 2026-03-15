from typing import Dict, DefaultDict
from collections import defaultdict

import time
import torch
import numpy as np

from training.metrics import compute_brats_region_metrics

def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    device,
) -> Dict[str, float]:

    model.train()
    first_batch_start = time.perf_counter()

    running_loss = 0.0
    correct_pixels = 0
    total_pixels = 0

    for batch_idx, (images, masks, meta) in enumerate(loader):

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if batch_idx == 0:
            print(f"[TRAIN] First batch ready in {time.perf_counter() - first_batch_start:.2f} sec")

        optimizer.zero_grad()

        logits = model(images)
        loss = loss_fn(logits, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(logits, dim=1)

        correct_pixels += (preds == masks).sum().item()
        total_pixels += masks.numel()

    avg_loss = running_loss / len(loader.dataset)
    pixel_acc = correct_pixels / total_pixels if total_pixels > 0 else 0.0

    return {
        "loss": avg_loss,
        "pixel_acc": pixel_acc,
    }


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    loss_fn,
    device,
    compute_hd95: bool = False,
) -> Dict[str, float]:

    model.eval()
    first_batch_start = time.perf_counter()

    running_loss = 0.0
    correct_pixels = 0
    total_pixels = 0

    pred_volumes: DefaultDict[str, dict] = defaultdict(dict)
    gt_volumes: DefaultDict[str, dict] = defaultdict(dict)

    with torch.no_grad():
        for batch_idx, (images, masks, meta) in enumerate(loader):

            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if batch_idx == 0:
                print(f"[VAL] First batch ready in {time.perf_counter() - first_batch_start:.2f} sec")

            logits = model(images)
            loss = loss_fn(logits, masks)

            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(logits, dim=1)

            correct_pixels += (preds == masks).sum().item()
            total_pixels += masks.numel()

            preds_np = preds.cpu().numpy()
            masks_np = masks.cpu().numpy()

            case_ids = meta["case_id"]
            slice_ids = meta["slice_idx"].cpu().numpy()

            for i in range(len(case_ids)):
                cid = case_ids[i]
                sid = int(slice_ids[i])

                pred_volumes[cid][sid] = preds_np[i]
                gt_volumes[cid][sid] = masks_np[i]

    avg_loss = running_loss / len(loader.dataset)
    pixel_acc = correct_pixels / total_pixels if total_pixels > 0 else 0.0

    dice_et = []
    dice_tc = []
    dice_wt = []

    hd95_et = []
    hd95_tc = []
    hd95_wt = []

    for cid in pred_volumes.keys():

        slice_ids = sorted(pred_volumes[cid].keys())

        pred_volume = np.stack([pred_volumes[cid][s] for s in slice_ids])
        gt_volume = np.stack([gt_volumes[cid][s] for s in slice_ids])

        metrics = compute_brats_region_metrics(
            pred_volume,
            gt_volume,
            compute_hd95=compute_hd95,
        )

        dice_et.append(metrics["dice_et"])
        dice_tc.append(metrics["dice_tc"])
        dice_wt.append(metrics["dice_wt"])

        hd95_et.append(metrics["hd95_et"])
        hd95_tc.append(metrics["hd95_tc"])
        hd95_wt.append(metrics["hd95_wt"])

    results = {
        "loss": avg_loss,
        "pixel_acc": pixel_acc,
        "dice_et": float(np.nanmean(dice_et)),
        "dice_tc": float(np.nanmean(dice_tc)),
        "dice_wt": float(np.nanmean(dice_wt)),
        "hd95_et": float(np.nanmean(hd95_et)) if len(hd95_et) > 0 else float("nan"),
        "hd95_tc": float(np.nanmean(hd95_tc)) if len(hd95_tc) > 0 else float("nan"),
        "hd95_wt": float(np.nanmean(hd95_wt)) if len(hd95_wt) > 0 else float("nan"),
    }

    return results