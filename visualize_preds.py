from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data.splits import get_fold_case_ids
from models.unet import UNet2D  # change only if your class is imported from a different file


# =========================
# CONFIG
# =========================
DATA_ROOT = Path(r"C:\Users\devan\Medical Image Computing\datasets\BraTs\Task01_BrainTumour")
PREPROC_ROOT = DATA_ROOT / "preprocessed"
CHECKPOINT = Path(r"C:\Users\devan\Medical Image Computing\Assignment-02\Brain-Tumor-Segmentation\outputs\checkpoints\fold_1_best.pt")
SAVE_PATH = Path(r"C:\Users\devan\Medical Image Computing\Assignment-02\Brain-Tumor-Segmentation\qualitative_example.png")

FOLD = 1
VAL_CASE_INDEX = 0          # which validation case to use
MODALITY_INDEX = 0          # which MRI modality to show in grayscale background
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# HELPERS
# =========================
def normalize_for_display(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    vmin = img.min()
    vmax = img.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    return (img - vmin) / (vmax - vmin)


def make_overlay(base_img: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """
    base_img: [H, W] grayscale
    mask: [H, W] with labels {0,1,2,3}
    Colors:
      WT-ish region components:
        label 2 -> green
        label 1 -> blue
        label 3 -> red
    """
    base = normalize_for_display(base_img)
    rgb = np.stack([base, base, base], axis=-1)

    color_mask = np.zeros_like(rgb)

    # label 2 -> green
    color_mask[mask == 2] = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # label 1 -> blue
    color_mask[mask == 1] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # label 3 -> red
    color_mask[mask == 3] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    fg = mask > 0
    rgb[fg] = (1.0 - alpha) * rgb[fg] + alpha * color_mask[fg]
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb


def find_best_tumor_slice(gt_volume: np.ndarray) -> int:
    """
    Pick slice with maximum tumor pixels.
    gt_volume: [D, H, W]
    """
    tumor_counts = [(gt_volume[i] > 0).sum() for i in range(gt_volume.shape[0])]
    return int(np.argmax(tumor_counts))


# =========================
# LOAD MODEL
# =========================
model = UNet2D(in_channels=4, out_channels=4).to(DEVICE)

ckpt = torch.load(CHECKPOINT, map_location=DEVICE)

if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
else:
    model.load_state_dict(ckpt)

model.eval()


# =========================
# LOAD ONE VALIDATION CASE
# =========================
_, val_case_ids = get_fold_case_ids(DATA_ROOT, fold=FOLD, num_folds=5, seed=42)

if VAL_CASE_INDEX >= len(val_case_ids):
    raise IndexError(f"VAL_CASE_INDEX={VAL_CASE_INDEX} is out of range for {len(val_case_ids)} validation cases.")

case_id = val_case_ids[VAL_CASE_INDEX]

img_path = PREPROC_ROOT / "imagesTr" / f"{case_id}.npy"
gt_path = PREPROC_ROOT / "labelsTr" / f"{case_id}.npy"

if not img_path.exists():
    raise FileNotFoundError(f"Missing image file: {img_path}")
if not gt_path.exists():
    raise FileNotFoundError(f"Missing label file: {gt_path}")

image_vol = np.load(img_path)   # [D, C, H, W]
gt_vol = np.load(gt_path)       # [D, H, W]

if image_vol.ndim != 4:
    raise ValueError(f"Expected image_vol shape [D,C,H,W], got {image_vol.shape}")
if gt_vol.ndim != 3:
    raise ValueError(f"Expected gt_vol shape [D,H,W], got {gt_vol.shape}")

slice_idx = find_best_tumor_slice(gt_vol)

image_slice = image_vol[slice_idx]              # [C, H, W]
gt_slice = gt_vol[slice_idx]                    # [H, W]
base_img = image_slice[MODALITY_INDEX]          # [H, W]


# =========================
# PREDICT ONE SLICE
# =========================
inp = torch.from_numpy(image_slice).unsqueeze(0).float().to(DEVICE)  # [1, C, H, W]

with torch.no_grad():
    logits = model(inp)
    pred_slice = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()  # [H, W]


# =========================
# MAKE OVERLAYS
# =========================
pred_overlay = make_overlay(base_img, pred_slice)
gt_overlay = make_overlay(base_img, gt_slice)


# =========================
# PLOT FIGURE
# =========================
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

axes[0].imshow(pred_overlay)
axes[0].set_title("Prediction")
axes[0].axis("off")

axes[1].imshow(gt_overlay)
axes[1].set_title("Ground Truth")
axes[1].axis("off")

fig.suptitle(f"{case_id} | slice {slice_idx}", fontsize=11)
plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved figure to: {SAVE_PATH}")
print(f"Case ID: {case_id}")
print(f"Slice index: {slice_idx}")
print(f"GT unique labels on slice: {np.unique(gt_slice).tolist()}")
print(f"Pred unique labels on slice: {np.unique(pred_slice).tolist()}")