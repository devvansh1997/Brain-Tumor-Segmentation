import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from data.splits import get_fold_case_ids
from models.unet2d import UNet2D  # adjust if needed

# ===== CONFIG =====
DATA_ROOT = Path("/lustre/fs1/home/de807845/med_img_computing/datasets/Brats/Task01_BrainTumour")
PREPROC_ROOT = DATA_ROOT / "preprocessed"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FOLD = 1
CHECKPOINT = "outputs/checkpoints/fold_1_best.pt"
SAVE_PATH = "qualitative_example.png"

# ===== LOAD MODEL =====
model = UNet2D(in_channels=4, out_channels=4)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ===== GET ONE VALIDATION CASE =====
_, val_ids = get_fold_case_ids(DATA_ROOT, fold=FOLD)
case_id = val_ids[0]  # just pick first case

img = np.load(PREPROC_ROOT / "imagesTr" / f"{case_id}.npy")  # [D,C,H,W]
gt  = np.load(PREPROC_ROOT / "labelsTr" / f"{case_id}.npy")  # [D,H,W]

# ===== PICK SLICE WITH TUMOR =====
slice_idx = None
for i in range(img.shape[0]):
    if np.any(gt[i] > 0):
        slice_idx = i
        break

assert slice_idx is not None, "No tumor slice found!"

image_slice = img[slice_idx]   # [C,H,W]
gt_slice = gt[slice_idx]       # [H,W]

# ===== RUN MODEL =====
with torch.no_grad():
    inp = torch.from_numpy(image_slice).unsqueeze(0).to(DEVICE)
    logits = model(inp)
    pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

# ===== HELPER: OVERLAY FUNCTION =====
def overlay_mask(base, mask):
    base = base.copy()
    base = (base - base.min()) / (base.max() - base.min() + 1e-8)

    rgb = np.stack([base]*3, axis=-1)

    # WT (green)
    rgb[mask > 0] = [0, 1, 0]

    # TC (blue)
    tc = np.logical_or(mask == 1, mask == 3)
    rgb[tc] = [0, 0, 1]

    # ET (red)
    rgb[mask == 3] = [1, 0, 0]

    return rgb

# Use FLAIR (usually channel 0 or 1 depending on dataset — adjust if needed)
mri = image_slice[0]

pred_overlay = overlay_mask(mri, pred)
gt_overlay   = overlay_mask(mri, gt_slice)

# ===== PLOT =====
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(pred_overlay)
plt.title("Prediction")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(gt_overlay)
plt.title("Ground Truth")
plt.axis("off")

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=300)
plt.close()

print(f"Saved qualitative example to: {SAVE_PATH}")