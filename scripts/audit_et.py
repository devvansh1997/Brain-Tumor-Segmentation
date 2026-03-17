import numpy as np
from pathlib import Path

labels_dir = Path("/lustre/fs1/home/de807845/med_img_computing/datasets/Brats/Task01_BrainTumour/preprocessed/labelsTr")

files = sorted(labels_dir.glob("*.npy"))
print("num label files:", len(files))

cases_with_et = 0
cases_without_et = 0

for p in files:
    arr = np.load(p, mmap_mode="r")   # [D, H, W]
    uniq = np.unique(arr)

    et_voxels = int((arr == 3).sum())
    tc_voxels = int(np.logical_or(arr == 1, arr == 3).sum())
    wt_voxels = int((arr > 0).sum())

    if et_voxels > 0:
        cases_with_et += 1
    else:
        cases_without_et += 1

    print(f"{p.name}: unique={uniq.tolist()} | ET={et_voxels} | TC={tc_voxels} | WT={wt_voxels}")

print()
print("cases_with_et:", cases_with_et)
print("cases_without_et:", cases_without_et)