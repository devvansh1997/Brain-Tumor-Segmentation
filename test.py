import numpy as np
from pathlib import Path
from data.splits import get_fold_case_ids

root=Path('/lustre/fs1/home/de807845/med_img_computing/datasets/Brats/Task01_BrainTumour')
lbl_dir=root/'preprocessed'/'labelsTr'
total=0
print('Checking ET-positive val cases per fold...') 

for fold in range(1,6):
    _, val_ids = get_fold_case_ids(root, fold=fold, num_folds=5, seed=42)
    et_pos=[]; et_neg=[]
    for cid in val_ids:
        arr=np.load(lbl_dir/f'{cid}.npy', mmap_mode='r')
        if np.any(arr==3): et_pos.append(cid)
        else: et_neg.append(cid)
    print(f'Fold {fold}: val_cases={len(val_ids)} | ET_pos={len(et_pos)} | ET_neg={len(et_neg)}')
    if len(et_neg)>0: print('  ET_neg_examples:', et_neg[:10])