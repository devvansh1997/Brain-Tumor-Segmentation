# Brain Tumor Segmentation using 2D U-Net

This repository contains a minimal and reproducible pipeline for brain tumor segmentation on the BraTS dataset using a 2D U-Net architecture. The project supports local debugging and scalable training on HPC clusters.

---

## Project Structure

```
Brain-Tumor-Segmentation/
│
├── data/
│   ├── dataset.py
│   └── transforms.py
│
├── models/
│   └── unet2d.py
│
├── scripts/
│   ├── train.py
│   ├── validate.py
│   └── visualize_preds.py
│
├── configs/
│   └── config.yaml
│
├── results/
├── logs/
└── README.md
```

---

## Setup

### Create environment

```bash
conda create -n med python=3.10
conda activate med
pip install -r requirements.txt
```

---

## Dataset

Download BraTS dataset and place it as:

```
DATA_DIR/Task01_BrainTumour/
```

Directory should contain:

```
imagesTr/
labelsTr/
imagesTs/
```

---

## Running Locally

Run small sample to verify pipeline:

```bash
python scripts/train.py \
    --epochs 2 \
    --batch_size 4 \
    --debug True
```

---

## Running on HPC

Example job script:

```bash
#!/bin/bash
#SBATCH --job-name=brats_unet
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

source .venv/bin/activate
python scripts/train.py
```

Submit job:

```bash
sbatch job.slurm
```

---

## Training Details

* Model: 2D U-Net
* Loss: Dice + CrossEntropy
* Optimizer: Adam
* Input: 2D slices from BraTS volumes

---

## Visualization

```bash
python scripts/visualize_preds.py
```
