import argparse
import time
from pathlib import Path

import torch

from data.data import build_dataloaders
from data.splits import get_case_ids, get_fold_case_ids
from models.model import build_model
from training.losses import DiceCrossEntropyLoss
from training.train import train_one_epoch, validate_one_epoch
from utils.io import prepare_output_dirs, save_json
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="BraTS 2D U-Net training entry point")
    parser.add_argument("--data_root", type=str, required=True, help="Path to Task01_BrainTumour")
    parser.add_argument("--fold", type=int, default=1, help="Fold number (1-5)")
    parser.add_argument("--run_all_folds", action="store_true", help="Run all 5 folds sequentially")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()


def run_fold(args, fold: int, total_cases: int):
    output_dirs = prepare_output_dirs("outputs")

    effective_epochs = 2 if args.debug else args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    train_case_ids, val_case_ids = get_fold_case_ids(
        data_root=args.data_root,
        fold=fold,
        num_folds=5,
        seed=args.seed,
    )

    if args.debug:
        train_case_ids = train_case_ids[:8]
        val_case_ids = val_case_ids[:2]

    preprocessed_root = Path(args.data_root) / "preprocessed"
    using_preprocessed = (
        (preprocessed_root / "imagesTr").exists() and
        (preprocessed_root / "labelsTr").exists()
    )

    print(f"Using preprocessed data: {using_preprocessed}")
    print(f"Preprocessed root: {preprocessed_root}")

    train_loader, val_loader = build_dataloaders(
        data_root=args.data_root,
        train_case_ids=train_case_ids,
        val_case_ids=val_case_ids,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        ignore_empty_slices=True,
        min_foreground_pixels=1,
    )
    print("-------- Dataloaders Built --------")

    model = build_model(
        in_channels=4,
        out_channels=4,
        base_channels=32,
    ).to(device)

    loss_fn = DiceCrossEntropyLoss(
        dice_weight=0.5,
        ce_weight=0.5,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("-------- Model + Loss + Optimizer Built --------")

    print("=" * 60)
    print(f"BraTS 2D U-Net Pipeline | Fold {fold}")
    print("=" * 60)
    print(f"Data root    : {args.data_root}")
    print(f"Fold         : {fold}")
    print(f"Batch size   : {args.batch_size}")
    print(f"Num workers  : {args.num_workers}")
    print(f"Epochs       : {effective_epochs}")
    print(f"LR           : {args.lr}")
    print(f"Seed         : {args.seed}")
    print(f"Debug        : {args.debug}")
    print(f"Device       : {device}")
    print(f"Model        : {model.__class__.__name__}")
    print(f"Loss         : {loss_fn.__class__.__name__}")
    print(f"Optimizer    : {optimizer.__class__.__name__}")
    print("-" * 60)
    print(f"Total cases  : {total_cases}")
    print(f"Train cases  : {len(train_case_ids)}")
    print(f"Val cases    : {len(val_case_ids)}")
    print(f"Train slices : {len(train_loader.dataset)}")
    print(f"Val slices   : {len(val_loader.dataset)}")
    print("=" * 60)

    best_val_loss = float("inf")
    best_ckpt_path = output_dirs["checkpoints"] / f"fold_{fold}_best.pt"

    fold_start_time = time.perf_counter()

    for epoch in range(1, effective_epochs + 1):
        epoch_start = time.perf_counter()

        train_start = time.perf_counter()
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )
        train_time = time.perf_counter() - train_start

        compute_hd95_this_epoch = (epoch == effective_epochs)

        val_start = time.perf_counter()
        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            compute_hd95=compute_hd95_this_epoch,
        )
        val_time = time.perf_counter() - val_start

        epoch_time = time.perf_counter() - epoch_start

        metrics_payload = {
            "fold": fold,
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "debug": args.debug,
            "seed": args.seed,
            "compute_hd95": compute_hd95_this_epoch,
            "train_time_sec": train_time,
            "val_time_sec": val_time,
            "epoch_time_sec": epoch_time,
        }

        save_json(
            metrics_payload,
            output_dirs["metrics"] / f"fold_{fold}_epoch_{epoch}.json"
        )

        checkpoint = {
            "epoch": epoch,
            "fold": fold,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "args": vars(args),
        }

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(checkpoint, best_ckpt_path)
            print(f"Saved new best checkpoint: {best_ckpt_path}")

        print(f"Epoch {epoch} results")
        print("-" * 60)
        print(f"Train loss      : {train_metrics['loss']:.6f}")
        print(f"Train pixel acc : {train_metrics['pixel_acc']:.6f}")
        print(f"Val loss        : {val_metrics['loss']:.6f}")
        print(f"Val pixel acc   : {val_metrics['pixel_acc']:.6f}")
        print(f"Dice ET         : {val_metrics['dice_et']:.4f}")
        print(f"Dice TC         : {val_metrics['dice_tc']:.4f}")
        print(f"Dice WT         : {val_metrics['dice_wt']:.4f}")

        if compute_hd95_this_epoch:
            print(f"HD95 ET         : {val_metrics['hd95_et']:.4f}")
            print(f"HD95 TC         : {val_metrics['hd95_tc']:.4f}")
            print(f"HD95 WT         : {val_metrics['hd95_wt']:.4f}")
        else:
            print("HD95 ET         : skipped")
            print("HD95 TC         : skipped")
            print("HD95 WT         : skipped")

        print(f"Train runtime   : {train_time:.2f} sec")
        print(f"Val runtime     : {val_time:.2f} sec")
        print(f"Epoch runtime   : {epoch_time:.2f} sec")
        print("=" * 60)

    fold_total_time = time.perf_counter() - fold_start_time
    print(f"\n======== FOLD {fold} RUNTIME SUMMARY ========")
    print(f"Fold runtime: {fold_total_time:.2f} seconds")
    print(f"Fold runtime: {fold_total_time/60:.2f} minutes")
    print("=============================================\n")


def main():
    print("-------- Main Started --------")
    start_time = time.perf_counter()

    args = parse_args()

    set_seed(args.seed, deterministic=True)

    print("-------- Args Processed --------")

    all_case_ids = get_case_ids(args.data_root)

    print("-------- Case ID done --------")

    if args.run_all_folds:
        for fold in range(1, 6):
            print(f"\n########## STARTING FOLD {fold} ##########")
            run_fold(args, fold, total_cases=len(all_case_ids))
            print(f"########## FINISHED FOLD {fold} ##########\n")
    else:
        run_fold(args, args.fold, total_cases=len(all_case_ids))

    total_time = time.perf_counter() - start_time
    print("======== RUNTIME SUMMARY ========")
    print(f"Total runtime: {total_time:.2f} seconds")
    print(f"Total runtime: {total_time/60:.2f} minutes")
    print("=================================")


if __name__ == "__main__":
    main()