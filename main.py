import argparse
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
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed, deterministic=True)
    output_dirs = prepare_output_dirs("outputs")

    effective_epochs = 1 if args.debug else args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_case_ids = get_case_ids(args.data_root)
    train_case_ids, val_case_ids = get_fold_case_ids(
        data_root=args.data_root,
        fold=args.fold,
        num_folds=5,
        seed=args.seed,
    )

    if args.debug:
        train_case_ids = train_case_ids[:8]
        val_case_ids = val_case_ids[:2]

    train_loader, val_loader = build_dataloaders(
        data_root=args.data_root,
        train_case_ids=train_case_ids,
        val_case_ids=val_case_ids,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        ignore_empty_slices=True,
        min_foreground_pixels=1,
    )

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

    print("=" * 60)
    print("BraTS 2D U-Net Pipeline")
    print("=" * 60)
    print(f"Data root    : {args.data_root}")
    print(f"Fold         : {args.fold}")
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
    print(f"Total cases  : {len(all_case_ids)}")
    print(f"Train cases  : {len(train_case_ids)}")
    print(f"Val cases    : {len(val_case_ids)}")
    print(f"Train slices : {len(train_loader.dataset)}")
    print(f"Val slices   : {len(val_loader.dataset)}")
    print("=" * 60)

    best_val_loss = float("inf")
    best_ckpt_path = output_dirs["checkpoints"] / f"fold_{args.fold}_best.pt"

    for epoch in range(1, effective_epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )

        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )

        metrics_payload = {
            "fold": args.fold,
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "debug": args.debug,
            "seed": args.seed,
        }

        save_json(
            metrics_payload,
            output_dirs["metrics"] / f"fold_{args.fold}_epoch_{epoch}.json"
        )

        checkpoint = {
            "epoch": epoch,
            "fold": args.fold,
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
        print(f"HD95 ET         : {val_metrics['hd95_et']:.4f}")
        print(f"HD95 TC         : {val_metrics['hd95_tc']:.4f}")
        print(f"HD95 WT         : {val_metrics['hd95_wt']:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    main()