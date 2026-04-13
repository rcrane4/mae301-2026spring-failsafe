"""
train.py
--------
Fine-tunes SegFormer-b0 on the OSF Ti-64 SEM fractography dataset.
Trains all three subsets (lack_of_fusion, keyhole, all_defects) separately.
CPU-optimized: small image size, small batch, few epochs.

Usage:
    python train.py
    python train.py --epochs 10 --image_size 256

Outputs (per subset):
    checkpoints/<subset>/best_model.pt   <- best checkpoint by val mIoU
    checkpoints/<subset>/last_model.pt   <- final epoch checkpoint
    checkpoints/<subset>/history.json    <- loss/mIoU per epoch
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import SegformerForSemanticSegmentation
import torch.nn.functional as F

from dataset import FractographyDataset

# ── Config ────────────────────────────────────────────────────────────────────
NUM_CLASSES  = 2          # background (0) + defect (1)
IMAGE_SIZE   = (256, 256) # smaller = faster on CPU; increase if you have time
BATCH_SIZE   = 2
EPOCHS       = 15
LR           = 6e-5
TRAIN_FRAC   = 0.8
WEIGHT_DECAY = 0.01

SUBSETS = ["lack_of_fusion", "keyhole", "all_defects"]
# ─────────────────────────────────────────────────────────────────────────────


def compute_miou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """Mean Intersection over Union across all classes."""
    ious = []
    preds   = preds.view(-1)
    targets = targets.view(-1)
    for cls in range(num_classes):
        pred_mask   = preds == cls
        target_mask = targets == cls
        intersection = (pred_mask & target_mask).sum().item()
        union        = (pred_mask | target_mask).sum().item()
        if union == 0:
            continue  # class not present in this batch
        ious.append(intersection / union)
    return float(np.mean(ious)) if ious else 0.0


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    Soft Dice loss for binary segmentation.
    Directly optimizes overlap — critical for imbalanced datasets.
    logits: (B, num_classes, H, W)
    targets: (B, H, W) integer labels
    """
    probs = torch.softmax(logits, dim=1)  # (B, C, H, W)
    # Focus on defect class (index 1)
    prob_defect   = probs[:, 1]           # (B, H, W)
    target_defect = (targets == 1).float()

    intersection = (prob_defect * target_defect).sum(dim=(1, 2))
    union        = prob_defect.sum(dim=(1, 2)) + target_defect.sum(dim=(1, 2))
    dice         = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def combined_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    defect_weight: float = 10.0,
    dice_weight: float = 0.5,
) -> torch.Tensor:
    """
    Weighted CE + Dice loss.
    defect_weight: how much extra to penalize missing defect pixels.
                   Start at 10x given ~6% defect pixels.
    dice_weight:   blend factor for Dice loss (0 = CE only, 1 = Dice only).
    """
    # Upsample logits to match mask size
    logits_up = F.interpolate(
        logits, size=targets.shape[-2:], mode="bilinear", align_corners=False
    )
    # Weighted cross-entropy
    weight = torch.tensor([1.0, defect_weight], device=logits.device)
    ce = F.cross_entropy(logits_up, targets, weight=weight)
    # Dice
    dl = dice_loss(logits_up, targets)
    return (1.0 - dice_weight) * ce + dice_weight * dl

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, masks in loader:
        images = images.to(device)
        masks  = masks.to(device)

        # Use HuggingFace built-in loss — passes labels at native resolution
        # SegFormer internally downsamples labels to match logit size
        outputs = model(pixel_values=images, labels=masks)
        loss    = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    total_loss = 0.0
    all_miou   = []

    for images, masks in loader:
        images = images.to(device)
        masks  = masks.to(device)

        outputs = model(pixel_values=images, labels=masks)
        loss    = outputs.loss
        logits  = outputs.logits  # (B, num_classes, H/4, W/4)

        # Upsample logits to mask size
        upsampled = F.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        preds = upsampled.argmax(dim=1)  # (B, H, W)

        total_loss += loss.item()
        all_miou.append(compute_miou(preds.cpu(), masks.cpu(), num_classes))

    return total_loss / len(loader), float(np.mean(all_miou))


def train_subset(subset: str, data_root: Path, args):
    subset_dir = data_root / subset
    if not subset_dir.exists():
        print(f"\n⚠️  Skipping '{subset}' — folder not found at {subset_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Training on subset: {subset}")
    print(f"{'='*60}")

    # Dataset
    full_ds = FractographyDataset(
        subset_dir,
        split="all",
        image_size=IMAGE_SIZE,
    )
    if len(full_ds) == 0:
        print(f"  ⚠️  No image/mask pairs found in {subset_dir}")
        return

    n_train = max(1, int(len(full_ds) * TRAIN_FRAC))
    n_val   = len(full_ds) - n_train
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    device = torch.device("cpu")
    id2label = {0: "background", 1: "defect"}
    label2id = {v: k for k, v in id2label.items()}

    print(f"  Loading SegFormer-b0 from HuggingFace...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0",
        num_labels=NUM_CLASSES,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(device)

    optimizer = torch.optim.AdamW([
        {"params": model.segformer.parameters(), "lr": args.lr},
        {"params": model.decode_head.parameters(), "lr": args.lr * 50},
    ], weight_decay=WEIGHT_DECAY)  
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Checkpoint dir
    ckpt_dir = Path("checkpoints") / subset
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history   = {"train_loss": [], "val_loss": [], "val_miou": []}
    best_miou = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_miou = evaluate(model, val_loader, device, NUM_CLASSES)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_mIoU={val_miou:.4f} | "
            f"{elapsed:.1f}s"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_miou"].append(val_miou)

        # Save best
        if val_miou >= best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
            print(f"    ✅ New best mIoU: {best_miou:.4f} — checkpoint saved")

    # Save last + history
    torch.save(model.state_dict(), ckpt_dir / "last_model.pt")
    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  Done. Best val mIoU: {best_miou:.4f}")
    print(f"  Checkpoints saved to: {ckpt_dir.resolve()}")
    return history


def plot_histories(histories: dict):
    """Save a training curve plot for all subsets."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("SegFormer-b0 Training — OSF Ti-64", fontweight="bold")

        for subset, h in histories.items():
            epochs = range(1, len(h["train_loss"]) + 1)
            axes[0].plot(epochs, h["train_loss"], label=f"{subset} train")
            axes[0].plot(epochs, h["val_loss"],   label=f"{subset} val", linestyle="--")
            axes[1].plot(epochs, h["val_miou"],   label=subset)

        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend(fontsize=7)
        axes[1].set_title("Val mIoU")
        axes[1].set_xlabel("Epoch")
        axes[1].legend(fontsize=7)

        out = Path("checkpoints/training_curves.png")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"\n📈 Training curves saved to: {out.resolve()}")
    except Exception as e:
        print(f"  (Could not save plot: {e})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs",   type=int, default=EPOCHS)
    parser.add_argument("--lr",       type=float, default=LR)
    parser.add_argument("--image_size", type=int, default=256,
                        help="Square image size (256 recommended for CPU)")
    args = parser.parse_args()

    # Override IMAGE_SIZE from arg
    IMAGE_SIZE = (args.image_size, args.image_size)
    # Patch dataset module so it uses the right size
    import dataset as ds_module
    ds_module.IMAGE_SIZE = IMAGE_SIZE

    data_root = Path(args.data_dir)
    histories = {}

    for subset in SUBSETS:
        h = train_subset(subset, data_root, args)
        if h:
            histories[subset] = h

    if histories:
        plot_histories(histories)
        print("\n✅ All subsets complete.")
        print("\nSummary:")
        for subset, h in histories.items():
            best = max(h["val_miou"])
            print(f"  {subset:20s}  best mIoU = {best:.4f}")
