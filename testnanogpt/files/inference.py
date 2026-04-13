"""
inference.py
------------
Loads a trained SegFormer checkpoint and runs inference on SEM images.
Saves a visualization grid showing: original image | predicted mask | overlay.

Usage:
    # Run on a specific subset's val images
    python inference.py --subset lack_of_fusion

    # Run on a specific image
    python inference.py --image path/to/image.png --subset keyhole

    # Run all three subsets
    python inference.py --subset all
"""

import argparse
import random
from pathlib import Path
from features import load_model, load_image_tensor


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation

from dataset import FractographyDataset, IMAGE_SIZE, NUM_CLASSES, MASK_SCALE

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE    = torch.device("cpu")
N_SAMPLES = 6   # images to visualize per subset
LABEL_MAP = {0: ("Background", "#1a1a2e"), 1: ("Defect", "#00d4ff")}
# ─────────────────────────────────────────────────────────────────────────────


def load_model(checkpoint_path: Path) -> SegformerForSemanticSegmentation:
    id2label = {0: "background", 1: "defect"}
    label2id = {v: k for k, v in id2label.items()}
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0",
        num_labels=NUM_CLASSES,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def load_raw_image(path: Path) -> np.ndarray:
    """Load SEM image as a displayable uint8 RGB array (handles 16-bit)."""
    arr = np.array(Image.open(path), dtype=np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    rgb = np.stack([arr, arr, arr], axis=-1)
    return (rgb * 255).astype(np.uint8)


def predict(model, image_tensor: torch.Tensor, target_size: tuple) -> np.ndarray:
    """Run inference and return (H, W) prediction mask as numpy array."""
    with torch.no_grad():
        outputs = model(pixel_values=image_tensor.unsqueeze(0))
        logits  = outputs.logits  # (1, C, H/4, W/4)
        upsampled = F.interpolate(
            logits, size=target_size, mode="bilinear", align_corners=False
        )
        pred = upsampled.squeeze(0).argmax(dim=0).numpy()  # (H, W)
    return pred


def colorize(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for val, (_, hex_color) in LABEL_MAP.items():
        r, g, b = tuple(int(hex_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        rgb[mask == val] = (r, g, b)
    return rgb


def compute_stats(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Compute per-image IoU and defect coverage."""
    pred_defect = pred == 1
    gt_defect   = gt == 1
    intersection = (pred_defect & gt_defect).sum()
    union        = (pred_defect | gt_defect).sum()
    iou          = intersection / union if union > 0 else float("nan")
    coverage_pred = pred_defect.sum() / pred.size * 100
    coverage_gt   = gt_defect.sum()   / gt.size   * 100
    return {"iou": iou, "pred_coverage": coverage_pred, "gt_coverage": coverage_gt}


def run_inference(subset: str, args):
    data_dir  = Path(args.data_dir) / subset
    ckpt_path = Path("checkpoints") / subset / "best_model.pt"
    out_dir   = Path("output") / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print(f"⚠️  Skipping '{subset}' — data not found at {data_dir}")
        return
    if not ckpt_path.exists():
        print(f"⚠️  Skipping '{subset}' — no checkpoint at {ckpt_path}")
        return

    print(f"\n{'='*60}")
    print(f"Inference: {subset}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"{'='*60}")

    model = load_model(ckpt_path)

    # Load dataset to get image/mask pairs
    ds = FractographyDataset(data_dir, split="all", image_size=IMAGE_SIZE)
    indices = list(range(len(ds)))
    random.seed(42)
    random.shuffle(indices)
    sample_indices = indices[:N_SAMPLES]

    # Build figure
    n = len(sample_indices)
    fig, axes = plt.subplots(n, 4, figsize=(16, n * 4))
    if n == 1:
        axes = [axes]
    fig.suptitle(
        f"SegFormer Inference — {subset.replace('_', ' ').title()}",
        fontsize=13, fontweight="bold"
    )

    ious = []
    for row, idx in enumerate(sample_indices):
        img_path, mask_path = ds.pairs[idx]
        img_tensor, gt_mask = ds[idx]

        # Raw image for display (16-bit safe)
        raw_img = load_raw_image(img_path)

        # GT mask (undo MASK_SCALE)
        gt_arr = gt_mask.numpy()  # already scaled by dataset

        # Predict
        pred = predict(model, img_tensor, target_size=IMAGE_SIZE)

        # Resize raw image to match prediction size for display
        raw_resized = np.array(
            Image.fromarray(raw_img).resize(
                (IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR
            )
        )

        # Stats
        stats = compute_stats(pred, gt_arr)
        ious.append(stats["iou"])

        # Colorize
        pred_colored = colorize(pred)
        gt_colored   = colorize(gt_arr)
        overlay = (raw_resized.astype(float) * 0.6 +
                   pred_colored.astype(float) * 0.4).astype(np.uint8)

        # Plot
        axes[row][0].imshow(raw_resized, cmap="gray")
        axes[row][0].set_title(f"Image\n{img_path.name}", fontsize=7)
        axes[row][0].axis("off")

        axes[row][1].imshow(gt_colored)
        axes[row][1].set_title(
            f"Ground Truth\n{stats['gt_coverage']:.1f}% defect", fontsize=7
        )
        axes[row][1].axis("off")

        axes[row][2].imshow(pred_colored)
        axes[row][2].set_title(
            f"Prediction\n{stats['pred_coverage']:.1f}% defect", fontsize=7
        )
        axes[row][2].axis("off")

        axes[row][3].imshow(overlay)
        iou_str = f"{stats['iou']:.3f}" if not np.isnan(stats["iou"]) else "N/A"
        axes[row][3].set_title(f"Overlay\nIoU={iou_str}", fontsize=7)
        axes[row][3].axis("off")

    # Legend
    patches = [
        mpatches.Patch(color=LABEL_MAP[0][1], label="Background"),
        mpatches.Patch(color=LABEL_MAP[1][1], label="Defect"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.01), fontsize=9)

    mean_iou = np.nanmean(ious)
    fig.text(0.5, -0.03, f"Mean IoU (these samples): {mean_iou:.4f}",
             ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out_path = out_dir / f"{subset}_inference.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Mean IoU (sampled): {mean_iou:.4f}")
    print(f"  Saved → {out_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, default="all",
                        help="lack_of_fusion | keyhole | all_defects | all")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--n", type=int, default=6,
                        help="Number of images to visualize")
    args = parser.parse_args()

    N_SAMPLES = args.n

    subsets = (
        ["lack_of_fusion", "keyhole", "all_defects"]
        if args.subset == "all"
        else [args.subset]
    )

    for subset in subsets:
        run_inference(subset, args)

    print("\n✅ Done. Check output/inference/")
