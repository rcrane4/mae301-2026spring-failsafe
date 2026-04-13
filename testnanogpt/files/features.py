"""
features.py
-----------
Week 3: Feature extraction + defect type classification.

Takes a trained SegFormer checkpoint, runs inference on an image,
and extracts quantitative morphological features from the predicted mask.
These features feed into:
  1. A rule-based defect classifier (lack_of_fusion vs keyhole vs clean)
  2. A structured feature dict consumed by the generative reasoning layer (Week 4)

Extracted features:
  - defect_area_fraction    : % of image that is defect
  - defect_count            : number of distinct defect regions
  - mean_pore_area          : mean area of individual defect blobs (px²)
  - max_pore_area           : largest single defect region
  - mean_aspect_ratio       : mean of (major_axis / minor_axis) per blob
                              → circular pores ≈ 1.0 (keyhole)
                              → elongated pores > 2.0 (lack of fusion)
  - spatial_concentration   : std of defect centroid positions (spread)
  - size_std                : std of pore areas (heterogeneity)
  - quadrant_distribution   : defect fraction per image quadrant

Usage:
    python features.py --image data/all_defects/images/001-Overview-EP04V24.png
                       --subset all_defects

    python features.py --subset all_defects --all   # run on all images in subset
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation

from dataset import FractographyDataset, IMAGE_SIZE, NUM_CLASSES, MASK_SCALE

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cpu")

# Rule-based classification thresholds (tunable)
# Lack of fusion: many small irregular pores, high aspect ratio
# Keyhole: fewer larger circular pores, low aspect ratio
THRESHOLDS = {
    "min_defect_fraction_to_classify": 0.002,
    "keyhole_max_aspect_ratio":        1.6,   # wider keyhole band
    "lof_min_count":                   20,    # need many blobs for LoF
}
# ─────────────────────────────────────────────────────────────────────────────
def load_model(checkpoint_path: Path) -> SegformerForSemanticSegmentation:
    from transformers import SegformerConfig

    config = SegformerConfig.from_pretrained("nvidia/mit-b0")
    config.num_labels = NUM_CLASSES
    config.id2label = {0: "background", 1: "defect"}
    config.label2id = {"background": 0, "defect": 1}

    model = SegformerForSemanticSegmentation(config)

    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    result = model.load_state_dict(state, strict=True)
    model.eval()
    return model
def load_image_tensor(path: Path, image_size: tuple) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = img.resize((image_size[1], image_size[0]), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    arr  = (arr - mean) / std
    return torch.from_numpy(arr).permute(2, 0, 1).float()
@torch.no_grad()
def predict_mask(model, image_tensor: torch.Tensor, target_size: tuple) -> np.ndarray:
    outputs = model(pixel_values=image_tensor.unsqueeze(0))
    logits  = outputs.logits
    upsampled = F.interpolate(
        logits, size=target_size, mode="bilinear", align_corners=False
    )
    pred = upsampled.squeeze(0).argmax(dim=0).numpy()
    return pred.astype(np.uint8)

def connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Simple flood-fill connected components (no scipy dependency).
    Returns (labeled_mask, num_components).
    """
    h, w   = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current_label = 0

    def neighbors(r, c):
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w:
                yield nr, nc

    for r in range(h):
        for c in range(w):
            if mask[r, c] == 1 and labels[r, c] == 0:
                current_label += 1
                stack = [(r, c)]
                labels[r, c] = current_label
                while stack:
                    cr, cc = stack.pop()
                    for nr, nc in neighbors(cr, cc):
                        if mask[nr, nc] == 1 and labels[nr, nc] == 0:
                            labels[nr, nc] = current_label
                            stack.append((nr, nc))

    return labels, current_label


def blob_properties(labels: np.ndarray, num_blobs: int) -> list[dict]:
    """Compute area, centroid, and aspect ratio for each labeled blob."""
    props = []
    for label_id in range(1, num_blobs + 1):
        ys, xs = np.where(labels == label_id)
        if len(ys) == 0:
            continue
        area = len(ys)
        cy, cx = ys.mean(), xs.mean()

        # Bounding box aspect ratio as proxy for shape
        h_bbox = ys.max() - ys.min() + 1
        w_bbox = xs.max() - xs.min() + 1
        major  = max(h_bbox, w_bbox)
        minor  = min(h_bbox, w_bbox)
        aspect_ratio = major / minor if minor > 0 else 1.0

        props.append({
            "area": area,
            "centroid": (float(cy), float(cx)),
            "aspect_ratio": float(aspect_ratio),
            "bbox": (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())),
        })
    return props


def extract_features(mask: np.ndarray, image_size: tuple) -> dict:
    """Extract quantitative morphological features from a binary prediction mask."""
    H, W        = image_size
    total_px    = H * W
    defect_px   = int((mask == 1).sum())
    defect_frac = defect_px / total_px

    if defect_px == 0:
        return {
            "defect_area_fraction":  0.0,
            "defect_count":          0,
            "mean_pore_area_px":     0.0,
            "max_pore_area_px":      0,
            "mean_aspect_ratio":     0.0,
            "spatial_concentration": 0.0,
            "size_std":              0.0,
            "quadrant_distribution": [0.0, 0.0, 0.0, 0.0],
            "defect_type":           "clean",
            "confidence":            "high",
        }

    # Connected components (note: slow for large masks — acceptable at 256×256)
    labels, n_blobs = connected_components(mask)
    props = blob_properties(labels, n_blobs)

    areas          = [p["area"] for p in props]
    aspect_ratios  = [p["aspect_ratio"] for p in props]
    centroids      = [p["centroid"] for p in props]

    mean_area    = float(np.mean(areas))   if areas else 0.0
    max_area     = int(max(areas))         if areas else 0
    mean_ar      = float(np.mean(aspect_ratios)) if aspect_ratios else 0.0
    size_std     = float(np.std(areas))    if areas else 0.0

    # Spatial concentration: std of centroid distances from image center
    if centroids:
        cy_center, cx_center = H / 2, W / 2
        dists = [math.sqrt((c[0]-cy_center)**2 + (c[1]-cx_center)**2)
                 for c in centroids]
        spatial_conc = float(np.std(dists))
    else:
        spatial_conc = 0.0

    # Quadrant distribution
    half_h, half_w = H // 2, W // 2
    quads = [
        float((mask[:half_h, :half_w] == 1).sum()),   # top-left
        float((mask[:half_h, half_w:] == 1).sum()),   # top-right
        float((mask[half_h:, :half_w] == 1).sum()),   # bottom-left
        float((mask[half_h:, half_w:] == 1).sum()),   # bottom-right
    ]
    total_defect = sum(quads) + 1e-8
    quad_dist = [q / total_defect for q in quads]

    # ── Rule-based classification ─────────────────────────────────────────────
    defect_type, confidence = classify_defect(defect_frac, n_blobs, mean_ar, mean_area)

    return {
        "defect_area_fraction":  round(defect_frac * 100, 3),  # as %
        "defect_count":          n_blobs,
        "mean_pore_area_px":     round(mean_area, 1),
        "max_pore_area_px":      max_area,
        "mean_aspect_ratio":     round(mean_ar, 3),
        "spatial_concentration": round(spatial_conc, 2),
        "size_std":              round(size_std, 1),
        "quadrant_distribution": [round(q, 3) for q in quad_dist],
        "defect_type":           defect_type,
        "confidence":            confidence,
    }


def classify_defect(
    defect_frac: float,
    count: int,
    mean_ar: float,
    mean_area: float,
) -> tuple[str, str]:
    """
    Rule-based defect classifier.
    Returns (defect_type, confidence).

    Lack of fusion:  many small irregular pores, higher aspect ratio
    Keyhole:         fewer larger circular pores, lower aspect ratio
    Mixed:           both morphologies present
    Clean:           below detection threshold
    """
    t = THRESHOLDS
    if defect_frac < t["min_defect_fraction_to_classify"]:
        return "clean", "high"

    is_circular  = mean_ar <= t["keyhole_max_aspect_ratio"]
    is_many      = count   >= t["lof_min_count"]

    if is_circular and not is_many:
        return "keyhole_porosity", "high"
    elif not is_circular and is_many:
        return "lack_of_fusion", "high"
    elif is_circular and is_many:
        return "mixed", "medium"
    else:
        return "lack_of_fusion", "medium"


def visualize_features(
    image_path: Path,
    mask: np.ndarray,
    features: dict,
    out_path: Path,
):
    """Save a single-image feature visualization."""
    raw = np.array(Image.open(image_path), dtype=np.float32)
    raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
    raw_resized = np.array(
        Image.fromarray((raw * 255).astype(np.uint8)).resize(
            (IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR
        )
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Feature Extraction — {image_path.name}\n"
        f"Defect Type: {features['defect_type'].upper()}  "
        f"(confidence: {features['confidence']})",
        fontsize=11, fontweight="bold"
    )

    # Image
    axes[0].imshow(raw_resized, cmap="gray")
    axes[0].set_title("SEM Image", fontsize=9)
    axes[0].axis("off")

    # Mask with blob labels
    overlay = np.stack([raw_resized, raw_resized, raw_resized], axis=-1).copy()
    overlay[mask == 1] = [0, 212, 255]  # cyan defects
    axes[1].imshow(overlay)
    axes[1].set_title(
        f"Prediction\n{features['defect_area_fraction']:.2f}% defect  |  "
        f"{features['defect_count']} blobs",
        fontsize=9
    )
    axes[1].axis("off")

    # Feature summary text
    axes[2].axis("off")
    feature_text = (
        f"Defect Area:       {features['defect_area_fraction']:.3f}%\n"
        f"Defect Count:      {features['defect_count']}\n"
        f"Mean Pore Area:    {features['mean_pore_area_px']:.1f} px²\n"
        f"Max Pore Area:     {features['max_pore_area_px']} px²\n"
        f"Mean Aspect Ratio: {features['mean_aspect_ratio']:.3f}\n"
        f"  (1.0=circle, >2=elongated)\n"
        f"Spatial Spread:    {features['spatial_concentration']:.2f}\n"
        f"Size Std Dev:      {features['size_std']:.1f}\n\n"
        f"Quadrant Distribution:\n"
        f"  TL:{features['quadrant_distribution'][0]:.2f}  "
        f"TR:{features['quadrant_distribution'][1]:.2f}\n"
        f"  BL:{features['quadrant_distribution'][2]:.2f}  "
        f"BR:{features['quadrant_distribution'][3]:.2f}\n\n"
        f"─────────────────────────\n"
        f"DEFECT TYPE:  {features['defect_type']}\n"
        f"CONFIDENCE:   {features['confidence']}"
    )
    axes[2].text(
        0.05, 0.95, feature_text,
        transform=axes[2].transAxes,
        fontsize=9, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#1a1a2e", alpha=0.8, edgecolor="#00d4ff"),
        color="white"
    )
    axes[2].set_title("Extracted Features", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path.resolve()}")


def run_on_image(image_path: Path, subset: str) -> dict:
    ckpt_path = Path("checkpoints") / subset / "best_model.pt"
    if not ckpt_path.exists():
        print(f"❌ No checkpoint at {ckpt_path}")
        return {}

    print(f"\nImage:    {image_path.name}")
    print(f"Subset:   {subset}")

    model      = load_model(ckpt_path)
    img_tensor = load_image_tensor(image_path, IMAGE_SIZE)
    mask       = predict_mask(model, img_tensor, IMAGE_SIZE)
    features   = extract_features(mask, IMAGE_SIZE)

    print(f"Defect type:   {features['defect_type']}  ({features['confidence']} confidence)")
    print(f"Defect area:   {features['defect_area_fraction']:.3f}%")
    print(f"Blob count:    {features['defect_count']}")
    print(f"Mean AR:       {features['mean_aspect_ratio']:.3f}")
    print(json.dumps(features, indent=2))

    out_path = Path("output/features") / f"{image_path.stem}_features.png"
    visualize_features(image_path, mask, features, out_path)

    return features


def run_on_subset(subset: str, data_dir: Path, n: int = 6):
    """Run feature extraction on n images from a subset and print summary."""
    subset_dir = data_dir / subset
    if not subset_dir.exists():
        print(f"⚠️  {subset_dir} not found")
        return

    ds = FractographyDataset(subset_dir, split="all", image_size=IMAGE_SIZE)
    ckpt_path = Path("checkpoints") / subset / "best_model.pt"
    if not ckpt_path.exists():
        print(f"⚠️  No checkpoint for {subset}")
        return

    model = load_model(ckpt_path)
    results = []

    print(f"\n{'='*60}")
    print(f"Feature extraction: {subset} ({min(n, len(ds))} images)")
    print(f"{'='*60}")

    for idx in range(min(n, len(ds))):
        img_path, _ = ds.pairs[idx]
        img_tensor  = load_image_tensor(img_path, IMAGE_SIZE)
        mask        = predict_mask(model, img_tensor, IMAGE_SIZE)
        features    = extract_features(mask, IMAGE_SIZE)
        features["image"] = img_path.name
        results.append(features)

        out_path = Path("output/features") / subset / f"{img_path.stem}_features.png"
        visualize_features(img_path, mask, features, out_path)

    # Summary
    print(f"\n  Classification summary:")
    from collections import Counter
    counts = Counter(r["defect_type"] for r in results)
    for dtype, count in counts.items():
        print(f"    {dtype:25s}: {count}")

    # Save results JSON
    json_out = Path("output/features") / f"{subset}_features.json"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    with open(json_out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Feature JSON → {json_out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",    type=str, default=None,
                        help="Path to a single SEM image")
    parser.add_argument("--subset",   type=str, default="all_defects",
                        help="lack_of_fusion | keyhole | all_defects")
    parser.add_argument("--all",      action="store_true",
                        help="Run on all images in subset (up to --n)")
    parser.add_argument("--n",        type=int, default=6,
                        help="Number of images to process in --all mode")
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()

    if args.image:
        run_on_image(Path(args.image), args.subset)
    else:
        subsets = (
            ["lack_of_fusion", "keyhole", "all_defects"]
            if args.subset == "all"
            else [args.subset]
        )
        for subset in subsets:
            run_on_subset(subset, Path(args.data_dir), n=args.n)
