"""
inspect_dataset.py
------------------
Inspects the OSF Ti-64 SEM fractography dataset after downloading.
Run after download_osf.py.

What this does:
  1. Scans the data/ directory and reports what it finds
  2. Detects mask format (grayscale int labels vs RGB color masks)
  3. Prints unique class label values found in masks
  4. Generates a visualization grid of image/mask pairs
  5. Saves visualization to output/inspection_grid.png

Usage:
    python inspect_dataset.py
    python inspect_dataset.py --data_dir path/to/your/data
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe; switch to "TkAgg" if you want interactive
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image


# ── Configurable label map ────────────────────────────────────────────────────
# Update this once you've inspected the actual class values in your masks.
# Keys = integer pixel values in mask PNGs.
LABEL_MAP = {
    0: ("Background",        "#1a1a2e"),
    1: ("Lack of Fusion",    "#e94560"),
    2: ("Keyhole",           "#0f3460"),
    3: ("Other Defect",      "#533483"),
    # Add more if you find additional class values
}

# Fallback colormap for unknown labels
CMAP = plt.cm.get_cmap("tab10")
# ─────────────────────────────────────────────────────────────────────────────


def find_image_mask_pairs(data_dir: Path) -> list[tuple[Path, Path]]:
    """
    Scan data_dir for image/mask pairs.
    Assumes masks live in a folder named 'masks' or 'mask',
    and images in 'images' or 'image', or are paired by filename.
    """
    pairs = []
    image_exts = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}

    # Strategy 1: look for images/ and masks/ sibling folders
    for images_dir in sorted(data_dir.rglob("images")):
        if not images_dir.is_dir():
            continue
        masks_dir = images_dir.parent / "masks"
        if not masks_dir.exists():
            masks_dir = images_dir.parent / "mask"
        if not masks_dir.exists():
            print(f"  ⚠️  Found images/ at {images_dir} but no masks/ sibling")
            continue
        for img_path in sorted(images_dir.iterdir()):
            if img_path.suffix.lower() not in image_exts:
                continue
            # Try matching by stem
            for ext in image_exts:
                mask_path = masks_dir / (img_path.stem + ext)
                if mask_path.exists():
                    pairs.append((img_path, mask_path))
                    break
            else:
                print(f"  ⚠️  No mask found for {img_path.name}")

    # Strategy 2: flat folder — files named *_image.* and *_mask.*
    if not pairs:
        for img_path in sorted(data_dir.rglob("*_image.*")):
            if img_path.suffix.lower() not in image_exts:
                continue
            stem = img_path.stem.replace("_image", "")
            for ext in image_exts:
                mask_path = img_path.parent / f"{stem}_mask{ext}"
                if mask_path.exists():
                    pairs.append((img_path, mask_path))
                    break

    return pairs


def inspect_mask(mask_path: Path) -> dict:
    """Return statistics about a mask file."""
    mask = np.array(Image.open(mask_path))
    info = {
        "shape": mask.shape,
        "dtype": str(mask.dtype),
        "mode": Image.open(mask_path).mode,
        "unique_values": sorted(np.unique(mask).tolist()),
        "min": int(mask.min()),
        "max": int(mask.max()),
    }
    return info


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert integer label mask to RGB image for visualization."""
    unique = np.unique(mask)
    rgb = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
    for val in unique:
        if val in LABEL_MAP:
            hex_color = LABEL_MAP[val][1].lstrip("#")
            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            color = (r, g, b)
        else:
            # fallback: use matplotlib colormap
            rgba = CMAP(val / max(unique.max(), 1))
            color = tuple(int(c * 255) for c in rgba[:3])
        rgb[mask == val] = color
    return rgb


def make_legend(unique_vals: list[int]) -> list[mpatches.Patch]:
    patches = []
    for val in unique_vals:
        label, hex_color = LABEL_MAP.get(val, (f"Class {val}", "#888888"))
        patches.append(mpatches.Patch(color=hex_color, label=f"{val}: {label}"))
    return patches


def visualize_pairs(
    pairs: list[tuple[Path, Path]],
    n: int = 6,
    output_path: Path = Path("output/inspection_grid.png"),
):
    """Save a grid of n image/mask/overlay triplets."""
    n = min(n, len(pairs))
    if n == 0:
        print("  No pairs to visualize.")
        return

    fig, axes = plt.subplots(n, 3, figsize=(12, n * 4))
    if n == 1:
        axes = [axes]

    fig.suptitle("OSF Ti-64 SEM Dataset — Inspection Grid\n(Image | Mask | Overlay)",
                 fontsize=13, fontweight="bold", y=1.01)

    all_unique = set()

    for i, (img_path, mask_path) in enumerate(pairs[:n]):
        img = np.array(Image.open(img_path).convert("RGB"))
        mask_pil = Image.open(mask_path)
        mask_arr = np.array(mask_pil)

        # If mask is RGB, convert to grayscale for inspection
        if mask_arr.ndim == 3:
            mask_arr = np.array(mask_pil.convert("L"))

        unique_vals = sorted(np.unique(mask_arr).tolist())
        all_unique.update(unique_vals)
        mask_rgb = colorize_mask(mask_arr)

        # Overlay: blend image and mask
        overlay = (img.astype(float) * 0.5 + mask_rgb.astype(float) * 0.5).astype(np.uint8)

        axes[i][0].imshow(img, cmap="gray" if img.ndim == 2 else None)
        axes[i][0].set_title(f"Image\n{img_path.name}", fontsize=8)
        axes[i][0].axis("off")

        axes[i][1].imshow(mask_rgb)
        axes[i][1].set_title(
            f"Mask  (classes: {unique_vals})\n{mask_path.name}", fontsize=8
        )
        axes[i][1].axis("off")

        axes[i][2].imshow(overlay)
        axes[i][2].set_title("Overlay", fontsize=8)
        axes[i][2].axis("off")

    # Add legend
    legend_patches = make_legend(sorted(all_unique))
    fig.legend(handles=legend_patches, loc="lower center", ncol=len(legend_patches),
               bbox_to_anchor=(0.5, -0.02), fontsize=9, title="Mask Classes Found")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Visualization saved to: {output_path.resolve()}")


def print_dataset_summary(data_dir: Path, pairs: list[tuple[Path, Path]]):
    print(f"\n{'='*60}")
    print(f"Dataset Summary — {data_dir.resolve()}")
    print(f"{'='*60}")
    print(f"Total image/mask pairs found: {len(pairs)}")

    if not pairs:
        print("\n⚠️  No pairs found. Check your data/ folder structure.")
        print("Expected layout:")
        print("  data/")
        print("    <subset>/")
        print("      images/  ← SEM images (.png or .tif)")
        print("      masks/   ← segmentation masks (.png)")
        return

    # Sample first few masks
    print(f"\nSampling first 5 masks for format inspection:")
    all_unique = set()
    for img_path, mask_path in pairs[:5]:
        info = inspect_mask(mask_path)
        print(f"\n  {mask_path.name}")
        print(f"    Mode:          {info['mode']}")
        print(f"    Shape:         {info['shape']}")
        print(f"    Dtype:         {info['dtype']}")
        print(f"    Unique values: {info['unique_values']}")
        print(f"    Value range:   [{info['min']}, {info['max']}]")
        all_unique.update(info["unique_values"])

    print(f"\n{'─'*40}")
    print(f"All unique class values across sampled masks: {sorted(all_unique)}")
    print("\nLabel interpretation:")
    for v in sorted(all_unique):
        label, _ = LABEL_MAP.get(v, (f"UNKNOWN — update LABEL_MAP in this script", "#888"))
        print(f"  {v:3d} → {label}")

    print(f"\n⚠️  NOTE: If all unique values are {{0, 255}}, masks are binary (defect/no-defect).")
    print("     If values are 0–N, masks are multi-class integer labels — ideal for SegFormer.")
    print("     If mode is 'RGB', masks encode class as color — you'll need to remap.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Path to downloaded dataset root")
    parser.add_argument("--n_vis", type=int, default=6,
                        help="Number of pairs to visualize")
    parser.add_argument("--output", type=str, default="output/inspection_grid.png",
                        help="Where to save the visualization grid")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ data_dir '{data_dir}' does not exist.")
        print("Run download_osf.py first, or set --data_dir to your data folder.")
        sys.exit(1)

    print("Scanning for image/mask pairs...")
    pairs = find_image_mask_pairs(data_dir)

    print_dataset_summary(data_dir, pairs)

    if pairs:
        print(f"\nGenerating visualization grid ({min(args.n_vis, len(pairs))} samples)...")
        visualize_pairs(pairs, n=args.n_vis, output_path=Path(args.output))


if __name__ == "__main__":
    main()
