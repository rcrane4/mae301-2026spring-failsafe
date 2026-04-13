"""
dataset.py
----------
PyTorch Dataset class for the OSF Ti-64 SEM fractography dataset.
Use this after running inspect_dataset.py to confirm your mask format.

Key decisions you may need to make after inspection:
  - If masks are binary (0/255): set NUM_CLASSES=2, update MASK_SCALE
  - If masks are RGB color: set COLOR_MASK=True and define COLOR_TO_LABEL
  - If masks are integer labels (0..N): use as-is (ideal case)

Usage:
    from dataset import FractographyDataset
    ds = FractographyDataset("data/", split="train")
    img, mask = ds[0]
"""

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
import random


# ── Config — update after running inspect_dataset.py ────────────────────────
NUM_CLASSES = 2       # update once you know how many classes are in your masks
IMAGE_SIZE  = (512, 512)  # resize target; SegFormer-b0 default input
MASK_SCALE  = 255
# If masks use RGB color encoding instead of integer labels, set this to True
# and populate COLOR_TO_LABEL below.
COLOR_MASK = False
COLOR_TO_LABEL: dict[tuple, int] = {
    # (R, G, B): class_index
    # e.g. (255, 0, 0): 1,
}
# ─────────────────────────────────────────────────────────────────────────────


def rgb_mask_to_label(mask_rgb: np.ndarray, color_to_label: dict) -> np.ndarray:
    """Convert an H×W×3 RGB mask to an H×W integer label mask."""
    label = np.zeros(mask_rgb.shape[:2], dtype=np.int64)
    for color, cls_idx in color_to_label.items():
        match = np.all(mask_rgb == np.array(color), axis=-1)
        label[match] = cls_idx
    return label


class FractographyDataset(Dataset):
    """
    OSF Ti-64 SEM Fractography Dataset.

    Args:
        data_dir:   Root of downloaded data (contains subfolders with images/ + masks/).
        split:      "train", "val", or "all" (no splitting, returns everything).
        transform:  Optional callable applied to both image and mask (augmentation).
        image_size: Resize target (H, W).
    """

    IMAGE_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "all",
        transform: Optional[Callable] = None,
        image_size: tuple[int, int] = IMAGE_SIZE,
    ):
        self.data_dir   = Path(data_dir)
        self.split      = split
        self.transform  = transform
        self.image_size = image_size
        self.pairs      = self._find_pairs()

        if not self.pairs:
            raise FileNotFoundError(
                f"No image/mask pairs found in {self.data_dir}. "
                "Run inspect_dataset.py to diagnose."
            )
    def _find_pairs(self) -> list[tuple[Path, Path]]:
        pairs = []
        for images_dir in sorted(self.data_dir.rglob("images_8bit")):
            if not images_dir.is_dir():
                continue
            masks_dir = images_dir.parent / "masks_8bit"
            if not masks_dir.exists():
                continue
            for img_path in sorted(images_dir.iterdir()):
                if img_path.suffix.lower() not in self.IMAGE_EXTS:
                    continue
                mask_path = masks_dir / img_path.name
                if mask_path.exists():
                    pairs.append((img_path, mask_path))
        return pairs
    def _load_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        arr  = (arr - mean) / std
        return torch.from_numpy(arr).permute(2, 0, 1).float()
    def _load_mask(self, path: Path) -> torch.Tensor:
        mask_pil = Image.open(path)

        if COLOR_MASK:
            mask_arr = np.array(mask_pil.convert("RGB"))
            mask_arr = rgb_mask_to_label(mask_arr, COLOR_TO_LABEL)
        else:
            mask_arr = np.array(mask_pil.convert("L"), dtype=np.int64)
            if MASK_SCALE > 1:
                mask_arr = mask_arr // MASK_SCALE  # e.g. 0/255 → 0/1

        mask_pil_resized = Image.fromarray(mask_arr.astype(np.uint8)).resize(
            (self.image_size[1], self.image_size[0]), Image.NEAREST  # NEAREST preserves labels
        )
        mask_arr = np.array(mask_pil_resized, dtype=np.int64)
        return torch.from_numpy(mask_arr).long()  # H×W

    def _augment(self, image: torch.Tensor, mask: torch.Tensor):
        """Shared spatial augmentations (applied identically to image and mask)."""
        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask.unsqueeze(0)).squeeze(0)

        # Random vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask  = TF.vflip(mask.unsqueeze(0)).squeeze(0)

        # Random 90° rotation
        k = random.choice([0, 1, 2, 3])
        if k:
            image = torch.rot90(image, k, dims=[1, 2])
            mask  = torch.rot90(mask, k, dims=[0, 1])

        return image, mask

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.pairs[idx]
        image = self._load_image(img_path)
        mask  = self._load_mask(mask_path)

        if self.split == "train" and self.transform is None:
            image, mask = self._augment(image, mask)
        elif self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask

    def __repr__(self) -> str:
        return (
            f"FractographyDataset("
            f"n={len(self)}, split='{self.split}', "
            f"image_size={self.image_size}, classes={NUM_CLASSES})"
        )


def get_dataloaders(
    data_dir: str | Path,
    batch_size: int = 4,
    train_frac: float = 0.8,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader) with 80/20 split.
    """
    full_dataset = FractographyDataset(data_dir, split="all")
    n_train = int(len(full_dataset) * train_frac)
    n_val   = len(full_dataset) - n_train
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

    # Override split tag so augmentation fires for train only
    train_ds.dataset.split = "train"

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")
    return train_loader, val_loader


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"

    try:
        ds = FractographyDataset(data_dir)
        print(ds)
        img, mask = ds[0]
        print(f"Image tensor: {img.shape}  dtype={img.dtype}  range=[{img.min():.2f}, {img.max():.2f}]")
        print(f"Mask tensor:  {mask.shape}  dtype={mask.dtype}  unique={mask.unique().tolist()}")
        print("\n✅ Dataset loads correctly.")
    except FileNotFoundError as e:
        print(f"❌ {e}")
