Objective and current MVP definition, What has been built so far, and technical approach: Product that takes in SEM images of Ti64, an alloy used in aircraft engines and race car components, where safety is critical and failure can be catastrophic, so identifying defects in manufacturing is extremely important.
The current MVP makes it so you can upload an SEM image of TI64 titanium via SegFormer, trained on a Ti64 dataset
of images and defect masks, and it will identify the defects and provide information and advice through a Claude server: 

https://huggingface.co/spaces/rcrane4/FailSafe

Current limitations: The dataset we trained the model on was originally 16bit, but the website (Gradio) we are using for an interface can only take in 8bit, so we had to convert the 16bit dataset to 8bit. Therefore the model is trained on 8bit images, so images must be converted to 8bit by the user before being uploaded for it to work.

Plan for phase 3: Fix the 8bit image issue, and make further optimizations to the current MVP for a more intuitive UI and more precise model training.

We have also provided all files used to train the model and the image dataset used, uploaded to the GitHub in /testnanogpt/
..
data
Create readme
2 weeks ago
app.py
Add files via upload
2 weeks ago
dataset.py
Add files via upload
2 weeks ago
diagnose.py
Add files via upload
2 weeks ago
download_osf.py
Add files via upload
2 weeks ago
features.py
Add files via upload
2 weeks ago
inference.py
Add files via upload
2 weeks ago
inspect_dataset.py
Add files via upload
2 weeks ago
setup.py
Add files via upload
2 weeks ago
test.py
Add files via upload
2 weeks ago
train.py
Add files via upload
app.py: """
app.py
------
FailureGPT — Gradio web interface.
Drag-and-drop SEM image → segmentation → features → AI diagnosis.
Usage:
pip install gradio
python app.py
Then open http://127.0.0.1:7860 in your browser.
"""
import json
import os
from pathlib import Path
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from dataset import IMAGE_SIZE, NUM_CLASSES
from features import (
load_model, load_image_tensor, predict_mask,
extract_features,
)
from diagnose import call_claude, format_diagnosis_report
# ── Load all three models at startup ─────────────────────────────────────────
SUBSETS = ["all_defects", "lack_of_fusion", "keyhole"]
MODELS = {}
print("Loading checkpoints...")
for subset in SUBSETS:
ckpt = Path("checkpoints") / subset / "best_model.pt"
if ckpt.exists():
MODELS[subset] = load_model(ckpt)
print(f" ✅ {subset}")
else:
print(f" ⚠️ {subset} — checkpoint not found")
# ─────────────────────────────────────────────────────────────────────────────
RISK_COLORS = {
"low": "#2ecc71",
"medium": "#f39c12",
"high": "#e74c3c",
"critical": "#8e44ad",
}
def run_pipeline(image: np.ndarray, subset: str) -> tuple:
if image is None:
return None, "No image provided.", "No image provided.", "—"
if subset not in MODELS:
return None, f"No checkpoint for '{subset}'.", "Train the model first.", "—"
model = MODELS[subset]
# Gradio gives H×W×3 uint8
arr = image.astype(np.float32)
if arr.ndim == 2:
arr = np.stack([arr]*3, axis=-1)
elif arr.shape[2] == 4:
arr = arr[:, :, :3]
# Normalize to [0,1]
arr_min, arr_max = arr.min(), arr.max()
arr_norm = (arr - arr_min) / (arr_max - arr_min + 1e-8) if arr_max > arr_min else arr / 255.0
# Build display copy
display_pil = Image.fromarray(
(arr_norm * 255).astype(np.uint8), mode="RGB"
).resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR)
display_arr = np.array(display_pil, dtype=np.uint8)
# ImageNet normalization for model
arr_model = np.array(display_pil, dtype=np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
arr_model = (arr_model - mean) / std
img_tensor = torch.from_numpy(arr_model).permute(2, 0, 1).float()
print(f"DEBUG display_arr: min={display_arr.min()} max={display_arr.max()}")
# ── Step 2: Segment ───────────────────────────────────────────────────────
mask = predict_mask(model, img_tensor, IMAGE_SIZE)
print(f"DEBUG mask: unique={np.unique(mask).tolist()} defect_px={( mask==1).sum()}")
# ── Step 3: Extract features ──────────────────────────────────────────────
features = extract_features(mask, IMAGE_SIZE)
# ── Step 4: Build overlay ─────────────────────────────────────────────────
# Replace the overlay build block with:
overlay = display_arr.copy()
# First apply cyan to defect pixels at full intensity
defect_mask = mask == 1
overlay[defect_mask] = [0, 212, 255]
# Blend only the background pixels, keep defects fully cyan
result = display_arr.copy()
result[~defect_mask] = display_arr[~defect_mask] # background unchanged
result[defect_mask] = (
display_arr[defect_mask].astype(float) * 0.3 +
np.array([0, 212, 255], dtype=float) * 0.7
).clip(0, 255).astype(np.uint8)
overlay = result
from PIL import Image as PILImage
PILImage.fromarray(overlay).save("output/debug_overlay.png")
print(f"DEBUG saved overlay to output/debug_overlay.png")
# ── Step 5: Format features text ──────────────────────────────────────────
feat_lines = [
f"Defect Area: {features['defect_area_fraction']:.3f}%",
f"Defect Count: {features['defect_count']} blobs",
f"Mean Pore Area: {features.get('mean_pore_area_px', 0):.1f} px²",
f"Max Pore Area: {features.get('max_pore_area_px', 0)} px²",
f"Mean Aspect Ratio: {features['mean_aspect_ratio']:.3f}",
f" (1.0=circular · >2.0=elongated)",
f"Spatial Spread: {features['spatial_concentration']:.2f}",
f"Size Std Dev: {features['size_std']:.1f}",
f"",
f"Quadrant Distribution:",
f" TL {features['quadrant_distribution'][0]:.2f} "
f"TR {features['quadrant_distribution'][1]:.2f}",
f" BL {features['quadrant_distribution'][2]:.2f} "
f"BR {features['quadrant_distribution'][3]:.2f}",
f"",
f"Rule-based type: {features['defect_type']}",
f"Confidence: {features['confidence']}",
]
features_text = "\n".join(feat_lines)
# ── Step 6: AI Diagnosis ──────────────────────────────────────────────────
if not os.environ.get("ANTHROPIC_API_KEY"):
diagnosis_text = (
"⚠️ ANTHROPIC_API_KEY not set.\n\n"
"Set it in your terminal:\n"
" $env:ANTHROPIC_API_KEY = 'sk-ant-...'\n\n"
"Features extracted successfully:\n\n"
+ features_text
)
risk_label = features["defect_type"].upper()
else:
diagnosis = call_claude(features, "uploaded_image")
diagnosis_text = format_diagnosis_report(features, diagnosis, "uploaded_image")
risk = diagnosis.get("crack_initiation_risk", "unknown")
mech = diagnosis.get("dominant_failure_mechanism", "unknown")
risk_label = f"{risk.upper()} RISK — {mech}"
# Ensure output is exactly what Gradio expects
overlay = overlay.astype(np.uint8)
assert overlay.ndim == 3 and overlay.shape[2] == 3
print(f"DEBUG overlay: shape={overlay.shape} dtype={overlay.dtype} min={overlay.min()} max={overlay.max()}")
return overlay, features_text, diagnosis_text, risk_label
# ── Gradio UI ─────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
body, .gradio-container {
background: #080c14 !important;
font-family: 'DM Sans', sans-serif !important;
color: #c8d6e5 !important;
}
.gradio-container {
max-width: 1400px !important;
margin: 0 auto !important;
}
/* Header */
#header {
text-align: center;
padding: 2rem 0 1rem;
border-bottom: 1px solid #1e3a5f;
margin-bottom: 1.5rem;
}
#header h1 {
font-family: 'Space Mono', monospace !important;
font-size: 2.4rem !important;
font-weight: 700 !important;
color: #00d4ff !important;
letter-spacing: -1px;
margin: 0;
}
#header p {
color: #5a7a9a;
font-size: 0.9rem;
margin: 0.4rem 0 0;
font-family: 'Space Mono', monospace;
}
/* Risk badge */
#risk_label textarea, #risk_label input {
font-family: 'Space Mono', monospace !important;
font-size: 1.1rem !important;
font-weight: 700 !important;
color: #00d4ff !important;
background: #0d1825 !important;
border: 2px solid #00d4ff !important;
border-radius: 6px !important;
text-align: center !important;
padding: 0.6rem !important;
}
/* Textboxes */
textarea {
font-family: 'Space Mono', monospace !important;
font-size: 0.78rem !important;
background: #0a1520 !important;
color: #a8c4dc !important;
border: 1px solid #1e3a5f !important;
border-radius: 6px !important;
line-height: 1.6 !important;
}
/* Labels */
label span {
font-family: 'Space Mono', monospace !important;
font-size: 0.72rem !important;
color: #4a7a9a !important;
letter-spacing: 1px !important;
text-transform: uppercase !important;
}
/* Buttons */
button.primary {
background: linear-gradient(135deg, #003d66, #006699) !important;
border: 1px solid #00d4ff !important;
color: #00d4ff !important;
font-family: 'Space Mono', monospace !important;
font-weight: 700 !important;
letter-spacing: 1px !important;
border-radius: 6px !important;
transition: all 0.2s !important;
}
button.primary:hover {
background: linear-gradient(135deg, #006699, #00aacc) !important;
box-shadow: 0 0 20px rgba(0, 212, 255, 0.3) !important;
}
button.secondary {
background: #0a1520 !important;
border: 1px solid #1e3a5f !important;
color: #5a7a9a !important;
font-family: 'Space Mono', monospace !important;
border-radius: 6px !important;
}
/* Dropdown */
select, .wrap {
background: #0a1520 !important;
border: 1px solid #1e3a5f !important;
color: #a8c4dc !important;
font-family: 'Space Mono', monospace !important;
}
/* Image panels */
.image-container {
border: 1px solid #1e3a5f !important;
border-radius: 8px !important;
overflow: hidden !important;
}
/* Panel blocks */
.block {
background: #0a1520 !important;
border: 1px solid #1e3a5f !important;
border-radius: 8px !important;
}
/* Footer note */
#footer {
text-align: center;
padding: 1rem 0;
color: #2a4a6a;
font-family: 'Space Mono', monospace;
font-size: 0.7rem;
border-top: 1px solid #1e3a5f;
margin-top: 1.5rem;
}
"""
with gr.Blocks(css=CSS, title="FailSafe") as demo:
gr.HTML("""
<div id="header">
<h1>⬡ FAILSAFE</h1>
<p>Ti-6Al-4V · LPBF Defect Analysis · SEM Fractography · Powered by SegFormer + Claude</p>
</div>
""")
with gr.Row():
# Left column — inputs
with gr.Column(scale=1):
image_input = gr.Image(
label="SEM FRACTOGRAPH — drag & drop or click to upload",
type="numpy",
height=500,
)
subset_input = gr.Dropdown(
choices=SUBSETS,
value="all_defects",
label="MODEL SUBSET",
)
with gr.Row():
run_btn = gr.Button("▶ ANALYZE", variant="primary", scale=3)
clear_btn = gr.Button("✕ CLEAR", variant="secondary", scale=1)
risk_output = gr.Textbox(
label="CRACK INITIATION RISK",
lines=1,
interactive=False,
elem_id="risk_label",
)
# Middle column — image output
with gr.Column(scale=1):
overlay_output = gr.Image(
label="DEFECT SEGMENTATION MAP",
height=500,
interactive=False,
)
features_output = gr.Textbox(
label="MORPHOLOGICAL FEATURES",
lines=14,
interactive=False,
)
# Right column — diagnosis
with gr.Column(scale=1):
diagnosis_output = gr.Textbox(
label="AI FAILURE DIAGNOSIS — Claude",
lines=28,
interactive=False,
)
gr.HTML("""
<div id="footer">
FailureGPT · ASU Mechanical Engineering · OSF Ti-64 Dataset ·
SegFormer-b0 fine-tuned · Claude Reasoning Layer
</div>
""")
# Wire up
run_btn.click(
fn=run_pipeline,
inputs=[image_input, subset_input],
outputs=[overlay_output, features_output, diagnosis_output, risk_output],
)
clear_btn.click(
fn=lambda: (None, None, "", "", ""),
outputs=[image_input, overlay_output, features_output, diagnosis_output, risk_output],
)
# Example images
example_images = list(Path("data/all_defects/images_8bit").glob("*.png"))[:3]
if example_images:
gr.Examples(
examples=[[str(p), "all_defects"] for p in example_images],
inputs=[image_input, subset_input],
label="EXAMPLE IMAGES",
)
if __name__ == "__main__":
demo.launch(
server_name="127.0.0.1",
server_port=7860,
share=False,
show_error=True,
)
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
NUM_CLASSES = 2 # update once you know how many classes are in your masks
IMAGE_SIZE = (512, 512) # resize target; SegFormer-b0 default input
MASK_SCALE = 255
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
data_dir: Root of downloaded data (contains subfolders with images/ + masks/).
split: "train", "val", or "all" (no splitting, returns everything).
transform: Optional callable applied to both image and mask (augmentation).
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
self.data_dir = Path(data_dir)
self.split = split
self.transform = transform
self.image_size = image_size
self.pairs = self._find_pairs()
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
std = np.array([0.229, 0.224, 0.225])
arr = (arr - mean) / std
return torch.from_numpy(arr).permute(2, 0, 1).float()
def _load_mask(self, path: Path) -> torch.Tensor:
mask_pil = Image.open(path)
if COLOR_MASK:
mask_arr = np.array(mask_pil.convert("RGB"))
mask_arr = rgb_mask_to_label(mask_arr, COLOR_TO_LABEL)
else:
mask_arr = np.array(mask_pil.convert("L"), dtype=np.int64)
if MASK_SCALE > 1:
mask_arr = mask_arr // MASK_SCALE # e.g. 0/255 → 0/1
mask_pil_resized = Image.fromarray(mask_arr.astype(np.uint8)).resize(
(self.image_size[1], self.image_size[0]), Image.NEAREST # NEAREST preserves labels
)
mask_arr = np.array(mask_pil_resized, dtype=np.int64)
return torch.from_numpy(mask_arr).long() # H×W
def _augment(self, image: torch.Tensor, mask: torch.Tensor):
"""Shared spatial augmentations (applied identically to image and mask)."""
# Random horizontal flip
if random.random() > 0.5:
image = TF.hflip(image)
mask = TF.hflip(mask.unsqueeze(0)).squeeze(0)
# Random vertical flip
if random.random() > 0.5:
image = TF.vflip(image)
mask = TF.vflip(mask.unsqueeze(0)).squeeze(0)
# Random 90° rotation
k = random.choice([0, 1, 2, 3])
if k:
image = torch.rot90(image, k, dims=[1, 2])
mask = torch.rot90(mask, k, dims=[0, 1])
return image, mask
def __len__(self) -> int:
return len(self.pairs)
def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
img_path, mask_path = self.pairs[idx]
image = self._load_image(img_path)
mask = self._load_mask(mask_path)
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
n_val = len(full_dataset) - n_train
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
print(f"Image tensor: {img.shape} dtype={img.dtype} range=[{img.min():.2f}, {img.max():.2f}]")
print(f"Mask tensor: {mask.shape} dtype={mask.dtype} unique={mask.unique().tolist()}")
print("\n✅ Dataset loads correctly.")
except FileNotFoundError as e:
print(f"❌ {e}")
"""
diagnose.py
-----------
Week 4: Generative reasoning layer.
Takes the feature dict output from features.py and calls the Claude API
to produce a structured engineering failure diagnosis.
The LLM receives:
- Quantitative morphological features from the segmentation
- Material context (Ti-6Al-4V, LPBF process)
- Defect type classification
And returns:
- Natural language diagnosis
- Crack initiation risk assessment
- Recommended follow-up actions
Usage:
# Single image full pipeline (segment → extract → diagnose)
python diagnose.py --image data/all_defects/images/001-Overview-EP04V24.png
--subset all_defects
# From existing feature JSON
python diagnose.py --json output/features/all_defects_features.json
# Interactive mode
python diagnose.py --interactive --subset all_defects
"""
import argparse
import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from transformers import SegformerForSemanticSegmentation
from dataset import FractographyDataset, IMAGE_SIZE, NUM_CLASSES
from features import (
load_model, load_image_tensor, predict_mask,
extract_features, visualize_features
)
# ── Anthropic API ─────────────────────────────────────────────────────────────
try:
import anthropic
HAS_ANTHROPIC = True
except ImportError:
HAS_ANTHROPIC = False
print("⚠️ anthropic package not found. Run: pip install anthropic")
# ─────────────────────────────────────────────────────────────────────────────
MATERIAL_CONTEXT = """
Material: Ti-6Al-4V (Grade 5 titanium alloy)
Process: Laser Powder Bed Fusion (LPBF) additive manufacturing
Application context: High-performance structural components (aerospace/defense)
Specimen type: Bend test bar, fractured in four-point bending
"""
SYSTEM_PROMPT = """You are an expert materials engineer specializing in fractography
and failure analysis of additively manufactured aerospace components.
You analyze quantitative defect features extracted from SEM (Scanning Electron Microscope)
images of Ti-6Al-4V fracture surfaces produced by Laser Powder Bed Fusion (LPBF).
Your role is to:
1. Interpret morphological defect features in the context of LPBF process physics
2. Assess crack initiation and propagation risk based on defect characteristics
3. Provide actionable engineering recommendations
4. Be precise and quantitative — reference the actual feature values in your diagnosis
Always structure your response as valid JSON with these exact keys:
{
"diagnosis_summary": "2-3 sentence plain English summary",
"defect_interpretation": "detailed interpretation of the morphological features",
"crack_initiation_risk": "low | medium | high | critical",
"risk_rationale": "why you assigned this risk level, referencing specific features",
"dominant_failure_mechanism": "e.g. lack of fusion porosity, keyhole porosity, mixed",
"critical_regions": "which quadrants or regions pose highest risk",
"recommendations": ["recommendation 1", "recommendation 2", "recommendation 3"],
"confidence": "low | medium | high",
"confidence_rationale": "why"
}
"""
def build_user_prompt(features: dict, image_name: str = "") -> str:
return f"""
Analyze the following defect features extracted from an SEM fractograph of a
Ti-6Al-4V LPBF test bar.
Material & Process Context:
{MATERIAL_CONTEXT}
Image: {image_name}
Extracted Morphological Features:
- Defect area fraction: {features.get('defect_area_fraction', 0):.3f}% of fracture surface
- Defect blob count: {features.get('defect_count', 0)} distinct pores/defects
- Mean pore area: {features.get('mean_pore_area_px', 0):.1f} px² (at 256×256 resolution)
- Max pore area: {features.get('max_pore_area_px', 0)} px²
- Mean aspect ratio: {features.get('mean_aspect_ratio', 0):.3f}
(1.0 = perfectly circular/keyhole, >2.0 = elongated/lack-of-fusion)
- Spatial spread (std): {features.get('spatial_concentration', 0):.2f} px
- Size heterogeneity: {features.get('size_std', 0):.1f} px² std dev
- Quadrant distribution:
Top-left: {features.get('quadrant_distribution', [0,0,0,0])[0]:.3f}
Top-right: {features.get('quadrant_distribution', [0,0,0,0])[1]:.3f}
Bottom-left: {features.get('quadrant_distribution', [0,0,0,0])[2]:.3f}
Bottom-right: {features.get('quadrant_distribution', [0,0,0,0])[3]:.3f}
- Rule-based defect type: {features.get('defect_type', 'unknown')}
(confidence: {features.get('confidence', 'unknown')})
Provide a structured engineering diagnosis as JSON.
"""
def call_claude(features: dict, image_name: str = "") -> dict:
"""Call Claude API and return parsed diagnosis dict."""
if not HAS_ANTHROPIC:
return {"error": "anthropic package not installed"}
client = anthropic.Anthropic() # uses ANTHROPIC_API_KEY env var
prompt = build_user_prompt(features, image_name)
try:
response = client.messages.create(
model="claude-sonnet-4-20250514",
max_tokens=1000,
system=SYSTEM_PROMPT,
messages=[{"role": "user", "content": prompt}]
)
raw_text = response.content[0].text.strip()
# Strip markdown code fences if present
if raw_text.startswith("```"):
raw_text = raw_text.split("```")[1]
if raw_text.startswith("json"):
raw_text = raw_text[4:]
raw_text = raw_text.strip()
diagnosis = json.loads(raw_text)
return diagnosis
except json.JSONDecodeError as e:
return {"error": f"JSON parse error: {e}", "raw": raw_text}
except Exception as e:
return {"error": str(e)}
def format_diagnosis_report(features: dict, diagnosis: dict, image_name: str = "") -> str:
"""Format a human-readable diagnosis report."""
sep = "=" * 60
lines = [
sep,
f"FAILURE ANALYSIS REPORT",
f"Image: {image_name}",
f"Material: Ti-6Al-4V (LPBF)",
sep,
"",
"QUANTITATIVE FEATURES",
f" Defect area: {features.get('defect_area_fraction', 0):.3f}%",
f" Defect count: {features.get('defect_count', 0)}",
f" Mean aspect ratio:{features.get('mean_aspect_ratio', 0):.3f}",
f" Rule-based type: {features.get('defect_type', 'unknown')}",
"",
]
if "error" in diagnosis:
lines += [f"⚠️ Diagnosis error: {diagnosis['error']}"]
return "\n".join(lines)
lines += [
"AI DIAGNOSIS",
f" Failure mechanism: {diagnosis.get('dominant_failure_mechanism', 'N/A')}",
f" Crack init. risk: {diagnosis.get('crack_initiation_risk', 'N/A').upper()}",
f" Critical regions: {diagnosis.get('critical_regions', 'N/A')}",
f" Confidence: {diagnosis.get('confidence', 'N/A')}",
"",
"SUMMARY",
f" {diagnosis.get('diagnosis_summary', '')}",
"",
"DEFECT INTERPRETATION",
f" {diagnosis.get('defect_interpretation', '')}",
"",
"RISK RATIONALE",
f" {diagnosis.get('risk_rationale', '')}",
"",
"RECOMMENDATIONS",
]
for i, rec in enumerate(diagnosis.get("recommendations", []), 1):
lines.append(f" {i}. {rec}")
lines.append(sep)
return "\n".join(lines)
def visualize_diagnosis(
image_path: Path,
mask: np.ndarray,
features: dict,
diagnosis: dict,
out_path: Path,
):
"""Save a full diagnosis visualization."""
raw = np.array(Image.open(image_path), dtype=np.float32)
raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
raw_resized = np.array(
Image.fromarray((raw * 255).astype(np.uint8)).resize(
(IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR
)
)
# Risk color
risk_colors = {
"low": "#2ecc71", "medium": "#f39c12",
"high": "#e74c3c", "critical": "#8e44ad"
}
risk = diagnosis.get("crack_initiation_risk", "medium")
risk_color = risk_colors.get(risk, "#888888")
fig = plt.figure(figsize=(18, 8))
fig.patch.set_facecolor("#0d0d1a")
# Title
mech = diagnosis.get("dominant_failure_mechanism", "Unknown")
fig.suptitle(
f"FailureGPT — {image_path.name}\n"
f"Mechanism: {mech} | Crack Risk: {risk.upper()}",
fontsize=12, fontweight="bold", color="white", y=1.01
)
# Image panel
ax1 = fig.add_subplot(1, 3, 1)
ax1.imshow(raw_resized, cmap="gray")
ax1.set_title("SEM Fractograph", color="white", fontsize=9)
ax1.axis("off")
ax1.set_facecolor("#0d0d1a")
# Segmentation overlay
ax2 = fig.add_subplot(1, 3, 2)
overlay = np.stack([raw_resized]*3, axis=-1).copy()
overlay[mask == 1] = [0, 212, 255]
ax2.imshow(overlay)
ax2.set_title(
f"Defect Map\n{features['defect_area_fraction']:.2f}% | "
f"{features['defect_count']} blobs | AR={features['mean_aspect_ratio']:.2f}",
color="white", fontsize=9
)
ax2.axis("off")
ax2.set_facecolor("#0d0d1a")
# Diagnosis text panel
ax3 = fig.add_subplot(1, 3, 3)
ax3.set_facecolor("#0d0d1a")
ax3.axis("off")
if "error" not in diagnosis:
summary = diagnosis.get("diagnosis_summary", "")
interp = diagnosis.get("defect_interpretation", "")
recs = diagnosis.get("recommendations", [])
conf = diagnosis.get("confidence", "")
# Word wrap helper
def wrap(text, width=42):
words, lines, line = text.split(), [], ""
for w in words:
if len(line) + len(w) + 1 <= width:
line += (" " if line else "") + w
else:
lines.append(line)
line = w
if line:
lines.append(line)
return "\n".join(lines)
report = (
f"RISK: {risk.upper()}\n"
f"{'─'*38}\n\n"
f"SUMMARY\n{wrap(summary)}\n\n"
f"INTERPRETATION\n{wrap(interp[:200])}\n\n"
f"RECOMMENDATIONS\n"
)
for i, r in enumerate(recs[:3], 1):
report += f"{i}. {wrap(r[:80])}\n"
report += f"\nConfidence: {conf}"
ax3.text(
0.05, 0.97, report,
transform=ax3.transAxes,
fontsize=7.5, verticalalignment="top",
fontfamily="monospace", color="white",
bbox=dict(
boxstyle="round", facecolor="#1a1a2e",
alpha=0.9, edgecolor=risk_color, linewidth=2
)
)
else:
ax3.text(
0.1, 0.5, f"API Error:\n{diagnosis['error']}",
transform=ax3.transAxes, color="red", fontsize=9
)
ax3.set_title("AI Diagnosis", color="white", fontsize=9)
plt.tight_layout()
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight",
facecolor="#0d0d1a")
plt.close()
print(f" Visualization → {out_path.resolve()}")
def run_full_pipeline(image_path: Path, subset: str, save_vis: bool = True) -> dict:
"""Full pipeline: image → segmentation → features → diagnosis."""
ckpt_path = Path("checkpoints") / subset / "best_model.pt"
if not ckpt_path.exists():
print(f"❌ No checkpoint at {ckpt_path}")
return {}
print(f"\n{'='*60}")
print(f"FailureGPT Pipeline")
print(f"Image: {image_path.name}")
print(f"Subset: {subset}")
print(f"{'='*60}")
# Step 1: Segment
print("Step 1/3: Segmenting...")
model = load_model(ckpt_path)
img_tensor = load_image_tensor(image_path, IMAGE_SIZE)
mask = predict_mask(model, img_tensor, IMAGE_SIZE)
# Step 2: Extract features
print("Step 2/3: Extracting features...")
features = extract_features(mask, IMAGE_SIZE)
print(f" → {features['defect_count']} blobs, "
f"{features['defect_area_fraction']:.2f}% defect, "
f"AR={features['mean_aspect_ratio']:.2f}")
# Step 3: Generate diagnosis
print("Step 3/3: Generating diagnosis...")
diagnosis = call_claude(features, image_path.name)
# Print report
report = format_diagnosis_report(features, diagnosis, image_path.name)
print(report)
# Save visualization
if save_vis:
out_path = Path("output/diagnosis") / f"{image_path.stem}_diagnosis.png"
visualize_diagnosis(image_path, mask, features, diagnosis, out_path)
# Save JSON
result = {"image": str(image_path), "features": features, "diagnosis": diagnosis}
json_out = Path("output/diagnosis") / f"{image_path.stem}_diagnosis.json"
json_out.parent.mkdir(parents=True, exist_ok=True)
with open(json_out, "w") as f:
json.dump(result, f, indent=2)
print(f" JSON → {json_out.resolve()}")
return result
def interactive_mode(subset: str, data_dir: Path):
"""Interactive CLI: pick an image, get a diagnosis."""
subset_dir = data_dir / subset
ds = FractographyDataset(subset_dir, split="all", image_size=IMAGE_SIZE)
print(f"\nAvailable images in '{subset}':")
for i, (img_path, _) in enumerate(ds.pairs[:20]):
print(f" [{i:2d}] {img_path.name}")
try:
idx = int(input("\nEnter image index: "))
img_path, _ = ds.pairs[idx]
run_full_pipeline(img_path, subset)
except (ValueError, IndexError) as e:
print(f"Invalid selection: {e}")
if __name__ == "__main__":
parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, default=None)
parser.add_argument("--subset", type=str, default="all_defects")
parser.add_argument("--json", type=str, default=None,
help="Path to existing features JSON from features.py")
parser.add_argument("--interactive", action="store_true")
parser.add_argument("--data_dir", type=str, default="data")
parser.add_argument("--n", type=int, default=3,
help="Number of images to process in batch mode")
args = parser.parse_args()
if args.interactive:
interactive_mode(args.subset, Path(args.data_dir))
elif args.json:
# Diagnose from existing feature JSON
with open(args.json) as f:
feature_list = json.load(f)
if isinstance(feature_list, list):
for item in feature_list[:args.n]:
diagnosis = call_claude(item, item.get("image", ""))
print(format_diagnosis_report(item, diagnosis, item.get("image", "")))
else:
diagnosis = call_claude(feature_list)
print(format_diagnosis_report(feature_list, diagnosis))
elif args.image:
run_full_pipeline(Path(args.image), args.subset)
else:
# Batch: run on first n images of subset
subset_dir = Path(args.data_dir) / args.subset
ds = FractographyDataset(subset_dir, split="all", image_size=IMAGE_SIZE)
for img_path, _ in list(ds.pairs)[:args.n]:
run_full_pipeline(img_path, args.subset)
"""
download_osf.py
---------------
Downloads the OSF Ti-64 SEM fractography dataset (osf.io/gdwyb).
The dataset has 3 sub-components:
- Lack of Fusion defects
- Keyhole defects
- All Defects (combined)
Each sub-dataset contains SEM images + ground truth segmentation masks.
Usage:
python download_osf.py
Output structure:
data/
lack_of_fusion/
images/
masks/
keyhole/
images/
masks/
all_defects/
images/
masks/
"""
import os
import requests
from pathlib import Path
from tqdm import tqdm
# OSF project GUIDs for each sub-dataset
# Inspect at: https://osf.io/gdwyb/
OSF_API = "https://api.osf.io/v2"
OSF_PROJECT_ID = "gdwyb" # top-level fractography project
DATA_DIR = Path("data")
def list_osf_files(node_id: str) -> list[dict]:
"""Recursively list all files in an OSF node."""
url = f"{OSF_API}/nodes/{node_id}/files/osfstorage/"
files = []
while url:
resp = requests.get(url, timeout=30)
resp.raise_for_status()
data = resp.json()
for item in data["data"]:
if item["attributes"]["kind"] == "file":
files.append({
"name": item["attributes"]["name"],
"path": item["attributes"]["materialized_path"],
"download": item["links"]["download"],
"size": item["attributes"]["size"],
})
elif item["attributes"]["kind"] == "folder":
# recurse into folders
folder_id = item["relationships"]["files"]["links"]["related"]["href"]
files.extend(list_osf_folder(folder_id))
url = data["links"].get("next")
return files
def list_osf_folder(url: str) -> list[dict]:
"""Recursively list files inside an OSF folder URL."""
files = []
while url:
resp = requests.get(url, timeout=30)
resp.raise_for_status()
data = resp.json()
for item in data["data"]:
if item["attributes"]["kind"] == "file":
files.append({
"name": item["attributes"]["name"],
"path": item["attributes"]["materialized_path"],
"download": item["links"]["download"],
"size": item["attributes"]["size"],
})
elif item["attributes"]["kind"] == "folder":
folder_url = item["relationships"]["files"]["links"]["related"]["href"]
files.extend(list_osf_folder(folder_url))
url = data["links"].get("next")
return files
def download_file(url: str, dest: Path):
"""Download a file with a progress bar."""
dest.parent.mkdir(parents=True, exist_ok=True)
if dest.exists():
print(f" [skip] {dest.name} already exists")
return
resp = requests.get(url, stream=True, timeout=60)
resp.raise_for_status()
total = int(resp.headers.get("content-length", 0))
with open(dest, "wb") as f, tqdm(
desc=dest.name, total=total, unit="B", unit_scale=True, leave=False
) as bar:
for chunk in resp.iter_content(chunk_size=8192):
f.write(chunk)
bar.update(len(chunk))
def download_osf_project(node_id: str, local_root: Path):
"""Download all files from an OSF node into local_root, preserving folder structure."""
print(f"\n📂 Fetching file list from OSF node: {node_id}")
try:
files = list_osf_files(node_id)
except Exception as e:
print(f" ⚠️ Could not list files: {e}")
print(" → You may need to download manually from https://osf.io/gdwyb/")
return []
print(f" Found {len(files)} files")
for f in files:
# strip leading slash from materialized path
rel_path = f["path"].lstrip("/")
dest = local_root / rel_path
print(f" ↓ {rel_path} ({f['size'] / 1024:.1f} KB)")
try:
download_file(f["download"], dest)
except Exception as e:
print(f" ⚠️ Failed: {e}")
return files
if __name__ == "__main__":
DATA_DIR.mkdir(exist_ok=True)
print("=" * 60)
print("OSF Ti-64 Fractography Dataset Downloader")
print("Project: https://osf.io/gdwyb/")
print("=" * 60)
files = download_osf_project(OSF_PROJECT_ID, DATA_DIR)
if files:
print(f"\n✅ Download complete. Files saved to: {DATA_DIR.resolve()}")
else:
print("\n⚠️ Automatic download failed.")
print("Manual download steps:")
print(" 1. Go to https://osf.io/gdwyb/")
print(" 2. Click each sub-component (Lack of Fusion, Key Hole, All Defects)")
print(" 3. Download the zip and extract into data/<subfolder>/")
print(" Expected structure:")
print(" data/lack_of_fusion/images/*.png (or .tif)")
print(" data/lack_of_fusion/masks/*.png")
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
- defect_area_fraction : % of image that is defect
- defect_count : number of distinct defect regions
- mean_pore_area : mean area of individual defect blobs (px²)
- max_pore_area : largest single defect region
- mean_aspect_ratio : mean of (major_axis / minor_axis) per blob
→ circular pores ≈ 1.0 (keyhole)
→ elongated pores > 2.0 (lack of fusion)
- spatial_concentration : std of defect centroid positions (spread)
- size_std : std of pore areas (heterogeneity)
- quadrant_distribution : defect fraction per image quadrant
Usage:
python features.py --image data/all_defects/images/001-Overview-EP04V24.png
--subset all_defects
python features.py --subset all_defects --all # run on all images in subset
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
"keyhole_max_aspect_ratio": 1.6, # wider keyhole band
"lof_min_count": 20, # need many blobs for LoF
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
std = np.array([0.229, 0.224, 0.225])
arr = (arr - mean) / std
return torch.from_numpy(arr).permute(2, 0, 1).float()
@torch.no_grad()
def predict_mask(model, image_tensor: torch.Tensor, target_size: tuple) -> np.ndarray:
outputs = model(pixel_values=image_tensor.unsqueeze(0))
logits = outputs.logits
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
h, w = mask.shape
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
major = max(h_bbox, w_bbox)
minor = min(h_bbox, w_bbox)
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
H, W = image_size
total_px = H * W
defect_px = int((mask == 1).sum())
defect_frac = defect_px / total_px
if defect_px == 0:
return {
"defect_area_fraction": 0.0,
"defect_count": 0,
"mean_pore_area_px": 0.0,
"max_pore_area_px": 0,
"mean_aspect_ratio": 0.0,
"spatial_concentration": 0.0,
"size_std": 0.0,
"quadrant_distribution": [0.0, 0.0, 0.0, 0.0],
"defect_type": "clean",
"confidence": "high",
}
# Connected components (note: slow for large masks — acceptable at 256×256)
labels, n_blobs = connected_components(mask)
props = blob_properties(labels, n_blobs)
areas = [p["area"] for p in props]
aspect_ratios = [p["aspect_ratio"] for p in props]
centroids = [p["centroid"] for p in props]
mean_area = float(np.mean(areas)) if areas else 0.0
max_area = int(max(areas)) if areas else 0
mean_ar = float(np.mean(aspect_ratios)) if aspect_ratios else 0.0
size_std = float(np.std(areas)) if areas else 0.0
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
float((mask[:half_h, :half_w] == 1).sum()), # top-left
float((mask[:half_h, half_w:] == 1).sum()), # top-right
float((mask[half_h:, :half_w] == 1).sum()), # bottom-left
float((mask[half_h:, half_w:] == 1).sum()), # bottom-right
]
total_defect = sum(quads) + 1e-8
quad_dist = [q / total_defect for q in quads]
# ── Rule-based classification ─────────────────────────────────────────────
defect_type, confidence = classify_defect(defect_frac, n_blobs, mean_ar, mean_area)
return {
"defect_area_fraction": round(defect_frac * 100, 3), # as %
"defect_count": n_blobs,
"mean_pore_area_px": round(mean_area, 1),
"max_pore_area_px": max_area,
"mean_aspect_ratio": round(mean_ar, 3),
"spatial_concentration": round(spatial_conc, 2),
"size_std": round(size_std, 1),
"quadrant_distribution": [round(q, 3) for q in quad_dist],
"defect_type": defect_type,
"confidence": confidence,
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
Lack of fusion: many small irregular pores, higher aspect ratio
Keyhole: fewer larger circular pores, lower aspect ratio
Mixed: both morphologies present
Clean: below detection threshold
"""
t = THRESHOLDS
if defect_frac < t["min_defect_fraction_to_classify"]:
return "clean", "high"
is_circular = mean_ar <= t["keyhole_max_aspect_ratio"]
is_many = count >= t["lof_min_count"]
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
f"Defect Type: {features['defect_type'].upper()} "
f"(confidence: {features['confidence']})",
fontsize=11, fontweight="bold"
)
# Image
axes[0].imshow(raw_resized, cmap="gray")
axes[0].set_title("SEM Image", fontsize=9)
axes[0].axis("off")
# Mask with blob labels
overlay = np.stack([raw_resized, raw_resized, raw_resized], axis=-1).copy()
overlay[mask == 1] = [0, 212, 255] # cyan defects
axes[1].imshow(overlay)
axes[1].set_title(
f"Prediction\n{features['defect_area_fraction']:.2f}% defect | "
f"{features['defect_count']} blobs",
fontsize=9
)
axes[1].axis("off")
# Feature summary text
axes[2].axis("off")
feature_text = (
f"Defect Area: {features['defect_area_fraction']:.3f}%\n"
f"Defect Count: {features['defect_count']}\n"
f"Mean Pore Area: {features['mean_pore_area_px']:.1f} px²\n"
f"Max Pore Area: {features['max_pore_area_px']} px²\n"
f"Mean Aspect Ratio: {features['mean_aspect_ratio']:.3f}\n"
f" (1.0=circle, >2=elongated)\n"
f"Spatial Spread: {features['spatial_concentration']:.2f}\n"
f"Size Std Dev: {features['size_std']:.1f}\n\n"
f"Quadrant Distribution:\n"
f" TL:{features['quadrant_distribution'][0]:.2f} "
f"TR:{features['quadrant_distribution'][1]:.2f}\n"
f" BL:{features['quadrant_distribution'][2]:.2f} "
f"BR:{features['quadrant_distribution'][3]:.2f}\n\n"
f"─────────────────────────\n"
f"DEFECT TYPE: {features['defect_type']}\n"
f"CONFIDENCE: {features['confidence']}"
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
print(f" Saved → {out_path.resolve()}")
def run_on_image(image_path: Path, subset: str) -> dict:
ckpt_path = Path("checkpoints") / subset / "best_model.pt"
if not ckpt_path.exists():
print(f"❌ No checkpoint at {ckpt_path}")
return {}
print(f"\nImage: {image_path.name}")
print(f"Subset: {subset}")
model = load_model(ckpt_path)
img_tensor = load_image_tensor(image_path, IMAGE_SIZE)
mask = predict_mask(model, img_tensor, IMAGE_SIZE)
features = extract_features(mask, IMAGE_SIZE)
print(f"Defect type: {features['defect_type']} ({features['confidence']} confidence)")
print(f"Defect area: {features['defect_area_fraction']:.3f}%")
print(f"Blob count: {features['defect_count']}")
print(f"Mean AR: {features['mean_aspect_ratio']:.3f}")
print(json.dumps(features, indent=2))
out_path = Path("output/features") / f"{image_path.stem}_features.png"
visualize_features(image_path, mask, features, out_path)
return features
def run_on_subset(subset: str, data_dir: Path, n: int = 6):
"""Run feature extraction on n images from a subset and print summary."""
subset_dir = data_dir / subset
if not subset_dir.exists():
print(f"⚠️ {subset_dir} not found")
return
ds = FractographyDataset(subset_dir, split="all", image_size=IMAGE_SIZE)
ckpt_path = Path("checkpoints") / subset / "best_model.pt"
if not ckpt_path.exists():
print(f"⚠️ No checkpoint for {subset}")
return
model = load_model(ckpt_path)
results = []
print(f"\n{'='*60}")
print(f"Feature extraction: {subset} ({min(n, len(ds))} images)")
print(f"{'='*60}")
for idx in range(min(n, len(ds))):
img_path, _ = ds.pairs[idx]
img_tensor = load_image_tensor(img_path, IMAGE_SIZE)
mask = predict_mask(model, img_tensor, IMAGE_SIZE)
features = extract_features(mask, IMAGE_SIZE)
features["image"] = img_path.name
results.append(features)
out_path = Path("output/features") / subset / f"{img_path.stem}_features.png"
visualize_features(img_path, mask, features, out_path)
# Summary
print(f"\n Classification summary:")
from collections import Counter
counts = Counter(r["defect_type"] for r in results)
for dtype, count in counts.items():
print(f" {dtype:25s}: {count}")
# Save results JSON
json_out = Path("output/features") / f"{subset}_features.json"
json_out.parent.mkdir(parents=True, exist_ok=True)
with open(json_out, "w") as f:
json.dump(results, f, indent=2)
print(f"\n Feature JSON → {json_out.resolve()}")
if __name__ == "__main__":
parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, default=None,
help="Path to a single SEM image")
parser.add_argument("--subset", type=str, default="all_defects",
help="lack_of_fusion | keyhole | all_defects")
parser.add_argument("--all", action="store_true",
help="Run on all images in subset (up to --n)")
parser.add_argument("--n", type=int, default=6,
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
DEVICE = torch.device("cpu")
N_SAMPLES = 6 # images to visualize per subset
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
logits = outputs.logits # (1, C, H/4, W/4)
upsampled = F.interpolate(
logits, size=target_size, mode="bilinear", align_corners=False
)
pred = upsampled.squeeze(0).argmax(dim=0).numpy() # (H, W)
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
gt_defect = gt == 1
intersection = (pred_defect & gt_defect).sum()
union = (pred_defect | gt_defect).sum()
iou = intersection / union if union > 0 else float("nan")
coverage_pred = pred_defect.sum() / pred.size * 100
coverage_gt = gt_defect.sum() / gt.size * 100
return {"iou": iou, "pred_coverage": coverage_pred, "gt_coverage": coverage_gt}
def run_inference(subset: str, args):
data_dir = Path(args.data_dir) / subset
ckpt_path = Path("checkpoints") / subset / "best_model.pt"
out_dir = Path("output") / "inference"
out_dir.mkdir(parents=True, exist_ok=True)
if not data_dir.exists():
print(f"⚠️ Skipping '{subset}' — data not found at {data_dir}")
return
if not ckpt_path.exists():
print(f"⚠️ Skipping '{subset}' — no checkpoint at {ckpt_path}")
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
gt_arr = gt_mask.numpy() # already scaled by dataset
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
gt_colored = colorize(gt_arr)
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
print(f" Mean IoU (sampled): {mean_iou:.4f}")
print(f" Saved → {out_path.resolve()}")
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
matplotlib.use("Agg") # headless-safe; switch to "TkAgg" if you want interactive
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
# ── Configurable label map ────────────────────────────────────────────────────
# Update this once you've inspected the actual class values in your masks.
# Keys = integer pixel values in mask PNGs.
LABEL_MAP = {
0: ("Background", "#1a1a2e"),
1: ("Lack of Fusion", "#e94560"),
2: ("Keyhole", "#0f3460"),
3: ("Other Defect", "#533483"),
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
print(f" ⚠️ Found images/ at {images_dir} but no masks/ sibling")
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
print(f" ⚠️ No mask found for {img_path.name}")
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
print(" No pairs to visualize.")
return
fig, axes = plt.subplots(n, 3, figsize=(12, n * 4))
if n == 1:
axes = [axes]
fig.suptitle("OSF Ti-64 SEM Dataset — Inspection Grid\n(Image | Mask | Overlay)",
fontsize=13, fontweight="bold", y=1.01)
all_unique = set()
for i, (img_path, mask_path) in enumerate(pairs[:n]):
raw = np.array(Image.open(img_path), dtype=np.float32)
raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
img = np.stack([raw, raw, raw], axis=-1)
mask_pil = Image.open(mask_path)
mask_arr = np.array(mask_pil)
# If mask is RGB, convert to grayscale for inspection
if mask_arr.ndim == 3:
mask_arr = np.array(mask_pil.convert("L"))
unique_vals = sorted(np.unique(mask_arr).tolist())
all_unique.update(unique_vals)
mask_rgb = colorize_mask(mask_arr)
# Overlay: blend image and mask
img_display = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
overlay = (img_display.astype(float) * 0.6 + mask_rgb.astype(float) * 0.4).astype(np.uint8)
axes[i][0].imshow(img, cmap="gray" if img.ndim == 2 else None)
axes[i][0].set_title(f"Image\n{img_path.name}", fontsize=8)
axes[i][0].axis("off")
axes[i][1].imshow(mask_rgb)
axes[i][1].set_title(
f"Mask (classes: {unique_vals})\n{mask_path.name}", fontsize=8
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
print("\n⚠️ No pairs found. Check your data/ folder structure.")
print("Expected layout:")
print(" data/")
print(" <subset>/")
print(" images/ ← SEM images (.png or .tif)")
print(" masks/ ← segmentation masks (.png)")
return
# Sample first few masks
print(f"\nSampling first 5 masks for format inspection:")
all_unique = set()
for img_path, mask_path in pairs[:5]:
info = inspect_mask(mask_path)
print(f"\n {mask_path.name}")
print(f" Mode: {info['mode']}")
print(f" Shape: {info['shape']}")
print(f" Dtype: {info['dtype']}")
print(f" Unique values: {info['unique_values']}")
print(f" Value range: [{info['min']}, {info['max']}]")
all_unique.update(info["unique_values"])
print(f"\n{'─'*40}")
print(f"All unique class values across sampled masks: {sorted(all_unique)}")
print("\nLabel interpretation:")
for v in sorted(all_unique):
label, _ = LABEL_MAP.get(v, (f"UNKNOWN — update LABEL_MAP in this script", "#888"))
print(f" {v:3d} → {label}")
print(f"\n⚠️ NOTE: If all unique values are {{0, 255}}, masks are binary (defect/no-defect).")
print(" If values are 0–N, masks are multi-class integer labels — ideal for SegFormer.")
print(" If mode is 'RGB', masks encode class as color — you'll need to remap.")
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
"""
FailureGPT — OSF Ti-64 Dataset Inspector
Run this first to install dependencies.
"""
import subprocess, sys
packages = [
"torch",
"torchvision",
"Pillow",
"matplotlib",
"numpy",
"requests",
"osfclient",
"tqdm",
]
for pkg in packages:
subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
print("✅ All dependencies installed.")
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
matplotlib.use("Agg") # headless-safe; switch to "TkAgg" if you want interactive
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
# ── Configurable label map ────────────────────────────────────────────────────
# Update this once you've inspected the actual class values in your masks.
# Keys = integer pixel values in mask PNGs.
LABEL_MAP = {
0: ("Background", "#1a1a2e"),
1: ("Lack of Fusion", "#e94560"),
2: ("Keyhole", "#0f3460"),
3: ("Other Defect", "#533483"),
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
print(f" ⚠️ Found images/ at {images_dir} but no masks/ sibling")
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
print(f" ⚠️ No mask found for {img_path.name}")
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
print(" No pairs to visualize.")
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
f"Mask (classes: {unique_vals})\n{mask_path.name}", fontsize=8
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
print("\n⚠️ No pairs found. Check your data/ folder structure.")
print("Expected layout:")
print(" data/")
print(" <subset>/")
print(" images/ ← SEM images (.png or .tif)")
print(" masks/ ← segmentation masks (.png)")
return
# Sample first few masks
print(f"\nSampling first 5 masks for format inspection:")
all_unique = set()
for img_path, mask_path in pairs[:5]:
info = inspect_mask(mask_path)
print(f"\n {mask_path.name}")
print(f" Mode: {info['mode']}")
print(f" Shape: {info['shape']}")
print(f" Dtype: {info['dtype']}")
print(f" Unique values: {info['unique_values']}")
print(f" Value range: [{info['min']}, {info['max']}]")
all_unique.update(info["unique_values"])
print(f"\n{'─'*40}")
print(f"All unique class values across sampled masks: {sorted(all_unique)}")
print("\nLabel interpretation:")
for v in sorted(all_unique):
label, _ = LABEL_MAP.get(v, (f"UNKNOWN — update LABEL_MAP in this script", "#888"))
print(f" {v:3d} → {label}")
print(f"\n⚠️ NOTE: If all unique values are {{0, 255}}, masks are binary (defect/no-defect).")
print(" If values are 0–N, masks are multi-class integer labels — ideal for SegFormer.")
print(" If mode is 'RGB', masks encode class as color — you'll need to remap.")
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
checkpoints/<subset>/best_model.pt <- best checkpoint by val mIoU
checkpoints/<subset>/last_model.pt <- final epoch checkpoint
checkpoints/<subset>/history.json <- loss/mIoU per epoch
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
NUM_CLASSES = 2 # background (0) + defect (1)
IMAGE_SIZE = (256, 256) # smaller = faster on CPU; increase if you have time
BATCH_SIZE = 2
EPOCHS = 15
LR = 6e-5
TRAIN_FRAC = 0.8
WEIGHT_DECAY = 0.01
SUBSETS = ["lack_of_fusion", "keyhole", "all_defects"]
# ─────────────────────────────────────────────────────────────────────────────
def compute_miou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
"""Mean Intersection over Union across all classes."""
ious = []
preds = preds.view(-1)
targets = targets.view(-1)
for cls in range(num_classes):
pred_mask = preds == cls
target_mask = targets == cls
intersection = (pred_mask & target_mask).sum().item()
union = (pred_mask | target_mask).sum().item()
if union == 0:
continue # class not present in this batch
ious.append(intersection / union)
return float(np.mean(ious)) if ious else 0.0
def dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
"""
Soft Dice loss for binary segmentation.
Directly optimizes overlap — critical for imbalanced datasets.
logits: (B, num_classes, H, W)
targets: (B, H, W) integer labels
"""
probs = torch.softmax(logits, dim=1) # (B, C, H, W)
# Focus on defect class (index 1)
prob_defect = probs[:, 1] # (B, H, W)
target_defect = (targets == 1).float()
intersection = (prob_defect * target_defect).sum(dim=(1, 2))
union = prob_defect.sum(dim=(1, 2)) + target_defect.sum(dim=(1, 2))
dice = (2.0 * intersection + smooth) / (union + smooth)
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
dice_weight: blend factor for Dice loss (0 = CE only, 1 = Dice only).
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
masks = masks.to(device)
# Use HuggingFace built-in loss — passes labels at native resolution
# SegFormer internally downsamples labels to match logit size
outputs = model(pixel_values=images, labels=masks)
loss = outputs.loss
optimizer.zero_grad()
loss.backward()
optimizer.step()
total_loss += loss.item()
return total_loss / len(loader)
@torch.no_grad()
def evaluate(model, loader, device, num_classes):
model.eval()
total_loss = 0.0
all_miou = []
for images, masks in loader:
images = images.to(device)
masks = masks.to(device)
outputs = model(pixel_values=images, labels=masks)
loss = outputs.loss
logits = outputs.logits # (B, num_classes, H/4, W/4)
# Upsample logits to mask size
upsampled = F.interpolate(
logits,
size=masks.shape[-2:],
mode="bilinear",
align_corners=False,
)
preds = upsampled.argmax(dim=1) # (B, H, W)
total_loss += loss.item()
all_miou.append(compute_miou(preds.cpu(), masks.cpu(), num_classes))
return total_loss / len(loader), float(np.mean(all_miou))
def train_subset(subset: str, data_root: Path, args):
subset_dir = data_root / subset
if not subset_dir.exists():
print(f"\n⚠️ Skipping '{subset}' — folder not found at {subset_dir}")
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
print(f" ⚠️ No image/mask pairs found in {subset_dir}")
return
n_train = max(1, int(len(full_ds) * TRAIN_FRAC))
n_val = len(full_ds) - n_train
train_ds, val_ds = random_split(
full_ds, [n_train, n_val],
generator=torch.Generator().manual_seed(42)
)
print(f" Train: {len(train_ds)} | Val: {len(val_ds)}")
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
# Model
device = torch.device("cpu")
id2label = {0: "background", 1: "defect"}
label2id = {v: k for k, v in id2label.items()}
print(f" Loading SegFormer-b0 from HuggingFace...")
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
history = {"train_loss": [], "val_loss": [], "val_miou": []}
best_miou = 0.0
for epoch in range(1, args.epochs + 1):
t0 = time.time()
train_loss = train_one_epoch(model, train_loader, optimizer, device)
val_loss, val_miou = evaluate(model, val_loader, device, NUM_CLASSES)
scheduler.step()
elapsed = time.time() - t0
print(
f" Epoch {epoch:02d}/{args.epochs} | "
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
print(f" ✅ New best mIoU: {best_miou:.4f} — checkpoint saved")
# Save last + history
torch.save(model.state_dict(), ckpt_dir / "last_model.pt")
with open(ckpt_dir / "history.json", "w") as f:
json.dump(history, f, indent=2)
print(f"\n Done. Best val mIoU: {best_miou:.4f}")
print(f" Checkpoints saved to: {ckpt_dir.resolve()}")
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
axes[0].plot(epochs, h["val_loss"], label=f"{subset} val", linestyle="--")
axes[1].plot(epochs, h["val_miou"], label=subset)
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
print(f" (Could not save plot: {e})")
if __name__ == "__main__":
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data")
parser.add_argument("--epochs", type=int, default=EPOCHS)
parser.add_argument("--lr", type=float, default=LR)
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
print(f" {subset:20s} best mIoU = {best:.4f}")

