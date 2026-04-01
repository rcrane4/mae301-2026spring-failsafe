"""
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
MODELS  = {}

print("Loading checkpoints...")
for subset in SUBSETS:
    ckpt = Path("checkpoints") / subset / "best_model.pt"
    if ckpt.exists():
        MODELS[subset] = load_model(ckpt)
        print(f"  ✅ {subset}")
    else:
        print(f"  ⚠️  {subset} — checkpoint not found")

# ─────────────────────────────────────────────────────────────────────────────

RISK_COLORS = {
    "low":      "#2ecc71",
    "medium":   "#f39c12",
    "high":     "#e74c3c",
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
    std  = np.array([0.229, 0.224, 0.225])
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
    result[~defect_mask] = display_arr[~defect_mask]  # background unchanged
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
        f"Defect Area:        {features['defect_area_fraction']:.3f}%",
        f"Defect Count:       {features['defect_count']} blobs",
        f"Mean Pore Area:     {features.get('mean_pore_area_px', 0):.1f} px²",
        f"Max Pore Area:      {features.get('max_pore_area_px', 0)} px²",
        f"Mean Aspect Ratio:  {features['mean_aspect_ratio']:.3f}",
        f"  (1.0=circular · >2.0=elongated)",
        f"Spatial Spread:     {features['spatial_concentration']:.2f}",
        f"Size Std Dev:       {features['size_std']:.1f}",
        f"",
        f"Quadrant Distribution:",
        f"  TL {features['quadrant_distribution'][0]:.2f}  "
        f"TR {features['quadrant_distribution'][1]:.2f}",
        f"  BL {features['quadrant_distribution'][2]:.2f}  "
        f"BR {features['quadrant_distribution'][3]:.2f}",
        f"",
        f"Rule-based type:    {features['defect_type']}",
        f"Confidence:         {features['confidence']}",
    ]
    features_text = "\n".join(feat_lines)

    # ── Step 6: AI Diagnosis ──────────────────────────────────────────────────
    if not os.environ.get("ANTHROPIC_API_KEY"):
        diagnosis_text = (
            "⚠️  ANTHROPIC_API_KEY not set.\n\n"
            "Set it in your terminal:\n"
            "  $env:ANTHROPIC_API_KEY = 'sk-ant-...'\n\n"
            "Features extracted successfully:\n\n"
            + features_text
        )
        risk_label = features["defect_type"].upper()
    else:
        diagnosis      = call_claude(features, "uploaded_image")
        diagnosis_text = format_diagnosis_report(features, diagnosis, "uploaded_image")
        risk       = diagnosis.get("crack_initiation_risk", "unknown")
        mech       = diagnosis.get("dominant_failure_mechanism", "unknown")
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
                label="SEM FRACTOGRAPH  —  drag & drop or click to upload",
                type="numpy",
                height=500,
            )
            subset_input = gr.Dropdown(
                choices=SUBSETS,
                value="all_defects",
                label="MODEL SUBSET",
            )
            with gr.Row():
                run_btn   = gr.Button("▶  ANALYZE", variant="primary", scale=3)
                clear_btn = gr.Button("✕  CLEAR",   variant="secondary", scale=1)

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
                label="AI FAILURE DIAGNOSIS  —  Claude",
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
