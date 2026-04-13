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
    print("⚠️  anthropic package not found. Run: pip install anthropic")
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
- Defect area fraction:    {features.get('defect_area_fraction', 0):.3f}% of fracture surface
- Defect blob count:       {features.get('defect_count', 0)} distinct pores/defects
- Mean pore area:          {features.get('mean_pore_area_px', 0):.1f} px² (at 256×256 resolution)
- Max pore area:           {features.get('max_pore_area_px', 0)} px²
- Mean aspect ratio:       {features.get('mean_aspect_ratio', 0):.3f}
  (1.0 = perfectly circular/keyhole, >2.0 = elongated/lack-of-fusion)
- Spatial spread (std):    {features.get('spatial_concentration', 0):.2f} px
- Size heterogeneity:      {features.get('size_std', 0):.1f} px² std dev
- Quadrant distribution:
    Top-left:     {features.get('quadrant_distribution', [0,0,0,0])[0]:.3f}
    Top-right:    {features.get('quadrant_distribution', [0,0,0,0])[1]:.3f}
    Bottom-left:  {features.get('quadrant_distribution', [0,0,0,0])[2]:.3f}
    Bottom-right: {features.get('quadrant_distribution', [0,0,0,0])[3]:.3f}
- Rule-based defect type:  {features.get('defect_type', 'unknown')}
  (confidence: {features.get('confidence', 'unknown')})

Provide a structured engineering diagnosis as JSON.
"""


def call_claude(features: dict, image_name: str = "") -> dict:
    """Call Claude API and return parsed diagnosis dict."""
    if not HAS_ANTHROPIC:
        return {"error": "anthropic package not installed"}

    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
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
        f"  Defect area:      {features.get('defect_area_fraction', 0):.3f}%",
        f"  Defect count:     {features.get('defect_count', 0)}",
        f"  Mean aspect ratio:{features.get('mean_aspect_ratio', 0):.3f}",
        f"  Rule-based type:  {features.get('defect_type', 'unknown')}",
        "",
    ]

    if "error" in diagnosis:
        lines += [f"⚠️  Diagnosis error: {diagnosis['error']}"]
        return "\n".join(lines)

    lines += [
        "AI DIAGNOSIS",
        f"  Failure mechanism: {diagnosis.get('dominant_failure_mechanism', 'N/A')}",
        f"  Crack init. risk:  {diagnosis.get('crack_initiation_risk', 'N/A').upper()}",
        f"  Critical regions:  {diagnosis.get('critical_regions', 'N/A')}",
        f"  Confidence:        {diagnosis.get('confidence', 'N/A')}",
        "",
        "SUMMARY",
        f"  {diagnosis.get('diagnosis_summary', '')}",
        "",
        "DEFECT INTERPRETATION",
        f"  {diagnosis.get('defect_interpretation', '')}",
        "",
        "RISK RATIONALE",
        f"  {diagnosis.get('risk_rationale', '')}",
        "",
        "RECOMMENDATIONS",
    ]
    for i, rec in enumerate(diagnosis.get("recommendations", []), 1):
        lines.append(f"  {i}. {rec}")
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
        f"Mechanism: {mech}  |  Crack Risk: {risk.upper()}",
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
        interp  = diagnosis.get("defect_interpretation", "")
        recs    = diagnosis.get("recommendations", [])
        conf    = diagnosis.get("confidence", "")

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
    print(f"  Visualization → {out_path.resolve()}")


def run_full_pipeline(image_path: Path, subset: str, save_vis: bool = True) -> dict:
    """Full pipeline: image → segmentation → features → diagnosis."""
    ckpt_path = Path("checkpoints") / subset / "best_model.pt"
    if not ckpt_path.exists():
        print(f"❌ No checkpoint at {ckpt_path}")
        return {}

    print(f"\n{'='*60}")
    print(f"FailureGPT Pipeline")
    print(f"Image:    {image_path.name}")
    print(f"Subset:   {subset}")
    print(f"{'='*60}")

    # Step 1: Segment
    print("Step 1/3: Segmenting...")
    model      = load_model(ckpt_path)
    img_tensor = load_image_tensor(image_path, IMAGE_SIZE)
    mask       = predict_mask(model, img_tensor, IMAGE_SIZE)

    # Step 2: Extract features
    print("Step 2/3: Extracting features...")
    features = extract_features(mask, IMAGE_SIZE)
    print(f"  → {features['defect_count']} blobs, "
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
    print(f"  JSON → {json_out.resolve()}")

    return result


def interactive_mode(subset: str, data_dir: Path):
    """Interactive CLI: pick an image, get a diagnosis."""
    subset_dir = data_dir / subset
    ds = FractographyDataset(subset_dir, split="all", image_size=IMAGE_SIZE)

    print(f"\nAvailable images in '{subset}':")
    for i, (img_path, _) in enumerate(ds.pairs[:20]):
        print(f"  [{i:2d}] {img_path.name}")

    try:
        idx = int(input("\nEnter image index: "))
        img_path, _ = ds.pairs[idx]
        run_full_pipeline(img_path, subset)
    except (ValueError, IndexError) as e:
        print(f"Invalid selection: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",       type=str, default=None)
    parser.add_argument("--subset",      type=str, default="all_defects")
    parser.add_argument("--json",        type=str, default=None,
                        help="Path to existing features JSON from features.py")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--data_dir",    type=str, default="data")
    parser.add_argument("--n",           type=int, default=3,
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
