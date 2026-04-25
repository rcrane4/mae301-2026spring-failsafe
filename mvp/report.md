MVP Report
Place in /mvp/report.md.

1. Executive Summary: Problem, solution, and what your MVP actually does.

Fractographic analysis of LPBF (Laser Powder Bed Fusion) aerospace components currently requires scarce expert knowledge, is slow, and doesn’t scale to production. FailSafe automates this end-to-end, where a user uploads a SEM (scanning electron microscopy) fractograph and receives pixel-level defect segmentation, morphological feature extraction, defect type classification (lack-of-fusion vs. keyhole porosity), crack initiation risk scoring, and actionable LPBF process recommendations. No expert is required. The MVP is live at huggingface.co/spaces/rcrane4/FailSafe.

2. User & Use Case: Clear persona and usage narrative.

The persona for this MVP is a process or quality engineer at an aerospace manufacturer producing LPBF Ti-6Al-4V components who needs to assess fractured specimens but lacks a dedicated fractography expert on staff. After a four-point bend test the bar fractures, the engineer images the fracture surface with an SEM and exports an 8-bit PNG. They upload it to FailSafe which returns in ~30 seconds:  segmented defect map, quantitative pore statistics (count, area fraction, aspect ratio, spatial distribution), a defect type classification, a risk level (low/medium/high/critical), and specific recommendations – e.g., “reduce laser power by _% to eliminate keyhole porosity in the 3rd quadrant.” The engineer can act on this without waiting days for an expert to review the SEM image.




<img width="368" height="879" alt="Screenshot 2026-04-25 125025" src="https://github.com/user-attachments/assets/10e92851-40cb-43b3-8a4a-ffb1b0b14bc1" />
















3. System Design: High-level architecture diagram (can be ASCII or image
in repo); where the model sits, how data flows through.



The vision model handles perception and Claude handles the engineering reasoning. The rule-based classifier bridges these and gives Claude a structured classification as context instead of raw pixels.
4. Data: Source(s), size, cleaning, splits.

The data source used to train the SegFormer model is an OSF Materials Data Segmentation Benchmark for Ti-6Al04V LPBF), from https://osf.io/gdwyb/overview. There are 300 images total across three subsets: lack_of_fusion (100), keyhole (100), and all_defects (100). Each of these are paired with pixel-level binary masks. 
The original images are 1024 x 1024 16-bit grayscale, with a pixel range 1328-65535. These were converted to 8-bit PNG via per-image min-max normalization for compatibility with Gradio’s upload component. The masks are binary (0/255), so no cleaning was required. The splits are 8020 train/val per subset (80 training images). No held-out test set was used, which is a known limitation given the small dataset size. The defect pixels are 1-8% of each image, which requies special handles (see next section, Models).

5. Models: What model(s) you used: agent workflow on top of frontier
models, supervised model(s), generative model(s), your own
nanoGPT variant, or pre-trained models from elsewhere; include any
fine-tuning, prompting, or workflow-design strategies.

For segmentation the SegFormer-b0 (fine-tuned) model was used:
Pre-trained nvidia/mit-b0 (ImageNet), fine tuned for binary semantic segmentation on each OSF subset separately.
Chosen over UNet because it uses a hierarchical Mix Transformer encoder, which captures multi-scale features, like fine crack edges + global fracture context, and can viably run on a CPU.
Input: 256 x 256 RGB, Output: binary defect mask
Loss: combined dice + weighted cross-entropy (0.5 * CE(w=[1,4]) + 0.5 * Dice) to counter class imbalances. Naive CE collapsed to all-background, but dice loss fixed this.
Differential LR: decade the head at 3 *10^-3 (50x the encoder’s 6 *10^-5) so the randomly-initialized head updates without destabilizing the pretrained weights.
Trained 15 epochs, with batch size 2, on CPU (trained for ~2 hours total across all three subsets)

For reasoning, Claude claude-sonnet-4-20250514 was used:
Receives a structured JSON with material metadata, all morphological features, and the rule-based classification
Returns a JSON diagnosis with fixed fields: summary, defect interpretation, risk level, rationale, failure mechanism, critical quadrants, recommendations, and confidence.
Adds value that the rule-based classifier cannot: it handles ambiguous aspect ratio ranges (1.3-1.6) and connects pore statistics to fracture mechanics and LPBF process physics.

6. Evaluation: Quantitative metrics where possible; qualitative assessment
(e.g., user examples, error analysis).

Quantitative (segmentation mIoU (Mean Intersection over Union)): 

Subset
Best val mIoU
lack_of_fusion
0.613
keyhole
0.619
all_defects
0.603


The published SegFormer fractography benchmark for mIoU (ScienceDirect 2024) is mIoU = 0.597. IoU (intersection over union) = True positives + false positives + false negatives). mIoU averages this across both classes, where a score of 1.0 is perfect, and 0.0 means there is no overlap between predicted and ground-truth masks.
Pixel accuracy is misleading here because defects are only 1-8% of each image. A model that predicts all background achieves around 94% accuracy while detecting nothing. mIoU penalizes this: missing all defect pixels collapses the defect-class IoU to 0, which drags the mean down regardless of the background accuracy. This is why mIoU is the meaningful metric for this task. An mIoU of ~0.61 means that the model’s predicted defect regions overlap with ground-truth masks about 61% of the time by area, after penalizing false positives (flagging healthy material) and false negatives (missing real defects). For a safety critical application, false negative (missed defects) are the more dangerous failure mode, which is why the loss function is weighted 4x more towards the defect class.
Qualitatively, the rule-based classifier performs well at the extremes (where there are clearly circular keyhole pores or clearly elongated lack of fusion pores), but struggles in the 1.3-1.6 aspect ratio range where defect types overlap. Furthermore, Claude’s reasoning outputs are coherent and materially grounded in examples reviewed manually, correctly connecting high area fraction and clustered quadrant distribution to elevated fatigue risk, and mapping keyhole porosity to excessive laser energy and recommending the appropriate next steps.

7. Limitations & Risks: Failure modes, biases, data issues, privacy concerns.

A major limitation is the relatively small training set: 80 images per subset is small for a deep learning model, so the model may overfit or fail to generalize to different SEM imaging conditions. Furthermore, Val mIoU is the only reported metric which risks overfitting to the validation split during hyperparameter tuning. The model is trained exclusively on Ti-6AL-4V LPBF, so performance on other alloys or other AM processes is untested. Additionally, 16-bit TIF inputs, which are the native SEM format are not supported for the Gradio web interface, requiring a preprocessing step to 8-bit that might introduce information loss. Fixed thresholds on aspect ratio could also misclassify ambiguous morphologies, which is a known failure mode when the mean AR falls between 1.3-1.6.

The Claude LLM reasoning layer has no ground-truth validation, so if given unusual combinations outside of its training distribution, it may give materially incorrect recommendations. Finally, there are privacy concerns if a user uploads proprietary SEM images, which can be logged by the public HuggingFace Space.

8. Next Steps: What you’d do with 2–3 more months (technical and
product).

Technical:
We would fine-tune the model on a GPU at 512x512 input resolution and incorporate an additional dataset from smartFRACS, https://huggingface.co/datasets/smartFRACs/material_fracturing/tree/main, for fatigue striation and crack propagation features, targeting mIoU > 0.7. 
We would replace the rule-based classifier with a small MLP trained on extracted morphological features with a proper test set for evaluation
We would add a 16-bit TIF -> 8-bit server side preprocessing step so engineers can upload unprocessed SEM images directly

Product:
We would build a structured results dashboard with exportable PDF reports which could be included in engineering disposition packages and reports.
We would add a batch mode for processing full inspection lots, which would output pass/fail flags against configurable aerospace acceptance thresholds for a batch of SEM images instead of individual images
We would extend to IN718 and AL7075 alloys to broaden applicability beyond Ti-6Al-4V.
