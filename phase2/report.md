Objective and current MVP definition, What has been built so far, and technical approach: Product that takes in SEM images of Ti64, an alloy used in aircraft engines and race car components, where safety is critical and failure can be catastrophic, so identifying defects in manufacturing is extremely important.
The current MVP makes it so you can upload an SEM image of TI64 titanium via SegFormer, trained on a Ti64 dataset
of images and defect masks, and it will identify the defects and provide information and advice through a Claude server: 

https://huggingface.co/spaces/rcrane4/FailSafe

<img width="596" height="459" alt="image" src="https://github.com/user-attachments/assets/df685236-1ca5-44bd-a0a2-c86b9585d621" />


<img width="762" height="482" alt="image" src="https://github.com/user-attachments/assets/5f81d90b-e1d9-4352-80ef-6c2611d402e8" />

<img width="974" height="867" alt="Screenshot 2026-03-30 222037" src="https://github.com/user-attachments/assets/fdd45d74-a37d-42ea-9aec-4f13cfb996fc" />


OUTPUT: "============================================================
FAILURE ANALYSIS REPORT
Image: uploaded_image
Material: Ti-6Al-4V (LPBF)
============================================================

QUANTITATIVE FEATURES
  Defect area:      17.200%
  Defect count:     123
  Mean aspect ratio:1.648
  Rule-based type:  lack_of_fusion

AI DIAGNOSIS
  Failure mechanism: lack_of_fusion porosity with secondary clustering at high-risk location (bottom-right quadrant)
  Crack init. risk:  HIGH
  Critical regions:  Bottom-right quadrant (57.9% of defect population) is the primary failure-risk zone. This region likely corresponds to either: (a) the final deposited layers (if quadrant mapping follows build progression), or (b) a localized zone of poor thermal contact or beam deflection. Secondary concern: the maximum pore (9036 px²) location within this quadrant acts as a stress concentration hotspot. Top-right quadrant (36.0%) represents secondary risk. Top-left and bottom-left (combined 6.1%) are relatively benign.
  Confidence:        high

SUMMARY
  This Ti-6Al-4V LPBF fracture surface exhibits a substantial defect burden (17.2% area fraction) dominated by lack-of-fusion porosity with highly non-uniform spatial distribution. The bottom-right quadrant concentration (57.9% of defects) combined with large maximum pore size (9036 px²) and high size heterogeneity indicates process control issues during layer deposition, particularly in final build stages or high-stress regions.

DEFECT INTERPRETATION
  The 123 distinct defects with mean aspect ratio of 1.648 confirm predominantly lack-of-fusion morphology rather than spherical keyhole pores (which cluster near 1.0). The mean pore area of 366.6 px² is moderate, but the maximum pore (9036 px²) is ~25× larger, indicating inconsistent inter-layer bonding and powder consolidation. The high size heterogeneity (std dev 1252.3 px²) suggests multiple failure modes: some regions achieved adequate fusion while others experienced localized powder bed discontinuities, beam misalignment, or recoater issues. The spatial spread (std = 81.42 px across 256×256 image) shows clustering rather than random distribution—the 57.9% concentration in bottom-right quadrant is statistically significant and suggests either build direction artifacts (layer accumulation effects) or localized thermal gradient collapse.

RISK RATIONALE
  Multiple risk factors elevate this to HIGH: (1) Defect area fraction of 17.2% exceeds typical accept/reject thresholds for aerospace structural components (commonly 5–10% tolerance); (2) The maximum pore size of 9036 px² represents a critical stress concentration that can initiate cracks under cyclic or high-strain loading—at realistic specimen scaling, this likely corresponds to 200–500 µm equivalent diameter; (3) Non-uniform quadrant distribution (bottom-right dominance) creates localized weakness zones and asymmetric stress state during bending; (4) Lack-of-fusion defects exhibit sharp unbonded walls and angular boundaries, which act as fatigue crack initiation sites with stress concentration factors (Kt) of 3–5×; (5) Aspect ratio of 1.648 indicates plate-like voids oriented parallel to build layers, which are mechanically the weakest orientation for transverse loading.

RECOMMENDATIONS
  1. Reject this component for primary structural aerospace use; downgrade to non-critical secondary structure only, pending further mechanical property validation via tensile/fatigue testing of samples from the same build.
  2. Conduct post-process X-ray micro-CT or acoustic microscopy to establish true 3D defect distribution and confirm whether bottom-right clustering extends into bulk material or is surface-limited; determine whether internal voids exceed 500 µm equivalent diameter.
  3. Root-cause investigation: audit LPBF process parameters for this build—specifically (i) laser power and scan speed uniformity, (ii) powder bed moisture and flowability, (iii) recoater blade contact and speed, (iv) inert gas purity and flow patterns, and (v) part position relative to build platform. Bottom-right clustering suggests directional process failure.
  4. If component must be retained, machine away the bottom 1–2 mm of the part (assuming that region corresponds to bottom-right quadrant) to remove maximum pores, then perform ultrasonic inspection or dye-penetrant testing on critical surfaces.
  5. Implement tighter in-process control: reduce allowed defect area fraction specification from current 17.2% to <8% via parameter optimization (lower scan speeds, higher laser power where thermal stability permits, improved powder delivery).
  6. For future builds, establish baseline defect maps by fractography or CT scanning of witness coupons from the same build platform location; correlate defects with LPBF machine log data (laser power, spot position, thermal signatures) to identify and correct process drift.
  7. If fatigue-critical application, perform conservative S–N testing on this material lot at stress levels 40–50% lower than assumed design strength until defect-conditional fatigue data is established.
============================================================"
___________________________________________________________________

Current limitations: The dataset we trained the model on was originally 16bit, but the website (Gradio) we are using for an interface can only take in 8bit, so we had to convert the 16bit dataset to 8bit. Therefore the model is trained on 8bit images, so images must be converted to 8bit by the user before being uploaded for it to work.

Plan for phase 3: Fix the 8bit image issue, and make further optimizations to the current MVP for a more intuitive UI and more precise model training.

_______________________________________________________________________

We have also provided all files used to train the model and the image dataset used, uploaded to the GitHub in /testnanogpt/
