# FailSafe

> End-to-end failure analysis for LPBF aerospace components · SegFormer + Claude · OSF Ti-64 Dataset

**Live demo:** https://huggingface.co/spaces/rcrane4/FailSafe

---

## Requirements

- Python 3.9+
- pip
- An Anthropic API key ([get one here](https://console.anthropic.com/))

---

## 1. Clone the repo

```bash
git clone https://github.com/rcrane4/mae301-2026spring-failsafe.git
cd mae301-2026spring-failsafe/phase2
```

---

## 2. Set up the environment

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies
python setup.py
```

---

## 3. Set your API key

FailSafe uses the Claude API for its engineering reasoning layer. You must set your Anthropic API key as an environment variable before running the app.

```bash
# Mac/Linux
export ANTHROPIC_API_KEY=sk-ant-...

# Windows PowerShell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

> **Where to get a key:** https://console.anthropic.com/ → API Keys → Create Key
>
> **Never commit your key to the repo.** If you want to persist it locally, add it to a `.env` file and load it with `python-dotenv` — but make sure `.env` is in your `.gitignore`.

---

## 4. Download the dataset

```bash
python download_osf.py
```

If the automatic download fails, manually download from https://osf.io/gdwyb/ and extract into:

```
data/lack_of_fusion/images/
data/lack_of_fusion/masks/
data/keyhole/images/
data/keyhole/masks/
data/all_defects/images/
data/all_defects/masks/
```

Then convert the 16-bit SEM images to 8-bit:

```bash
python -c "
import numpy as np
from PIL import Image
from pathlib import Path
for subset in ['all_defects', 'lack_of_fusion', 'keyhole']:
    for folder in ['images', 'masks']:
        src = Path(f'data/{subset}/{folder}')
        dst = Path(f'data/{subset}/{folder}_8bit')
        dst.mkdir(exist_ok=True)
        for p in src.glob('*.png'):
            arr = np.array(Image.open(p), dtype=np.float32)
            if folder == 'images':
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
                Image.fromarray((arr * 255).astype(np.uint8)).save(dst / p.name)
            else:
                Image.open(p).save(dst / p.name)
"
```

---

## 5. Run the minimal demo

Pre-trained checkpoints are included in `checkpoints/`. You do **not** need to retrain to run the demo.

```bash
python app.py
```

Then open **http://localhost:7860** in your browser.

**OR JUST OPEN** https://huggingface.co/spaces/rcrane4/FailSafe

Upload any 8-bit PNG SEM fractograph and the pipeline will return:
- Segmented defect mask
- Morphological feature stats
- Defect type classification (lack-of-fusion / keyhole / mixed)
- Crack initiation risk level
- Engineering recommendations from Claude

> **Image requirements:** 8-bit PNG only. 16-bit TIFs are not supported via the web interface.

---

## 6. Optional: retrain from scratch

```bash
# Inspect dataset first
python inspect_dataset.py

# Train all three subsets (15 epochs each, ~2 hrs on CPU)
python train.py --epochs 15

# Visualize predictions
python inference.py --subset all
```

---

## Project structure

```
phase2/
├── app.py                 Gradio web interface (start here)
├── diagnose.py            Claude API reasoning layer
├── features.py            Feature extraction + rule-based classifier
├── train.py               SegFormer fine-tuning
├── inference.py           Prediction visualization
├── dataset.py             PyTorch Dataset class
├── download_osf.py        Dataset downloader
├── inspect_dataset.py     Dataset inspection utility
├── setup.py               Dependency installer
├── requirements.txt       HuggingFace Spaces dependencies
├── data/                  SEM images + masks (downloaded separately)
├── checkpoints/           Pre-trained model weights
└── output/                Inference, feature, and diagnosis outputs
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ANTHROPIC_API_KEY not set` | Run the `export` command in step 3 before launching the app |
| App launches but diagnosis fails | Check that your API key is valid at console.anthropic.com |
| Image uploads as all-white | You uploaded a 16-bit image — convert to 8-bit PNG first |
| `download_osf.py` fails | Download manually from https://osf.io/gdwyb/ and extract per step 4 |
| Low mIoU / model predicts all background | Ensure you are using the provided checkpoints, not a partially-trained model |
