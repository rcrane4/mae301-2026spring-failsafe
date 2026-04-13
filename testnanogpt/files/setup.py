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
