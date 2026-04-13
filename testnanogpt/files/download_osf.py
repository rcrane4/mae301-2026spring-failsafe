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
OSF_PROJECT_ID = "gdwyb"  # top-level fractography project

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
        print(f"  [skip] {dest.name} already exists")
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
        print(f"  ⚠️  Could not list files: {e}")
        print("  → You may need to download manually from https://osf.io/gdwyb/")
        return []

    print(f"  Found {len(files)} files")
    for f in files:
        # strip leading slash from materialized path
        rel_path = f["path"].lstrip("/")
        dest = local_root / rel_path
        print(f"  ↓ {rel_path} ({f['size'] / 1024:.1f} KB)")
        try:
            download_file(f["download"], dest)
        except Exception as e:
            print(f"    ⚠️  Failed: {e}")
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
        print("\n⚠️  Automatic download failed.")
        print("Manual download steps:")
        print("  1. Go to https://osf.io/gdwyb/")
        print("  2. Click each sub-component (Lack of Fusion, Key Hole, All Defects)")
        print("  3. Download the zip and extract into data/<subfolder>/")
        print("     Expected structure:")
        print("       data/lack_of_fusion/images/*.png (or .tif)")
        print("       data/lack_of_fusion/masks/*.png")
