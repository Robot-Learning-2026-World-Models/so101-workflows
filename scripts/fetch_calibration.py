#!/usr/bin/env python3
"""Fetch so101-calibration dataset from HuggingFace into ./calibration/"""

from huggingface_hub import snapshot_download
import os

REPO_ID = "rl26-world-models/so101-calibration"
DEST = os.path.join(os.path.dirname(os.path.dirname(__file__)), "calibration")

if __name__ == "__main__":
    print(f"Downloading {REPO_ID} → {DEST}")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=DEST,
        ignore_patterns=["*.gitattributes", ".gitattributes"],
    )
    print("Done.")
