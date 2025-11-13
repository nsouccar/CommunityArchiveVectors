#!/usr/bin/env python3
"""
Download organized_by_community.pkl from Modal volume to local machine
WARNING: This is a very large file (10-20+ GB)
"""

import modal
import pickle
from pathlib import Path

app = modal.App("download-organized-pickle")
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy")
volume = modal.Volume.from_name("tweet-vectors-large", create_if_missing=False)

@app.function(volumes={"/data": volume}, image=image)
def download_pickle():
    """Download the organized_by_community.pkl file"""
    import os

    organized_path = Path("/data/organized_by_community.pkl")

    if not organized_path.exists():
        print("Error: organized_by_community.pkl does not exist!")
        return None

    # Get file size
    file_size_bytes = organized_path.stat().st_size
    file_size_gb = file_size_bytes / (1024**3)

    print(f"\n{'='*80}")
    print(f"DOWNLOADING organized_by_community.pkl")
    print(f"{'='*80}")
    print(f"File size: {file_size_gb:.2f} GB ({file_size_bytes:,} bytes)")
    print(f"This may take several minutes...")
    print(f"{'='*80}\n")

    # Read the pickle file
    print("Reading pickle file from Modal volume...")
    with open(organized_path, 'rb') as f:
        data = pickle.load(f)

    print("✓ Pickle file loaded successfully")

    return data

@app.local_entrypoint()
def main():
    print("\n" + "="*80)
    print("STARTING DOWNLOAD")
    print("="*80)

    # Download from Modal
    data = download_pickle.remote()

    if data is None:
        print("Download failed!")
        return

    # Save locally
    local_path = Path("organized_by_community.pkl")

    print("\nSaving to local file...")
    with open(local_path, 'wb') as f:
        pickle.dump(data, f)

    # Get local file size
    file_size_bytes = local_path.stat().st_size
    file_size_gb = file_size_bytes / (1024**3)

    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE!")
    print("="*80)
    print(f"Saved to: {local_path.absolute()}")
    print(f"File size: {file_size_gb:.2f} GB ({file_size_bytes:,} bytes)")
    print(f"\nTo open in Finder, run:")
    print(f"  open -R {local_path.absolute()}")
    print("="*80 + "\n")
