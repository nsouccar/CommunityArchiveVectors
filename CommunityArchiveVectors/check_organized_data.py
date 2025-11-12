#!/usr/bin/env python3
"""
Check what years exist in organized_by_community.pkl on Modal volume
"""

import modal
import pickle
from pathlib import Path

app = modal.App("check-organized-data")
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy")
volume = modal.Volume.from_name("tweet-vectors-large", create_if_missing=False)

@app.function(volumes={"/data": volume}, image=image)
def check_data():
    organized_path = Path("/data/organized_by_community.pkl")

    if not organized_path.exists():
        print("organized_by_community.pkl does not exist!")
        return

    print("Loading organized data...")
    with open(organized_path, 'rb') as f:
        organized = pickle.load(f)

    print("\n" + "="*60)
    print("Years available in organized_by_community.pkl:")
    print("="*60)
    for year in sorted(organized.keys()):
        months = len(organized[year])
        total_communities = sum(len(organized[year][month]) for month in organized[year])
        print(f"  {year}: {months} months, {total_communities} total communities")

    print("\n" + "="*60)
    print("Checking for existing topic files...")
    print("="*60)
    data_dir = Path("/data")

    topic_files = sorted(data_dir.glob("topics_year_*.pkl"))
    print(f"\nFound {len(topic_files)} topic files:")
    for f in topic_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name} ({size_mb:.1f} MB)")

@app.local_entrypoint()
def main():
    check_data.remote()
