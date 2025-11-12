#!/usr/bin/env python3
"""
Debug script to check the structure of organized_by_community.pkl
"""

import modal
import pickle
from pathlib import Path

app = modal.App("debug-organized-structure")
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy")
volume = modal.Volume.from_name("tweet-vectors-large", create_if_missing=False)

@app.function(volumes={"/data": volume}, image=image)
def debug_structure():
    organized_path = Path("/data/organized_by_community.pkl")

    print("Loading organized data...")
    with open(organized_path, 'rb') as f:
        organized = pickle.load(f)

    print("\n" + "="*60)
    print("Top-level keys in organized:")
    print("="*60)
    for key in list(organized.keys())[:15]:  # First 15 keys
        print(f"  Key: {key!r} (type: {type(key).__name__})")

    print("\n" + "="*60)
    print("Testing year lookups:")
    print("="*60)

    target_years = [2020, 2021, 2022, 2023, 2024]
    for year in target_years:
        exists = year in organized
        print(f"  {year} (int) in organized: {exists}")

        # Also try string version
        str_year = str(year)
        exists_str = str_year in organized
        print(f"  {str_year!r} (str) in organized: {exists_str}")

    # Check if any year keys match
    print("\n" + "="*60)
    print("Checking which years are actually in organized:")
    print("="*60)
    year_like_keys = [k for k in organized.keys() if isinstance(k, (int, str)) and (str(k).isdigit() if isinstance(k, str) else True)]
    for key in sorted(year_like_keys)[:20]:
        print(f"  Found: {key!r} (type: {type(key).__name__})")

@app.local_entrypoint()
def main():
    debug_structure.remote()
