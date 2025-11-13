#!/usr/bin/env python3
"""
Inspect the actual structure of a single community in organized_by_community.pkl
"""

import modal
import pickle
from pathlib import Path

app = modal.App("inspect-community-structure")
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy")
volume = modal.Volume.from_name("tweet-vectors-large", create_if_missing=False)

@app.function(volumes={"/data": volume}, image=image)
def inspect_structure():
    organized_path = Path("/data/organized_by_community.pkl")

    print("Loading organized data...")
    with open(organized_path, 'rb') as f:
        organized = pickle.load(f)

    print("\n" + "="*80)
    print("INSPECTING A SINGLE COMMUNITY")
    print("="*80)

    # Get 2022 data
    year_2022 = organized['2022']
    first_month = list(year_2022.keys())[0]
    month_data = year_2022[first_month]
    first_comm_id = list(month_data.keys())[0]
    community_data = month_data[first_comm_id]

    print(f"\nYear: 2022, Month: {first_month}, Community: {first_comm_id}")
    print(f"Community data type: {type(community_data)}")

    if isinstance(community_data, dict):
        print(f"\nCommunity data keys: {list(community_data.keys())}")
        for key, value in community_data.items():
            print(f"\n  Key: {key!r}")
            print(f"    Type: {type(value)}")
            if hasattr(value, '__len__') and not isinstance(value, str):
                print(f"    Length: {len(value)}")
                if hasattr(value, 'shape'):
                    print(f"    Shape: {value.shape}")
                # Show first few items if it's a list/array
                if isinstance(value, list) and len(value) > 0:
                    print(f"    First 3 items: {value[:3]}")
            else:
                print(f"    Value: {value}")

    elif hasattr(community_data, 'shape'):
        # It's a numpy array
        print(f"\nDirect numpy array")
        print(f"  Shape: {community_data.shape}")
        print(f"  Dtype: {community_data.dtype}")

    else:
        print(f"\nCommunity data: {community_data}")

@app.local_entrypoint()
def main():
    inspect_structure.remote()
