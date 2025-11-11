"""
Quick diagnostic script to check what keys are in the batch metadata
"""

import modal
import pickle

app = modal.App("check-metadata")
volume = modal.Volume.from_name("tweet-vectors-large", create_if_missing=False)

@app.function(volumes={"/data": volume}, timeout=300)
def check_first_batch():
    """Check the keys in the first batch file"""
    from pathlib import Path

    print("Checking metadata structure in batch files...\n")

    # First, list files in the batches directory
    batches_dir = Path("/data/batches")
    print(f"Contents of {batches_dir}:")
    batch_files = sorted(batches_dir.glob("*.pkl"))
    for f in batch_files[:5]:
        print(f"  {f.name}")
    print(f"  ... (total {len(batch_files)} files)\n")

    if not batch_files:
        print("ERROR: No batch files found!")
        return

    # Use the first batch file
    batch_file = batch_files[0]
    print(f"Reading {batch_file.name}...\n")

    with open(batch_file, 'rb') as f:
        batch_data = pickle.load(f)

    metadata = batch_data['metadata']

    print(f"Total tweets in batch 1: {len(metadata)}")
    print("\nFirst tweet metadata keys:")
    print(list(metadata[0].keys()))
    print("\nFirst tweet full content:")
    print(metadata[0])
    print("\n" + "="*80)

    # Check a few more tweets to see if the structure is consistent
    print("\nChecking keys in first 5 tweets:")
    for i in range(min(5, len(metadata))):
        keys = list(metadata[i].keys())
        print(f"  Tweet {i+1}: {keys}")

    # Try to find which key contains the tweet text
    print("\n" + "="*80)
    print("Looking for tweet text content...")

    for key in metadata[0].keys():
        value = metadata[0].get(key)
        if isinstance(value, str) and len(value) > 20:  # Likely to be tweet text
            print(f"\nKey '{key}' contains:")
            print(f"  {value[:200]}...")  # First 200 chars

@app.local_entrypoint()
def main():
    check_first_batch.remote()
