#!/usr/bin/env python3
"""
Reconstruct metadata.pkl from batch files.

This reads the metadata from batches 1-54 (the ones that were successfully
inserted before the crash) and creates the metadata.pkl file.
"""

import modal

app = modal.App("reconstruct-metadata")

image = modal.Image.debian_slim()

vector_volume = modal.Volume.from_name("tweet-vectors-volume", create_if_missing=False)


@app.function(
    image=image,
    volumes={"/data": vector_volume},
    timeout=600,
)
def reconstruct_metadata(num_batches=54):
    """
    Reconstruct metadata from batch files.

    Args:
        num_batches: How many batches were successfully inserted (default 54)
    """
    import pickle
    import glob
    import os

    vector_volume.reload()

    print("="*80)
    print("RECONSTRUCTING METADATA")
    print("="*80 + "\n")

    # Find all batch files
    batch_files = sorted(glob.glob("/data/batches/batch_*.pkl"))

    if len(batch_files) == 0:
        print("❌ No batch files found!")
        return False

    print(f"📦 Found {len(batch_files)} total batch files")
    print(f"📝 Reconstructing metadata from batches 1-{num_batches}\n")

    # Read metadata from successfully inserted batches
    all_metadata = []

    for batch_num in range(1, num_batches + 1):
        batch_file = f"/data/batches/batch_{batch_num:04d}.pkl"

        if not os.path.exists(batch_file):
            print(f"⚠️  Batch {batch_num} not found: {batch_file}")
            continue

        print(f"Reading batch {batch_num}/{num_batches}...", end="\r")

        with open(batch_file, "rb") as f:
            batch_data = pickle.load(f)
            metadata = batch_data["metadata"]
            all_metadata.extend(metadata)

    print(f"\n✅ Loaded metadata for {len(all_metadata):,} vectors\n")

    # Create metadata dictionary (keyed by tweet_id)
    print("📊 Creating metadata dictionary...")
    metadata_dict = {meta["tweet_id"]: meta for meta in all_metadata}

    print(f"✅ Created dictionary with {len(metadata_dict):,} unique tweets\n")

    # Save metadata
    print("💾 Saving metadata to /data/metadata.pkl...")
    metadata_pkg = {
        "metadata": metadata_dict,
        "count": len(all_metadata)
    }

    with open("/data/metadata.pkl", "wb") as f:
        pickle.dump(metadata_pkg, f)

    print("✅ Metadata saved successfully\n")

    # Commit to volume
    vector_volume.commit()

    print("="*80)
    print("✅ METADATA RECONSTRUCTION COMPLETE!")
    print("="*80)
    print(f"Total vectors: {len(all_metadata):,}")
    print(f"Unique tweets: {len(metadata_dict):,}")
    print(f"Batches processed: 1-{num_batches}")
    print("="*80 + "\n")

    return True


@app.local_entrypoint()
def main():
    """
    Reconstruct metadata for the 5.4M vectors (batches 1-54).
    """
    print("\nThis will reconstruct the metadata for batches 1-54")
    print("(the 5.4M vectors that were successfully inserted)\n")

    success = reconstruct_metadata.remote(num_batches=54)

    if success:
        print("🎉 Success! Your 5.4M vector database now has complete metadata.")
        print("You can now use the database for searches.\n")
    else:
        print("❌ Failed to reconstruct metadata.\n")
