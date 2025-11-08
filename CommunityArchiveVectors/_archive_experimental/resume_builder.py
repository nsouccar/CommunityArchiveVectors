#!/usr/bin/env python3
"""
Resume Database Builder - Continue from batch 55 where we left off

This script:
1. Opens the existing 5.4M vector database (batches 1-54)
2. Continues inserting batches 55-64
3. Regenerates complete metadata from all batch files
"""

import modal
import os
import time
from datetime import datetime

app = modal.App("resume-tweet-builder")

image = (
    modal.Image.debian_slim()
    .pip_install("corenn-py", "numpy")
)

vector_volume = modal.Volume.from_name("tweet-vectors-large", create_if_missing=False)


@app.function(
    image=image,
    volumes={"/data": vector_volume},
    cpu=8.0,
    memory=32768,  # 32 GB RAM
    timeout=21600,  # 6 hour timeout
)
def resume_database_build(start_batch=55):
    """
    Resume building the database from a specific batch.

    Args:
        start_batch: Which batch to start from (default 55)
    """
    from corenn_py import CoreNN
    import numpy as np
    import pickle
    import glob
    import time

    vector_volume.reload()

    # Find all batch files
    batch_files = sorted(glob.glob("/data/batches/batch_*.pkl"))

    if len(batch_files) == 0:
        print("❌ No batch files found!")
        return False, 0

    print(f"📦 Found {len(batch_files)} batch files")
    print(f"🔄 Resuming from batch {start_batch}\n")
    print("="*80)
    print("RESUMING DATABASE BUILD")
    print("="*80 + "\n")

    db_path = "/data/corenn_db"

    # Check if database exists
    if not os.path.exists(db_path):
        print("❌ Database not found! Cannot resume.")
        return False, 0

    # Open existing database
    print("🔓 Opening existing database...")
    db = CoreNN.open(db_path)
    print("✅ Database opened successfully\n")

    # Track what we're adding
    vectors_added = 0
    insert_times = []

    # Calculate starting vector count
    existing_vectors = (start_batch - 1) * 100_000
    print(f"📊 Database already has: {existing_vectors:,} vectors")
    print(f"🎯 Adding batches {start_batch}-{len(batch_files)}\n")

    # Process remaining batches
    remaining_batches = batch_files[start_batch-1:]

    for batch_num, batch_file in enumerate(remaining_batches, start_batch):
        batch_start = time.time()

        print(f"Batch {batch_num}/{len(batch_files)}: {batch_file}")

        # Load batch data
        with open(batch_file, "rb") as f:
            batch_data = pickle.load(f)

        embeddings = batch_data["embeddings"]
        metadata = batch_data["metadata"]

        # Prepare vectors
        keys = [meta["tweet_id"] for meta in metadata]
        vectors = np.array(embeddings, dtype=np.float32)

        # Normalize for cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vectors = vectors / norms

        # Insert vectors
        insert_start = time.time()
        print(f"   📥 Inserting {len(keys):,} vectors...")

        db.insert_f32(keys, vectors)

        insert_time = time.time() - insert_start
        insert_times.append(insert_time)

        print(f"   ✅ Inserted in {insert_time:.1f}s ({len(keys)/insert_time:.0f} vectors/sec)")

        vectors_added += len(keys)
        total_vectors = existing_vectors + vectors_added

        batch_time = time.time() - batch_start
        print(f"   ⏱️  Total batch time: {batch_time:.1f}s")
        print(f"   📊 Database now has: {total_vectors:,} vectors\n")

    print("\n" + "="*80)
    print("REGENERATING COMPLETE METADATA")
    print("="*80 + "\n")

    # Regenerate metadata from ALL batch files (1-64)
    print("📝 Reading metadata from all batches...")
    all_metadata = []

    for batch_file in batch_files:
        with open(batch_file, "rb") as f:
            batch_data = pickle.load(f)
            all_metadata.extend(batch_data["metadata"])

    print(f"✅ Loaded metadata for {len(all_metadata):,} vectors\n")

    # Save complete metadata
    print("💾 Saving complete metadata...")
    metadata_dict = {meta["tweet_id"]: meta for meta in all_metadata}
    with open("/data/metadata.pkl", "wb") as f:
        pickle.dump({"metadata": metadata_dict, "count": len(all_metadata)}, f)

    print("✅ Metadata saved\n")

    vector_volume.commit()

    print("\n" + "="*80)
    print("✅ BUILD COMPLETE!")
    print("="*80)
    print(f"Vectors added this session: {vectors_added:,}")
    print(f"Total vectors in database: {existing_vectors + vectors_added:,}")
    print(f"Batches processed: {start_batch}-{len(batch_files)}")
    if insert_times:
        print(f"Average insert time: {sum(insert_times)/len(insert_times):.1f}s")
        print(f"Slowest insert: {max(insert_times):.1f}s")
        print(f"Fastest insert: {min(insert_times):.1f}s")
    print("="*80 + "\n")

    return True, existing_vectors + vectors_added


@app.function(
    image=image,
    volumes={"/data": vector_volume},
    timeout=300,
)
def check_volume_space():
    """
    Check available space on the Modal Volume.
    """
    import subprocess

    vector_volume.reload()

    print("="*80)
    print("CHECKING VOLUME SPACE")
    print("="*80 + "\n")

    # Check disk usage
    result = subprocess.run(["df", "-h", "/data"], capture_output=True, text=True)
    print(result.stdout)

    # Check database size
    if os.path.exists("/data/corenn_db"):
        result = subprocess.run(["du", "-sh", "/data/corenn_db"], capture_output=True, text=True)
        print(f"Current database size: {result.stdout.strip()}")

    # Check batch files size
    if os.path.exists("/data/batches"):
        result = subprocess.run(["du", "-sh", "/data/batches"], capture_output=True, text=True)
        print(f"Batch files size: {result.stdout.strip()}")

    print("\n" + "="*80 + "\n")


@app.local_entrypoint()
def main():
    """
    Resume building the database from where we left off.
    """
    start_time = datetime.now()

    print("="*80)
    print("RESUME CORENN DATABASE BUILDER")
    print("="*80 + "\n")

    # Step 1: Check available space
    print("Step 1: Checking available storage space...\n")
    check_volume_space.remote()

    # Step 2: Resume from batch 55
    print("Step 2: Resuming database build from batch 55...\n")
    success, total_vectors = resume_database_build.remote(start_batch=55)

    elapsed = datetime.now() - start_time

    if success:
        print(f"🎉 SUCCESS! Database now has {total_vectors:,} vectors")
        print(f"⏱️  Resume time: {elapsed}\n")
    else:
        print(f"❌ Build failed after {elapsed}\n")
