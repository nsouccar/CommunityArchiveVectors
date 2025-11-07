#!/usr/bin/env python3
"""
Incremental Database Builder - Build CoreNN with 64 sequential 100K inserts

This uses CoreNN's backedge delta strategy by:
1. Creating fresh database with batch 1 (100K vectors)
2. Incrementally adding batches 2-64 using db.insert_f32()
3. All in ONE continuous session (database stays open)
4. Only 64 total insert operations

This should work because:
- Much larger batches than original (100K vs 100 tweets)
- Fresh database with no fragmentation
- CoreNN's backedge deltas handle incremental adds efficiently
"""

import modal
import os
import time
from datetime import datetime

app = modal.App("incremental-tweet-builder")

image = (
    modal.Image.debian_slim()
    .pip_install("corenn-py", "numpy")
)

vector_volume = modal.Volume.from_name("tweet-vectors-volume", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/data": vector_volume},
    timeout=300,
)
def cleanup_old_database():
    """
    Delete old database to start fresh.
    Keeps the batch files (embeddings) - only deletes the CoreNN database.
    """
    import shutil
    import os

    vector_volume.reload()

    print("🗑️  Cleaning up old database...\n")

    # Remove old database
    if os.path.exists("/data/corenn_db"):
        shutil.rmtree("/data/corenn_db")
        print("✅ Deleted old CoreNN database")

    # Remove old metadata
    if os.path.exists("/data/metadata.pkl"):
        os.remove("/data/metadata.pkl")
        print("✅ Deleted old metadata")

    # Keep batch files! They're expensive to regenerate
    batch_count = 0
    if os.path.exists("/data/batches"):
        import glob
        batch_files = glob.glob("/data/batches/batch_*.pkl")
        batch_count = len(batch_files)

    print(f"✅ Kept {batch_count} batch files (embeddings)\n")

    vector_volume.commit()
    return batch_count


@app.function(
    image=image,
    volumes={"/data": vector_volume},
    cpu=8.0,
    memory=32768,  # 32 GB RAM - enough for each operation
    timeout=21600,  # 6 hour timeout (should be plenty)
)
def build_database_incrementally():
    """
    Build database incrementally using 64 batches.

    Strategy:
    - Batch 1: Create new database
    - Batches 2-64: Incremental inserts using backedge deltas
    - Track time for each insert to detect slowdowns
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

    print(f"📦 Found {len(batch_files)} batch files\n")
    print("="*80)
    print("INCREMENTAL DATABASE BUILD")
    print("="*80 + "\n")

    db_path = "/data/corenn_db"
    db = None
    all_metadata = []
    total_vectors = 0
    insert_times = []

    # Process each batch
    for batch_num, batch_file in enumerate(batch_files, 1):
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

        # First batch: CREATE database
        if batch_num == 1:
            print("   🆕 Creating new database...")
            db = CoreNN.create(db_path, {"dim": 1024})
            print("   ✅ Database created")

        # Insert vectors
        insert_start = time.time()
        print(f"   📥 Inserting {len(keys):,} vectors...")

        db.insert_f32(keys, vectors)

        insert_time = time.time() - insert_start
        insert_times.append(insert_time)

        print(f"   ✅ Inserted in {insert_time:.1f}s ({len(keys)/insert_time:.0f} vectors/sec)")

        # Track metadata
        all_metadata.extend(metadata)
        total_vectors += len(keys)

        batch_time = time.time() - batch_start
        print(f"   ⏱️  Total batch time: {batch_time:.1f}s")
        print(f"   📊 Database now has: {total_vectors:,} vectors\n")

        # Check for slowdowns
        if batch_num > 1:
            avg_time = sum(insert_times) / len(insert_times)
            if insert_time > avg_time * 2:
                print(f"   ⚠️  WARNING: Insert time {insert_time:.1f}s is slower than average {avg_time:.1f}s")
                print(f"   This might indicate scaling issues...\n")

    # Save metadata
    print("💾 Saving metadata...")
    metadata_dict = {meta["tweet_id"]: meta for meta in all_metadata}
    with open("/data/metadata.pkl", "wb") as f:
        pickle.dump({"metadata": metadata_dict, "count": total_vectors}, f)

    vector_volume.commit()

    print("\n" + "="*80)
    print("✅ BUILD COMPLETE!")
    print("="*80)
    print(f"Total vectors: {total_vectors:,}")
    print(f"Total batches: {len(batch_files)}")
    print(f"Average insert time: {sum(insert_times)/len(insert_times):.1f}s")
    print(f"Slowest insert: {max(insert_times):.1f}s (batch {insert_times.index(max(insert_times))+1})")
    print(f"Fastest insert: {min(insert_times):.1f}s (batch {insert_times.index(min(insert_times))+1})")
    print("="*80 + "\n")

    return True, total_vectors


@app.local_entrypoint()
def main():
    """
    Main entry point: Clean up old database, then build incrementally.
    """
    start_time = datetime.now()

    print("="*80)
    print("INCREMENTAL CORENN DATABASE BUILDER")
    print("="*80 + "\n")

    # Step 1: Clean up old database
    batch_count = cleanup_old_database.remote()

    if batch_count == 0:
        print("❌ No batch files found! Run offline_builder.py first to generate embeddings.\n")
        return

    print(f"Ready to build database from {batch_count} batches\n")

    # Step 2: Build database incrementally
    success, total_vectors = build_database_incrementally.remote()

    elapsed = datetime.now() - start_time

    if success:
        print(f"🎉 SUCCESS! Built database with {total_vectors:,} vectors")
        print(f"⏱️  Total time: {elapsed}\n")
    else:
        print(f"❌ Build failed after {elapsed}\n")
