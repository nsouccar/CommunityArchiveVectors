#!/usr/bin/env python3
"""
Build on Ephemeral Disk Strategy

This script:
1. Copies existing 5.4M database from Modal Volume to ephemeral disk
2. Copies batch files to ephemeral disk
3. Continues building batches 55-64 on fast local ephemeral disk
4. Generates complete metadata
5. Copies final completed database back to Modal Volume

Why this works:
- Ephemeral disk: 150GB of fast local NVMe storage
- 50-100x faster than network storage (0.1ms vs 1-5ms latency)
- Plenty of space for build + compaction overhead
- Only one final copy to persistent storage
"""

import modal
import os
import time
from datetime import datetime

app = modal.App("build-on-ephemeral")

image = (
    modal.Image.debian_slim()
    .pip_install("corenn-py", "numpy")
)

vector_volume = modal.Volume.from_name("tweet-vectors-volume", create_if_missing=False)


@app.function(
    image=image,
    volumes={"/persistent": vector_volume},
    ephemeral_disk=524288,  # 512GB ephemeral disk (minimum allowed)
    cpu=8.0,
    memory=32768,  # 32 GB RAM
    timeout=21600,  # 6 hours
)
def build_on_ephemeral_disk(start_batch=55):
    """
    Build database on fast ephemeral disk, then save to persistent storage.
    """
    from corenn_py import CoreNN
    import numpy as np
    import pickle
    import glob
    import subprocess

    vector_volume.reload()

    print("="*80)
    print("BUILD ON EPHEMERAL DISK")
    print("="*80 + "\n")

    # Check available space
    print("📊 Storage Overview:\n")
    result = subprocess.run(["df", "-h", "/tmp"], capture_output=True, text=True)
    print("Ephemeral disk:")
    print(result.stdout)

    print("\n" + "="*80)
    print("STEP 1: COPY TO EPHEMERAL DISK")
    print("="*80 + "\n")

    # Copy database from persistent to ephemeral
    print("📦 Copying database to ephemeral disk (33GB)...")
    print("   This will take ~5-8 minutes...")
    copy_start = time.time()

    result = subprocess.run(
        ["cp", "-r", "/persistent/corenn_db", "/tmp/"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"❌ Failed to copy database: {result.stderr}")
        return False, 0

    copy_time = time.time() - copy_start
    print(f"✅ Database copied in {copy_time:.1f}s\n")

    # Copy batch files
    print("📦 Copying batch files to ephemeral disk (57GB)...")
    print("   This will take ~8-12 minutes...")
    copy_start = time.time()

    result = subprocess.run(
        ["cp", "-r", "/persistent/batches", "/tmp/"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"❌ Failed to copy batches: {result.stderr}")
        return False, 0

    copy_time = time.time() - copy_start
    print(f"✅ Batch files copied in {copy_time:.1f}s\n")

    # Verify space after copy
    result = subprocess.run(["df", "-h", "/tmp"], capture_output=True, text=True)
    print("Ephemeral disk after copy:")
    print(result.stdout)

    print("\n" + "="*80)
    print("STEP 2: RESUME BUILDING ON EPHEMERAL")
    print("="*80 + "\n")

    # Open database on ephemeral
    db_path = "/tmp/corenn_db"
    batch_files = sorted(glob.glob("/tmp/batches/batch_*.pkl"))

    print(f"🔓 Opening database on ephemeral disk...")
    db = CoreNN.open(db_path)
    print("✅ Database opened successfully\n")

    existing_vectors = (start_batch - 1) * 100_000
    print(f"📊 Database has: {existing_vectors:,} vectors")
    print(f"🎯 Adding batches {start_batch}-{len(batch_files)}\n")

    # Build remaining batches on fast ephemeral disk
    vectors_added = 0
    insert_times = []
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

        # Normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vectors = vectors / norms

        # Insert on fast ephemeral disk!
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
    print("STEP 3: GENERATE METADATA")
    print("="*80 + "\n")

    # Generate metadata from ALL batches
    print("📝 Reading metadata from all batches...")
    all_metadata = []

    for batch_file in batch_files:
        with open(batch_file, "rb") as f:
            batch_data = pickle.load(f)
            all_metadata.extend(batch_data["metadata"])

    print(f"✅ Loaded metadata for {len(all_metadata):,} vectors\n")

    print("💾 Saving metadata on ephemeral...")
    metadata_dict = {meta["tweet_id"]: meta for meta in all_metadata}
    with open("/tmp/metadata.pkl", "wb") as f:
        pickle.dump({"metadata": metadata_dict, "count": len(all_metadata)}, f)

    print("✅ Metadata saved\n")

    print("\n" + "="*80)
    print("STEP 4: COPY BACK TO PERSISTENT STORAGE")
    print("="*80 + "\n")

    # Copy completed database back to persistent storage
    print("📦 Copying completed database to persistent storage...")
    print("   (This is the final completed 6.4M database)")
    print("   This will take ~8-12 minutes...")
    copy_start = time.time()

    # First, backup old database
    if os.path.exists("/persistent/corenn_db"):
        print("   📦 Backing up old database...")
        subprocess.run(["mv", "/persistent/corenn_db", "/persistent/corenn_db.backup"])

    # Copy new database
    result = subprocess.run(
        ["cp", "-r", "/tmp/corenn_db", "/persistent/"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"❌ Failed to copy database back: {result.stderr}")
        # Restore backup
        subprocess.run(["mv", "/persistent/corenn_db.backup", "/persistent/corenn_db"])
        return False, 0

    copy_time = time.time() - copy_start
    print(f"✅ Database copied back in {copy_time:.1f}s\n")

    # Copy metadata
    print("📦 Copying metadata to persistent storage...")
    result = subprocess.run(
        ["cp", "/tmp/metadata.pkl", "/persistent/"],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("✅ Metadata copied\n")

    # Commit to volume
    print("💾 Committing changes to Modal Volume...")
    vector_volume.commit()
    print("✅ Changes committed\n")

    print("\n" + "="*80)
    print("✅ BUILD COMPLETE!")
    print("="*80)
    print(f"Vectors added this session: {vectors_added:,}")
    print(f"Total vectors in database: {existing_vectors + vectors_added:,}")
    print(f"Batches processed: {start_batch}-{len(batch_files)}")
    if insert_times:
        print(f"\nInsertion performance on ephemeral disk:")
        print(f"  Average insert time: {sum(insert_times)/len(insert_times):.1f}s")
        print(f"  Fastest insert: {min(insert_times):.1f}s")
        print(f"  Slowest insert: {max(insert_times):.1f}s")
    print("\nDatabase and metadata saved to persistent storage!")
    print("="*80 + "\n")

    return True, existing_vectors + vectors_added


@app.local_entrypoint()
def main():
    """
    Build remaining batches on ephemeral disk, then save to persistent storage.
    """
    start_time = datetime.now()

    print("="*80)
    print("EPHEMERAL DISK BUILD STRATEGY")
    print("="*80)
    print("\nStrategy:")
    print("1. Copy 5.4M database to fast ephemeral disk (~8 min)")
    print("2. Copy batch files to ephemeral disk (~10 min)")
    print("3. Build remaining batches on ephemeral (~10 min)")
    print("4. Copy completed 6.4M database back to persistent (~10 min)")
    print("\nEstimated total time: ~40 minutes")
    print("="*80 + "\n")

    success, total_vectors = build_on_ephemeral_disk.remote(start_batch=55)

    elapsed = datetime.now() - start_time

    if success:
        print(f"🎉 SUCCESS! Database has {total_vectors:,} vectors")
        print(f"⏱️  Total time: {elapsed}\n")
        print("Your 6.4M vector database is ready to use!")
    else:
        print(f"❌ Build failed after {elapsed}\n")
