#!/usr/bin/env python3
"""
Upload Embeddings to Supabase
Reads embeddings from Modal batch files and uploads to Supabase tweets table
"""

import modal
import os

app = modal.App("upload-to-supabase")

image = modal.Image.debian_slim().pip_install("supabase")
volume = modal.Volume.from_name("tweet-vectors-volume", create_if_missing=False)
secrets = modal.Secret.from_name("tweet-vectors-secrets")

@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=[secrets],
    timeout=7200,  # 2 hours per batch
)
def upload_batch_to_supabase(batch_number: int):
    """
    Upload one batch file of embeddings to Supabase
    """
    from supabase import create_client
    import pickle
    import time

    print(f"\n{'='*80}")
    print(f"UPLOADING BATCH {batch_number} TO SUPABASE")
    print(f"{'='*80}\n")

    # Connect to Supabase
    supabase_url = os.getenv("SUPABASE_URL") or "https://uahtfiujbblvjectcpli.supabase.co"
    supabase_key = os.getenv("SUPABASE_KEY") or "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVhaHRmaXVqYmJsdmplY3RjcGxpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjE2NzE1ODIsImV4cCI6MjA3NzI0NzU4Mn0.JwOlQ6wpeqrrgFItoeqcYFqB__0L_LZTcl2q_DDiXc4"

    supabase = create_client(supabase_url, supabase_key)

    # Load batch file
    batch_file = f"/data/batches/batch_{batch_number:04d}.pkl"

    if not os.path.exists(batch_file):
        print(f"❌ Batch file not found: {batch_file}")
        return 0

    print(f"📥 Loading {batch_file}...")
    with open(batch_file, "rb") as f:
        batch_data = pickle.load(f)

    embeddings = batch_data["embeddings"]
    metadata = batch_data["metadata"]
    count = len(embeddings)

    print(f"✅ Loaded {count:,} embeddings\n")

    # Upload in smaller chunks (100 at a time to avoid timeouts)
    CHUNK_SIZE = 100
    uploaded = 0
    failed = 0

    for i in range(0, count, CHUNK_SIZE):
        chunk_end = min(i + CHUNK_SIZE, count)
        chunk_embeddings = embeddings[i:chunk_end]
        chunk_metadata = metadata[i:chunk_end]

        # Prepare update data
        updates = []
        for emb, meta in zip(chunk_embeddings, chunk_metadata):
            updates.append({
                "tweet_id": meta["tweet_id"],
                "embedding": emb  # Supabase will automatically handle vector format
            })

        # Upload chunk
        try:
            # Use upsert to handle existing tweets
            response = supabase.table("tweets").upsert(
                updates,
                on_conflict="tweet_id"
            ).execute()

            uploaded += len(updates)

            if (uploaded % 10000 == 0) or (uploaded == count):
                print(f"   Uploaded {uploaded:,}/{count:,} embeddings...")

        except Exception as e:
            failed += len(updates)
            print(f"   ⚠️ Failed to upload chunk {i}-{chunk_end}: {e}")
            # Continue with next chunk

        # Rate limiting
        time.sleep(0.1)

    print(f"\n✅ Batch {batch_number} complete!")
    print(f"   Uploaded: {uploaded:,}")
    print(f"   Failed: {failed:,}\n")

    return uploaded


@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=300,
)
def list_batches():
    """List all available batch files"""
    import glob

    volume.reload()
    batch_files = sorted(glob.glob("/data/batches/batch_*.pkl"))

    print(f"\n{'='*80}")
    print(f"AVAILABLE BATCH FILES")
    print(f"{'='*80}\n")
    print(f"Found {len(batch_files)} batch files\n")

    for i, bf in enumerate(batch_files, 1):
        import os
        size_mb = os.path.getsize(bf) / 1024 / 1024
        print(f"  {i:2d}. {os.path.basename(bf)} ({size_mb:.1f} MB)")

    return len(batch_files)


@app.local_entrypoint()
def main(batch: int = None, start: int = 1, end: int = None):
    """
    Upload embeddings to Supabase

    Examples:
        # List available batches
        modal run upload_embeddings_to_supabase.py

        # Upload single batch
        modal run upload_embeddings_to_supabase.py --batch 1

        # Upload range of batches
        modal run upload_embeddings_to_supabase.py --start 1 --end 10
    """
    from datetime import datetime
    import time

    start_time = datetime.now()

    print(f"\n{'='*80}")
    print(f"SUPABASE EMBEDDING UPLOAD")
    print(f"{'='*80}\n")

    # List batches first
    total_batches = list_batches.remote()

    if batch is not None:
        # Upload single batch
        print(f"\n📤 Uploading batch {batch}...")
        uploaded = upload_batch_to_supabase.remote(batch)
        print(f"\n✅ Uploaded {uploaded:,} embeddings from batch {batch}")

    elif end is not None:
        # Upload range of batches
        end = min(end, total_batches)
        print(f"\n📤 Uploading batches {start} to {end}...")

        for batch_num in range(start, end + 1):
            print(f"\n--- Batch {batch_num}/{end} ---")
            uploaded = upload_batch_to_supabase.remote(batch_num)
            print(f"✅ Uploaded {uploaded:,} embeddings")
            time.sleep(2)  # Small delay between batches

        print(f"\n{'='*80}")
        print(f"✅ UPLOAD COMPLETE!")
        print(f"{'='*80}")
        print(f"Uploaded batches {start} to {end}")

    else:
        # No batch specified - just list
        print("\nTo upload embeddings, run:")
        print("  modal run upload_embeddings_to_supabase.py --start 1 --end 64")
        print("  (or specify --batch N for a single batch)")

    duration = datetime.now() - start_time
    print(f"\nTotal time: {duration}")
