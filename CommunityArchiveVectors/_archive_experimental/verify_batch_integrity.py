#!/usr/bin/env python3
"""
Verify that batch files have matching tweet_id, text, and embeddings
by checking if tweet text matches what's in Supabase
"""
import modal
import os

app = modal.App("verify-batch-integrity")

image = modal.Image.debian_slim().pip_install("supabase", "voyageai", "numpy")

vector_volume = modal.Volume.from_name("tweet-vectors-volume", create_if_missing=True)
secrets = modal.Secret.from_name("tweet-vectors-secrets")

@app.function(
    image=image,
    secrets=[secrets],
    volumes={"/data": vector_volume},
    timeout=600,
)
def verify_batch_data():
    """Check if batch file data matches Supabase data"""
    import pickle
    import glob
    from supabase import create_client

    print("🔍 Verifying batch file integrity...\n")

    # Connect to Supabase
    supabase = create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_KEY"]
    )

    # Reload volume
    vector_volume.reload()

    # Find batch files
    batch_files = sorted(glob.glob("/data/batches/batch_*.pkl"))

    if not batch_files:
        print("❌ No batch files found!")
        return

    print(f"📂 Found {len(batch_files)} batch files")
    print(f"   Checking first batch: {batch_files[0]}\n")

    # Load first batch
    with open(batch_files[0], "rb") as f:
        batch_data = pickle.load(f)

    embeddings = batch_data["embeddings"]
    metadata = batch_data["metadata"]

    print(f"✅ Loaded batch with {len(metadata)} tweets\n")

    # Check first 5 tweets
    print("🔬 Verifying first 5 tweets match Supabase:\n")

    mismatches = 0
    for i in range(min(5, len(metadata))):
        meta = metadata[i]
        tweet_id = meta["tweet_id"]
        batch_text = meta["full_text"]
        batch_date = meta.get("created_at")

        # Fetch from Supabase
        response = supabase.table("tweets").select(
            "tweet_id, full_text, created_at"
        ).eq("tweet_id", tweet_id).execute()

        if not response.data or len(response.data) == 0:
            print(f"❌ Tweet {tweet_id} NOT FOUND in Supabase!")
            mismatches += 1
            continue

        supabase_tweet = response.data[0]
        supabase_text = supabase_tweet["full_text"]
        supabase_date = supabase_tweet.get("created_at")

        # Compare
        text_match = batch_text == supabase_text
        date_match = batch_date == supabase_date

        print(f"Tweet {i+1} (ID: {tweet_id}):")
        print(f"  Text match: {'✓' if text_match else '✗'}")
        print(f"  Date match: {'✓' if date_match else '✗'}")

        if not text_match:
            print(f"  Batch text: {batch_text[:100]}...")
            print(f"  Supabase:   {supabase_text[:100]}...")
            mismatches += 1

        if not date_match:
            print(f"  Batch date: {batch_date}")
            print(f"  Supabase:   {supabase_date}")
            mismatches += 1

        print()

    if mismatches > 0:
        print(f"⚠️  Found {mismatches} mismatches!")
        print("   The batch files contain DIFFERENT data than Supabase!")
    else:
        print("✅ All samples match - batch files are correct")

    return {"mismatches": mismatches}

@app.local_entrypoint()
def main():
    result = verify_batch_data.remote()
    print(f"\nResult: {result}")
