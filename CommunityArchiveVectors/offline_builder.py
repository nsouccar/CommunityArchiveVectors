#!/usr/bin/env python3
"""
Offline Database Builder - Build CoreNN database with ALL tweets from Supabase

Strategy:
1. Generate ALL embeddings first (in batches, keep in memory)
2. Build ENTIRE CoreNN database in ONE operation (not incremental)
3. Upload to Modal volume

This avoids the incremental add scaling problem!
"""

import modal
import os

app = modal.App("offline-tweet-builder")

image = (
    modal.Image.debian_slim()
    .pip_install("supabase", "voyageai", "corenn-py", "numpy")
)

vector_volume = modal.Volume.from_name("tweet-vectors-volume", create_if_missing=True)
secrets = modal.Secret.from_name("tweet-vectors-secrets")


@app.function(
    image=image,
    secrets=[secrets],
    volumes={"/data": vector_volume},
    timeout=10800,  # 3 hours for large batches
)
def generate_embeddings_batch(start_tweet_id, batch_number, batch_size=100000):
    """
    Generate embeddings for a batch of tweets and SAVE to Modal volume.
    Returns: (last_tweet_id, count)

    Saves to: /data/batch_{batch_number}_embeddings.pkl
    """
    from supabase import create_client
    import voyageai
    import pickle
    import os

    print(f"\n{'='*80}")
    print(f"GENERATING BATCH: Starting from tweet_id > {start_tweet_id}")
    print(f"Target: {batch_size:,} tweets")
    print(f"{'='*80}\n")

    supabase = create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_KEY"]
    )

    vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

    # Fetch tweets
    print(f"📥 Fetching tweets from Supabase...")
    all_tweets = []
    last_tweet_id = start_tweet_id

    while len(all_tweets) < batch_size:
        fetch_count = min(1000, batch_size - len(all_tweets))
        query = supabase.table("tweets").select(
            "tweet_id, full_text, reply_to_tweet_id, created_at, account_id, "
            "retweet_count, favorite_count"
        ).order("tweet_id", desc=False).limit(fetch_count)

        if last_tweet_id is not None:
            query = query.gt("tweet_id", last_tweet_id)

        response = query.execute()

        if not response.data:
            print(f"✅ No more tweets. Retrieved {len(all_tweets):,} total.")
            break

        all_tweets.extend(response.data)
        last_tweet_id = response.data[-1]["tweet_id"]

        if len(all_tweets) % 10000 == 0:
            print(f"   Fetched {len(all_tweets):,} tweets...")

    if not all_tweets:
        print("⚠️  No tweets found!")
        return [], [], last_tweet_id, 0

    print(f"✅ Retrieved {len(all_tweets):,} tweets\n")

    # Fetch usernames
    print("👤 Fetching usernames...")
    unique_account_ids = list({t["account_id"] for t in all_tweets if t.get("account_id")})
    username_map = {}

    for i in range(0, len(unique_account_ids), 1000):
        batch = unique_account_ids[i:i+1000]
        accounts = supabase.table("all_account").select("account_id, username").in_("account_id", batch).execute()
        for account in accounts.data:
            username_map[account["account_id"]] = account["username"]

    print(f"✅ Found {len(username_map):,} usernames\n")

    # Fetch parent tweets
    print("🔗 Fetching parent tweets for replies...")
    reply_tweet_ids = [t["reply_to_tweet_id"] for t in all_tweets if t.get("reply_to_tweet_id")]
    parent_tweets = {}

    if reply_tweet_ids:
        for i in range(0, len(reply_tweet_ids), 1000):
            batch = reply_tweet_ids[i:i+1000]
            parents = supabase.table("tweets").select("tweet_id, full_text").in_("tweet_id", batch).execute()
            for parent in parents.data:
                parent_tweets[parent["tweet_id"]] = parent["full_text"]

    print(f"✅ Found {len(parent_tweets):,} parent tweets\n")

    # Prepare texts
    print("🔄 Preparing texts for embedding...")
    texts_to_embed = []
    metadata_list = []

    for tweet in all_tweets:
        if not tweet.get("full_text"):
            continue

        username = username_map.get(tweet.get("account_id"), "unknown")
        text_to_embed = f"@{username}: {tweet['full_text']}"

        if tweet.get("reply_to_tweet_id") and tweet["reply_to_tweet_id"] in parent_tweets:
            parent_text = parent_tweets[tweet["reply_to_tweet_id"]]
            text_to_embed = f"Replying to: {parent_text}\n\n{text_to_embed}"

        texts_to_embed.append(text_to_embed)
        metadata_list.append({
            "tweet_id": tweet["tweet_id"],
            "full_text": tweet["full_text"],
            "username": username,
            "account_id": tweet.get("account_id"),
            "created_at": tweet.get("created_at"),
            "retweet_count": tweet.get("retweet_count", 0),
            "favorite_count": tweet.get("favorite_count", 0),
            "reply_to_tweet_id": tweet.get("reply_to_tweet_id")
        })

    print(f"✅ Prepared {len(texts_to_embed):,} texts\n")

    # Generate embeddings
    print("🤖 Generating Voyage AI embeddings...")
    all_embeddings = []
    EMBED_BATCH_SIZE = 128

    for i in range(0, len(texts_to_embed), EMBED_BATCH_SIZE):
        batch_texts = texts_to_embed[i:i+EMBED_BATCH_SIZE]
        batch_num = i//EMBED_BATCH_SIZE + 1
        total_batches = (len(texts_to_embed) + EMBED_BATCH_SIZE - 1)//EMBED_BATCH_SIZE

        if batch_num % 100 == 0 or batch_num == 1 or batch_num == total_batches:
            print(f"   Batch {batch_num}/{total_batches}...")

        result = vo.embed(batch_texts, model="voyage-3", input_type="document")
        all_embeddings.extend(result.embeddings)

    print(f"✅ Generated {len(all_embeddings):,} embeddings\n")

    # Save to Modal volume (NOT return to laptop!)
    os.makedirs("/data/batches", exist_ok=True)
    batch_file = f"/data/batches/batch_{batch_number:04d}.pkl"

    print(f"💾 Saving batch to {batch_file}...")
    with open(batch_file, "wb") as f:
        pickle.dump({
            "embeddings": all_embeddings,
            "metadata": metadata_list,
            "last_tweet_id": last_tweet_id,
            "count": len(all_embeddings)
        }, f)

    vector_volume.commit()
    print(f"✅ Batch saved to Modal volume\n")
    print(f"Last tweet_id in this batch: {last_tweet_id}\n")

    return last_tweet_id, len(all_embeddings)


@app.function(
    image=image,
    volumes={"/data": vector_volume},
    cpu=8.0,
    memory=65536,  # 64 GB RAM for loading all embeddings at once
    timeout=10800,  # 3 hours
)
def build_fresh_database():
    """
    Build FRESH CoreNN database from ALL batch files in /data/batches.
    TRUE one-shot build: Load ALL embeddings into RAM, then ONE insert.
    """
    from corenn_py import CoreNN
    import numpy as np
    import pickle
    import os
    import glob

    print(f"\n{'='*80}")
    print(f"BUILDING FRESH DATABASE FROM BATCH FILES")
    print(f"TRUE ONE-SHOT BUILD: Load all → Insert once")
    print(f"{'='*80}\n")

    # Load the volume to see latest files
    vector_volume.reload()

    # Find all batch files
    batch_files = sorted(glob.glob("/data/batches/batch_*.pkl"))
    print(f"📂 Found {len(batch_files)} batch files to process\n")

    if not batch_files:
        print("❌ No batch files found!")
        return 0

    # STEP 1: Load ALL embeddings into RAM
    print("📥 Loading ALL embeddings into RAM...")
    all_embeddings = []
    all_metadata = []

    for batch_num, batch_file in enumerate(batch_files, 1):
        print(f"   Loading batch {batch_num}/{len(batch_files)}: {os.path.basename(batch_file)}")

        with open(batch_file, "rb") as f:
            batch_data = pickle.load(f)

        all_embeddings.extend(batch_data["embeddings"])
        all_metadata.extend(batch_data["metadata"])

    total_vectors = len(all_embeddings)
    print(f"\n✅ Loaded {total_vectors:,} embeddings into RAM (~{total_vectors * 4 / 1024 / 1024:.1f} GB)\n")

    # STEP 2: Prepare ALL vectors at once
    print("🔄 Preparing vectors...")
    keys = [meta["tweet_id"] for meta in all_metadata]
    vectors = np.array(all_embeddings, dtype=np.float32)

    # Normalize for cosine similarity
    print("   Normalizing vectors...")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vectors = vectors / norms

    # STEP 3: Create database and do ONE SINGLE INSERT
    print(f"\n🔨 Creating CoreNN database...")
    os.makedirs("/data", exist_ok=True)
    db_path = "/data/corenn_db"
    db = CoreNN.create(db_path, {"dim": 1024})

    print(f"⚡ ONE SINGLE INSERT of {total_vectors:,} vectors...")
    db.insert_f32(keys, vectors)
    print(f"✅ Index built!\n")

    # STEP 4: Save metadata
    print("💾 Saving metadata...")
    metadata_dict = {meta["tweet_id"]: meta for meta in all_metadata}
    with open("/data/metadata.pkl", "wb") as f:
        pickle.dump({"metadata": metadata_dict, "count": total_vectors}, f)

    print("💾 Committing volume...")
    vector_volume.commit()

    print(f"\n🎉 Database build complete!")
    print(f"   Total vectors: {total_vectors:,}")
    print(f"   Method: ONE SINGLE INSERT (true one-shot build)\n")

    return total_vectors


@app.function(
    image=image,
    volumes={"/data": vector_volume},
    timeout=300,
)
def check_resume_point():
    """
    Check if there are existing batch files and find resume point.
    Returns: (can_resume, last_batch_number, last_tweet_id, total_embeddings)
    """
    import glob
    import pickle
    import os

    print("\n🔍 CHECKING FOR EXISTING BATCHES")
    print("="*80)

    # Reload volume to see current files
    vector_volume.reload()

    # Check for batch files
    if not os.path.exists("/data/batches"):
        print("   No /data/batches/ directory found")
        print("✅ Starting fresh build\n")
        return False, 0, None, 0

    batch_files = sorted(glob.glob("/data/batches/batch_*.pkl"))

    if not batch_files:
        print("   /data/batches/ exists but no batch files found")
        print("✅ Starting fresh build\n")
        return False, 0, None, 0

    # Found existing batches - get resume info
    print(f"   📂 Found {len(batch_files)} existing batch files!")

    # Load the last batch to get resume point
    last_batch_file = batch_files[-1]
    with open(last_batch_file, "rb") as f:
        last_batch_data = pickle.load(f)

    last_batch_number = len(batch_files)
    last_tweet_id = last_batch_data["last_tweet_id"]

    # Count total embeddings from all batches
    total_embeddings = sum(
        pickle.load(open(bf, "rb"))["count"]
        for bf in batch_files
    )

    print(f"   Last batch: {last_batch_number}")
    print(f"   Last tweet_id: {last_tweet_id}")
    print(f"   Total embeddings: {total_embeddings:,}")
    print(f"✅ Can resume from batch {last_batch_number + 1}\n")

    return True, last_batch_number, last_tweet_id, total_embeddings


@app.function(
    image=image,
    volumes={"/data": vector_volume},
    timeout=300,
)
def cleanup_volume():
    """
    Clean Modal volume completely - remove ALL files for fresh start
    """
    import shutil
    import os

    print("\n🧹 CLEANING MODAL VOLUME FOR FRESH START")
    print("="*80)

    # Reload volume to see current files
    vector_volume.reload()

    # Remove old database
    if os.path.exists("/data/corenn_db"):
        shutil.rmtree("/data/corenn_db")
        print("   ✅ Removed /data/corenn_db")

    # Remove old metadata
    if os.path.exists("/data/metadata.pkl"):
        os.remove("/data/metadata.pkl")
        print("   ✅ Removed /data/metadata.pkl")

    # Remove ALL batch files
    if os.path.exists("/data/batches"):
        shutil.rmtree("/data/batches")
        print("   ✅ Removed /data/batches/")

    # Commit the cleanup
    vector_volume.commit()

    print("✅ Volume cleaned - ready for fresh build\n")
    return True


@app.local_entrypoint()
def main():
    """
    Build entire database offline:
    0. Clean Modal volume for fresh start
    1. Generate ALL embeddings in batches
    2. Build database in ONE operation
    """
    from datetime import datetime

    print(f"\n{'='*80}")
    print(f"OFFLINE DATABASE BUILDER")
    print(f"Building CoreNN database with ALL tweets from Supabase")
    print(f"{'='*80}\n")

    start_time = datetime.now()

    # Step 0: Check for existing batches (resume capability)
    can_resume, last_batch, last_tweet_id, existing_embeddings = check_resume_point.remote()

    if can_resume:
        print(f"🔄 RESUMING from batch {last_batch + 1}")
        print(f"   Already have {existing_embeddings:,} embeddings saved\n")
        current_tweet_id = last_tweet_id
        batch_count = last_batch
        total_embeddings = existing_embeddings
    else:
        print("🆕 STARTING FRESH BUILD\n")
        current_tweet_id = None
        batch_count = 0
        total_embeddings = 0

    # Step 1: Generate ALL embeddings and save to Modal volume
    print("STEP 1: Generate All Embeddings (saved to Modal volume)\n")

    batch_size = 100000  # 100K tweets per batch

    while True:
        batch_count += 1
        print(f"Starting batch {batch_count}...")
        print(f"Total embeddings so far: {total_embeddings:,}")

        # Generate batch with automatic retry logic
        MAX_RETRIES = 3
        retry_count = 0
        batch_succeeded = False

        while retry_count <= MAX_RETRIES and not batch_succeeded:
            try:
                # Generate batch (saves to Modal volume, returns only tweet_id and count)
                last_tweet_id, count = generate_embeddings_batch.remote(
                    current_tweet_id,
                    batch_count,  # Pass batch number for filename
                    batch_size
                )
                batch_succeeded = True  # Success!

            except Exception as e:
                retry_count += 1
                if retry_count <= MAX_RETRIES:
                    wait_time = 30 * (2 ** (retry_count - 1))  # 30s, 60s, 120s
                    print(f"⚠️  Batch {batch_count} failed (attempt {retry_count}/{MAX_RETRIES})")
                    print(f"   Error: {str(e)[:100]}")
                    print(f"   Retrying in {wait_time} seconds...\n")
                    import time
                    time.sleep(wait_time)
                else:
                    print(f"❌ Batch {batch_count} failed after {MAX_RETRIES} attempts!")
                    print(f"   Last error: {str(e)}")
                    print(f"   Stopping build. Resume later to continue from batch {batch_count}.\n")
                    raise  # Re-raise to stop the build

        if count == 0:
            print(f"✅ No more tweets! Total batches: {batch_count - 1}\n")
            break

        current_tweet_id = last_tweet_id
        total_embeddings += count

        print(f"Batch {batch_count} complete: +{count:,} embeddings")
        print(f"Total so far: {total_embeddings:,} embeddings (saved to Modal volume)\n")

        # Safety check: If batch returned less than requested, we're done
        if count < batch_size:
            print(f"✅ Reached end of tweets! Total: {total_embeddings:,} embeddings\n")
            break

    if total_embeddings == 0:
        print("❌ No embeddings generated! Check Supabase connection.")
        return

    # Step 2: Build database from batch files on Modal
    print(f"\n{'='*80}")
    print("STEP 2: Build CoreNN Database from Batch Files")
    print(f"{'='*80}\n")

    final_count = build_fresh_database.remote()

    end_time = datetime.now()
    duration = end_time - start_time
    hours = duration.total_seconds() / 3600

    print(f"\n{'='*80}")
    print(f"✅ BUILD COMPLETE!")
    print(f"{'='*80}")
    print(f"Duration: {hours:.1f} hours")
    print(f"Final vector count: {final_count:,}")
    print(f"Batches processed: {batch_count}")
    print(f"{'='*80}\n")
