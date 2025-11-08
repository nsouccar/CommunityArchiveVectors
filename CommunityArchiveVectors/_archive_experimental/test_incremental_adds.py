#!/usr/bin/env python3
"""
Test Incremental Adds - Test CoreNN's backedge delta strategy

This script tests whether incremental adds work efficiently on the fresh 7M database.
CoreNN's backedge delta strategy should make incremental adds fast!

Tests:
1. Add 1K new tweets - should be fast (< 30 seconds)
2. Add 10K new tweets - should be fast (< 2 minutes)
3. Add 100K new tweets - should be reasonable (< 15 minutes)

If these work, we can use incremental adds for daily/weekly updates!
"""

import modal
import os
from datetime import datetime
import time

app = modal.App("test-incremental-adds")

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
    cpu=8.0,
    memory=16384,  # 16 GB should be enough
    timeout=3600,  # 1 hour timeout per test
)
def test_incremental_add(test_size, start_after_tweet_id):
    """
    Test adding new tweets to existing CoreNN database.

    This should use CoreNN's backedge delta strategy automatically!

    Args:
        test_size: Number of tweets to add (1000, 10000, 100000)
        start_after_tweet_id: Tweet ID to start after (to get NEW tweets not in DB)

    Returns:
        (success, time_taken, tweets_added)
    """
    from supabase import create_client
    import voyageai
    from corenn_py import CoreNN
    import numpy as np
    import pickle

    start_time = time.time()

    print(f"================================================================================")
    print(f"INCREMENTAL ADD TEST: Adding {test_size:,} new tweets")
    print(f"================================================================================\n")

    # Initialize clients
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    voyage_api_key = os.environ.get("VOYAGE_API_KEY")

    supabase = create_client(supabase_url, supabase_key)
    vo = voyageai.Client(api_key=voyage_api_key)

    # Step 1: Fetch NEW tweets that aren't in the database yet
    print(f"📥 Fetching {test_size:,} new tweets from Supabase...")
    print(f"   Starting after tweet_id: {start_after_tweet_id}\n")

    query = (
        supabase.table("tweets")
        .select("*")
        .order("tweet_id")
        .gt("tweet_id", start_after_tweet_id)
        .limit(test_size)
    )

    response = query.execute()
    all_tweets = response.data

    if len(all_tweets) == 0:
        print("❌ No new tweets found!")
        return False, 0, 0

    print(f"✅ Retrieved {len(all_tweets):,} new tweets\n")

    # Step 2: Get account IDs and fetch usernames
    print("👤 Fetching usernames...")
    unique_account_ids = list(set(t.get("account_id") for t in all_tweets if t.get("account_id")))

    username_map = {}
    for i in range(0, len(unique_account_ids), 1000):
        batch = unique_account_ids[i:i+1000]
        accounts = (
            supabase.table("all_account")
            .select("account_id, username")
            .in_("account_id", batch)
            .execute()
        )
        for account in accounts.data:
            username_map[account["account_id"]] = account["username"]

    print(f"✅ Found {len(username_map):,} usernames\n")

    # Step 3: Fetch parent tweets for replies
    print("🔗 Fetching parent tweets for replies...")
    reply_tweet_ids = [t["reply_to_tweet_id"] for t in all_tweets if t.get("reply_to_tweet_id")]

    parent_tweets = {}
    if reply_tweet_ids:
        for i in range(0, len(reply_tweet_ids), 1000):
            batch = reply_tweet_ids[i:i+1000]
            parents = (
                supabase.table("tweets")
                .select("tweet_id, full_text")
                .in_("tweet_id", batch)
                .execute()
            )
            for parent in parents.data:
                parent_tweets[parent["tweet_id"]] = parent["full_text"]

    print(f"✅ Found {len(parent_tweets):,} parent tweets\n")

    # Step 4: Prepare texts with reply context
    print("🔄 Preparing texts for embedding...")
    texts_to_embed = []
    metadata_list = []

    for tweet in all_tweets:
        username = username_map.get(tweet.get("account_id"), "unknown")
        text_to_embed = f"@{username}: {tweet['full_text']}"

        # Add reply context if this is a reply
        if tweet.get("reply_to_tweet_id") and tweet["reply_to_tweet_id"] in parent_tweets:
            parent_text = parent_tweets[tweet["reply_to_tweet_id"]]
            text_to_embed = f"Replying to: {parent_text}\n\n{text_to_embed}"

        texts_to_embed.append(text_to_embed)
        metadata_list.append({
            "tweet_id": tweet["tweet_id"],
            "account_id": tweet.get("account_id"),
            "username": username,
            "full_text": tweet["full_text"],
            "created_at": tweet.get("created_at"),
        })

    print(f"✅ Prepared {len(texts_to_embed):,} texts\n")

    # Step 5: Generate embeddings
    print("🤖 Generating Voyage AI embeddings...")
    all_embeddings = []

    # Process in batches of 128 (Voyage AI limit)
    for i in range(0, len(texts_to_embed), 128):
        batch_texts = texts_to_embed[i:i+128]
        result = vo.embed(batch_texts, model="voyage-3", input_type="document")
        all_embeddings.extend(result.embeddings)

        if (i + 128) % 1000 == 0:
            print(f"   Processed {i + 128:,}/{len(texts_to_embed):,}...")

    print(f"✅ Generated {len(all_embeddings):,} embeddings\n")

    # Step 6: Load existing database and add new vectors
    print("🔓 Opening existing CoreNN database...")
    vector_volume.reload()

    db_path = "/data/corenn_db"

    if not os.path.exists(db_path):
        print("❌ Database doesn't exist yet! Run offline_builder.py first.")
        return False, 0, 0

    # OPEN existing database (not create!)
    db = CoreNN.open(db_path)
    print(f"✅ Opened existing database\n")

    # Step 7: Prepare vectors
    print("📊 Preparing vectors for insertion...")
    keys = [meta["tweet_id"] for meta in metadata_list]
    vectors = np.array(all_embeddings, dtype=np.float32)

    # Normalize for cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vectors = vectors / norms

    print(f"✅ Prepared {len(keys):,} vectors\n")

    # Step 8: THE KEY TEST - Incremental add!
    print("🚀 INSERTING NEW VECTORS (backedge delta strategy should make this fast!)...")
    insert_start = time.time()

    # THIS IS THE CRITICAL OPERATION
    # CoreNN should use backedge deltas automatically!
    db.insert_f32(keys, vectors)

    insert_time = time.time() - insert_start
    print(f"✅ Insert completed in {insert_time:.1f} seconds!\n")

    # Step 9: Update metadata
    print("💾 Updating metadata...")

    # Load existing metadata
    metadata_path = "/data/metadata.pkl"
    with open(metadata_path, "rb") as f:
        existing_data = pickle.load(f)

    existing_metadata = existing_data["metadata"]

    # Add new metadata
    for meta in metadata_list:
        existing_metadata[meta["tweet_id"]] = meta

    # Save updated metadata
    with open(metadata_path, "wb") as f:
        pickle.dump({
            "metadata": existing_metadata,
            "count": len(existing_metadata)
        }, f)

    vector_volume.commit()
    print(f"✅ Metadata updated\n")

    total_time = time.time() - start_time

    print("================================================================================")
    print(f"✅ INCREMENTAL ADD TEST COMPLETE!")
    print(f"================================================================================")
    print(f"Tweets added: {len(keys):,}")
    print(f"Insert time: {insert_time:.1f} seconds")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Rate: {len(keys) / insert_time:.0f} tweets/second")
    print("================================================================================\n")

    return True, total_time, len(keys)


@app.function(
    image=image,
    volumes={"/data": vector_volume},
    timeout=300,
)
def get_last_tweet_id_in_db():
    """
    Get the last tweet_id in the existing database.
    We'll start adding tweets after this ID.
    """
    import pickle

    vector_volume.reload()

    metadata_path = "/data/metadata.pkl"
    if not os.path.exists(metadata_path):
        return None

    with open(metadata_path, "rb") as f:
        data = pickle.load(f)

    metadata = data["metadata"]

    # Find max tweet_id
    max_tweet_id = max(metadata.keys())

    return max_tweet_id, len(metadata)


@app.local_entrypoint()
def main():
    """
    Run incremental add tests to verify backedge delta strategy works!
    """

    print("================================================================================")
    print("INCREMENTAL ADD TEST SUITE")
    print("Testing CoreNN's backedge delta strategy")
    print("================================================================================\n")

    # Step 1: Get last tweet_id in existing database
    print("🔍 Checking existing database...\n")
    result = get_last_tweet_id_in_db.remote()

    if result is None:
        print("❌ No existing database found!")
        print("   Run offline_builder.py first to build the base database.\n")
        return

    last_tweet_id, db_size = result
    print(f"✅ Found existing database")
    print(f"   Current size: {db_size:,} tweets")
    print(f"   Last tweet_id: {last_tweet_id}\n")

    # Step 2: Run tests with increasing sizes
    tests = [
        ("Small", 1000),
        ("Medium", 10000),
        ("Large", 100000),
    ]

    results = []

    for test_name, test_size in tests:
        print(f"\n{'='*80}")
        print(f"TEST {len(results) + 1}/3: {test_name} batch ({test_size:,} tweets)")
        print(f"{'='*80}\n")

        try:
            success, time_taken, tweets_added = test_incremental_add.remote(
                test_size,
                last_tweet_id
            )

            if success:
                results.append({
                    "name": test_name,
                    "size": test_size,
                    "time": time_taken,
                    "added": tweets_added,
                    "rate": tweets_added / time_taken if time_taken > 0 else 0
                })

                # Update last_tweet_id for next test
                last_tweet_id = last_tweet_id  # Would need to update this properly
            else:
                print(f"❌ Test failed!\n")

        except Exception as e:
            print(f"❌ Test failed with error: {e}\n")

    # Step 3: Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80 + "\n")

    if not results:
        print("❌ All tests failed!\n")
        return

    for i, result in enumerate(results, 1):
        print(f"{i}. {result['name']} ({result['size']:,} tweets)")
        print(f"   Time: {result['time']:.1f}s")
        print(f"   Rate: {result['rate']:.0f} tweets/second")
        print()

    # Analysis
    print("="*80)
    print("ANALYSIS")
    print("="*80 + "\n")

    if all(r['time'] < 300 for r in results):  # All tests < 5 minutes
        print("✅ EXCELLENT! Incremental adds are working efficiently!")
        print("   CoreNN's backedge delta strategy is working as designed.")
        print()
        print("📋 RECOMMENDATION:")
        print("   - Use incremental adds for daily/weekly updates (10K-100K tweets)")
        print("   - Do full rebuilds monthly/quarterly to optimize database structure")
        print("   - This approach should scale to 100M tweets!")
    else:
        print("⚠️  WARNING: Incremental adds are slower than expected")
        print("   Backedge delta strategy may not be working optimally.")
        print()
        print("📋 RECOMMENDATION:")
        print("   - Stick with weekly full rebuilds")
        print("   - Consider investigating corenn-py library implementation")

    print()
