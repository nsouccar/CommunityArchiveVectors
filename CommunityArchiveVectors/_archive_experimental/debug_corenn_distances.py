#!/usr/bin/env python3
"""
Debug CoreNN distance calculations to understand the metric
"""
import modal
import os

app = modal.App("debug-corenn")

image = modal.Image.debian_slim().pip_install("corenn-py", "numpy", "voyageai")

vector_volume = modal.Volume.from_name("tweet-vectors-volume", create_if_missing=False)
secrets = modal.Secret.from_name("tweet-vectors-secrets")

@app.function(
    image=image,
    volumes={"/data": vector_volume},
    secrets=[secrets],
    timeout=600,
)
def debug_distances():
    """
    Test CoreNN with known vectors to understand distance calculation
    """
    from corenn_py import CoreNN
    import numpy as np
    import pickle
    import voyageai

    print("🔬 DEBUGGING CORENN DISTANCE METRIC\n")

    # Load database
    vector_volume.reload()
    db = CoreNN.open("/data/corenn_db")

    # Load metadata
    with open("/data/metadata.pkl", "rb") as f:
        data = pickle.load(f)
        metadata_dict = data["metadata"]

    print("✅ Loaded database\n")

    # Test 1: Generate embedding for a specific tweet's text
    # Let's find a tweet and re-embed its exact text to see if we get distance=0

    # Pick the first tweet from metadata
    sample_tweet_id = list(metadata_dict.keys())[0]
    sample_meta = metadata_dict[sample_tweet_id]
    sample_text = sample_meta["full_text"]
    sample_username = sample_meta["username"]

    print(f"📝 Sample tweet (ID: {sample_tweet_id}):")
    print(f"   Username: @{sample_username}")
    print(f"   Text: {sample_text[:100]}...\n")

    # Recreate the EXACT text that was embedded in the database
    text_to_embed = f"@{sample_username}: {sample_text}"

    print(f"🔄 Embedding the exact same text: '{text_to_embed[:80]}...'\n")

    # Generate embedding using Voyage AI (same as database)
    vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    result = vo.embed(
        texts=[text_to_embed],
        model="voyage-3",
        input_type="document"  # Same as database build
    )
    query_embedding = result.embeddings[0]

    # Convert to 2D array and normalize (same as search)
    query_vector = np.array([query_embedding], dtype=np.float32)
    norm = np.linalg.norm(query_vector)
    if norm > 0:
        query_vector = query_vector / norm

    # Search for this exact tweet
    print("🔎 Searching for the exact same tweet...\n")
    results_list = db.query_f32(query_vector, 5)
    results = results_list[0] if results_list else []

    print(f"Top 5 results:\n")
    for i, (tweet_id, distance) in enumerate(results, 1):
        meta = metadata_dict.get(tweet_id, {})
        is_same = "⭐ EXACT MATCH!" if tweet_id == sample_tweet_id else ""

        print(f"Result {i}: {is_same}")
        print(f"  Tweet ID: {tweet_id}")
        print(f"  Raw distance: {distance}")
        print(f"  Text: {meta.get('full_text', 'N/A')[:80]}...")
        print()

    # Test 2: Try with "query" input_type to see if it makes a difference
    print("\n" + "="*80)
    print("TEST 2: Using input_type='query' instead of 'document'")
    print("="*80 + "\n")

    result2 = vo.embed(
        texts=[text_to_embed],
        model="voyage-3",
        input_type="query"  # Different from database
    )
    query_embedding2 = result2.embeddings[0]
    query_vector2 = np.array([query_embedding2], dtype=np.float32)
    norm2 = np.linalg.norm(query_vector2)
    if norm2 > 0:
        query_vector2 = query_vector2 / norm2

    results_list2 = db.query_f32(query_vector2, 5)
    results2 = results_list2[0] if results_list2 else []

    print(f"Top 5 results with input_type='query':\n")
    for i, (tweet_id, distance) in enumerate(results2, 1):
        meta = metadata_dict.get(tweet_id, {})
        is_same = "⭐ EXACT MATCH!" if tweet_id == sample_tweet_id else ""

        print(f"Result {i}: {is_same}")
        print(f"  Tweet ID: {tweet_id}")
        print(f"  Raw distance: {distance}")
        print(f"  Text: {meta.get('full_text', 'N/A')[:80]}...")
        print()

    # Test 3: Understand the distance metric
    print("\n" + "="*80)
    print("DISTANCE METRIC ANALYSIS")
    print("="*80 + "\n")

    # Get the top result's distance
    if results:
        best_distance = results[0][1]
        print(f"Best distance (same tweet with input_type='document'): {best_distance}")
        print(f"Expected: 0.0 for identical vectors")
        print(f"Actual: {best_distance}")
        print()

        if best_distance > 0.01:
            print("⚠️  WARNING: Distance is not near 0 for the exact same tweet!")
            print("   This suggests either:")
            print("   1. Voyage AI embeddings are not deterministic")
            print("   2. Text reconstruction doesn't match original")
            print("   3. Normalization issue")
            print()

    if results2:
        best_distance2 = results2[0][1]
        print(f"Best distance (same tweet with input_type='query'): {best_distance2}")
        print()

        if best_distance2 < best_distance:
            print("✅ input_type='query' gives better results!")
        else:
            print("❌ input_type='document' is better (as expected)")

@app.local_entrypoint()
def main():
    debug_distances.remote()
