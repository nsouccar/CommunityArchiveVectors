#!/usr/bin/env python3
"""
Semantic Search on 6.4M Tweet Database

This script allows you to search for tweets using natural language queries.
"""

import modal
import os

app = modal.App("search-tweets")

image = (
    modal.Image.debian_slim()
    .pip_install("corenn-py", "numpy", "openai")
)

vector_volume = modal.Volume.from_name("tweet-vectors-volume", create_if_missing=False)


@app.function(
    image=image,
    volumes={"/data": vector_volume},
    secrets=[modal.Secret.from_name("openai-secret")],
    cpu=4.0,
    memory=16384,
    timeout=300,
)
def search_tweets(query: str, k: int = 10):
    """
    Search for tweets similar to the query.

    Args:
        query: Natural language search query
        k: Number of results to return (default 10)

    Returns:
        List of tweet results with metadata
    """
    from corenn_py import CoreNN
    import numpy as np
    import pickle
    from openai import OpenAI

    vector_volume.reload()

    print(f"🔍 Searching for: '{query}'\n")

    # Step 1: Load the database
    print("📂 Opening CoreNN database...")
    db = CoreNN.open("/data/corenn_db")
    print("✅ Database opened\n")

    # Step 2: Load metadata
    print("📝 Loading metadata...")
    with open("/data/metadata.pkl", "rb") as f:
        metadata_pkg = pickle.load(f)
        metadata_dict = metadata_pkg["metadata"]
        total_vectors = metadata_pkg["count"]
    print(f"✅ Loaded metadata for {total_vectors:,} tweets\n")

    # Step 3: Convert query to embedding
    print("🔄 Converting query to embedding...")
    client = OpenAI()
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=query,
        dimensions=1024
    )
    query_embedding = response.data[0].embedding

    # Convert to 2D array (shape: [1, 1024]) for query_f32
    query_vector = np.array([query_embedding], dtype=np.float32)

    # Normalize for cosine similarity
    norm = np.linalg.norm(query_vector)
    if norm > 0:
        query_vector = query_vector / norm
    print("✅ Query embedded\n")

    # Step 4: Search the database
    print(f"🔎 Searching database for top {k} results...")
    results_list = db.query_f32(query_vector, k)
    results = results_list[0] if results_list else []
    print(f"✅ Found {len(results)} results\n")

    # Step 5: Format results with metadata
    print("="*80)
    print("SEARCH RESULTS")
    print("="*80 + "\n")

    formatted_results = []

    for i, (tweet_id, distance) in enumerate(results, 1):
        # Get metadata for this tweet
        meta = metadata_dict.get(tweet_id, {})

        # Convert distance to cosine similarity (range: -1 to 1)
        cosine_similarity = 1 - distance
        # Normalize to 0-1 range for display (0% = opposite, 100% = identical)
        normalized_similarity = (cosine_similarity + 1) / 2

        result = {
            "rank": i,
            "tweet_id": tweet_id,
            "similarity": normalized_similarity,
            "text": meta.get("full_text", "N/A"),
            "username": meta.get("username", "N/A"),
            "created_at": meta.get("created_at", "N/A"),
        }

        formatted_results.append(result)

        # Print result
        print(f"Result #{i} (Similarity: {normalized_similarity:.4f})")
        print(f"  Tweet ID: {tweet_id}")
        print(f"  Username: @{result['username']}")
        print(f"  Date: {result['created_at']}")
        print(f"  Text: {result['text'][:200]}{'...' if len(result['text']) > 200 else ''}")
        print()

    print("="*80 + "\n")

    return formatted_results


@app.local_entrypoint()
def main(query: str = "artificial intelligence and machine learning", k: int = 10):
    """
    Search for tweets matching the query.

    Usage:
        modal run search_tweets.py --query "your search query" --k 10
    """
    results = search_tweets.remote(query, k)

    print(f"✅ Search complete! Found {len(results)} results for: '{query}'")
