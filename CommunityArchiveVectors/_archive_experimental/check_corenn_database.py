#!/usr/bin/env python3
"""
Check what's actually in the built CoreNN database
"""
import modal
import os

app = modal.App("check-corenn-db")

image = modal.Image.debian_slim().pip_install("supabase", "corenn-py", "numpy", "openai")

vector_volume = modal.Volume.from_name("tweet-vectors-volume", create_if_missing=True)
secrets = modal.Secret.from_name("tweet-vectors-secrets")

@app.function(
    image=image,
    secrets=[secrets],
    volumes={"/data": vector_volume},
    timeout=600,
)
def test_search_in_corenn():
    """Search for 'love' in the CoreNN database and see what we get"""
    from corenn_py import CoreNN
    import pickle
    import numpy as np
    from openai import OpenAI

    print("🔍 Testing search in CoreNN database...\n")

    # Load database
    vector_volume.reload()

    db_path = "/data/corenn_db"
    db = CoreNN.open(db_path)
    print(f"✅ Opened CoreNN database\n")

    # Load metadata
    with open("/data/metadata.pkl", "rb") as f:
        data = pickle.load(f)
        metadata_dict = data.get("metadata", {})
        count = data.get("count", 0)

    print(f"📊 Database has {count:,} vectors\n")
    print(f"📊 Metadata has {len(metadata_dict):,} entries\n")

    # Generate embedding for "love" using OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input="love"
    )
    query_embedding = response.data[0].embedding

    # Convert to 2D array and normalize
    query_vector = np.array([query_embedding], dtype=np.float32)
    norm = np.linalg.norm(query_vector)
    if norm > 0:
        query_vector = query_vector / norm

    print("🔎 Searching for 'love' in CoreNN database...\n")

    # Search
    results_list = db.query_f32(query_vector, 10)
    results = results_list[0] if results_list else []

    print(f"Found {len(results)} results:\n")

    for i, (tweet_id, distance) in enumerate(results, 1):
        # Get metadata
        meta = metadata_dict.get(tweet_id, {})

        # Calculate similarity
        cosine_similarity = 1 - distance
        normalized_similarity = (cosine_similarity + 1) / 2

        print(f"Result {i}:")
        print(f"  Tweet ID: {tweet_id}")
        print(f"  Similarity: {normalized_similarity*100:.1f}%")
        print(f"  Date: {meta.get('created_at', 'N/A')}")
        print(f"  Text: {meta.get('full_text', 'N/A')[:100]}...")
        print()

@app.local_entrypoint()
def main():
    test_search_in_corenn.remote()
