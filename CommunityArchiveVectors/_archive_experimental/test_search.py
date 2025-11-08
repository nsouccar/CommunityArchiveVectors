#!/usr/bin/env python3
"""
Quick test of CoreNN search to verify the API works
"""

import modal

app = modal.App("test-search")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "corenn-py",
    "numpy"
)

vector_volume = modal.Volume.from_name("tweet-vectors-volume", create_if_missing=False)

@app.function(
    image=image,
    volumes={"/data": vector_volume},
    timeout=120,
)
def test_corenn_search():
    """Test CoreNN search with a random query vector"""
    from corenn_py import CoreNN
    import numpy as np
    import time

    print("📂 Opening CoreNN database...")
    start = time.time()
    db = CoreNN.open("/data/corenn_db")
    print(f"✅ Database opened in {time.time() - start:.2f}s")

    print("\n🔍 Creating random query vector (1024D)...")
    query_vector = np.random.randn(1, 1024).astype(np.float32)
    norm = np.linalg.norm(query_vector)
    if norm > 0:
        query_vector = query_vector / norm
    print("✅ Query vector created and normalized")

    print("\n🔎 Testing CoreNN query_f32 method...")
    start = time.time()
    results_list = db.query_f32(query_vector, 10)
    search_time = time.time() - start

    print(f"✅ Search completed in {search_time*1000:.1f}ms")
    print(f"Results type: {type(results_list)}")
    print(f"Results length: {len(results_list)}")

    if results_list and len(results_list) > 0:
        results = results_list[0]
        print(f"First result list length: {len(results)}")
        print("\nTop 3 results:")
        for i, (key, distance) in enumerate(results[:3], 1):
            print(f"  {i}. Key: {key}, Distance: {distance:.4f}")

    return {
        "search_time_ms": search_time * 1000,
        "num_results": len(results_list[0]) if results_list else 0
    }


@app.local_entrypoint()
def main():
    print("Testing CoreNN search...\n")
    result = test_corenn_search.remote()
    print(f"\n✅ Test completed successfully!")
    print(f"Search took: {result['search_time_ms']:.1f}ms")
    print(f"Returned: {result['num_results']} results")
