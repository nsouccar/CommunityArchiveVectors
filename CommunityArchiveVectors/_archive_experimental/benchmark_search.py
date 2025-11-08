"""
Benchmark CoreNN search performance on the 6.4M vector database
"""
import modal
import time

app = modal.App("benchmark-search")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "corenn-py",
    "numpy"
)

vector_volume = modal.Volume.from_name("tweet-vectors-volume", create_if_missing=False)

@app.function(
    image=image,
    volumes={"/data": vector_volume},
    cpu=8.0,  # More CPU for faster performance
    memory=32768,  # 32GB RAM to keep everything in memory
    timeout=600,
)
def benchmark_corenn_search():
    from corenn_py import CoreNN
    import numpy as np

    vector_volume.reload()

    print("="*80)
    print("CoreNN Search Performance Benchmark")
    print("="*80)

    # STEP 1: Open database
    print("\n📂 Opening CoreNN database...")
    open_start = time.time()
    db = CoreNN.open("/data/corenn_db")
    open_time = time.time() - open_start
    print(f"✅ Database opened in {open_time:.2f}s")

    # STEP 2: Prepare test query vector
    print("\n🔍 Preparing test query vector (1024D, normalized)...")
    query_vector = np.random.randn(1, 1024).astype(np.float32)
    norm = np.linalg.norm(query_vector)
    if norm > 0:
        query_vector = query_vector / norm

    # STEP 3: Run multiple search benchmarks
    print("\n🔎 Running search benchmarks...")
    print("-"*80)

    search_times = []
    num_tests = 5
    k = 10

    for i in range(num_tests):
        search_start = time.time()
        results_list = db.query_f32(query_vector, k)
        search_time = (time.time() - search_start) * 1000  # Convert to milliseconds
        search_times.append(search_time)

        results = results_list[0] if results_list else []
        print(f"  Test {i+1}: {search_time:.1f}ms ({len(results)} results)")

    # STEP 4: Display statistics
    print("-"*80)
    print(f"\n📊 Performance Statistics:")
    print(f"  Database size: 6,399,999 vectors")
    print(f"  Dimension: 1024")
    print(f"  K (results returned): {k}")
    print(f"\n  Search times:")
    print(f"    Average: {sum(search_times)/len(search_times):.1f}ms")
    print(f"    Fastest: {min(search_times):.1f}ms")
    print(f"    Slowest: {max(search_times):.1f}ms")
    print(f"    Median: {sorted(search_times)[len(search_times)//2]:.1f}ms")

    print("\n" + "="*80)

    if sum(search_times)/len(search_times) > 1000:
        print("⚠️  WARNING: Search is very slow (>1 second)")
        print("   Expected: <100ms for proper Vamana index")
        print("   This suggests the index may not be optimized")
    else:
        print("✅ Search performance is reasonable")

    print("="*80)

    return {
        "average_ms": sum(search_times)/len(search_times),
        "fastest_ms": min(search_times),
        "slowest_ms": max(search_times),
        "database_open_time_s": open_time
    }


@app.local_entrypoint()
def main():
    """Run the benchmark"""
    result = benchmark_corenn_search.remote()
    print(f"\n✅ Benchmark complete!")
    print(f"   Average search time: {result['average_ms']:.1f}ms")
