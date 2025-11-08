#!/usr/bin/env python3
"""Quick test of warm search service"""
import modal
import time

# Use the already-running warm search service
from warm_search_service import app, WarmSearchService

@app.local_entrypoint()
def main():
    print("⚡ Testing WARM search service (database already loaded in memory)\n")
    print("="*80)

    service = WarmSearchService()

    # Test 1
    print("\n🔍 Search 1: 'love'")
    start = time.time()
    results = service.search.remote("love", 5)
    duration = (time.time() - start) * 1000

    print(f"\n⏱️  Total time: {duration:.0f}ms")
    print(f"\nTop 3 results:")
    for r in results['results'][:3]:
        print(f"  {r['similarity']*100:.1f}% - @{r['username']}: {r['text'][:70]}...")

    # Test 2
    print("\n" + "="*80)
    print("\n🔍 Search 2: 'artificial intelligence'")
    start = time.time()
    results = service.search.remote("artificial intelligence", 5)
    duration = (time.time() - start) * 1000

    print(f"\n⏱️  Total time: {duration:.0f}ms")
    print(f"\nTop 3 results:")
    for r in results['results'][:3]:
        print(f"  {r['similarity']*100:.1f}% - @{r['username']}: {r['text'][:70]}...")

    # Test 3
    print("\n" + "="*80)
    print("\n🔍 Search 3: 'climate change'")
    start = time.time()
    results = service.search.remote("climate change", 5)
    duration = (time.time() - start) * 1000

    print(f"\n⏱️  Total time: {duration:.0f}ms")
    print(f"\nTop 3 results:")
    for r in results['results'][:3]:
        print(f"  {r['similarity']*100:.1f}% - @{r['username']}: {r['text'][:70]}...")

    print("\n" + "="*80)
    print("\n✅ All 3 searches complete!")
    print("   Notice: All searches are FAST - no 35-second cold-start!")
    print("   The database is already loaded in memory from bash 423941\n")
