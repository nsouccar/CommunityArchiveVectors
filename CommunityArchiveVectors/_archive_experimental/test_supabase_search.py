#!/usr/bin/env python3
"""
Test Supabase Search - Run this while embeddings are uploading
Tests semantic search as the database grows
"""

import os
from supabase import create_client
import voyageai
import time

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

def test_search():
    """Test semantic search on current Supabase data"""

    print("\n" + "="*80)
    print("TESTING SUPABASE SEMANTIC SEARCH")
    print("="*80 + "\n")

    # Connect to Supabase
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    voyage_key = os.getenv("VOYAGE_API_KEY")

    if not all([supabase_url, supabase_key, voyage_key]):
        print("❌ Missing environment variables!")
        print("Make sure .env file has:")
        print("  SUPABASE_URL=...")
        print("  SUPABASE_KEY=...")
        print("  VOYAGE_API_KEY=...")
        return

    supabase = create_client(supabase_url, supabase_key)
    voyage = voyageai.Client(api_key=voyage_key)

    # Check how many embeddings are uploaded
    print("📊 Checking database status...")
    response = supabase.table("tweets").select("tweet_id", count="exact").not_("embedding", "is", None).execute()
    count = response.count

    print(f"✅ Current embeddings in database: {count:,}")
    print(f"   Target: 6,400,000")
    print(f"   Progress: {(count/6400000*100):.1f}%\n")

    if count == 0:
        print("⚠️  No embeddings uploaded yet. Start upload first!")
        return

    # Test search
    test_queries = [
        "artificial intelligence",
        "climate change",
        "bitcoin"
    ]

    for query in test_queries:
        print(f"🔍 Testing search: '{query}'")
        print("-" * 80)

        start_time = time.time()

        # Generate embedding
        result = voyage.embed([query], model="voyage-3", input_type="query")
        query_embedding = result.embeddings[0]

        # Search using RPC function
        response = supabase.rpc(
            "search_tweets",
            {
                "query_embedding": query_embedding,
                "match_count": 5
            }
        ).execute()

        search_time = (time.time() - start_time) * 1000

        print(f"⏱️  Search time: {search_time:.2f}ms")
        print(f"📝 Found {len(response.data)} results\n")

        # Show top 3 results
        for i, tweet in enumerate(response.data[:3], 1):
            similarity = tweet['similarity'] * 100
            print(f"  {i}. [{similarity:.1f}%] @{tweet['username']}")
            print(f"     {tweet['full_text'][:100]}...")
            print()

        print()

    print("="*80)
    print("✅ TEST COMPLETE")
    print("="*80)
    print(f"\nCurrent database size: {count:,} tweets")
    print("You can run this script again as more data uploads!\n")

if __name__ == "__main__":
    test_search()
