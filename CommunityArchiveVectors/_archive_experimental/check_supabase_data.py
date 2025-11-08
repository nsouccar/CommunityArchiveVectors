#!/usr/bin/env python3
"""
Check what data is actually in Supabase
"""
from supabase import create_client
import os
from datetime import datetime

# Load from environment
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    print("❌ Missing SUPABASE_URL or SUPABASE_KEY environment variables")
    exit(1)

print(f"🔗 Connecting to Supabase: {supabase_url}")
supabase = create_client(supabase_url, supabase_key)

# Query for tweets containing "love"
print("\n📊 Checking tweets containing 'love'...")
response = supabase.table("tweets").select(
    "tweet_id, full_text, created_at, account_id"
).ilike("full_text", "%love%").limit(10).execute()

print(f"\n✅ Found {len(response.data)} tweets:\n")
for tweet in response.data:
    print(f"Tweet ID: {tweet['tweet_id']}")
    print(f"Created: {tweet.get('created_at', 'N/A')}")
    print(f"Text: {tweet.get('full_text', 'N/A')[:100]}...")
    print()

# Check for future dates (after 2024)
print("\n🔮 Checking for tweets with dates in 2025 or later...")
response = supabase.table("tweets").select(
    "tweet_id, full_text, created_at"
).gte("created_at", "2025-01-01").limit(10).execute()

if response.data:
    print(f"⚠️  Found {len(response.data)} tweets from 2025+:")
    for tweet in response.data:
        print(f"  - Tweet {tweet['tweet_id']}: {tweet.get('created_at')}")
else:
    print("✅ No tweets from 2025+ found in Supabase")

# Check date range in database
print("\n📅 Checking date range in database...")
response = supabase.table("tweets").select(
    "created_at"
).order("created_at", desc=False).limit(1).execute()

if response.data:
    print(f"Earliest tweet: {response.data[0].get('created_at', 'N/A')}")

response = supabase.table("tweets").select(
    "created_at"
).order("created_at", desc=True).limit(1).execute()

if response.data:
    print(f"Latest tweet: {response.data[0].get('created_at', 'N/A')}")

# Count total tweets
print("\n📈 Total tweet count...")
response = supabase.table("tweets").select("*", count="exact", head=True).execute()
print(f"Total tweets in Supabase: {response.count:,}")
