import os
from supabase import create_client

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_KEY"]
)

# Search for tweets containing "coding" or "code" in the text
print("Searching for tweets about coding from Oct 1, 2024...")

response = supabase.table("tweets").select(
    "tweet_id, full_text, created_at"
).gte(
    "created_at", "2024-10-01"
).or_(
    "full_text.ilike.%coding%,full_text.ilike.%code%,full_text.ilike.%programming%,full_text.ilike.%software%"
).limit(20).execute()

print(f"\nFound {len(response.data)} tweets containing coding-related words:\n")

for i, tweet in enumerate(response.data, 1):
    print(f"{i}. {tweet['full_text'][:150]}...")
    print(f"   Date: {tweet['created_at']}")
    print()
