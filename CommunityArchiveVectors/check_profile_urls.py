"""Check if Supabase has profile image URLs stored."""
import os
from supabase import create_client

# Initialize Supabase client
supabase = create_client(
    os.environ.get("SUPABASE_URL"),
    os.environ.get("SUPABASE_KEY")
)

# Get a sample tweet to see what fields are available
response = supabase.table("tweets").select("*").limit(1).execute()

if response.data:
    print("Available fields in tweets table:")
    print(list(response.data[0].keys()))
    print("\nSample record:")
    for key, value in response.data[0].items():
        if isinstance(value, str) and len(value) > 100:
            print(f"  {key}: {value[:100]}...")
        else:
            print(f"  {key}: {value}")
else:
    print("No data found")
