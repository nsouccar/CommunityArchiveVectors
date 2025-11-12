"""
Create a simple username -> avatar_url mapping JSON file.
No downloads, just a mapping for the frontend to use.
"""

import json
import os
from pathlib import Path
from supabase import create_client

def main():
    print("Creating username → avatar URL mapping...")

    supabase = create_client(
        os.environ.get("SUPABASE_URL"),
        os.environ.get("SUPABASE_KEY")
    )

    # Get all profiles with avatars and their usernames from tweets
    print("Querying Supabase...")

    # Join all_profile with tweets to get usernames
    query = """
    SELECT DISTINCT ON (p.account_id)
        t.account_display_name as username,
        p.avatar_media_url
    FROM all_profile p
    JOIN tweets t ON p.account_id = t.account_id
    WHERE p.avatar_media_url IS NOT NULL
    """

    response = supabase.rpc('exec_sql', {'query': query}).execute()

    # Alternative approach if RPC doesn't work: fetch separately and join in Python
    print("Fetching profiles...")
    profiles_response = supabase.table("all_profile")\
        .select("account_id, avatar_media_url")\
        .not_("avatar_media_url", "is", None)\
        .execute()

    profiles = {p['account_id']: p['avatar_media_url'] for p in profiles_response.data}
    account_ids = list(profiles.keys())

    print(f"Found {len(profiles)} profiles with avatars")
    print("Fetching usernames...")

    mapping = {}
    batch_size = 1000

    for i in range(0, len(account_ids), batch_size):
        batch = account_ids[i:i + batch_size]

        tweets_response = supabase.table("tweets")\
            .select("account_id, account_display_name")\
            .in_("account_id", batch)\
            .limit(len(batch))\
            .execute()

        for tweet in tweets_response.data:
            account_id = tweet['account_id']
            username = tweet.get('account_display_name')

            if username and account_id in profiles and username not in mapping:
                # Upgrade to higher quality image
                url = profiles[account_id].replace('_normal', '_400x400')
                mapping[username] = url

        print(f"  Processed {min(i + batch_size, len(account_ids))}/{len(account_ids)}...")

    # Save mapping
    output_file = Path("frontend/public/avatar_mapping.json")
    with open(output_file, 'w') as f:
        json.dump(mapping, f, indent=2)

    print(f"\n✓ Created {output_file}")
    print(f"  {len(mapping)} username → avatar URL mappings")
    print(f"\nFrontend usage: const avatar = avatarMap[username];")

if __name__ == "__main__":
    main()
