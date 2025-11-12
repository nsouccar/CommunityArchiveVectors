"""
Add profile images to all clustering data files by looking up usernames in avatar_urls.json
"""

import json
from pathlib import Path
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

def main():
    # Load avatar URLs
    avatar_path = Path("frontend/public/avatar_urls.json")
    with open(avatar_path, 'r') as f:
        avatar_urls = json.load(f)

    print(f"Loaded {len(avatar_urls)} avatar URLs")

    # Initialize Supabase client
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment")

    supabase = create_client(supabase_url, supabase_key)

    # Process each year's clustering data
    years = [2012, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    data_dir = Path("frontend/public/data")

    print(f"\n{'='*80}")
    print(f"ADDING PROFILE IMAGES TO CLUSTERING DATA")
    print(f"{'='*80}\n")

    for year in years:
        year_file = data_dir / f"topics_year_{year}_summary.json"

        if not year_file.exists():
            print(f"⚠  Skipped {year}: file not found")
            continue

        # Load the clustering data
        with open(year_file, 'r') as f:
            data = json.load(f)

        print(f"Processing {year}...")

        communities = data.get('communities', {})
        total_topics = 0
        total_tweets_enriched = 0

        # Get all unique tweet IDs from all communities
        all_tweet_ids = set()
        for comm_id, topics in communities.items():
            for topic in topics:
                tweet_ids = topic.get('tweet_ids', [])
                all_tweet_ids.update(tweet_ids)
                total_topics += 1

        print(f"  Found {len(all_tweet_ids)} unique tweets across {total_topics} topics")

        # Fetch tweet data from Supabase in batches
        tweet_id_to_username = {}
        batch_size = 1000
        tweet_ids_list = list(all_tweet_ids)

        for i in range(0, len(tweet_ids_list), batch_size):
            batch = tweet_ids_list[i:i+batch_size]

            # Fetch tweets with their accounts
            response = supabase.table('tweets').select('tweet_id, account_id, accounts(username)').in_('tweet_id', batch).execute()

            for tweet in response.data:
                tweet_id = str(tweet['tweet_id'])
                username = tweet.get('accounts', {}).get('username') if tweet.get('accounts') else None
                if username:
                    tweet_id_to_username[tweet_id] = username

            print(f"  Processed {min(i+batch_size, len(tweet_ids_list))}/{len(tweet_ids_list)} tweets")

        print(f"  Mapped {len(tweet_id_to_username)} tweets to usernames")

        # Now enrich each topic with sample tweets that have profile images
        for comm_id, topics in communities.items():
            for topic in topics:
                tweet_ids = topic.get('tweet_ids', [])

                # Add sample tweets with profile images (take first 5)
                sample_tweets = []
                for tweet_id in tweet_ids[:5]:
                    username = tweet_id_to_username.get(str(tweet_id))
                    if username:
                        profile_image_url = avatar_urls.get(username)
                        sample_tweets.append({
                            'tweet_id': tweet_id,
                            'username': username,
                            'profile_image_url': profile_image_url
                        })
                        if profile_image_url:
                            total_tweets_enriched += 1

                topic['sample_tweets'] = sample_tweets

        # Save the enriched data
        with open(year_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓  {year}: Enriched {total_tweets_enriched} tweets with profile images\n")

    print(f"{'='*80}")
    print(f"COMPLETE!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
