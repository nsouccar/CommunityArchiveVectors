#!/usr/bin/env python3
"""
Count total number of tweets in organized_by_community.pkl
"""

import modal
import pickle
from pathlib import Path

app = modal.App("count-organized-tweets")
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy")
volume = modal.Volume.from_name("tweet-vectors-large", create_if_missing=False)

@app.function(volumes={"/data": volume}, image=image)
def count_tweets():
    organized_path = Path("/data/organized_by_community.pkl")

    if not organized_path.exists():
        print("organized_by_community.pkl does not exist!")
        return

    print("Loading organized data...")
    with open(organized_path, 'rb') as f:
        organized = pickle.load(f)

    print("\n" + "="*80)
    print("COUNTING TWEETS IN organized_by_community.pkl")
    print("="*80)

    total_tweets = 0
    year_counts = {}

    # Years are stored as strings, sort them properly
    for year in sorted(organized.keys(), key=lambda x: int(x) if x.isdigit() else 0):
        year_total = 0
        year_data = organized[year]

        print(f"\nProcessing year {year}...")
        print(f"  Type: {type(year_data)}")
        print(f"  Keys: {list(year_data.keys())[:5] if isinstance(year_data, dict) else 'N/A'}")

        # Check the actual structure
        if isinstance(year_data, dict):
            for month in year_data:
                month_data = year_data[month]

                if isinstance(month_data, dict):
                    for community_id, community_data in month_data.items():
                        # Structure is: community_data = {'tweets': [...], 'embeddings': np.array}
                        if isinstance(community_data, dict) and 'tweets' in community_data:
                            num_tweets = len(community_data['tweets'])
                            year_total += num_tweets

        year_counts[year] = year_total
        total_tweets += year_total
        print(f"  {year}: {year_total:,} tweets")

    print("\n" + "="*80)
    print(f"TOTAL TWEETS: {total_tweets:,}")
    print("="*80)

    if total_tweets >= 6_000_000:
        print(f"✓ This appears to be the FULL dataset (6M+ tweets)")
    else:
        print(f"⚠ This might be filtered data (expected ~6M, found {total_tweets:,})")

@app.local_entrypoint()
def main():
    count_tweets.remote()
