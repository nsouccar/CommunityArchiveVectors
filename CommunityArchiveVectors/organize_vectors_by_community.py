"""
Organize tweet vectors by (year, month, community)

Step 1 of the modular approach:
- Load network data (user → community per year)
- Load metadata (tweets with username, timestamp, tweet_id)
- Match tweets to communities
- Fetch embeddings from Modal batches
- Save organized structure for later clustering
"""

import modal
import pickle
import json
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import numpy as np

app = modal.App("organize-community-vectors")
volume = modal.Volume.from_name("tweet-vectors-large", create_if_missing=False)

# Image with dependencies
image = modal.Image.debian_slim().pip_install("numpy")

@app.function(
    volumes={"/data": volume},
    image=image,
    timeout=7200,  # 2 hours
    cpu=8,
    memory=32768,  # 32GB RAM
)
def organize_vectors(network_data_json: str, metadata_pkl: bytes):
    """
    Organize vectors by (year, month, community)
    """
    print("=" * 80)
    print("ORGANIZING VECTORS BY YEAR, MONTH, COMMUNITY")
    print("=" * 80)
    print()

    # Step 1: Load network animation data (user → community per year)
    print("Step 1: Loading network animation data...")
    network_data = json.loads(network_data_json)

    # Build lookup: {year: {username: community}}
    user_communities = {}
    for year_data in network_data['years']:
        year = year_data['year']
        user_communities[year] = {}
        for node in year_data['nodes']:
            username = node['id']
            community = node['community']
            user_communities[year][username] = community

    print(f"✓ Loaded network data for {len(network_data['years'])} years")
    print()

    # Step 2: Load metadata (tweet info)
    print("Step 2: Loading tweet metadata...")
    metadata = pickle.loads(metadata_pkl)

    print(f"✓ Loaded {len(metadata):,} tweets")
    print()

    # Step 3: Organize tweets by (year, month, community)
    print("Step 3: Organizing tweets by year/month/community...")

    # Structure: {year: {month: {community: [tweet_ids]}}}
    organized = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    tweets_matched = 0
    tweets_unmatched = 0

    for tweet in metadata:
        username = tweet.get('username')
        timestamp = tweet.get('created_at')
        tweet_id = tweet.get('tweet_id')

        if not username or not timestamp or not tweet_id:
            continue

        # Parse timestamp
        try:
            # Handle different timestamp formats
            if isinstance(timestamp, str):
                # Try ISO format first
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except:
                    # Try other common formats
                    dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
            else:
                # Assume it's already a datetime
                dt = timestamp

            year = str(dt.year)
            month = dt.month

        except Exception as e:
            tweets_unmatched += 1
            continue

        # Find community for this user in this year
        if year in user_communities and username in user_communities[year]:
            community = user_communities[year][username]
            organized[year][month][community].append({
                'tweet_id': tweet_id,
                'username': username,
                'text': tweet.get('text', ''),
                'timestamp': timestamp
            })
            tweets_matched += 1
        else:
            tweets_unmatched += 1

    print(f"✓ Matched {tweets_matched:,} tweets to communities")
    print(f"  Unmatched: {tweets_unmatched:,} tweets")
    print()

    # Step 4: Print summary
    print("=" * 80)
    print("ORGANIZATION SUMMARY")
    print("=" * 80)
    print()

    total_groups = 0
    for year in sorted(organized.keys()):
        year_tweets = sum(
            len(tweets)
            for month_data in organized[year].values()
            for tweets in month_data.values()
        )
        num_months = len(organized[year])
        num_communities = len(set(
            comm
            for month_data in organized[year].values()
            for comm in month_data.keys()
        ))

        print(f"Year {year}:")
        print(f"  {num_months} months")
        print(f"  {num_communities} communities")
        print(f"  {year_tweets:,} tweets")
        print()

        total_groups += sum(
            len(month_data)
            for month_data in organized[year].values()
        )

    print(f"Total (year, month, community) groups: {total_groups}")
    print()

    # Step 5: Load embeddings from batches
    print("=" * 80)
    print("STEP 5: LOADING EMBEDDINGS")
    print("=" * 80)
    print()

    # Get all tweet IDs we need
    all_tweet_ids = set()
    for year_data in organized.values():
        for month_data in year_data.values():
            for comm_tweets in month_data.values():
                for tweet in comm_tweets:
                    all_tweet_ids.add(tweet['tweet_id'])

    print(f"Need embeddings for {len(all_tweet_ids):,} unique tweets")
    print()

    # Load embeddings from batches
    print("Loading from batch files...")
    batches_dir = Path("/data/batches")
    batch_files = sorted(batches_dir.glob("batch_*.pkl"))

    tweet_embeddings = {}

    for i, batch_file in enumerate(batch_files, 1):
        with open(batch_file, 'rb') as f:
            batch_data = pickle.load(f)
            batch_metadata = batch_data['metadata']
            batch_embeddings = batch_data['embeddings']

            for meta, emb in zip(batch_metadata, batch_embeddings):
                tweet_id = meta['tweet_id']
                if tweet_id in all_tweet_ids:
                    tweet_embeddings[tweet_id] = emb

        if i % 10 == 0:
            print(f"  Processed {i}/{len(batch_files)} batches, found {len(tweet_embeddings):,} embeddings")

    print(f"✓ Loaded {len(tweet_embeddings):,} embeddings")
    print()

    # Step 6: Create final organized structure with embeddings
    print("=" * 80)
    print("STEP 6: CREATING FINAL STRUCTURE")
    print("=" * 80)
    print()

    organized_with_embeddings = {}

    for year, year_data in organized.items():
        organized_with_embeddings[year] = {}

        for month, month_data in year_data.items():
            organized_with_embeddings[year][month] = {}

            for community, tweets in month_data.items():
                # Filter to only tweets with embeddings
                tweets_with_embeddings = []
                embeddings_list = []

                for tweet in tweets:
                    tweet_id = tweet['tweet_id']
                    if tweet_id in tweet_embeddings:
                        tweets_with_embeddings.append(tweet)
                        embeddings_list.append(tweet_embeddings[tweet_id])

                if len(tweets_with_embeddings) > 0:
                    organized_with_embeddings[year][month][str(community)] = {
                        'tweets': tweets_with_embeddings,
                        'embeddings': np.array(embeddings_list),
                        'num_tweets': len(tweets_with_embeddings)
                    }

    # Step 7: Save organized data
    print("Saving organized data...")
    output_path = "/data/organized_vectors.pkl"

    with open(output_path, 'wb') as f:
        pickle.dump(organized_with_embeddings, f)

    print(f"✓ Saved to {output_path}")
    print()

    # Step 8: Create summary JSON (without embeddings for readability)
    print("Creating summary JSON...")

    summary = {
        'years': {},
        'total_tweets': tweets_matched,
        'total_groups': total_groups
    }

    for year, year_data in organized_with_embeddings.items():
        summary['years'][year] = {}

        for month, month_data in year_data.items():
            summary['years'][year][month] = {}

            for community, data in month_data.items():
                summary['years'][year][month][community] = {
                    'num_tweets': data['num_tweets'],
                    'sample_usernames': list(set(t['username'] for t in data['tweets'][:10])),
                    'embedding_shape': data['embeddings'].shape
                }

    summary_path = "/data/organized_vectors_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"✓ Saved summary to {summary_path}")
    print()

    print("=" * 80)
    print("ORGANIZATION COMPLETE!")
    print("=" * 80)
    print()
    print(f"Total tweets organized: {tweets_matched:,}")
    print(f"Total (year, month, community) groups: {total_groups}")
    print()
    print("Next steps:")
    print("  1. Review the summary JSON to verify structure")
    print("  2. Run k-means clustering on each group")
    print("  3. Generate topic labels with LLM")
    print()

    return summary

@app.local_entrypoint()
def main():
    """Run vector organization"""
    print()
    print("=" * 80)
    print("ORGANIZE VECTORS BY COMMUNITY")
    print("=" * 80)
    print()
    print("This script will:")
    print("  1. Load network data (user → community per year)")
    print("  2. Load tweet metadata")
    print("  3. Match tweets to communities")
    print("  4. Fetch embeddings from Modal batches")
    print("  5. Organize by (year, month, community)")
    print("  6. Save for later clustering")
    print()
    print("Estimated time: 30-60 minutes")
    print()

    # Load local files
    print("Loading local files...")

    with open('network_animation_data.json', 'r') as f:
        network_data_json = f.read()
    print("✓ Loaded network_animation_data.json")

    with open('metadata.pkl', 'rb') as f:
        metadata_pkl = f.read()
    print("✓ Loaded metadata.pkl")
    print()

    # Run on Modal with the data
    summary = organize_vectors.remote(network_data_json, metadata_pkl)

    # Save summary locally too
    with open('organized_vectors_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print()
    print("=" * 80)
    print("ALL DONE!")
    print("=" * 80)
    print()
    print("Summary saved to: organized_vectors_summary.json")
    print("Full data saved to Modal volume: /data/organized_vectors.pkl")
    print()
    print("You can now:")
    print("  1. Review the summary to verify organization")
    print("  2. Run clustering script (to be created next)")
    print()
