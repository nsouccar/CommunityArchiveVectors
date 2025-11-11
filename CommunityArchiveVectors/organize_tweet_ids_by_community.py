"""
Organize tweet IDs by (year, month, community) - MEMORY EFFICIENT VERSION

This script:
1. Loads network data (user → community per year)
2. Loads metadata (tweets with username, timestamp, tweet_id)
3. Matches tweets to communities
4. Saves just the organization structure (no embeddings!)
5. Later, clustering script will load embeddings one group at a time

This avoids loading 56.5GB of embeddings into memory all at once.
"""

import modal
import pickle
import json
from datetime import datetime
from collections import defaultdict

app = modal.App("organize-tweet-ids")
volume = modal.Volume.from_name("tweet-vectors-large", create_if_missing=False)

# Minimal image - we're not loading embeddings
image = modal.Image.debian_slim()

@app.function(
    volumes={"/data": volume},
    image=image,
    timeout=3600,  # 1 hour should be plenty
    cpu=4,
    memory=8192,  # Only 8GB - we're just organizing metadata
)
def organize_tweet_ids(network_data_json: str, metadata_pkl: bytes):
    """
    Organize tweet IDs by (year, month, community) without loading embeddings
    """
    print("=" * 80)
    print("ORGANIZING TWEET IDS BY YEAR, MONTH, COMMUNITY")
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
    print(f"  Years: {sorted([y['year'] for y in network_data['years']])}")
    print()

    # Step 2: Load metadata (tweet info)
    print("Step 2: Loading tweet metadata...")
    metadata = pickle.loads(metadata_pkl)

    print(f"✓ Loaded {len(metadata):,} tweets")
    print()

    # Step 3: Organize tweets by (year, month, community)
    print("Step 3: Organizing tweets by year/month/community...")

    # Structure: {year: {month: {community: [tweet_info]}}}
    organized = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    tweets_matched = 0
    tweets_unmatched = 0
    tweets_no_timestamp = 0
    tweets_no_username = 0

    for tweet in metadata:
        username = tweet.get('username')
        timestamp = tweet.get('created_at')
        tweet_id = tweet.get('tweet_id')

        if not username:
            tweets_no_username += 1
            continue

        if not timestamp:
            tweets_no_timestamp += 1
            continue

        if not tweet_id:
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
                    try:
                        dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                    except:
                        dt = datetime.strptime(timestamp, '%a %b %d %H:%M:%S %z %Y')
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

            # Store lightweight info (not embeddings!)
            organized[year][month][community].append({
                'tweet_id': tweet_id,
                'username': username,
                'text': tweet.get('text', ''),
                'timestamp': str(timestamp) if not isinstance(timestamp, str) else timestamp
            })
            tweets_matched += 1
        else:
            tweets_unmatched += 1

    print(f"✓ Matched {tweets_matched:,} tweets to communities")
    print(f"  Unmatched (not in network): {tweets_unmatched:,}")
    print(f"  Skipped (no username): {tweets_no_username:,}")
    print(f"  Skipped (no timestamp): {tweets_no_timestamp:,}")
    print()

    # Step 4: Print detailed summary
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
        communities_in_year = set()
        for month_data in organized[year].values():
            for comm in month_data.keys():
                communities_in_year.add(comm)
        num_communities = len(communities_in_year)

        print(f"Year {year}:")
        print(f"  {num_months} months with data")
        print(f"  {num_communities} communities")
        print(f"  {year_tweets:,} tweets")

        # Show sample month breakdown
        for month in sorted(organized[year].keys())[:3]:
            month_tweets = sum(len(tweets) for tweets in organized[year][month].values())
            print(f"    Month {month}: {month_tweets:,} tweets across {len(organized[year][month])} communities")

        if len(organized[year]) > 3:
            print(f"    ... and {len(organized[year]) - 3} more months")
        print()

        total_groups += sum(
            len(month_data)
            for month_data in organized[year].values()
        )

    print(f"Total (year, month, community) groups: {total_groups}")
    print()

    # Step 5: Save organized structure
    print("=" * 80)
    print("SAVING ORGANIZATION")
    print("=" * 80)
    print()

    # Convert defaultdicts to regular dicts for serialization
    organized_dict = {}
    for year, year_data in organized.items():
        organized_dict[year] = {}
        for month, month_data in year_data.items():
            organized_dict[year][month] = {}
            for community, tweets in month_data.items():
                organized_dict[year][month][str(community)] = tweets

    # Save to Modal volume
    output_path = "/data/organized_tweet_ids.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(organized_dict, f)
    print(f"✓ Saved organized structure to {output_path}")

    # Create human-readable JSON summary
    summary = {
        'total_tweets': tweets_matched,
        'total_groups': total_groups,
        'years': {}
    }

    for year, year_data in organized_dict.items():
        summary['years'][year] = {
            'months': {},
            'total_tweets': sum(
                len(tweets)
                for month_data in year_data.values()
                for tweets in month_data.values()
            ),
            'num_communities': len(set(
                comm
                for month_data in year_data.values()
                for comm in month_data.keys()
            ))
        }

        for month, month_data in year_data.items():
            summary['years'][year]['months'][month] = {
                'communities': {},
                'total_tweets': sum(len(tweets) for tweets in month_data.values())
            }

            for community, tweets in month_data.items():
                summary['years'][year]['months'][month]['communities'][community] = {
                    'num_tweets': len(tweets),
                    'sample_usernames': list(set(t['username'] for t in tweets[:10])),
                    'sample_tweet_ids': [t['tweet_id'] for t in tweets[:5]]
                }

    summary_path = "/data/organized_tweet_ids_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary to {summary_path}")

    # Commit changes to volume
    volume.commit()
    print(f"✓ Committed changes to volume")
    print()

    print("=" * 80)
    print("ORGANIZATION COMPLETE!")
    print("=" * 80)
    print()
    print(f"Total tweets organized: {tweets_matched:,}")
    print(f"Total (year, month, community) groups: {total_groups}")
    print()
    print("Files saved to Modal volume:")
    print(f"  1. {output_path} - Full organization (tweet IDs + metadata)")
    print(f"  2. {summary_path} - Human-readable summary")
    print()
    print("Next steps:")
    print("  1. Review the summary JSON to verify structure")
    print("  2. Create clustering script that:")
    print("     - Loads one (year, month, community) group at a time")
    print("     - Fetches embeddings for just those tweet IDs")
    print("     - Runs k-means clustering")
    print("     - Generates topic labels with LLM")
    print("     - Saves results")
    print()

    return summary

@app.local_entrypoint()
def main():
    """Run tweet ID organization"""
    print()
    print("=" * 80)
    print("ORGANIZE TWEET IDS BY COMMUNITY (MEMORY EFFICIENT)")
    print("=" * 80)
    print()
    print("This script will:")
    print("  1. Load network data (user → community per year)")
    print("  2. Load tweet metadata")
    print("  3. Match tweets to communities")
    print("  4. Organize by (year, month, community)")
    print("  5. Save organization structure (NO embeddings loaded!)")
    print()
    print("Memory usage: ~8GB (very light)")
    print("Estimated time: 5-10 minutes")
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
    summary = organize_tweet_ids.remote(network_data_json, metadata_pkl)

    # Save summary locally too
    with open('organized_tweet_ids_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 80)
    print("ALL DONE!")
    print("=" * 80)
    print()
    print("Summary saved locally to: organized_tweet_ids_summary.json")
    print("Full data saved to Modal volume: /data/organized_tweet_ids.pkl")
    print()
    print("You can now:")
    print("  1. Review the summary to verify organization")
    print("  2. Run clustering script (to be created next)")
    print()
