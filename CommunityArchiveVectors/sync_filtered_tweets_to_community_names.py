"""
Sync filtered tweet IDs from topics_year_*_summary.json to community_names_*.json
"""

import json
from pathlib import Path

def main():
    years = [2012, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    data_dir = Path("frontend/public/data")

    print(f"\n{'='*80}")
    print(f"SYNCING FILTERED TWEET IDS TO COMMUNITY NAMES")
    print(f"{'='*80}\n")

    for year in years:
        topics_file = data_dir / f"topics_year_{year}_summary.json"
        names_file = data_dir / f"community_names_{year}.json"

        if not topics_file.exists():
            print(f"⚠ Skipped {year}: topics file not found")
            continue

        if not names_file.exists():
            print(f"⚠ Skipped {year}: community names file not found")
            continue

        # Load both files
        with open(topics_file, 'r') as f:
            topics_data = json.load(f)

        with open(names_file, 'r') as f:
            names_data = json.load(f)

        # Sync tweet IDs from topics to community names
        synced_count = 0
        for community in names_data.get('communities', []):
            community_id = str(community['community_id'])

            if community_id not in topics_data['communities']:
                print(f"  Warning: Community {community_id} not found in topics file")
                continue

            # Get filtered topics from topics_data
            filtered_topics = topics_data['communities'][community_id]
            filtered_topics_map = {t['cluster_id']: t for t in filtered_topics}

            # Update tweet_ids in community names
            for topic in community.get('topics', []):
                cluster_id = topic['cluster_id']

                if cluster_id not in filtered_topics_map:
                    print(f"  Warning: Cluster {cluster_id} not found in filtered topics")
                    continue

                # Skip topics that don't have tweet_ids field
                if 'tweet_ids' not in topic:
                    print(f"  Warning: Cluster {cluster_id} missing tweet_ids field, skipping")
                    continue

                filtered_topic = filtered_topics_map[cluster_id]
                old_count = len(topic['tweet_ids'])
                new_count = len(filtered_topic['tweet_ids'])

                if old_count != new_count:
                    topic['tweet_ids'] = filtered_topic['tweet_ids']
                    topic['num_tweets'] = new_count
                    synced_count += 1
                    print(f"  {year} Community {community_id} Cluster {cluster_id}: {old_count} -> {new_count} tweets")

        # Save updated community names file
        if synced_count > 0:
            with open(names_file, 'w') as f:
                json.dump(names_data, f, indent=2)
            print(f"✓ {year}: Updated {synced_count} clusters\n")
        else:
            print(f"✓ {year}: Already in sync (no changes)\n")

    print(f"{'='*80}")
    print(f"COMPLETE!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
