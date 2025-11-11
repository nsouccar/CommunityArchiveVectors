"""
Organize embeddings by (year, month, community) directly on Modal

Reads from batch files on Modal volume, organizes by community, saves back to volume.
Memory efficient - processes batches one at a time.
"""

import modal
import pickle
import json
from datetime import datetime
from collections import defaultdict
from pathlib import Path

app = modal.App("organize-community-embeddings")
volume = modal.Volume.from_name("tweet-vectors-large", create_if_missing=False)

image = modal.Image.debian_slim().pip_install("numpy")

@app.function(
    volumes={"/data": volume},
    image=image,
    timeout=7200,  # 2 hours
    cpu=8,
    memory=16384,  # 16GB - enough for processing batches
)
def organize_embeddings(network_data_json: str):
    """
    Organize embeddings by (year, month, community)
    Reads metadata from batch files, organizes, saves with embeddings
    """
    import numpy as np

    print("=" * 80)
    print("ORGANIZING EMBEDDINGS BY (YEAR, MONTH, COMMUNITY)")
    print("=" * 80)
    print()

    # Step 1: Parse network data to get user → community mapping
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

    # Step 2: Process batch files one at a time (memory efficient!)
    print("Step 2: Processing batch files...")
    batches_dir = Path("/data/batches")
    batch_files = sorted(batches_dir.glob("batch_*.pkl"))

    print(f"Found {len(batch_files)} batch files to process")
    print()

    # Structure: {year: {month: {community: {'tweets': [...], 'embeddings': [...]}}}}
    organized = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'tweets': [], 'embeddings': []})))

    stats = {
        'total_tweets': 0,
        'matched': 0,
        'unmatched': 0,
        'no_username': 0,
        'no_timestamp': 0,
    }

    # Process each batch file
    for i, batch_file in enumerate(batch_files, 1):
        if i % 10 == 0:
            print(f"Processing batch {i}/{len(batch_files)}...")

        with open(batch_file, 'rb') as f:
            batch_data = pickle.load(f)

        batch_metadata = batch_data['metadata']
        batch_embeddings = batch_data['embeddings']

        # Process each tweet in this batch
        for meta, emb in zip(batch_metadata, batch_embeddings):
            stats['total_tweets'] += 1

            username = meta.get('username')
            timestamp = meta.get('created_at')
            tweet_id = meta.get('tweet_id')

            if not username:
                stats['no_username'] += 1
                continue

            if not timestamp:
                stats['no_timestamp'] += 1
                continue

            # Parse timestamp
            try:
                if isinstance(timestamp, str):
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    except:
                        try:
                            dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                        except:
                            dt = datetime.strptime(timestamp, '%a %b %d %H:%M:%S %z %Y')
                else:
                    dt = timestamp

                year = str(dt.year)
                month = dt.month

            except Exception as e:
                stats['unmatched'] += 1
                continue

            # Find community for this user in this year
            if year in user_communities and username in user_communities[year]:
                community = str(user_communities[year][username])

                # Add to organized structure
                organized[year][month][community]['tweets'].append({
                    'tweet_id': tweet_id,
                    'username': username,
                    'text': meta.get('full_text', ''),  # Changed from 'text' to 'full_text'
                    'timestamp': str(timestamp) if not isinstance(timestamp, str) else timestamp
                })
                organized[year][month][community]['embeddings'].append(emb)

                stats['matched'] += 1
            else:
                stats['unmatched'] += 1

    print()
    print(f"✓ Processed all {len(batch_files)} batch files")
    print(f"  Total tweets: {stats['total_tweets']:,}")
    print(f"  Matched to communities: {stats['matched']:,}")
    print(f"  Unmatched: {stats['unmatched']:,}")
    print(f"  Skipped (no username): {stats['no_username']:,}")
    print(f"  Skipped (no timestamp): {stats['no_timestamp']:,}")
    print()

    # Step 3: Convert embeddings lists to numpy arrays
    print("Step 3: Converting to numpy arrays...")
    for year in organized:
        for month in organized[year]:
            for community in organized[year][month]:
                emb_list = organized[year][month][community]['embeddings']
                organized[year][month][community]['embeddings'] = np.array(emb_list)

    print("✓ Converted embeddings to numpy arrays")
    print()

    # Step 4: Print summary
    print("=" * 80)
    print("ORGANIZATION SUMMARY")
    print("=" * 80)
    print()

    total_groups = 0
    for year in sorted(organized.keys()):
        year_tweets = sum(
            len(organized[year][month][community]['tweets'])
            for month in organized[year]
            for community in organized[year][month]
        )
        num_months = len(organized[year])
        communities_in_year = set()
        for month in organized[year]:
            for community in organized[year][month]:
                communities_in_year.add(community)
        num_communities = len(communities_in_year)

        print(f"Year {year}:")
        print(f"  {num_months} months with data")
        print(f"  {num_communities} communities")
        print(f"  {year_tweets:,} tweets")

        # Show sample months
        for month in sorted(organized[year].keys())[:3]:
            month_tweets = sum(len(organized[year][month][c]['tweets']) for c in organized[year][month])
            print(f"    Month {month}: {month_tweets:,} tweets across {len(organized[year][month])} communities")

        if len(organized[year]) > 3:
            print(f"    ... and {len(organized[year]) - 3} more months")
        print()

        total_groups += sum(len(organized[year][month]) for month in organized[year])

    print(f"Total (year, month, community) groups: {total_groups}")
    print()

    # Step 5: Save organized data
    print("=" * 80)
    print("SAVING TO MODAL VOLUME")
    print("=" * 80)
    print()

    # Convert defaultdicts to regular dicts
    organized_dict = {}
    for year in organized:
        organized_dict[year] = {}
        for month in organized[year]:
            organized_dict[year][month] = dict(organized[year][month])

    # Save full data with embeddings
    output_path = "/data/organized_by_community.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(organized_dict, f)
    print(f"✓ Saved to {output_path}")

    # Create summary JSON (without embeddings)
    summary = {
        'total_tweets': stats['matched'],
        'total_groups': total_groups,
        'years': {}
    }

    for year in organized_dict:
        summary['years'][year] = {
            'months': {},
            'total_tweets': sum(
                len(organized_dict[year][month][c]['tweets'])
                for month in organized_dict[year]
                for c in organized_dict[year][month]
            ),
            'num_communities': len(set(
                c for month in organized_dict[year] for c in organized_dict[year][month]
            ))
        }

        for month in organized_dict[year]:
            summary['years'][year]['months'][month] = {
                'communities': {},
                'total_tweets': sum(len(organized_dict[year][month][c]['tweets']) for c in organized_dict[year][month])
            }

            for community in organized_dict[year][month]:
                data = organized_dict[year][month][community]
                summary['years'][year]['months'][month]['communities'][community] = {
                    'num_tweets': len(data['tweets']),
                    'embedding_shape': list(data['embeddings'].shape),
                    'sample_usernames': list(set(t['username'] for t in data['tweets'][:10]))
                }

    summary_path = "/data/organized_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary to {summary_path}")

    # Commit to volume
    volume.commit()
    print(f"✓ Committed changes to volume")
    print()

    print("=" * 80)
    print("ORGANIZATION COMPLETE!")
    print("=" * 80)
    print()
    print(f"Files saved to Modal volume:")
    print(f"  1. {output_path} - Full data with embeddings")
    print(f"  2. {summary_path} - Human-readable summary")
    print()
    print("Next step: Run clustering on each (year, month, community) group")
    print()

    return summary

@app.local_entrypoint()
def main():
    """Run organization"""
    print()
    print("=" * 80)
    print("ORGANIZE EMBEDDINGS BY (YEAR, MONTH, COMMUNITY)")
    print("=" * 80)
    print()
    print("This will:")
    print("  1. Read all 64 batch files from Modal volume")
    print("  2. Match tweets to communities using network data")
    print("  3. Organize by (year, month, community)")
    print("  4. Save organized structure back to Modal volume")
    print()
    print("Memory: 16GB (processes batches one at a time)")
    print("Time: ~20-30 minutes")
    print()

    # Load network data
    print("Loading network_animation_data.json...")
    with open('network_animation_data.json', 'r') as f:
        network_data_json = f.read()
    print("✓ Loaded network data (6.5 MB)")
    print()

    # Run on Modal
    summary = organize_embeddings.remote(network_data_json)

    # Save summary locally
    with open('organized_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 80)
    print("DONE!")
    print("=" * 80)
    print()
    print("Summary saved locally: organized_summary.json")
    print("Full data on Modal: /data/organized_by_community.pkl")
    print()
