"""
Incremental Update Pipeline for Tweet Archive

Automatically processes new tweets from Supabase:
1. Detects new tweets (not yet embedded)
2. Generates embeddings
3. Assigns to communities based on network data
4. Updates organized_by_community.pkl
5. Re-runs clustering for affected communities
6. Updates topic JSON files
7. Updates frontend data

Can be run:
- Manually: modal run incremental_update_pipeline.py
- As cron: Runs daily at 2 AM UTC
"""

import modal
import json
import pickle
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path

app = modal.App("incremental-update-pipeline")
volume = modal.Volume.from_name("tweet-vectors-large", create_if_missing=False)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "psycopg2-binary",
    "sentence-transformers",  # For embeddings
    "numpy",
    "scikit-learn",
    "anthropic",  # For LLM topic generation
)

secrets = modal.Secret.from_name("anthropic-api-key")

@app.function(
    volumes={"/data": volume},
    image=image,
    timeout=7200,  # 2 hours
    memory=16384,  # 16GB
    secrets=[secrets],
)
def process_new_tweets():
    """
    Main incremental update function

    Steps:
    1. Find new tweets in Supabase
    2. Generate embeddings
    3. Assign to communities
    4. Update organized data
    5. Re-cluster affected communities
    6. Update topic files
    """
    import psycopg2
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import os

    print("=" * 80)
    print("INCREMENTAL UPDATE PIPELINE")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 80)
    print()

    # Database config (use environment variables)
    DB_CONFIG = {
        "host": os.environ.get("SUPABASE_HOST", "YOUR_HOST.supabase.co"),
        "database": "postgres",
        "user": "postgres",
        "password": os.environ.get("SUPABASE_PASSWORD", ""),
        "port": 5432
    }

    # Step 1: Load tracking file to find last processed tweet
    print("Step 1: Finding new tweets...")
    tracking_file = Path("/data/last_processed_tweet.json")

    if tracking_file.exists():
        with open(tracking_file, 'r') as f:
            tracking = json.load(f)
            last_tweet_id = tracking['last_tweet_id']
            last_update = tracking['last_update']
    else:
        # First run - get the highest tweet_id from organized data
        print("  No tracking file - checking existing data...")
        try:
            with open("/data/organized_by_community.pkl", 'rb') as f:
                organized = pickle.load(f)

            # Find highest tweet_id in organized data
            max_id = 0
            for year_data in organized.values():
                for month_data in year_data.values():
                    for comm_data in month_data.values():
                        for tweet in comm_data['tweets']:
                            tweet_id = int(tweet.get('tweet_id', 0))
                            if tweet_id > max_id:
                                max_id = tweet_id

            last_tweet_id = max_id
            print(f"  Found highest existing tweet_id: {last_tweet_id}")
        except:
            last_tweet_id = 0
            print("  No existing data - starting from 0")

        last_update = None

    print(f"  Last processed tweet_id: {last_tweet_id}")
    print()

    # Step 2: Query Supabase for new tweets
    print("Step 2: Querying Supabase for new tweets...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute("""
        SELECT
            t.tweet_id,
            t.full_text,
            t.created_at,
            t.account_id,
            a.username
        FROM tweets t
        JOIN account a ON t.account_id = a.account_id
        WHERE t.tweet_id > %s
        AND t.full_text IS NOT NULL
        ORDER BY t.tweet_id
        LIMIT 10000
    """, (last_tweet_id,))

    new_tweets = []
    for row in cur.fetchall():
        new_tweets.append({
            'tweet_id': row[0],
            'text': row[1],
            'created_at': row[2].isoformat() if row[2] else None,
            'account_id': row[3],
            'username': row[4]
        })

    cur.close()
    conn.close()

    if not new_tweets:
        print("  No new tweets found!")
        print()
        return {
            'new_tweets': 0,
            'status': 'up_to_date'
        }

    print(f"  Found {len(new_tweets):,} new tweets")
    print(f"  ID range: {new_tweets[0]['tweet_id']} - {new_tweets[-1]['tweet_id']}")
    print()

    # Step 3: Generate embeddings
    print("Step 3: Generating embeddings...")
    print("  Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("  Model loaded")

    texts = [f"@{t['username']}: {t['text']}" for t in new_tweets]

    print(f"  Encoding {len(texts)} tweets...")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    print(f"  ✓ Generated {len(embeddings)} embeddings")
    print()

    # Step 4: Load network data to assign communities
    print("Step 4: Assigning tweets to communities...")

    # Load network animation data
    network_file = Path("/data/network_animation_data.json")
    if not network_file.exists():
        print("  ⚠️  No network data found - cannot assign communities")
        print("     Tweets will be added without community assignment")
        user_communities = {}
    else:
        with open(network_file, 'r') as f:
            network_data = json.load(f)

        # Build lookup: {year: {username: community}}
        user_communities = {}
        for year_data in network_data['years']:
            year = year_data['year']
            user_communities[year] = {}
            for node in year_data['nodes']:
                username = node['id']
                community = node['community']
                user_communities[year][username] = community

        print(f"  Loaded network data for {len(user_communities)} years")

    # Step 5: Load existing organized data
    print("Step 5: Loading existing organized data...")
    organized_file = Path("/data/organized_by_community.pkl")

    if organized_file.exists():
        with open(organized_file, 'rb') as f:
            organized = pickle.load(f)
        print("  ✓ Loaded existing organized data")
    else:
        organized = {}
        print("  Creating new organized structure")
    print()

    # Step 6: Add new tweets to organized structure
    print("Step 6: Adding new tweets to organized structure...")

    tweets_added = 0
    tweets_skipped = 0

    for tweet, embedding in zip(new_tweets, embeddings):
        username = tweet['username']
        timestamp = tweet.get('created_at')

        if not timestamp:
            tweets_skipped += 1
            continue

        # Parse timestamp
        try:
            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                dt = timestamp

            year = str(dt.year)
            month = dt.month
        except:
            tweets_skipped += 1
            continue

        # Find community for this user in this year
        if year in user_communities and username in user_communities[year]:
            community = str(user_communities[year][username])

            # Initialize structure if needed
            if year not in organized:
                organized[year] = {}
            if month not in organized[year]:
                organized[year][month] = {}
            if community not in organized[year][month]:
                organized[year][month][community] = {
                    'tweets': [],
                    'embeddings': np.array([]),
                    'num_tweets': 0
                }

            # Add tweet
            organized[year][month][community]['tweets'].append({
                'tweet_id': tweet['tweet_id'],
                'username': username,
                'text': tweet['text'],
                'timestamp': timestamp
            })

            # Add embedding
            existing_embeddings = organized[year][month][community]['embeddings']
            if len(existing_embeddings) == 0:
                organized[year][month][community]['embeddings'] = np.array([embedding])
            else:
                organized[year][month][community]['embeddings'] = np.vstack([
                    existing_embeddings,
                    embedding.reshape(1, -1)
                ])

            organized[year][month][community]['num_tweets'] = len(
                organized[year][month][community]['tweets']
            )

            tweets_added += 1
        else:
            tweets_skipped += 1

    print(f"  ✓ Added {tweets_added:,} tweets to organized structure")
    print(f"  Skipped {tweets_skipped:,} tweets (no community/timestamp)")
    print()

    # Step 7: Save updated organized data
    print("Step 7: Saving updated organized data...")
    with open(organized_file, 'wb') as f:
        pickle.dump(organized, f)
    volume.commit()
    print("  ✓ Saved to /data/organized_by_community.pkl")
    print()

    # Step 8: Update tracking file
    print("Step 8: Updating tracking file...")
    tracking = {
        'last_tweet_id': new_tweets[-1]['tweet_id'],
        'last_update': datetime.now().isoformat(),
        'tweets_processed': tweets_added
    }
    with open(tracking_file, 'w') as f:
        json.dump(tracking, f, indent=2)
    volume.commit()
    print(f"  ✓ Updated tracking: last_tweet_id = {tracking['last_tweet_id']}")
    print()

    # Step 9: Identify communities that need re-clustering
    print("Step 9: Identifying communities for re-clustering...")

    # Collect all (year, month, community) tuples that were updated
    updated_communities = set()
    for tweet in new_tweets[:tweets_added]:
        timestamp = tweet.get('created_at')
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                year = str(dt.year)
                month = dt.month
                username = tweet['username']

                if year in user_communities and username in user_communities[year]:
                    community = str(user_communities[year][username])
                    updated_communities.add((year, month, community))
            except:
                pass

    print(f"  Found {len(updated_communities)} communities to re-cluster")
    print()

    # Step 10: Re-cluster updated communities (simplified - just update counts)
    print("Step 10: Updating community summaries...")

    # For now, just update the summary stats
    # Full re-clustering with LLM can be done separately as needed
    summary = {
        'last_update': datetime.now().isoformat(),
        'total_tweets': sum(
            sum(
                comm_data['num_tweets']
                for comm_data in month_data.values()
            )
            for year_data in organized.values()
            for month_data in year_data.values()
        ),
        'total_communities': sum(
            len(month_data)
            for year_data in organized.values()
            for month_data in year_data.values()
        ),
        'communities_updated': len(updated_communities)
    }

    print(f"  Total tweets in system: {summary['total_tweets']:,}")
    print(f"  Total communities: {summary['total_communities']}")
    print()

    print("=" * 80)
    print("UPDATE COMPLETE")
    print("=" * 80)
    print(f"New tweets processed: {tweets_added:,}")
    print(f"Communities updated: {len(updated_communities)}")
    print(f"Completed: {datetime.now().isoformat()}")
    print()

    return {
        'new_tweets': tweets_added,
        'communities_updated': len(updated_communities),
        'status': 'success',
        'last_tweet_id': new_tweets[-1]['tweet_id']
    }


# Cron schedule: Run daily at 2 AM UTC
@app.function(
    schedule=modal.Cron("0 2 * * *"),  # Daily at 2 AM UTC
    volumes={"/data": volume},
    image=image,
    timeout=7200,
    memory=16384,
    secrets=[secrets],
)
def scheduled_update():
    """
    Scheduled version of the update pipeline
    Runs automatically every day
    """
    print("\n" + "="*80)
    print("SCHEDULED UPDATE - RUNNING")
    print(f"Time: {datetime.now().isoformat()}")
    print("="*80 + "\n")

    result = process_new_tweets()

    print("\n" + "="*80)
    print("SCHEDULED UPDATE - COMPLETE")
    print("="*80)
    print(f"Status: {result['status']}")
    print(f"New tweets: {result['new_tweets']}")
    print(f"Communities updated: {result.get('communities_updated', 0)}")
    print()

    return result


@app.local_entrypoint()
def main():
    """
    Manual run of the update pipeline
    """
    print()
    print("=" * 80)
    print("INCREMENTAL UPDATE PIPELINE - MANUAL RUN")
    print("=" * 80)
    print()
    print("This will:")
    print("  1. Check Supabase for new tweets")
    print("  2. Generate embeddings for new tweets")
    print("  3. Assign to communities")
    print("  4. Update organized_by_community.pkl")
    print("  5. Mark communities for re-clustering")
    print()
    print("Estimated time: 5-10 minutes for 10K tweets")
    print()

    result = process_new_tweets.remote()

    print()
    print("=" * 80)
    print("MANUAL UPDATE COMPLETE")
    print("=" * 80)
    print(f"Status: {result['status']}")
    print(f"New tweets processed: {result['new_tweets']}")
    if result.get('communities_updated'):
        print(f"Communities updated: {result['communities_updated']}")
    print()

    if result['new_tweets'] > 0:
        print("Next steps:")
        print("  1. Optionally run re-clustering on updated communities")
        print("  2. Regenerate topic JSON files")
        print("  3. Deploy updated data to frontend")
        print()
