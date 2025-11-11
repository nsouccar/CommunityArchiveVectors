"""
Extract embeddings from Modal's persistent volume for clustering
"""
import modal
import pickle
import random
from collections import defaultdict

app = modal.App("extract-embeddings")

volume = modal.Volume.from_name("tweet-vectors-large", create_if_missing=False)

@app.function(
    volumes={"/data": volume},
    timeout=3600,
)
def extract_community_embeddings(sample_size_per_community=1000):
    """
    Extract embeddings for sampled tweets from each community
    """
    import numpy as np
    from collections import Counter

    print("Loading metadata...")
    with open('/data/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)['metadata']
    print(f"Loaded {len(metadata):,} tweets")

    print("\nLoading user communities...")
    with open('/data/user_communities.pkl', 'rb') as f:
        user_to_community = pickle.load(f)
    print(f"Loaded {len(user_to_community):,} users")

    # Group tweets by community
    print("\nGrouping tweets by community...")
    community_tweets = defaultdict(list)

    for tweet_id, tweet_data in metadata.items():
        username = tweet_data.get('username')
        if username in user_to_community:
            comm_id = user_to_community[username]
            community_tweets[comm_id].append({
                'tweet_id': tweet_id,
                'text': tweet_data.get('full_text', ''),
                'username': username,
                'created_at': tweet_data.get('created_at', '')
            })

    # Sort by size
    sorted_communities = sorted(
        community_tweets.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )

    print(f"\nFound {len(community_tweets)} communities")
    for i, (comm_id, tweets) in enumerate(sorted_communities[:10], 1):
        print(f"  {i}. Community {comm_id}: {len(tweets):,} tweets")

    # Load embeddings file
    print("\nLoading embeddings from Modal volume...")
    # The embeddings are stored in the CoreNN database files
    # We need to check what format they're in

    import os
    files = os.listdir('/data')
    print(f"Files in /data: {files}")

    # Look for embedding files
    embedding_files = [f for f in files if 'embed' in f.lower() or f.endswith('.npy') or f.endswith('.bin')]
    print(f"Found embedding files: {embedding_files}")

    # Check if there's a corenn_db directory
    if 'corenn_db' in files:
        corenn_files = os.listdir('/data/corenn_db')
        print(f"Files in corenn_db: {corenn_files[:20]}")

    # Sample tweets from each community
    sampled_data = {}

    for comm_id, tweets in sorted_communities[:10]:
        print(f"\nProcessing Community {comm_id}...")

        # Sample tweets
        if len(tweets) > sample_size_per_community:
            sampled = random.sample(tweets, sample_size_per_community)
        else:
            sampled = tweets

        sampled_data[comm_id] = {
            'num_tweets': len(tweets),
            'num_users': len(set(t['username'] for t in tweets)),
            'sampled_tweets': sampled,
            'top_users': [u for u, _ in Counter(t['username'] for t in tweets).most_common(10)]
        }

        print(f"  Sampled {len(sampled)} tweets")

    # Save sampled data
    print("\nSaving sampled community data...")
    with open('/data/community_samples.pkl', 'wb') as f:
        pickle.dump(sampled_data, f)

    print("✓ Saved to /data/community_samples.pkl")

    return {
        'num_communities': len(sampled_data),
        'total_sampled_tweets': sum(len(d['sampled_tweets']) for d in sampled_data.values()),
        'files_in_data': files,
        'embedding_files': embedding_files
    }

@app.local_entrypoint()
def main():
    """Run extraction and download results"""
    print("="*80)
    print("EXTRACTING EMBEDDINGS FROM MODAL")
    print("="*80)

    result = extract_community_embeddings.remote()

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Communities processed: {result['num_communities']}")
    print(f"Total sampled tweets: {result['total_sampled_tweets']:,}")
    print(f"\nFiles in Modal /data:")
    for f in result['files_in_data']:
        print(f"  - {f}")

    if result['embedding_files']:
        print(f"\nEmbedding files found:")
        for f in result['embedding_files']:
            print(f"  - {f}")

    print("\nSaved community samples to Modal volume at /data/community_samples.pkl")
    print("Next step: Download this file and run clustering locally")
