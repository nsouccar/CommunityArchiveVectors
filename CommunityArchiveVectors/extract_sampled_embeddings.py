"""
Extract embeddings for sampled tweets from each community
This creates a small file (~100MB) instead of transferring all 58GB
"""
import modal
import pickle

app = modal.App("extract-sampled-embeddings")

volume = modal.Volume.from_name("tweet-vectors-large", create_if_missing=False)

image = modal.Image.debian_slim().pip_install("numpy")

@app.function(
    volumes={"/data": volume},
    image=image,
    timeout=3600,
    cpu=4,
    memory=16384,  # 16GB RAM to handle batch files
)
def extract_embeddings_for_communities():
    """
    Extract embeddings for sampled tweets from each community
    """
    import numpy as np
    from collections import defaultdict, Counter
    import random

    random.seed(42)

    print("="*80)
    print("STEP 1: LOAD METADATA AND COMMUNITIES")
    print("="*80)

    # Load metadata
    print("\nLoading metadata...")
    with open('/data/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)['metadata']
    print(f"✓ Loaded {len(metadata):,} tweets")

    # Load user communities
    print("\nLoading user communities...")
    with open('/data/user_communities.pkl', 'rb') as f:
        user_to_community = pickle.load(f)
    print(f"✓ Loaded {len(user_to_community):,} users")

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

    # Sort by size and take top 10
    sorted_communities = sorted(
        community_tweets.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )[:10]

    print(f"\n✓ Found {len(community_tweets)} communities")
    print("\nTop 10 communities:")
    for i, (comm_id, tweets) in enumerate(sorted_communities, 1):
        print(f"  {i}. Community {comm_id}: {len(tweets):,} tweets")

    # Sample tweets from each community
    sample_size = 1000
    sampled_communities = {}

    for comm_id, tweets in sorted_communities:
        if len(tweets) > sample_size:
            sampled = random.sample(tweets, sample_size)
        else:
            sampled = tweets

        sampled_communities[comm_id] = {
            'tweets': sampled,
            'num_total_tweets': len(tweets),
            'num_users': len(set(t['username'] for t in tweets)),
            'top_users': [u for u, _ in Counter(t['username'] for t in tweets).most_common(10)]
        }

    # Get list of all tweet IDs we need embeddings for
    all_tweet_ids = set()
    for comm_data in sampled_communities.values():
        for tweet in comm_data['tweets']:
            all_tweet_ids.add(tweet['tweet_id'])

    print(f"\n✓ Sampled {len(all_tweet_ids):,} total tweets across all communities")

    print("\n" + "="*80)
    print("STEP 2: LOAD BATCH FILES AND EXTRACT EMBEDDINGS")
    print("="*80)

    # Load all batch files and build tweet_id → embedding mapping
    tweet_embeddings = {}

    for batch_num in range(1, 65):  # 64 batches
        batch_file = f'/data/batches/batch_{batch_num:04d}.pkl'

        print(f"\nProcessing batch {batch_num}/64...")

        try:
            with open(batch_file, 'rb') as f:
                batch_data = pickle.load(f)

            embeddings_list = batch_data['embeddings']
            metadata_list = batch_data['metadata']

            # Find tweets we need from this batch
            found_in_batch = 0
            for i, (emb, meta) in enumerate(zip(embeddings_list, metadata_list)):
                tweet_id = meta['tweet_id']
                if tweet_id in all_tweet_ids:
                    tweet_embeddings[tweet_id] = np.array(emb)
                    found_in_batch += 1

            print(f"  ✓ Found {found_in_batch} needed tweets in this batch")
            print(f"  ✓ Total collected: {len(tweet_embeddings):,}/{len(all_tweet_ids):,}")

            # Stop early if we found everything
            if len(tweet_embeddings) >= len(all_tweet_ids):
                print(f"\n✓ Found all needed embeddings!")
                break

        except Exception as e:
            print(f"  ✗ Error loading batch {batch_num}: {e}")

    print("\n" + "="*80)
    print("STEP 3: SAVE EXTRACTED DATA")
    print("="*80)

    # Save the extracted data
    output = {
        'communities': sampled_communities,
        'embeddings': tweet_embeddings
    }

    output_file = '/data/sampled_community_embeddings.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(output, f)

    # Calculate file size
    import os
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)

    print(f"\n✓ Saved to {output_file}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Communities: {len(sampled_communities)}")
    print(f"  Embeddings: {len(tweet_embeddings):,}")

    return {
        'num_communities': len(sampled_communities),
        'num_embeddings': len(tweet_embeddings),
        'file_size_mb': file_size_mb,
        'output_file': output_file
    }

@app.local_entrypoint()
def main():
    """Run extraction"""
    print("="*80)
    print("EXTRACTING SAMPLED EMBEDDINGS FROM MODAL")
    print("="*80)

    result = extract_embeddings_for_communities.remote()

    print("\n" + "="*80)
    print("✅ EXTRACTION COMPLETE")
    print("="*80)
    print(f"Communities: {result['num_communities']}")
    print(f"Embeddings: {result['num_embeddings']:,}")
    print(f"File size: {result['file_size_mb']:.1f} MB")
    print(f"\nFile saved to Modal volume: {result['output_file']}")
    print("\nNext step: Download this file to run clustering locally")
    print("  modal volume get tweet-vectors-large sampled_community_embeddings.pkl sampled_community_embeddings.pkl")
