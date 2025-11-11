"""
Generate community topics using k-means clustering + LLM on Modal
This runs where the embeddings already are - no transfer needed!
"""

import modal
import pickle
import json
import random
from pathlib import Path

app = modal.App("generate-community-topics")
volume = modal.Volume.from_name("tweet-vectors-large", create_if_missing=False)

# Image with dependencies
image = modal.Image.debian_slim().pip_install(
    "anthropic",
    "scikit-learn",
    "numpy"
)

@app.function(
    volumes={"/data": volume},
    image=image,
    timeout=3600,  # 1 hour
    cpu=8,  # More CPUs for faster k-means
    memory=16384,  # 16GB RAM
    secrets=[modal.Secret.from_name("anthropic-key")]
)
def generate_topics_for_all_communities():
    """
    Generate topic labels for all communities using k-means + Claude
    """
    import numpy as np
    from sklearn.cluster import KMeans
    from anthropic import Anthropic
    import os

    print("=" * 80)
    print("GENERATING COMMUNITY TOPICS WITH K-MEANS + LLM")
    print("=" * 80)
    print()

    # Initialize Claude client
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Load metadata to get community assignments
    print("Loading metadata...")
    metadata_file = "/data/metadata.json"
    with open(metadata_file, 'r') as f:
        all_metadata = json.load(f)

    print(f"Loaded {len(all_metadata):,} tweets")
    print()

    # Group tweets by community
    print("Grouping tweets by community...")
    communities = {}
    for tweet in all_metadata:
        comm_id = tweet.get('community')
        if comm_id is not None:
            if comm_id not in communities:
                communities[comm_id] = []
            communities[comm_id].append(tweet)

    print(f"Found {len(communities)} communities")
    print()

    # Sort communities by size
    sorted_communities = sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)

    # Load embeddings from batches
    print("Loading embeddings from batches...")
    batches_dir = Path("/data/batches")
    batch_files = sorted(batches_dir.glob("batch_*.pkl"))

    # Create tweet_id -> embedding mapping
    tweet_embeddings = {}
    for batch_file in batch_files:
        with open(batch_file, 'rb') as f:
            batch_data = pickle.load(f)
            batch_metadata = batch_data['metadata']
            batch_embeddings = batch_data['embeddings']

            for meta, emb in zip(batch_metadata, batch_embeddings):
                tweet_id = meta['tweet_id']
                tweet_embeddings[tweet_id] = emb

    print(f"Loaded embeddings for {len(tweet_embeddings):,} tweets")
    print()

    # Process each community
    results = {
        "num_communities": len(communities),
        "total_tweets_analyzed": len(all_metadata),
        "communities": {}
    }

    # Process top 10 largest communities for now
    for comm_id, tweets in sorted_communities[:10]:
        print("=" * 80)
        print(f"COMMUNITY {comm_id}: {len(tweets):,} tweets")
        print("=" * 80)
        print()

        # Sample tweets (max 1000 per community for clustering)
        sample_size = min(1000, len(tweets))
        sampled_tweets = random.sample(tweets, sample_size)

        # Get embeddings for sampled tweets
        embeddings_list = []
        valid_tweets = []
        for tweet in sampled_tweets:
            tweet_id = tweet['tweet_id']
            if tweet_id in tweet_embeddings:
                embeddings_list.append(tweet_embeddings[tweet_id])
                valid_tweets.append(tweet)

        if len(embeddings_list) < 50:
            print(f"⚠️  Skipping community {comm_id}: not enough embeddings ({len(embeddings_list)})")
            print()
            continue

        print(f"Got embeddings for {len(embeddings_list):,} tweets")
        print()

        # Convert to numpy array
        X = np.array(embeddings_list)

        # Run k-means clustering (5 clusters per community)
        print("Running k-means clustering (5 clusters)...")
        n_clusters = min(5, len(embeddings_list) // 20)  # At least 20 tweets per cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        print(f"Clustered into {n_clusters} groups")
        print()

        # For each cluster, get representative tweets and use LLM to label
        topics = []
        for cluster_id in range(n_clusters):
            cluster_tweets = [valid_tweets[i] for i in range(len(labels)) if labels[i] == cluster_id]

            # Sample 20 tweets from this cluster for LLM
            sample_for_llm = random.sample(cluster_tweets, min(20, len(cluster_tweets)))
            tweet_texts = [t['text'] for t in sample_for_llm]

            # Create prompt for Claude
            prompt = f"""Analyze these {len(tweet_texts)} tweets from a Twitter community cluster and generate a concise topic label (2-5 words).

Tweets:
{chr(10).join(f'{i+1}. {text[:200]}' for i, text in enumerate(tweet_texts))}

Based on the common themes in these tweets, provide a single topic label that captures what this cluster is discussing. Examples of good labels:
- "Love and Relationships"
- "Ukraine and Russia War"
- "Crypto and Bitcoin"
- "AI and Technology"
- "Climate Change Politics"

Respond with ONLY the topic label, nothing else."""

            # Call Claude Haiku (fast and cheap)
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=50,
                messages=[{"role": "user", "content": prompt}]
            )

            topic_label = response.content[0].text.strip()

            topics.append({
                "topic": topic_label,
                "num_tweets": len(cluster_tweets),
                "sample_tweets": tweet_texts[:3]  # Save 3 example tweets
            })

            print(f"  Cluster {cluster_id + 1}: {topic_label} ({len(cluster_tweets)} tweets)")

        print()

        # Save community results
        results["communities"][str(comm_id)] = {
            "num_tweets": len(tweets),
            "num_users": len(set(t['username'] for t in tweets)),
            "top_users": [u for u, _ in sorted(
                [(username, sum(1 for t in tweets if t['username'] == username))
                 for username in set(t['username'] for t in tweets)],
                key=lambda x: x[1], reverse=True
            )[:10]],
            "topics": topics
        }

    print()
    print("=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    print()

    # Save results to volume
    output_file = "/data/community_topics_kmeans.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Saved to {output_file}")
    print()

    # Also return the results so we can save locally
    return results

@app.local_entrypoint()
def main():
    """Run topic generation and save results locally"""
    print()
    print("=" * 80)
    print("GENERATING COMMUNITY TOPICS")
    print("=" * 80)
    print()
    print("This will:")
    print("  1. Load embeddings from Modal volume (already there!)")
    print("  2. Run k-means clustering on each community")
    print("  3. Use Claude to generate topic labels")
    print("  4. Save results to JSON")
    print()
    print("Estimated time: 10-20 minutes")
    print()

    # Run the function
    results = generate_topics_for_all_communities.remote()

    # Save locally
    output_file = "community_topics_llm.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print()
    print(f"✓ Results saved to {output_file}")
    print(f"✓ Processed {results['num_communities']} communities")
    print(f"✓ Analyzed {results['total_tweets_analyzed']:,} total tweets")
    print()
    print("You can now copy this file to frontend/public/ and deploy!")
    print()
