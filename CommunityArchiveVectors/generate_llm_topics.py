#!/usr/bin/env python3
"""
Generate semantic topic labels for communities using:
1. K-means clustering on tweet embeddings
2. LLM analysis of cluster contents to generate topic labels
"""

import pickle
import json
import numpy as np
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
import anthropic
import os
import random

def load_data():
    """Load metadata"""
    print("Loading metadata...")
    with open('metadata.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['metadata']

def build_network_and_communities(metadata, min_interactions=2):
    """Build network and detect communities"""
    import networkx as nx
    from networkx.algorithms import community as nx_community

    print("\nBuilding network...")

    # Build mapping: tweet_id -> username
    tweet_to_user = {}
    for tweet_id, tweet_data in metadata.items():
        tweet_to_user[tweet_id] = tweet_data.get('username', 'unknown')

    # Count interactions
    interactions = defaultdict(int)
    for tweet_id, tweet_data in metadata.items():
        from_user = tweet_data.get('username')
        reply_to_id = tweet_data.get('reply_to_tweet_id')

        if from_user and reply_to_id and reply_to_id in tweet_to_user:
            to_user = tweet_to_user[reply_to_id]
            if from_user != to_user:
                interactions[(from_user, to_user)] += 1

    # Build graph
    G = nx.DiGraph()
    for (from_user, to_user), count in interactions.items():
        if count >= min_interactions:
            G.add_edge(from_user, to_user, weight=count)

    print(f"Network: {G.number_of_nodes():,} users, {G.number_of_edges():,} interactions")

    # Detect communities
    print("Detecting communities...")
    G_undirected = G.to_undirected()
    communities = nx_community.louvain_communities(G_undirected, seed=42)

    user_to_community = {}
    for comm_id, community in enumerate(communities):
        if len(community) >= 5:
            for user in community:
                user_to_community[user] = comm_id

    print(f"Found {len(set(user_to_community.values()))} communities (5+ users)")

    return user_to_community

def get_community_tweets(metadata, user_to_community):
    """Get all tweets for each community"""
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

    return community_tweets

def get_embeddings_from_supabase(tweet_ids):
    """Fetch embeddings from Supabase"""
    from supabase import create_client

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")

    supabase = create_client(url, key)

    # Fetch embeddings in batches
    batch_size = 1000
    embeddings = {}

    for i in range(0, len(tweet_ids), batch_size):
        batch = tweet_ids[i:i+batch_size]
        response = supabase.table('tweets').select('tweet_id, embedding').in_('tweet_id', batch).execute()

        for row in response.data:
            if row['embedding']:
                embeddings[row['tweet_id']] = np.array(row['embedding'])

    return embeddings

def cluster_and_analyze_community(comm_id, tweets, n_clusters=5, sample_size=1000):
    """
    Cluster tweets and generate topic labels using LLM
    """
    print(f"\n{'='*80}")
    print(f"COMMUNITY {comm_id}: {len(tweets):,} tweets")
    print(f"{'='*80}")

    # Sample tweets if too many
    if len(tweets) > sample_size:
        sampled_tweets = random.sample(tweets, sample_size)
        print(f"Sampled {sample_size} tweets for clustering")
    else:
        sampled_tweets = tweets

    # Get tweet IDs
    tweet_ids = [t['tweet_id'] for t in sampled_tweets]

    # Fetch embeddings from Supabase
    print(f"Fetching embeddings from Supabase...")
    embeddings_dict = get_embeddings_from_supabase(tweet_ids)

    # Create matrix of embeddings
    valid_tweets = []
    embedding_matrix = []

    for tweet in sampled_tweets:
        if tweet['tweet_id'] in embeddings_dict:
            valid_tweets.append(tweet)
            embedding_matrix.append(embeddings_dict[tweet['tweet_id']])

    if len(embedding_matrix) < n_clusters:
        print(f"Not enough embeddings ({len(embedding_matrix)}) for clustering")
        return None

    embedding_matrix = np.array(embedding_matrix)
    print(f"Clustering {len(embedding_matrix)} embeddings into {n_clusters} topics...")

    # Run k-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embedding_matrix)

    # Group tweets by cluster
    clusters = defaultdict(list)
    for tweet, label in zip(valid_tweets, labels):
        clusters[label].append(tweet)

    # For each cluster, get LLM to generate a topic label
    print(f"\nGenerating topic labels with LLM...")

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    topics = []

    for cluster_id, cluster_tweets in sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True):
        # Sample tweets to show LLM (max 20)
        sample_tweets = random.sample(cluster_tweets, min(20, len(cluster_tweets)))

        # Create prompt for LLM
        tweets_text = "\n\n".join([
            f"{i+1}. @{t['username']}: {t['text'][:200]}"
            for i, t in enumerate(sample_tweets)
        ])

        prompt = f"""Here are {len(sample_tweets)} representative tweets from a cluster within a Twitter community:

{tweets_text}

Based on these tweets, generate a short, descriptive topic label (2-5 words) that captures what this cluster is discussing. Examples of good labels:
- "Ukraine and Russia war"
- "AI and machine learning"
- "Love and relationships"
- "Startup funding and VCs"
- "Climate change policy"

Respond with ONLY the topic label, nothing else."""

        try:
            message = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}]
            )

            topic_label = message.content[0].text.strip()

            # Remove quotes if LLM added them
            topic_label = topic_label.strip('"').strip("'")

            topics.append({
                'label': topic_label,
                'num_tweets': len(cluster_tweets),
                'example_tweets': [
                    {
                        'username': t['username'],
                        'text': t['text'],
                        'created_at': t['created_at']
                    }
                    for t in cluster_tweets[:3]
                ]
            })

            print(f"  Topic {cluster_id + 1}: \"{topic_label}\" ({len(cluster_tweets)} tweets)")

        except Exception as e:
            print(f"  Error generating topic label for cluster {cluster_id}: {e}")
            topics.append({
                'label': f'Topic {cluster_id + 1}',
                'num_tweets': len(cluster_tweets),
                'example_tweets': []
            })

    return topics

def save_clusters_without_labels(community_clusters, filename='community_clusters.json'):
    """Save clusters to file without LLM labels (for review/debugging)"""
    with open(filename, 'w') as f:
        json.dump(community_clusters, f, indent=2)
    print(f"\nSaved clusters to {filename}")

def label_clusters_with_llm(community_clusters):
    """Label all clusters using LLM in batch"""
    print("\n" + "="*80)
    print("LABELING CLUSTERS WITH LLM")
    print("="*80)

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nWARNING: ANTHROPIC_API_KEY not set. Skipping LLM labeling.")
        print("Run: export ANTHROPIC_API_KEY='your-key' to enable LLM labeling")
        return community_clusters

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    total_clusters = sum(len(comm['clusters']) for comm in community_clusters.values())
    print(f"\nLabeling {total_clusters} clusters across {len(community_clusters)} communities...")

    cluster_count = 0

    for comm_id, comm_data in community_clusters.items():
        print(f"\nCommunity {comm_id}:")

        for cluster in comm_data['clusters']:
            cluster_count += 1

            # Sample tweets to show LLM (max 15 for faster processing)
            sample_tweets = cluster['example_tweets'][:15]

            # Create prompt for LLM
            tweets_text = "\n\n".join([
                f"{i+1}. @{t['username']}: {t['text'][:200]}"
                for i, t in enumerate(sample_tweets)
            ])

            prompt = f"""Here are {len(sample_tweets)} representative tweets from a cluster within a Twitter community:

{tweets_text}

Based on these tweets, generate a short, descriptive topic label (2-5 words) that captures what this cluster is discussing. Examples of good labels:
- "Ukraine and Russia war"
- "AI and machine learning"
- "Love and relationships"
- "Startup funding and VCs"
- "Climate change policy"

Respond with ONLY the topic label, nothing else."""

            try:
                message = client.messages.create(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=100,
                    messages=[{"role": "user", "content": prompt}]
                )

                topic_label = message.content[0].text.strip()
                topic_label = topic_label.strip('"').strip("'")

                cluster['label'] = topic_label

                print(f"  [{cluster_count}/{total_clusters}] Cluster {cluster['cluster_id']}: \"{topic_label}\" ({cluster['num_tweets']} tweets)")

            except Exception as e:
                print(f"  Error labeling cluster {cluster['cluster_id']}: {e}")
                cluster['label'] = f"Topic {cluster['cluster_id']}"

    return community_clusters

def main(skip_llm=False):
    """
    Main function to generate community topics

    Args:
        skip_llm: If True, skip LLM labeling (useful for testing clustering)
    """
    # Load data
    metadata = load_data()

    # Build network and detect communities
    user_to_community = build_network_and_communities(metadata, min_interactions=2)

    # Get tweets for each community
    community_tweets = get_community_tweets(metadata, user_to_community)

    # Sort communities by size
    sorted_communities = sorted(
        community_tweets.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )

    print("\n" + "="*80)
    print("STEP 1: CLUSTERING COMMUNITIES")
    print("="*80)

    # First pass: cluster all communities and save clusters
    community_clusters = {}

    for comm_id, tweets in sorted_communities[:10]:
        print(f"\n{'='*80}")
        print(f"COMMUNITY {comm_id}: {len(tweets):,} tweets")
        print(f"{'='*80}")

        # Sample tweets if too many
        sample_size = 1000
        if len(tweets) > sample_size:
            sampled_tweets = random.sample(tweets, sample_size)
            print(f"Sampled {sample_size} tweets for clustering")
        else:
            sampled_tweets = tweets

        # Get tweet IDs
        tweet_ids = [t['tweet_id'] for t in sampled_tweets]

        # Fetch embeddings from Supabase
        print(f"Fetching embeddings from Supabase...")
        embeddings_dict = get_embeddings_from_supabase(tweet_ids)

        # Create matrix of embeddings
        valid_tweets = []
        embedding_matrix = []

        for tweet in sampled_tweets:
            if tweet['tweet_id'] in embeddings_dict:
                valid_tweets.append(tweet)
                embedding_matrix.append(embeddings_dict[tweet['tweet_id']])

        n_clusters = 5
        if len(embedding_matrix) < n_clusters:
            print(f"Not enough embeddings ({len(embedding_matrix)}) for clustering")
            continue

        embedding_matrix = np.array(embedding_matrix)
        print(f"Clustering {len(embedding_matrix)} embeddings into {n_clusters} topics...")

        # Run k-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embedding_matrix)

        # Group tweets by cluster
        clusters = defaultdict(list)
        for tweet, label in zip(valid_tweets, labels):
            clusters[label].append(tweet)

        # Store cluster information (without labels yet)
        cluster_info = []
        for cluster_id, cluster_tweets in sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True):
            cluster_info.append({
                'cluster_id': cluster_id,
                'num_tweets': len(cluster_tweets),
                'example_tweets': [
                    {
                        'username': t['username'],
                        'text': t['text'],
                        'created_at': t['created_at']
                    }
                    for t in cluster_tweets[:20]  # Save top 20 examples
                ],
                'label': f'Cluster {cluster_id}'  # Placeholder
            })
            print(f"  Cluster {cluster_id}: {len(cluster_tweets)} tweets")

        # Get unique users
        users = set(t['username'] for t in tweets)

        community_clusters[str(comm_id)] = {
            'num_tweets': len(tweets),
            'num_users': len(users),
            'clusters': cluster_info,
            'top_users': [u for u, _ in Counter(t['username'] for t in tweets).most_common(10)]
        }

    # Save clusters without labels first
    save_clusters_without_labels(community_clusters, 'community_clusters_unlabeled.json')

    if not skip_llm:
        # Step 2: Label clusters with LLM
        print("\n" + "="*80)
        print("STEP 2: LABELING WITH LLM")
        print("="*80)

        community_clusters = label_clusters_with_llm(community_clusters)

    # Convert to final format with topics
    community_topics = {}
    for comm_id, comm_data in community_clusters.items():
        community_topics[comm_id] = {
            'num_tweets': comm_data['num_tweets'],
            'num_users': comm_data['num_users'],
            'topics': [
                {
                    'label': cluster['label'],
                    'num_tweets': cluster['num_tweets'],
                    'example_tweets': cluster['example_tweets'][:3]  # Only save top 3 for final JSON
                }
                for cluster in comm_data['clusters']
            ],
            'top_users': comm_data['top_users']
        }

    # Save results
    output = {
        'num_communities': len(community_topics),
        'total_tweets_analyzed': sum(ct['num_tweets'] for ct in community_topics.values()),
        'communities': community_topics
    }

    with open('community_topics.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total communities analyzed: {len(community_topics)}")
    print(f"Total tweets analyzed: {output['total_tweets_analyzed']:,}")
    print(f"\nResults saved to community_topics.json")

    # Print summary
    print("\nTop Communities by Size:")
    for i, (comm_id, data) in enumerate(sorted(community_topics.items(), key=lambda x: x[1]['num_tweets'], reverse=True), 1):
        print(f"\n{i}. Community {comm_id}")
        print(f"   {data['num_tweets']:,} tweets from {data['num_users']} users")
        print(f"   Topics:")
        for topic in data['topics'][:3]:
            print(f"     - {topic['label']} ({topic['num_tweets']} tweets)")

if __name__ == "__main__":
    import sys
    skip_llm = '--skip-llm' in sys.argv
    main(skip_llm=skip_llm)
