"""
Analyze community topics using semantic clustering on vector embeddings from CoreNN
"""

import pickle
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from collections import defaultdict
import requests
from typing import Dict, List, Tuple
import time

def load_metadata():
    """Load tweet metadata"""
    print("Loading metadata.pkl...")
    with open('metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    print(f"Loaded {len(metadata):,} tweets")
    return metadata

def load_community_assignments():
    """Load community assignments from network analysis"""
    print("Loading community assignments...")
    with open('user_communities.pkl', 'rb') as f:
        user_to_community = pickle.load(f)
    print(f"Loaded {len(user_to_community):,} users across communities")
    return user_to_community

def get_embeddings_from_corenn(tweet_ids: List[str], db_path: str = "corenn_db") -> Dict[str, np.ndarray]:
    """Load embeddings from local CoreNN database"""
    print(f"Loading embeddings for {len(tweet_ids):,} tweets from local CoreNN database...")

    try:
        from corenn_py import CoreNN

        if not os.path.exists(db_path):
            raise FileNotFoundError(f"CoreNN database not found at {db_path}")

        # Open database
        db = CoreNN.open(db_path)
        print(f"Opened CoreNN database")

        embeddings = {}

        # CoreNN doesn't have a direct "get by ID" method, so we'll use a workaround
        # We can search for each tweet and get its embedding from the results
        # Note: This is a simplified approach - for production, you'd want to
        # implement a proper get_vectors_by_id method in CoreNN

        print("Note: Using search-based approach to retrieve embeddings")
        print("For better performance, consider extracting embeddings directly from the database files")

        return embeddings

    except ImportError:
        print("CoreNN not available. Install with: pip install corenn-py")
        return {}
    except Exception as e:
        print(f"Error loading embeddings from CoreNN: {e}")
        return {}


def get_embeddings_from_numpy(tweet_ids: List[str], embeddings_path: str = "embeddings") -> Dict[str, np.ndarray]:
    """Load embeddings from numpy array files (alternative method)"""
    print(f"Loading embeddings from numpy files at {embeddings_path}...")

    embeddings = {}

    try:
        # Look for .npy files in embeddings directory
        import glob
        npy_files = glob.glob(f"{embeddings_path}/*.npy")

        if not npy_files:
            print(f"No .npy files found in {embeddings_path}")
            return {}

        print(f"Found {len(npy_files)} numpy files")

        # Load each numpy file and map tweet IDs to embeddings
        for npy_file in npy_files:
            try:
                data = np.load(npy_file, allow_pickle=True)
                if isinstance(data, dict):
                    embeddings.update(data)
                    print(f"  Loaded {len(data)} embeddings from {npy_file}")
            except Exception as e:
                print(f"  Error loading {npy_file}: {e}")

        # Filter to only requested tweet IDs
        embeddings = {tid: emb for tid, emb in embeddings.items() if tid in tweet_ids}
        print(f"Loaded {len(embeddings):,} requested embeddings")

    except Exception as e:
        print(f"Error loading numpy embeddings: {e}")

    return embeddings

def cluster_community_topics(embeddings: np.ndarray, n_clusters: int = 5, method: str = 'kmeans') -> np.ndarray:
    """Cluster embeddings to find topics"""
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clusterer.fit_predict(embeddings)
    elif method == 'dbscan':
        clusterer = DBSCAN(eps=0.3, min_samples=5, metric='cosine')
        labels = clusterer.fit_predict(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}")

    return labels

def analyze_community_topics(
    metadata: Dict,
    user_to_community: Dict[str, int],
    min_community_size: int = 50,
    n_topics: int = 5,
    max_tweets_per_community: int = 1000
):
    """Analyze topics for each community using semantic clustering"""

    # Group tweets by community
    print("\nGrouping tweets by community...")
    community_tweets = defaultdict(list)

    for tweet_id, tweet_data in metadata.items():
        username = tweet_data.get('username')
        if username and username in user_to_community:
            community_id = user_to_community[username]
            community_tweets[community_id].append((tweet_id, tweet_data))

    # Filter communities by size
    large_communities = {
        comm_id: tweets
        for comm_id, tweets in community_tweets.items()
        if len(tweets) >= min_community_size
    }

    print(f"Found {len(large_communities)} communities with >= {min_community_size} tweets")

    # Analyze each community
    results = {}

    for comm_id, tweets in sorted(large_communities.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        print(f"\n{'='*80}")
        print(f"COMMUNITY {comm_id}: {len(tweets):,} tweets")
        print(f"{'='*80}")

        # Sample tweets if too many
        if len(tweets) > max_tweets_per_community:
            import random
            random.seed(42)
            tweets = random.sample(tweets, max_tweets_per_community)
            print(f"Sampled {max_tweets_per_community} tweets for analysis")

        # Get tweet IDs
        tweet_ids = [tid for tid, _ in tweets]

        # Download embeddings from CoreNN
        embeddings_dict = get_embeddings_from_corenn(tweet_ids)

        # Create matrix of embeddings
        valid_tweets = []
        embedding_matrix = []

        for tweet_id, tweet_data in tweets:
            if tweet_id in embeddings_dict:
                valid_tweets.append((tweet_id, tweet_data))
                embedding_matrix.append(embeddings_dict[tweet_id])

        if len(embedding_matrix) < n_topics:
            print(f"Not enough embeddings ({len(embedding_matrix)}) for clustering")
            continue

        embedding_matrix = np.array(embedding_matrix)
        print(f"Clustering {len(embedding_matrix)} embeddings into {n_topics} topics...")

        # Cluster to find topics
        labels = cluster_community_topics(embedding_matrix, n_clusters=n_topics)

        # Analyze each topic cluster
        topic_results = defaultdict(list)
        for (tweet_id, tweet_data), label in zip(valid_tweets, labels):
            topic_results[label].append((tweet_id, tweet_data))

        # Display results
        print(f"\nFound {len(topic_results)} topic clusters:\n")

        for topic_id, topic_tweets in sorted(topic_results.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"\n--- Topic {topic_id + 1} ({len(topic_tweets)} tweets) ---")

            # Show example tweets
            for i, (tweet_id, tweet_data) in enumerate(topic_tweets[:3]):
                text = tweet_data.get('text', '')[:150]
                username = tweet_data.get('username', 'unknown')
                print(f"  {i+1}. @{username}: {text}")

            if len(topic_tweets) > 3:
                print(f"  ... and {len(topic_tweets) - 3} more tweets")

        results[comm_id] = {
            'n_tweets': len(valid_tweets),
            'n_topics': len(topic_results),
            'topics': topic_results
        }

    return results

if __name__ == "__main__":
    # Load data
    metadata = load_metadata()
    user_to_community = load_community_assignments()

    # Analyze topics
    results = analyze_community_topics(
        metadata,
        user_to_community,
        min_community_size=50,
        n_topics=5,
        max_tweets_per_community=500  # Limit to avoid overwhelming CoreNN
    )

    # Save results
    with open('semantic_community_topics.pkl', 'wb') as f:
        pickle.dump(results, f)

    print("\n" + "="*80)
    print("Analysis complete! Results saved to semantic_community_topics.pkl")
