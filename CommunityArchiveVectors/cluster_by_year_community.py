#!/usr/bin/env python3
"""
K-means clustering and topic labeling - OPTIMIZED VERSION

Clusters by (year, community) instead of (year, month, community)
- Combines all months together for each year/community
- Saves incrementally after each year
- Much faster: ~500-600 groups instead of 3,127

For each (year, community) group:
1. Run k-means clustering on tweet embeddings
2. Sample representative tweets from each cluster
3. Use Claude API to generate topic labels
4. Filter out irrelevant clusters
5. Save topics with metadata
"""

import modal
import pickle
import numpy as np
from collections import defaultdict
from sklearn.cluster import MiniBatchKMeans
import anthropic
import os
import json
from pathlib import Path

# Modal setup
app = modal.App("cluster-topics-by-year")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "scikit-learn",
    "numpy",
    "anthropic"
)
volume = modal.Volume.from_name("tweet-vectors-large", create_if_missing=False)

# Secrets for Claude API
secrets = modal.Secret.from_name("anthropic-api-key")

def determine_num_clusters(num_tweets: int) -> int:
    """
    Determine optimal number of clusters based on dataset size.
    """
    if num_tweets < 50:
        return max(1, min(2, num_tweets // 20))
    elif num_tweets < 200:
        return max(3, min(5, num_tweets // 40))
    elif num_tweets < 500:
        return max(5, min(8, num_tweets // 60))
    elif num_tweets < 2000:
        return max(8, min(12, num_tweets // 150))
    else:
        return max(10, min(15, num_tweets // 200))

def generate_topic_label(tweets: list, api_key: str) -> dict:
    """
    Use Claude API to generate a concise topic label for a cluster.
    """
    client = anthropic.Anthropic(api_key=api_key)

    # Sample up to 15 representative tweets
    sample = tweets[:15]

    tweets_text = "\n\n".join([f"Tweet {i+1}: {t.get('text', '')}" for i, t in enumerate(sample)])

    prompt = f"""You are analyzing a cluster of {len(tweets)} tweets from a Twitter community. Based on these sample tweets, generate a concise topic label.

Sample tweets from this cluster:
{tweets_text}

Please analyze these tweets and provide:
1. A concise 2-5 word topic label (e.g., "AI Safety Discussions", "Ukraine War Updates", "Crypto Market Analysis")
2. A 1-sentence description of what this cluster is about
3. Whether this cluster represents substantive discussion (not just greetings, spam, or meaningless chatter)
4. Your confidence level: high, medium, or low

Respond in JSON format:
{{
  "topic": "2-5 word topic label",
  "description": "One sentence description",
  "is_relevant": true or false,
  "confidence": "high" or "medium" or "low",
  "reasoning": "Brief explanation of your assessment"
}}
"""

    try:
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = response.content[0].text

        # Parse JSON response
        result = json.loads(response_text)

        return result

    except json.JSONDecodeError as e:
        print(f"Error calling Claude API: JSON parsing failed - {e}")
        print(f"Response text: {response_text[:200] if 'response_text' in locals() else 'No response'}")
        return {
            'topic': 'Unknown Topic',
            'description': 'Could not parse JSON response',
            'is_relevant': True,
            'confidence': 'low',
            'reasoning': f'JSON parse error: {str(e)}'
        }
    except Exception as e:
        print(f"Error calling Claude API: {e}")
        return {
            'topic': 'Unknown Topic',
            'description': 'Could not generate description',
            'is_relevant': True,
            'confidence': 'low',
            'reasoning': f'API error: {str(e)}'
        }

@app.function(
    volumes={"/data": volume},
    image=image,
    secrets=[secrets],
    timeout=14400,  # 4 hours (increased from 3)
    cpu=8,
    memory=32768,  # 32GB for k-means
)
def cluster_and_label_by_year():
    """
    Main function: cluster and label by (year, community) groups.
    Saves incrementally after each year.
    """

    print("\n" + "="*80)
    print("CLUSTERING AND TOPIC LABELING BY (YEAR, COMMUNITY)")
    print("="*80)

    # Load organized data
    print("\nLoading organized data from Modal volume...")
    organized_path = Path("/data/organized_by_community.pkl")

    with open(organized_path, 'rb') as f:
        organized = pickle.load(f)

    print(f"✓ Loaded organized data")

    # Get Claude API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment")

    # Process each year
    all_years_stats = []

    for year in sorted(organized.keys()):
        # Check if results already exist for this year
        output_path = Path(f"/data/topics_year_{year}.pkl")
        summary_path = Path(f"/data/topics_year_{year}_summary.json")

        if output_path.exists() and summary_path.exists():
            print(f"\n{'='*80}")
            print(f"SKIPPING year {year} - Results already exist")
            print(f"{'='*80}")
            print(f"  Found: {output_path}")
            print(f"  Found: {summary_path}")
            continue

        print(f"\n{'='*80}")
        print(f"Processing year {year}")
        print(f"{'='*80}")

        year_topics = {}
        year_stats = {
            'year': year,
            'total_communities': 0,
            'communities_processed': 0,
            'communities_skipped_too_small': 0,
            'total_clusters': 0,
            'clusters_filtered_llm': 0,
            'clusters_kept': 0,
        }

        # Combine all months together for each community
        communities_combined = defaultdict(lambda: {'tweets': [], 'embeddings': []})

        for month in organized[year]:
            for community in organized[year][month]:
                group_data = organized[year][month][community]
                communities_combined[community]['tweets'].extend(group_data['tweets'])
                communities_combined[community]['embeddings'].extend(group_data['embeddings'])

        print(f"\nCombined {len(organized[year])} months into {len(communities_combined)} communities")

        # Process each community
        for community in sorted(communities_combined.keys()):
            year_stats['total_communities'] += 1

            group_data = communities_combined[community]
            num_tweets = len(group_data['tweets'])

            # Skip if too few tweets
            if num_tweets < 20:
                print(f"  Community {community}: SKIP ({num_tweets} tweets - too small)")
                year_stats['communities_skipped_too_small'] += 1
                continue

            print(f"  Community {community}: {num_tweets} tweets", end=" ")

            # Determine number of clusters
            n_clusters = determine_num_clusters(num_tweets)
            print(f"→ {n_clusters} clusters", end="")

            # Run k-means clustering
            embeddings = np.array(group_data['embeddings'])

            if n_clusters == 1:
                # Just one cluster
                labels = np.zeros(len(embeddings), dtype=int)
            else:
                kmeans = MiniBatchKMeans(
                    n_clusters=n_clusters,
                    batch_size=min(1000, len(embeddings)),
                    random_state=42,
                    n_init=3
                )
                labels = kmeans.fit_predict(embeddings)

            # Group tweets by cluster
            clusters = defaultdict(list)
            for tweet, label in zip(group_data['tweets'], labels):
                clusters[int(label)].append(tweet)

            # Generate topic labels for each cluster
            community_topics = []

            for cluster_id in sorted(clusters.keys()):
                cluster_tweets = clusters[cluster_id]
                year_stats['total_clusters'] += 1

                # Generate topic label using Claude
                topic_result = generate_topic_label(cluster_tweets, api_key)

                # Filter based on LLM assessment
                if not topic_result.get('is_relevant', True):
                    print(f"\n    Cluster {cluster_id}: {len(cluster_tweets)} tweets - FILTERED by LLM ({topic_result.get('reasoning', 'No reason')})")
                    year_stats['clusters_filtered_llm'] += 1
                    continue

                year_stats['clusters_kept'] += 1

                print(f"\n    Cluster {cluster_id}: {len(cluster_tweets)} tweets - '{topic_result['topic']}' ({topic_result['confidence']} confidence)")

                # Store topic
                community_topics.append({
                    'cluster_id': cluster_id,
                    'topic': topic_result['topic'],
                    'description': topic_result['description'],
                    'confidence': topic_result['confidence'],
                    'num_tweets': len(cluster_tweets),
                    'tweet_ids': [t['tweet_id'] for t in cluster_tweets],  # All tweet IDs
                    'sample_tweets': [
                        {
                            'tweet_id': t['tweet_id'],
                            'username': t['username'],
                            'text': t['text'],
                            'timestamp': t['timestamp']
                        }
                        for t in cluster_tweets[:5]  # Save 5 sample tweets
                    ]
                })

            year_topics[community] = community_topics
            year_stats['communities_processed'] += 1

            print(f"  ✓ {len(community_topics)} topics for community {community}")

        # Save this year's results incrementally
        print(f"\n{'='*80}")
        print(f"SAVING YEAR {year} RESULTS")
        print(f"{'='*80}\n")

        # Save full topics data
        output_path = Path(f"/data/topics_year_{year}.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump(year_topics, f)
        print(f"✓ Saved topics to {output_path}")

        # Save summary
        summary = {
            'year': year,
            'stats': year_stats,
            'communities': {}
        }

        for community, topics in year_topics.items():
            summary['communities'][community] = [
                {
                    'cluster_id': t['cluster_id'],
                    'topic': t['topic'],
                    'description': t['description'],
                    'confidence': t['confidence'],
                    'num_tweets': t['num_tweets'],
                    'tweet_ids': t['tweet_ids']
                }
                for t in topics
            ]

        summary_path = Path(f"/data/topics_year_{year}_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Saved summary to {summary_path}")

        # Commit to volume
        volume.commit()
        print(f"✓ Committed year {year} to Modal volume\n")

        # Print year stats
        print(f"Year {year} Statistics:")
        print(f"  Total communities: {year_stats['total_communities']}")
        print(f"  Processed: {year_stats['communities_processed']}")
        print(f"  Skipped (< 20 tweets): {year_stats['communities_skipped_too_small']}")
        print(f"  Total clusters created: {year_stats['total_clusters']}")
        print(f"  Filtered by LLM: {year_stats['clusters_filtered_llm']}")
        print(f"  Kept (high quality): {year_stats['clusters_kept']}")
        if year_stats['total_clusters'] > 0:
            print(f"  Filtering rate: {year_stats['clusters_filtered_llm'] / year_stats['total_clusters'] * 100:.1f}%")

        all_years_stats.append(year_stats)

    # Print final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}\n")

    total_communities = sum(s['total_communities'] for s in all_years_stats)
    total_processed = sum(s['communities_processed'] for s in all_years_stats)
    total_clusters = sum(s['total_clusters'] for s in all_years_stats)
    total_filtered = sum(s['clusters_filtered_llm'] for s in all_years_stats)
    total_kept = sum(s['clusters_kept'] for s in all_years_stats)

    print(f"Total communities: {total_communities}")
    print(f"  Processed: {total_processed}")
    print(f"Total clusters: {total_clusters}")
    print(f"  Filtered: {total_filtered}")
    print(f"  Kept: {total_kept}")
    if total_clusters > 0:
        print(f"  Overall filtering rate: {total_filtered / total_clusters * 100:.1f}%")

    return {
        'years_processed': [s['year'] for s in all_years_stats],
        'total_communities': total_communities,
        'total_clusters': total_clusters,
        'stats': all_years_stats
    }

@app.local_entrypoint()
def main():
    """Run clustering and topic labeling."""
    print("Starting clustering and topic labeling by (year, community)...")
    print("This will process ~500-600 groups (much faster than 3,127!)")
    print("Saves incrementally after each year")
    print("Estimated time: 30-60 minutes")
    print()

    result = cluster_and_label_by_year.remote()

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    print(f"\nProcessed years: {', '.join(result['years_processed'])}")
    print(f"Total communities: {result['total_communities']}")
    print(f"Total clusters: {result['total_clusters']}")
    print("\nResults saved to Modal volume:")
    for year in result['years_processed']:
        print(f"  - /data/topics_year_{year}.pkl")
        print(f"  - /data/topics_year_{year}_summary.json")
    print("\nNext: Export topics to frontend for visualization")
