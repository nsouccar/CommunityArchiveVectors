#!/usr/bin/env python3
"""
K-means clustering and topic labeling for organized community embeddings.

For each (year, month, community) group:
1. Run k-means clustering on tweet embeddings
2. Sample representative tweets from each cluster
3. Use Claude API to generate topic labels
4. Filter out irrelevant clusters (mentions, short replies)
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

# Modal setup
app = modal.App("cluster-topics")
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

    Rules:
    - < 50 tweets: 1-2 clusters
    - 50-200: 3-5 clusters
    - 200-500: 5-8 clusters
    - 500-2000: 8-12 clusters
    - > 2000: 10-15 clusters
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

def is_cluster_relevant(tweets: list) -> tuple[bool, str]:
    """
    No heuristic filtering - let the LLM handle ALL quality assessment.

    Returns: (is_relevant, reason_for_filtering)
    """
    # Everything passes - LLM will filter based on content quality
    return True, ""

def generate_topic_label(tweets: list, api_key: str) -> dict:
    """
    Use Claude API to generate a concise topic label for a cluster.

    Returns: {
        'topic': str,  # 2-5 word topic label
        'description': str,  # 1 sentence description
        'is_relevant': bool,  # LLM's assessment of relevance
        'confidence': str  # 'high', 'medium', 'low'
    }
    """
    client = anthropic.Anthropic(api_key=api_key)

    # Sample up to 15 representative tweets
    sample = tweets[:15]

    # DEBUG: Print first tweet to see what keys it has
    if len(sample) > 0:
        print(f"DEBUG - First tweet keys: {sample[0].keys()}")
        print(f"DEBUG - First tweet content: {sample[0]}")

    tweets_text = "\n\n".join([f"Tweet {i+1}: {t.get('text', t.get('tweet_text', t.get('full_text', 'NO TEXT')))}" for i, t in enumerate(sample)])

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
  "topic": "concise topic label",
  "description": "one sentence description",
  "is_relevant": true/false,
  "confidence": "high/medium/low",
  "reasoning": "brief explanation of your assessment"
}}"""

    try:
        message = client.messages.create(
            model="claude-3-5-haiku-20241022",  # Fast and cheap
            max_tokens=500,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = message.content[0].text.strip()

        # Handle empty response
        if not response_text:
            return {
                'topic': 'Unknown Topic',
                'description': 'Empty API response',
                'is_relevant': True,
                'confidence': 'low',
                'reasoning': 'API returned empty response'
            }

        # Parse JSON response
        # Find JSON in the response (might have markdown code blocks)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        # Try to parse JSON
        result = json.loads(response_text)
        return result

    except json.JSONDecodeError as e:
        print(f"Error calling Claude API: JSON parsing failed - {e}")
        print(f"Response text: {response_text[:200] if 'response_text' in locals() else 'No response'}")
        return {
            'topic': 'Unknown Topic',
            'description': 'Could not parse JSON response',
            'is_relevant': True,  # Default to keeping it
            'confidence': 'low',
            'reasoning': f'JSON parse error: {str(e)}'
        }
    except Exception as e:
        print(f"Error calling Claude API: {e}")
        return {
            'topic': 'Unknown Topic',
            'description': 'Could not generate description',
            'is_relevant': True,  # Default to keeping it
            'confidence': 'low',
            'reasoning': f'API error: {str(e)}'
        }

@app.function(
    volumes={"/data": volume},
    image=image,
    secrets=[secrets],
    timeout=10800,  # 3 hours
    cpu=8,
    memory=32768,  # 32GB for k-means
)
def cluster_and_label_all_groups():
    """
    Main function: cluster and label all (year, month, community) groups.
    """
    from pathlib import Path

    print("\n" + "="*80)
    print("CLUSTERING AND TOPIC LABELING")
    print("="*80)

    # Load organized data
    print("\nLoading organized data from Modal volume...")
    organized_path = Path("/data/organized_by_community.pkl")

    with open(organized_path, 'rb') as f:
        organized = pickle.load(f)

    print(f"✓ Loaded organized data")

    # Get Claude API key from environment
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment")

    # Process each group
    all_topics = {}
    stats = {
        'total_groups': 0,
        'groups_processed': 0,
        'groups_skipped_too_small': 0,
        'total_clusters': 0,
        'clusters_filtered_heuristic': 0,
        'clusters_filtered_llm': 0,
        'clusters_kept': 0,
    }

    for year in sorted(organized.keys()):
        print(f"\n{'='*80}")
        print(f"Processing year {year}")
        print(f"{'='*80}")

        if year not in all_topics:
            all_topics[year] = {}

        for month in sorted(organized[year].keys()):
            print(f"\n  Month {month}:")

            if month not in all_topics[year]:
                all_topics[year][month] = {}

            for community in sorted(organized[year][month].keys()):
                stats['total_groups'] += 1

                group_data = organized[year][month][community]
                num_tweets = len(group_data['tweets'])

                # Skip if too few tweets
                if num_tweets < 20:
                    print(f"    Community {community}: SKIP ({num_tweets} tweets - too small)")
                    stats['groups_skipped_too_small'] += 1
                    continue

                print(f"    Community {community}: {num_tweets} tweets", end=" ")

                # Determine number of clusters
                n_clusters = determine_num_clusters(num_tweets)
                print(f"→ {n_clusters} clusters", end="")

                # Run k-means clustering
                embeddings = np.array(group_data['embeddings'])

                if n_clusters == 1:
                    # Just one cluster - all tweets in cluster 0
                    labels = np.zeros(num_tweets, dtype=int)
                else:
                    kmeans = MiniBatchKMeans(
                        n_clusters=n_clusters,
                        batch_size=min(1000, num_tweets),
                        random_state=42,
                        n_init=3
                    )
                    labels = kmeans.fit_predict(embeddings)

                # Organize tweets by cluster
                clusters = defaultdict(list)
                for i, label in enumerate(labels):
                    clusters[label].append(group_data['tweets'][i])

                # Process each cluster
                community_topics = []

                for cluster_id in sorted(clusters.keys()):
                    cluster_tweets = clusters[cluster_id]
                    stats['total_clusters'] += 1

                    # Filter 1: Heuristic checks
                    is_relevant, filter_reason = is_cluster_relevant(cluster_tweets)

                    if not is_relevant:
                        stats['clusters_filtered_heuristic'] += 1
                        print(f"\n      Cluster {cluster_id}: {len(cluster_tweets)} tweets - FILTERED ({filter_reason})")
                        continue

                    # Generate topic label with Claude
                    topic_data = generate_topic_label(cluster_tweets, api_key)

                    # Filter 2: LLM assessment
                    if not topic_data['is_relevant']:
                        stats['clusters_filtered_llm'] += 1
                        print(f"\n      Cluster {cluster_id}: {len(cluster_tweets)} tweets - FILTERED by LLM ({topic_data.get('reasoning', 'not relevant')})")
                        continue

                    # Keep this cluster!
                    stats['clusters_kept'] += 1

                    community_topics.append({
                        'cluster_id': int(cluster_id),
                        'topic': topic_data['topic'],
                        'description': topic_data['description'],
                        'confidence': topic_data['confidence'],
                        'num_tweets': len(cluster_tweets),
                        'sample_tweets': cluster_tweets[:5]  # Keep 5 examples
                    })

                    print(f"\n      Cluster {cluster_id}: {len(cluster_tweets)} tweets - '{topic_data['topic']}' ({topic_data['confidence']} confidence)")

                if community_topics:
                    all_topics[year][month][community] = community_topics
                    stats['groups_processed'] += 1
                    print(f"    ✓ {len(community_topics)} topics for community {community}")
                else:
                    print(f"    ✗ No topics passed filtering")

    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    output_path = Path("/data/community_topics.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(all_topics, f)
    print(f"✓ Saved topics to {output_path}")

    # Save human-readable summary
    summary_path = Path("/data/community_topics_summary.json")

    # Create summary (without full tweet data)
    summary = {}
    for year in all_topics:
        summary[year] = {}
        for month in all_topics[year]:
            summary[year][month] = {}
            for community in all_topics[year][month]:
                summary[year][month][community] = [
                    {
                        'cluster_id': t['cluster_id'],
                        'topic': t['topic'],
                        'description': t['description'],
                        'confidence': t['confidence'],
                        'num_tweets': t['num_tweets']
                    }
                    for t in all_topics[year][month][community]
                ]

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary to {summary_path}")

    # Commit volume
    volume.commit()
    print("✓ Committed changes to Modal volume")

    # Print statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"Total (year, month, community) groups: {stats['total_groups']}")
    print(f"  Processed: {stats['groups_processed']}")
    print(f"  Skipped (< 20 tweets): {stats['groups_skipped_too_small']}")
    print(f"\nTotal clusters created: {stats['total_clusters']}")
    print(f"  Filtered by heuristics: {stats['clusters_filtered_heuristic']}")
    print(f"  Filtered by LLM: {stats['clusters_filtered_llm']}")
    print(f"  Kept (high quality): {stats['clusters_kept']}")
    print(f"\nFiltering rate: {(stats['clusters_filtered_heuristic'] + stats['clusters_filtered_llm']) / stats['total_clusters'] * 100:.1f}%")

    return {
        'stats': stats,
        'output_files': [str(output_path), str(summary_path)]
    }

@app.local_entrypoint()
def main():
    """Run clustering and topic labeling."""
    print("Starting clustering and topic labeling on Modal...")
    print("This will process all 3,127 (year, month, community) groups")
    print("Estimated time: 1-2 hours")
    print()

    result = cluster_and_label_all_groups.remote()

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    print(f"\nResults saved to Modal volume:")
    for filepath in result['output_files']:
        print(f"  - {filepath}")
    print("\nNext: Export topics to frontend for visualization")
