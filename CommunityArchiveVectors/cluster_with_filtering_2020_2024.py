#!/usr/bin/env python3
"""
K-means clustering with integrated tweet-level filtering for years 2020-2024

This script reclusters years 2020-2024 which lost their tweet_ids data.
It includes less aggressive filtering than the standalone filter:

1. Run k-means clustering on tweet embeddings
2. Generate topic labels using Claude
3. Filter out irrelevant CLUSTERS (existing behavior)
4. NEW: For each kept cluster, filter tweets to keep only relevant ones
5. Save topics with filtered tweet_ids

Estimate: ~2 hours for years 2020-2024 (610 clusters total)
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
app = modal.App("cluster-with-filtering-2020-2024")
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

def filter_cluster_tweets(cluster_tweets: list, topic: str, description: str, api_key: str) -> list:
    """
    Filter tweets within a cluster to keep only those relevant to the cluster topic.
    Less aggressive than standalone filtering - just removes clearly off-topic content.

    Returns: list of tweet IDs to keep
    """
    if len(cluster_tweets) <= 5:
        # Don't filter very small clusters
        return [t['tweet_id'] for t in cluster_tweets]

    client = anthropic.Anthropic(api_key=api_key)

    # Prepare tweet list for Claude
    tweets_for_analysis = [
        {
            'id': t['tweet_id'],
            'text': t['text'][:200]  # Truncate to first 200 chars to save tokens
        }
        for t in cluster_tweets[:50]  # Analyze up to 50 tweets
    ]

    tweets_json = json.dumps(tweets_for_analysis, indent=2)

    prompt = f"""You are filtering tweets in a cluster about: "{topic}"

Cluster description: {description}

Here are the tweets (showing first 50):
{tweets_json}

For each tweet, determine if it's RELEVANT to this cluster topic. Be LENIENT - only mark as irrelevant if the tweet is clearly off-topic, spam, or meaningless.

Examples of what to KEEP:
- Tweets that discuss the topic directly
- Related discussions and questions
- Personal experiences related to the topic
- Tangentially related content that adds context

Examples of what to REMOVE:
- Complete non-sequiturs
- Spam or promotional content
- Generic greetings with no substance
- Tweets in another language (unless topic is about that language)

Respond with a JSON array of tweet IDs to KEEP (not remove):
{{
  "keep_ids": ["1234567890", "9876543210", ...]
}}
"""

    try:
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = response.content[0].text
        result = json.loads(response_text)

        keep_ids_set = set(result.get('keep_ids', []))

        # For tweets we didn't analyze (>50), keep them all
        all_tweet_ids = [t['tweet_id'] for t in cluster_tweets]
        analyzed_ids = [t['tweet_id'] for t in cluster_tweets[:50]]
        unanalyzed_ids = [tid for tid in all_tweet_ids if tid not in analyzed_ids]

        # Combine: keep the ones Claude said to keep + all unanalyzed ones
        final_keep_ids = list(keep_ids_set) + unanalyzed_ids

        return final_keep_ids

    except Exception as e:
        print(f"    Warning: Filtering failed ({e}), keeping all tweets")
        return [t['tweet_id'] for t in cluster_tweets]

@app.function(
    volumes={"/data": volume},
    image=image,
    secrets=[secrets],
    timeout=14400,  # 4 hours
    cpu=8,
    memory=32768,  # 32GB for k-means
)
def cluster_and_filter_years():
    """
    Recluster years 2020-2024 with integrated filtering.
    Saves incrementally after each year.
    """

    print("\n" + "="*80)
    print("CLUSTERING WITH FILTERING FOR YEARS 2020-2024")
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

    # Test with 2020 first, then process remaining years
    # IMPORTANT: Years are stored as strings in organized data
    target_years = ['2022']
    all_years_stats = []

    for year in target_years:
        if year not in organized:
            print(f"\n Year {year} not found in organized data, skipping...")
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
            'total_tweets_original': 0,
            'total_tweets_kept': 0,
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
                year_stats['total_tweets_original'] += len(cluster_tweets)

                # Generate topic label using Claude
                topic_result = generate_topic_label(cluster_tweets, api_key)

                # Filter based on LLM assessment (cluster-level)
                if not topic_result.get('is_relevant', True):
                    print(f"\n    Cluster {cluster_id}: {len(cluster_tweets)} tweets - FILTERED by LLM ({topic_result.get('reasoning', 'No reason')})")
                    year_stats['clusters_filtered_llm'] += 1
                    continue

                year_stats['clusters_kept'] += 1

                # NEW: Filter tweets within this cluster
                print(f"\n    Cluster {cluster_id}: {len(cluster_tweets)} tweets - '{topic_result['topic']}' ({topic_result['confidence']} confidence)")
                print(f"      Filtering tweets...", end=" ")

                filtered_tweet_ids = filter_cluster_tweets(
                    cluster_tweets,
                    topic_result['topic'],
                    topic_result['description'],
                    api_key
                )

                tweets_removed = len(cluster_tweets) - len(filtered_tweet_ids)
                removal_pct = (tweets_removed / len(cluster_tweets) * 100) if len(cluster_tweets) > 0 else 0

                print(f"Kept {len(filtered_tweet_ids)}/{len(cluster_tweets)} tweets ({removal_pct:.0f}% removed)")

                year_stats['total_tweets_kept'] += len(filtered_tweet_ids)

                # Store topic with filtered tweet IDs
                community_topics.append({
                    'cluster_id': cluster_id,
                    'topic': topic_result['topic'],
                    'description': topic_result['description'],
                    'confidence': topic_result['confidence'],
                    'num_tweets': len(filtered_tweet_ids),  # Filtered count
                    'tweet_ids': filtered_tweet_ids,  # Only filtered tweets
                    'sample_tweets': [
                        {
                            'tweet_id': t['tweet_id'],
                            'username': t['username'],
                            'text': t['text'],
                            'timestamp': t['timestamp']
                        }
                        for t in cluster_tweets[:5] if t['tweet_id'] in filtered_tweet_ids
                    ][:5]  # Up to 5 sample tweets from filtered set
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
        print(f"  Filtered by LLM (cluster-level): {year_stats['clusters_filtered_llm']}")
        print(f"  Kept (high quality): {year_stats['clusters_kept']}")
        print(f"  Total tweets (original): {year_stats['total_tweets_original']}")
        print(f"  Total tweets (filtered): {year_stats['total_tweets_kept']}")
        if year_stats['total_tweets_original'] > 0:
            tweet_removal_pct = (year_stats['total_tweets_original'] - year_stats['total_tweets_kept']) / year_stats['total_tweets_original'] * 100
            print(f"  Tweet-level filtering: {tweet_removal_pct:.1f}% removed")

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
    total_tweets_orig = sum(s['total_tweets_original'] for s in all_years_stats)
    total_tweets_kept = sum(s['total_tweets_kept'] for s in all_years_stats)

    print(f"Total communities: {total_communities}")
    print(f"  Processed: {total_processed}")
    print(f"Total clusters: {total_clusters}")
    print(f"  Filtered (cluster-level): {total_filtered}")
    print(f"  Kept: {total_kept}")
    print(f"Total tweets (original): {total_tweets_orig}")
    print(f"Total tweets (filtered): {total_tweets_kept}")
    if total_tweets_orig > 0:
        print(f"  Overall tweet removal: {(total_tweets_orig - total_tweets_kept) / total_tweets_orig * 100:.1f}%")

    return {
        'years_processed': [s['year'] for s in all_years_stats],
        'total_communities': total_communities,
        'total_clusters': total_clusters,
        'stats': all_years_stats
    }

@app.local_entrypoint()
def main():
    """Run clustering with filtering for years 2020-2024."""
    print("Starting clustering with integrated filtering...")
    print("Target years: 2020, 2021, 2022, 2023, 2024")
    print("Estimated time: ~2 hours")
    print()

    result = cluster_and_filter_years.remote()

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    print(f"\nProcessed years: {', '.join(str(y) for y in result['years_processed'])}")
    print(f"Total communities: {result['total_communities']}")
    print(f"Total clusters: {result['total_clusters']}")
    print("\nResults saved to Modal volume:")
    for year in result['years_processed']:
        print(f"  - /data/topics_year_{year}.pkl")
        print(f"  - /data/topics_year_{year}_summary.json")
    print("\nNext: Download and use these filtered files!")
