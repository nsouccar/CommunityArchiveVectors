"""
Regenerate Topics for Updated Communities

After incremental update adds new tweets, this script:
1. Identifies which communities were updated
2. Re-runs topic clustering on those communities
3. Updates the topic JSON files
4. Syncs to frontend data directory

Run after: incremental_update_pipeline.py
"""

import modal
import json
import pickle
import random
from pathlib import Path
from collections import defaultdict

app = modal.App("regenerate-topics-incremental")
volume = modal.Volume.from_name("tweet-vectors-large", create_if_missing=False)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "anthropic",
    "numpy",
    "scikit-learn",
)

secrets = modal.Secret.from_name("anthropic-api-key")

@app.function(
    volumes={"/data": volume},
    image=image,
    timeout=7200,  # 2 hours
    cpu=4,
    memory=8192,
    secrets=[secrets],
)
def regenerate_updated_topics(year_filter=None):
    """
    Regenerate topics for communities that have been updated

    Similar to recluster_2024_topics.py but runs on all years with recent updates
    """
    from anthropic import Anthropic
    import os
    import numpy as np
    from datetime import datetime

    print("=" * 80)
    print("REGENERATING TOPICS FOR UPDATED COMMUNITIES")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 80)
    print()

    # Initialize Claude client
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Load organized data
    print("Loading organized data...")
    with open("/data/organized_by_community.pkl", 'rb') as f:
        organized = pickle.load(f)
    print("✓ Loaded organized data")

    # Load community names
    community_names = {}
    names_file = "/data/all_community_names.json"
    try:
        with open(names_file, 'r') as f:
            community_names_data = json.load(f)

        # Extract names for all years
        for year_key in community_names_data.keys():
            if "communities" in community_names_data[year_key]:
                for comm in community_names_data[year_key]["communities"]:
                    community_names[f"{year_key}_{comm['community_id']}"] = comm["name"]

        print(f"✓ Loaded {len(community_names)} community names")
    except FileNotFoundError:
        print("⚠️  Community names file not found")
    print()

    # Determine which years to process
    if year_filter:
        years_to_process = [year_filter]
    else:
        years_to_process = sorted(organized.keys())

    print(f"Processing {len(years_to_process)} years: {years_to_process}")
    print()

    all_results = {}

    for year_key in years_to_process:
        print("=" * 80)
        print(f"PROCESSING YEAR {year_key}")
        print("=" * 80)
        print()

        if year_key not in organized:
            print(f"No data for {year_key}")
            print()
            continue

        # Combine all months for each community
        communities_combined = defaultdict(lambda: {'tweets': [], 'embeddings': []})

        for month in organized[year_key]:
            for community in organized[year_key][month]:
                group_data = organized[year_key][month][community]
                communities_combined[community]['tweets'].extend(group_data['tweets'])
                if len(group_data.get('embeddings', [])) > 0:
                    communities_combined[community]['embeddings'].append(group_data['embeddings'])

        # Combine embeddings
        for community in communities_combined:
            if communities_combined[community]['embeddings']:
                communities_combined[community]['embeddings'] = np.vstack(
                    communities_combined[community]['embeddings']
                )

        print(f"Combined into {len(communities_combined)} communities")

        # Process each community
        year_results = {
            "year": year_key,
            "stats": {
                "year": year_key,
                "total_communities": len(communities_combined),
                "communities_processed": 0,
                "communities_skipped_too_small": 0,
                "total_clusters": 0,
                "clusters_filtered_llm": 0,
                "clusters_kept": 0,
                "total_tweets_original": 0,
                "total_tweets_kept": 0
            },
            "communities": {}
        }

        # Sort by size
        sorted_communities = sorted(
            communities_combined.items(),
            key=lambda x: len(x[1]['tweets']),
            reverse=True
        )

        for comm_id, data in sorted_communities:
            tweets = data['tweets']
            print(f"Community {comm_id}: {len(tweets)} tweets")

            # Get community name
            comm_name = community_names.get(f"{year_key}_{comm_id}", f"Community {comm_id}")
            print(f"Name: {comm_name}")

            # Skip if too small
            if len(tweets) < 50:
                print(f"⚠️  Skipping - too few tweets")
                year_results["stats"]["communities_skipped_too_small"] += 1
                print()
                continue

            # Sample tweets for analysis
            sample_size = min(100, len(tweets))
            sampled_tweets = random.sample(tweets, sample_size)
            tweet_texts = [t["text"] for t in sampled_tweets]

            # Create prompt for Claude
            prompt = f"""Analyze these tweets from the "{comm_name}" community and identify 3-5 main topics.

Tweets:
{chr(10).join(f"{i+1}. {text[:200]}" for i, text in enumerate(tweet_texts[:20]))}

Return ONLY a JSON object (no explanations or markdown) with this exact structure:
{{
  "topics": [
    {{
      "name": "Topic Name (2-5 words)",
      "description": "One sentence description",
      "relevant": true,
      "tweet_indices": [1, 2, 3]
    }}
  ]
}}

Mark "relevant": false for spam, generic mentions, or off-topic content."""

            try:
                # Call Claude
                print("  Calling LLM...")
                response = client.messages.create(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=2000,
                    temperature=0.3,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )

                # Parse response
                response_text = response.content[0].text

                # Extract JSON
                json_str = None
                if "```json" in response_text:
                    json_str = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    json_str = response_text.split("```")[1].split("```")[0].strip()
                elif "{" in response_text and "}" in response_text:
                    start = response_text.index("{")
                    brace_count = 0
                    end = start
                    for i in range(start, len(response_text)):
                        if response_text[i] == "{":
                            brace_count += 1
                        elif response_text[i] == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                end = i + 1
                                break
                    json_str = response_text[start:end].strip()
                else:
                    json_str = response_text.strip()

                llm_result = json.loads(json_str)

                # Process topics
                community_topics = []
                for topic_data in llm_result.get("topics", []):
                    if not topic_data.get("relevant", True):
                        year_results["stats"]["clusters_filtered_llm"] += 1
                        continue

                    # Get tweets for this topic
                    topic_tweets = []
                    tweet_indices = topic_data.get("tweet_indices", [])
                    for idx in tweet_indices:
                        if 0 <= idx - 1 < len(sampled_tweets):
                            topic_tweets.append(sampled_tweets[idx - 1])

                    # If no specific indices, sample from community
                    if not topic_tweets:
                        topic_tweets = random.sample(tweets, min(20, len(tweets)))

                    topic_tweet_ids = [t["tweet_id"] for t in topic_tweets]

                    # Include ALL tweets with full data
                    all_tweets_data = [
                        {
                            "tweet_id": t["tweet_id"],
                            "username": t.get("username", "unknown"),
                            "text": t.get("text", ""),
                            "timestamp": t.get("timestamp", "")
                        }
                        for t in topic_tweets
                    ]

                    community_topics.append({
                        "cluster_id": len(community_topics),
                        "topic": topic_data["name"],
                        "description": topic_data["description"],
                        "confidence": "high",
                        "num_tweets": len(topic_tweet_ids),
                        "tweet_ids": topic_tweet_ids,
                        "sample_tweets": all_tweets_data
                    })

                    year_results["stats"]["clusters_kept"] += 1
                    year_results["stats"]["total_tweets_kept"] += len(topic_tweet_ids)

                if community_topics:
                    year_results["communities"][str(comm_id)] = community_topics
                    year_results["stats"]["communities_processed"] += 1
                    year_results["stats"]["total_clusters"] += len(community_topics)
                    print(f"  ✓ Generated {len(community_topics)} topics")
                else:
                    print(f"  ⚠️  No relevant topics found")
                    year_results["stats"]["communities_skipped_too_small"] += 1

            except Exception as e:
                print(f"  ❌ Error: {e}")
                year_results["stats"]["communities_skipped_too_small"] += 1

            print()
            year_results["stats"]["total_tweets_original"] += len(tweets)

        # Save results for this year
        output_file = f"/data/topics_year_{year_key}_summary.json"
        with open(output_file, 'w') as f:
            json.dump(year_results, f, indent=2)

        print(f"✓ Saved topics for {year_key}")
        print(f"  Processed: {year_results['stats']['communities_processed']} communities")
        print(f"  Total topics: {year_results['stats']['total_clusters']}")
        print()

        all_results[year_key] = year_results

    volume.commit()

    print("=" * 80)
    print("TOPIC REGENERATION COMPLETE")
    print("=" * 80)
    print(f"Years processed: {len(all_results)}")
    print(f"Total topics generated: {sum(r['stats']['total_clusters'] for r in all_results.values())}")
    print()

    return all_results


@app.local_entrypoint()
def main():
    """
    Run topic regeneration
    """
    print()
    print("=" * 80)
    print("REGENERATE TOPICS - INCREMENTAL")
    print("=" * 80)
    print()
    print("This will:")
    print("  1. Load organized_by_community.pkl")
    print("  2. For each community, identify topics with LLM")
    print("  3. Update topic JSON files")
    print()
    print("Note: Processes ALL years by default")
    print("      To process specific year, edit the script")
    print()
    print("Estimated time: 20-60 minutes depending on updates")
    print()

    # Run regeneration
    results = regenerate_updated_topics.remote()

    print()
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print()
    for year, result in results.items():
        print(f"Year {year}:")
        print(f"  Communities: {result['stats']['communities_processed']}")
        print(f"  Topics: {result['stats']['total_clusters']}")
    print()
    print("Next steps:")
    print("  1. Download updated topic files:")
    print("     modal volume get tweet-vectors-large /topics_year_*_summary.json")
    print("  2. Copy to frontend/public/data/")
    print("  3. Redeploy frontend")
    print()
