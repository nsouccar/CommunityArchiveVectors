"""
Regenerate 2024 topics using LLM-based semantic filtering.

For each community:
1. Load the community name
2. Sample tweets from that community
3. Use Claude to identify relevant themes/topics
4. Group tweets into those topics
5. Filter out irrelevant content
"""

import modal
import json
import random
import pickle
from pathlib import Path
from collections import defaultdict

app = modal.App("recluster-2024-topics")
volume = modal.Volume.from_name("tweet-vectors-large", create_if_missing=False)

image = modal.Image.debian_slim().pip_install(
    "anthropic",
    "numpy",
)

@app.function(
    volumes={"/data": volume},
    image=image,
    timeout=7200,  # 2 hours
    cpu=4,
    memory=8192,
    secrets=[modal.Secret.from_name("anthropic-api-key")]
)
def regenerate_2024_topics():
    """
    Regenerate topics for 2024 using LLM-based semantic analysis
    """
    from anthropic import Anthropic
    import os

    print("=" * 80)
    print("REGENERATING 2024 TOPICS WITH LLM SEMANTIC FILTERING")
    print("=" * 80)
    print()

    # Initialize Claude client
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Load organized data from pickle file
    print("Loading organized data from Modal volume...")
    organized_path = Path("/data/organized_by_community.pkl")
    with open(organized_path, 'rb') as f:
        organized = pickle.load(f)
    print("✓ Loaded organized data")

    # Try to load community names (optional)
    community_names = {}
    names_file = "/data/all_community_names.json"
    try:
        print("Loading community names...")
        with open(names_file, 'r') as f:
            community_names_data = json.load(f)

        # Get 2024 community names
        if "2024" in community_names_data:
            for comm in community_names_data["2024"]["communities"]:
                community_names[comm["community_id"]] = comm["name"]

        print(f"✓ Found {len(community_names)} community names for 2024")
    except FileNotFoundError:
        print("⚠️  Community names file not found - will use generic names")
    print()

    # Process 2024 communities
    year_key = "2024"
    results = {
        "year": "2024",
        "stats": {
            "year": "2024",
            "total_communities": 0,
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

    if year_key not in organized:
        print(f"No data found for 2024")
        return results

    # Combine all months together for each community
    communities_combined = defaultdict(lambda: {'tweets': []})

    for month in organized[year_key]:
        for community in organized[year_key][month]:
            group_data = organized[year_key][month][community]
            communities_combined[community]['tweets'].extend(group_data['tweets'])

    print(f"Combined {len(organized[year_key])} months into {len(communities_combined)} communities")

    # Extract tweets by community
    communities_tweets = {comm_id: data['tweets'] for comm_id, data in communities_combined.items()}

    results["stats"]["total_communities"] = len(communities_tweets)

    # Sort by size
    sorted_communities = sorted(
        communities_tweets.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )

    print(f"Processing {len(sorted_communities)} communities for 2024...")
    print()

    # Process each community
    for comm_id, tweets in sorted_communities:
        print("-" * 80)
        print(f"Community {comm_id}: {len(tweets)} tweets")

        # Get community name
        comm_name = community_names.get(comm_id, f"Community {comm_id}")
        print(f"Name: {comm_name}")

        # Skip if too small
        if len(tweets) < 50:
            print(f"⚠️  Skipping - too few tweets ({len(tweets)} < 50)")
            results["stats"]["communities_skipped_too_small"] += 1
            print()
            continue

        # Sample tweets (max 100 for analysis)
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

            # Debug: print first 200 chars of response
            print(f"    Response preview: {response_text[:200]}...")

            # Extract JSON from response with multiple strategies
            json_str = None

            # Strategy 1: Look for ```json code blocks
            if "```json" in response_text:
                try:
                    json_str = response_text.split("```json")[1].split("```")[0].strip()
                except:
                    pass

            # Strategy 2: Look for any ``` code blocks
            if not json_str and "```" in response_text:
                try:
                    json_str = response_text.split("```")[1].split("```")[0].strip()
                except:
                    pass

            # Strategy 3: Look for JSON object with { }
            if not json_str and "{" in response_text and "}" in response_text:
                try:
                    start = response_text.index("{")
                    # Find the matching closing brace
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
                except:
                    pass

            # Strategy 4: Use the whole response as-is
            if not json_str:
                json_str = response_text.strip()

            if not json_str:
                raise ValueError("Empty response from LLM")

            llm_result = json.loads(json_str)

            # Process topics
            community_topics = []
            for topic_data in llm_result.get("topics", []):
                if not topic_data.get("relevant", True):
                    results["stats"]["clusters_filtered_llm"] += 1
                    continue

                # Get tweets for this topic
                topic_tweets = []
                tweet_indices = topic_data.get("tweet_indices", [])
                for idx in tweet_indices:
                    if 0 <= idx - 1 < len(sampled_tweets):
                        topic_tweets.append(sampled_tweets[idx - 1])

                # If no specific indices, just sample some tweets from the community
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
                    "confidence": "high",  # Mark as high since LLM filtered it
                    "num_tweets": len(topic_tweet_ids),
                    "tweet_ids": topic_tweet_ids,
                    "sample_tweets": all_tweets_data  # Now contains ALL tweets, not just 5
                })

                results["stats"]["clusters_kept"] += 1
                results["stats"]["total_tweets_kept"] += len(topic_tweet_ids)

            if community_topics:
                results["communities"][str(comm_id)] = community_topics
                results["stats"]["communities_processed"] += 1
                results["stats"]["total_clusters"] += len(community_topics)
                print(f"  ✓ Generated {len(community_topics)} topics")
            else:
                print(f"  ⚠️  No relevant topics found")
                results["stats"]["communities_skipped_too_small"] += 1

        except json.JSONDecodeError as e:
            print(f"  ❌ JSON Error: {e}")
            if 'response_text' in locals():
                print(f"    Full response (first 500 chars):")
                print(f"    {response_text[:500]}")
            results["stats"]["communities_skipped_too_small"] += 1
        except Exception as e:
            print(f"  ❌ Error: {e}")
            if 'response_text' in locals():
                print(f"    Full response (first 500 chars):")
                print(f"    {response_text[:500]}")
            results["stats"]["communities_skipped_too_small"] += 1

        print()
        results["stats"]["total_tweets_original"] += len(tweets)

    # Save results
    output_file = "/data/topics_year_2024_summary.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"Processed: {results['stats']['communities_processed']} communities")
    print(f"Total topics: {results['stats']['total_clusters']}")
    print(f"Filtered: {results['stats']['clusters_filtered_llm']} irrelevant topics")
    print()

    return results

@app.local_entrypoint()
def main():
    print("Starting 2024 topic regeneration...")
    print("This will:")
    print("  1. Load 2024 community data and names")
    print("  2. For each community, use LLM to identify relevant topics")
    print("  3. Filter out irrelevant/unknown content")
    print("  4. Save high-confidence topics")
    print()
    print("Estimated time: 20-30 minutes")
    print()

    # Run the function
    results = regenerate_2024_topics.remote()

    print()
    print("=" * 80)
    print("DONE!")
    print("=" * 80)
    print()
    print(f"✓ Processed {results['stats']['communities_processed']} communities")
    print(f"✓ Generated {results['stats']['total_clusters']} topics")
    print(f"✓ Filtered {results['stats']['clusters_filtered_llm']} irrelevant topics")
    print()
    print("Next steps:")
    print("  1. Run: modal run download_topics.py")
    print("  2. Run: python3 combine_local_topics.py")
    print("  3. Deploy frontend")
    print()
