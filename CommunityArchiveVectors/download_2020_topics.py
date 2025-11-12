#!/usr/bin/env python3
"""
Download 2020 topics from Modal volume and convert to JSON for frontend
"""

import modal
import pickle
import json
from pathlib import Path

app = modal.App("download-2020-topics")
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy")
volume = modal.Volume.from_name("tweet-vectors-large", create_if_missing=False)

@app.function(volumes={"/data": volume}, image=image)
def download_and_convert():
    """Download topics_year_2020.pkl and convert to JSON"""

    topics_path = Path("/data/topics_year_2020.pkl")

    if not topics_path.exists():
        print(f"ERROR: {topics_path} does not exist!")
        return None

    print(f"Loading {topics_path}...")
    with open(topics_path, 'rb') as f:
        topics_data = pickle.load(f)

    print(f"Loaded data for year 2020")
    print(f"Number of communities: {len(topics_data)}")

    # Convert to JSON-serializable format matching the 2019 structure
    json_output = {
        "year": "2020",
        "stats": {
            "year": "2020",
            "total_communities": len(topics_data),
            "communities_processed": len(topics_data),
            "total_clusters": sum(len(topics) for topics in topics_data.values()),
        },
        "communities": {}
    }

    total_topics = 0

    for community_id, topics in topics_data.items():
        json_output["communities"][str(community_id)] = []

        for i, topic in enumerate(topics):
            # Match the exact format of 2019 topics
            topic_summary = {
                "cluster_id": i,
                "topic": topic.get("topic", "Unknown Topic"),
                "description": topic.get("description", ""),
                "confidence": topic.get("confidence", "unknown"),
                "num_tweets": len(topic.get("tweet_ids", [])),
                "tweet_ids": topic.get("tweet_ids", [])  # Use tweet_ids, not sample_tweets!
            }
            json_output["communities"][str(community_id)].append(topic_summary)
            total_topics += 1

    print(f"Total topics: {total_topics}")

    return json_output

@app.local_entrypoint()
def main():
    print("Downloading 2020 topics from Modal volume...")
    data = download_and_convert.remote()

    if data is None:
        print("Failed to download data")
        return

    # Save to frontend directory
    output_path = Path("frontend/public/data/topics_year_2020_summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n✅ Saved to {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")
