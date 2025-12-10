"""
Update profile_image_url in embedded tweets to use local avatar paths.

This script reads the updated avatar_urls.json (with local paths) and
updates all profile_image_url fields in the topic JSON files.

Usage:
    python update_embedded_avatar_urls.py
"""

import json
from pathlib import Path

TOPIC_FILES = [
    "topics_year_2012_summary.json",
    "topics_year_2018_summary.json",
    "topics_year_2019_summary.json",
    "topics_year_2020_summary.json",
    "topics_year_2021_summary.json",
    "topics_year_2022_summary.json",
    "topics_year_2023_summary.json",
    "topics_year_2024_summary.json",
    "topics_year_2025_summary.json",
]
TOPIC_DIR = Path("frontend/public/data")
AVATAR_FILE = Path("frontend/public/avatar_urls.json")


def main():
    print("=" * 80)
    print("UPDATING EMBEDDED AVATAR URLS")
    print("=" * 80)

    # Load new avatar URLs
    print("\nLoading avatar URLs...")
    with open(AVATAR_FILE) as f:
        avatar_urls = json.load(f)

    local_count = sum(1 for url in avatar_urls.values() if url.startswith('/avatars/'))
    print(f"  Loaded {len(avatar_urls)} avatar URLs ({local_count} local)")

    # Process each topic file
    total_updated = 0

    for filename in TOPIC_FILES:
        filepath = TOPIC_DIR / filename
        if not filepath.exists():
            print(f"\nSkipping {filename} - not found")
            continue

        print(f"\nProcessing {filename}...")

        with open(filepath) as f:
            data = json.load(f)

        if "communities" not in data:
            print(f"  No communities in {filename}")
            continue

        file_updated = 0

        for community_id, topics in data["communities"].items():
            for topic in topics:
                sample_tweets = topic.get("sample_tweets", [])

                for tweet in sample_tweets:
                    # Update main tweet's avatar
                    if tweet.get("all_account"):
                        username = tweet["all_account"].get("username")
                        if username and username in avatar_urls:
                            tweet["all_account"]["profile_image_url"] = avatar_urls[username]
                            file_updated += 1

                    # Update parent tweet's avatar
                    if tweet.get("parent_tweet") and tweet["parent_tweet"].get("all_account"):
                        parent_username = tweet["parent_tweet"]["all_account"].get("username")
                        if parent_username and parent_username in avatar_urls:
                            tweet["parent_tweet"]["all_account"]["profile_image_url"] = avatar_urls[parent_username]
                            file_updated += 1

        # Save updated file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"  Updated {file_updated} avatar URLs")
        total_updated += file_updated

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"\nTotal avatar URLs updated: {total_updated}")


if __name__ == "__main__":
    main()
