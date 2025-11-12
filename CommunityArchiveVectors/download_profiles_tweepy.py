"""
Download profile images using Twitter's API via Tweepy.
Much more reliable than unavatar.io since it's directly from Twitter.
"""

import json
import os
from pathlib import Path
import requests
import tweepy
from typing import Set
import time

def get_unique_usernames() -> list[str]:
    """Extract unique usernames from network animation data."""
    network_file = Path("frontend/public/network_animation_data.json")

    with open(network_file, 'r') as f:
        data = json.load(f)

    usernames = set()

    for year_data in data.get("years", []):
        for node in year_data.get("nodes", []):
            username = node.get("id")
            if username:
                usernames.add(username)

    print(f"Found {len(usernames)} unique users in network data")
    return sorted(list(usernames))

def setup_twitter_client():
    """Setup Tweepy client with your Twitter API credentials."""
    # You'll need to set these environment variables
    bearer_token = os.environ.get("TWITTER_BEARER_TOKEN")

    if not bearer_token:
        print("ERROR: Please set TWITTER_BEARER_TOKEN environment variable")
        print("You can get this from your Twitter Developer Portal")
        return None

    client = tweepy.Client(bearer_token=bearer_token)
    return client

def download_image(url: str, output_path: Path) -> bool:
    """Download image from URL."""
    try:
        # Get original quality image (remove _normal suffix)
        url = url.replace('_normal', '_400x400')

        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return True
        return False
    except Exception as e:
        return False

def main():
    print("🐦 Downloading profile images using Twitter API (Tweepy)")
    print("=" * 60)

    # Setup
    client = setup_twitter_client()
    if not client:
        return

    usernames = get_unique_usernames()
    output_dir = Path("frontend/public/profile-images")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Twitter API allows 100 users per request
    batch_size = 100
    total = len(usernames)
    downloaded = 0
    failed = 0
    skipped = 0

    print(f"\n📥 Processing {total} users in batches of {batch_size}")
    print(f"📁 Output directory: {output_dir}\n")

    for i in range(0, total, batch_size):
        batch = usernames[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size

        print(f"Batch {batch_num}/{total_batches}: Processing {len(batch)} users...")

        try:
            # Fetch user data from Twitter API
            response = client.get_users(
                usernames=batch,
                user_fields=['profile_image_url']
            )

            if response.data:
                for user in response.data:
                    username = user.username
                    profile_image_url = user.profile_image_url
                    output_path = output_dir / f"{username}.jpg"

                    # Skip if already downloaded
                    if output_path.exists():
                        skipped += 1
                        continue

                    # Download image
                    if download_image(profile_image_url, output_path):
                        downloaded += 1
                        if downloaded % 10 == 0:
                            print(f"  ✓ Downloaded {downloaded} images...")
                    else:
                        failed += 1

            # Handle users not found
            if response.errors:
                for error in response.errors:
                    failed += 1

            # Rate limit: Twitter API v2 allows 300 requests per 15 min window
            # Sleep to avoid hitting rate limits
            time.sleep(1)

        except tweepy.errors.TooManyRequests:
            print("⚠️  Rate limited! Waiting 15 minutes...")
            time.sleep(15 * 60)
        except Exception as e:
            print(f"❌ Error processing batch: {e}")
            failed += len(batch)

    print("\n" + "=" * 60)
    print("✅ Download Complete!")
    print(f"Downloaded: {downloaded}")
    print(f"Skipped (already existed): {skipped}")
    print(f"Failed: {failed}")
    print(f"Total: {total}")

if __name__ == "__main__":
    main()
