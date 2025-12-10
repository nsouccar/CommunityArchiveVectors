"""
Embed tweets directly into topic JSON files.

This script fetches tweets from Supabase for all topics and embeds them
directly into the topic JSON files as `sample_tweets`. This eliminates
the need for API calls during page load.

Usage:
    python embed_tweets_in_topics.py

Requires SUPABASE_URL and SUPABASE_KEY environment variables.
"""

import json
import os
from pathlib import Path
from supabase import create_client
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Configuration
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
MAX_TWEETS_PER_TOPIC = 50
BATCH_SIZE = 500  # Supabase query batch size


def load_avatar_urls() -> dict[str, str]:
    """Load username to avatar URL mapping."""
    if AVATAR_FILE.exists():
        with open(AVATAR_FILE) as f:
            return json.load(f)
    print(f"Warning: {AVATAR_FILE} not found, profile images will be unavailable")
    return {}


def create_supabase_client():
    """Create Supabase client from environment variables."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables required")

    return create_client(url, key)


def fetch_tweets_batch(supabase, tweet_ids: list[str]) -> dict[str, dict]:
    """Fetch tweets by IDs in batches."""
    tweets = {}

    for i in range(0, len(tweet_ids), BATCH_SIZE):
        batch = tweet_ids[i:i + BATCH_SIZE]

        response = supabase.table("tweets").select(
            "tweet_id, full_text, created_at, account_id, retweet_count, favorite_count, reply_to_tweet_id"
        ).in_("tweet_id", batch).execute()

        for tweet in response.data:
            tweets[tweet["tweet_id"]] = tweet

    return tweets


def fetch_accounts_batch(supabase, account_ids: list[str]) -> dict[str, dict]:
    """Fetch account info by IDs in batches."""
    accounts = {}
    unique_ids = list(set(account_ids))

    for i in range(0, len(unique_ids), BATCH_SIZE):
        batch = unique_ids[i:i + BATCH_SIZE]

        response = supabase.table("account").select(
            "account_id, username, account_display_name"
        ).in_("account_id", batch).execute()

        for account in response.data:
            accounts[account["account_id"]] = account

    return accounts


def build_tweet_object(
    tweet: dict,
    accounts: dict[str, dict],
    avatar_urls: dict[str, str],
    parent_tweet: dict | None = None
) -> dict:
    """Build a tweet object matching the expected UI structure."""
    account_id = tweet.get("account_id")
    account = accounts.get(account_id, {})
    username = account.get("username", "")

    result = {
        "tweet_id": tweet.get("tweet_id"),
        "full_text": tweet.get("full_text", ""),
        "created_at": tweet.get("created_at"),
        "retweet_count": tweet.get("retweet_count", 0),
        "favorite_count": tweet.get("favorite_count", 0),
        "reply_to_tweet_id": tweet.get("reply_to_tweet_id"),
        "all_account": {
            "username": username,
            "account_display_name": account.get("account_display_name", username),
            "profile_image_url": avatar_urls.get(username)
        }
    }

    if parent_tweet:
        parent_account_id = parent_tweet.get("account_id")
        parent_account = accounts.get(parent_account_id, {})
        parent_username = parent_account.get("username", "")

        result["parent_tweet"] = {
            "tweet_id": parent_tweet.get("tweet_id"),
            "full_text": parent_tweet.get("full_text", ""),
            "created_at": parent_tweet.get("created_at"),
            "all_account": {
                "username": parent_username,
                "account_display_name": parent_account.get("account_display_name", parent_username),
                "profile_image_url": avatar_urls.get(parent_username)
            }
        }

    return result


def process_topic_file(filepath: Path, supabase, avatar_urls: dict[str, str]) -> tuple[int, int]:
    """Process a single topic file and embed tweets. Returns (topics_processed, tweets_embedded)."""
    print(f"\nProcessing {filepath.name}...")

    with open(filepath) as f:
        data = json.load(f)

    if "communities" not in data:
        print(f"  No communities found in {filepath.name}")
        return 0, 0

    # Collect all tweet IDs we need to fetch
    all_tweet_ids = set()
    topics_to_process = []

    for community_id, topics in data["communities"].items():
        for topic in topics:
            # Only process high confidence topics
            if topic.get("confidence") != "high":
                continue

            tweet_ids = topic.get("tweet_ids", [])
            if not tweet_ids:
                continue

            # Take first N tweet IDs
            tweet_ids = tweet_ids[:MAX_TWEETS_PER_TOPIC]
            all_tweet_ids.update(tweet_ids)
            topics_to_process.append((community_id, topic, tweet_ids))

    if not all_tweet_ids:
        print(f"  No tweets to fetch for {filepath.name}")
        return 0, 0

    print(f"  Found {len(topics_to_process)} high-confidence topics")
    print(f"  Fetching {len(all_tweet_ids)} unique tweets...")

    # Fetch all tweets
    tweets = fetch_tweets_batch(supabase, list(all_tweet_ids))
    print(f"  Fetched {len(tweets)} tweets from database")

    # Collect parent tweet IDs for replies
    parent_ids = set()
    for tweet in tweets.values():
        reply_to = tweet.get("reply_to_tweet_id")
        if reply_to:
            parent_ids.add(reply_to)

    # Fetch parent tweets
    parent_tweets = {}
    if parent_ids:
        print(f"  Fetching {len(parent_ids)} parent tweets...")
        parent_tweets = fetch_tweets_batch(supabase, list(parent_ids))
        print(f"  Fetched {len(parent_tweets)} parent tweets")

    # Collect all account IDs
    all_account_ids = set()
    for tweet in tweets.values():
        all_account_ids.add(tweet.get("account_id"))
    for tweet in parent_tweets.values():
        all_account_ids.add(tweet.get("account_id"))

    # Fetch account info
    print(f"  Fetching {len(all_account_ids)} account profiles...")
    accounts = fetch_accounts_batch(supabase, list(all_account_ids))
    print(f"  Fetched {len(accounts)} accounts")

    # Build sample_tweets for each topic
    total_embedded = 0
    for community_id, topic, tweet_ids in topics_to_process:
        sample_tweets = []

        for tweet_id in tweet_ids:
            if tweet_id not in tweets:
                continue

            tweet = tweets[tweet_id]
            parent = None

            reply_to = tweet.get("reply_to_tweet_id")
            if reply_to and reply_to in parent_tweets:
                parent = parent_tweets[reply_to]

            sample_tweets.append(build_tweet_object(tweet, accounts, avatar_urls, parent))

        topic["sample_tweets"] = sample_tweets
        total_embedded += len(sample_tweets)

    # Save updated file
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  Embedded {total_embedded} tweets across {len(topics_to_process)} topics")
    return len(topics_to_process), total_embedded


def main():
    print("=" * 80)
    print("EMBEDDING TWEETS INTO TOPIC FILES")
    print("=" * 80)

    # Load avatar URLs
    print("\nLoading avatar URLs...")
    avatar_urls = load_avatar_urls()
    print(f"  Loaded {len(avatar_urls)} avatar mappings")

    # Create Supabase client
    print("\nConnecting to Supabase...")
    supabase = create_supabase_client()
    print("  Connected!")

    # Process each topic file
    total_topics = 0
    total_tweets = 0
    start_time = time.time()

    for filename in TOPIC_FILES:
        filepath = TOPIC_DIR / filename

        if not filepath.exists():
            print(f"\nSkipping {filename} - file not found")
            continue

        topics, tweets = process_topic_file(filepath, supabase, avatar_urls)
        total_topics += topics
        total_tweets += tweets

    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"\nProcessed {len(TOPIC_FILES)} files in {elapsed:.1f} seconds")
    print(f"Total topics updated: {total_topics}")
    print(f"Total tweets embedded: {total_tweets}")
    print("\nThe frontend will now load tweets instantly from the JSON files!")


if __name__ == "__main__":
    main()
