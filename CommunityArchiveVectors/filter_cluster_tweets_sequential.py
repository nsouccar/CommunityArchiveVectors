"""
SEQUENTIAL VERSION: Filter tweets without aggressive parallelization

IMPROVEMENTS:
- Removes orphaned replies (replies without parent tweet in database)
- Optionally filters low-engagement tweets
- Shows parent tweets even if deleted from Twitter
- **PROCESSES BATCHES SEQUENTIALLY** to avoid rate limits
- **SAVES INCREMENTALLY** after each community

Uses Claude API to evaluate if each tweet belongs to its cluster.
"""

import modal
import json
from typing import List, Dict, Optional
import anthropic
import os

app = modal.App("filter-cluster-tweets-sequential")

# Supabase setup
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "anthropic",
        "supabase"
    )
)

@app.function(
    image=image,
    secrets=[
        modal.Secret.from_dict({
            "SUPABASE_URL": SUPABASE_URL,
            "SUPABASE_KEY": SUPABASE_KEY,
        }),
        modal.Secret.from_name("anthropic-api-key")
    ],
    timeout=3600,
)
def filter_cluster(
    year: str,
    community: str,
    cluster_id: int,
    topic: str,
    description: str,
    tweet_ids: List[str],
    min_engagement: int = 0
) -> Dict:
    """
    Filter tweets in a cluster, keeping only those that match the topic.
    PROCESSES BATCHES SEQUENTIALLY to avoid rate limits.
    """
    from supabase import create_client
    import anthropic
    import os
    import json
    import re

    print(f"Filtering cluster {cluster_id}: {topic}")
    print(f"Original tweet count: {len(tweet_ids)}")

    # Initialize clients
    supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Fetch tweets from Supabase with reply and engagement data
    print("Fetching tweets from Supabase...")
    tweets_data = {}
    fetch_batch_size = 100

    for i in range(0, len(tweet_ids), fetch_batch_size):
        batch = tweet_ids[i:i+fetch_batch_size]
        response = supabase.from_("tweets").select(
            "tweet_id, full_text, reply_to_tweet_id, retweet_count, favorite_count"
        ).in_("tweet_id", batch).execute()

        for t in response.data:
            tweets_data[t["tweet_id"]] = {
                "text": t["full_text"],
                "reply_to": t.get("reply_to_tweet_id"),
                "retweets": t.get("retweet_count", 0) or 0,
                "likes": t.get("favorite_count", 0) or 0
            }

    print(f"Fetched {len(tweets_data)} tweets")

    # Fetch parent tweets for replies to provide context
    parent_tweet_ids = list(set(t["reply_to"] for t in tweets_data.values() if t["reply_to"]))
    parent_tweets = {}

    if parent_tweet_ids:
        print(f"Fetching {len(parent_tweet_ids)} parent tweets for context...")
        for i in range(0, len(parent_tweet_ids), fetch_batch_size):
            batch = parent_tweet_ids[i:i+fetch_batch_size]
            response = supabase.from_("tweets").select(
                "tweet_id, full_text"
            ).in_("tweet_id", batch).execute()

            for t in response.data:
                parent_tweets[t["tweet_id"]] = t["full_text"]

        print(f"Fetched {len(parent_tweets)} parent tweets")

    # First pass: Check for orphaned replies and low engagement
    orphaned_replies = []
    low_engagement_tweets = []
    valid_tweet_ids = []

    print("Checking for orphaned replies and low engagement...")
    for tid in tweet_ids:
        tweet_info = tweets_data.get(tid)
        if not tweet_info:
            continue

        # Check if it's an orphaned reply
        if tweet_info["reply_to"]:
            parent_id = tweet_info["reply_to"]
            if parent_id not in parent_tweets:
                print(f"  Removing orphaned reply {tid} (parent {parent_id} not found): {tweet_info['text'][:80]}...")
                orphaned_replies.append(tid)
                continue

        # Check engagement threshold
        total_engagement = tweet_info["retweets"] + tweet_info["likes"]
        if min_engagement > 0 and total_engagement < min_engagement:
            print(f"  Removing low-engagement tweet {tid} (engagement: {total_engagement}): {tweet_info['text'][:80]}...")
            low_engagement_tweets.append(tid)
            continue

        valid_tweet_ids.append(tid)

    print(f"After pre-filtering: {len(valid_tweet_ids)} tweets remain")
    print(f"  Removed {len(orphaned_replies)} orphaned replies")
    print(f"  Removed {len(low_engagement_tweets)} low-engagement tweets")

    # Second pass: Batch evaluate tweets with Claude SEQUENTIALLY
    batch_size = 200  # Larger batches since we're sequential
    kept_ids = []
    removed_ids = orphaned_replies + low_engagement_tweets

    print(f"Processing {len(valid_tweet_ids)} tweets in batches of {batch_size} (SEQUENTIAL)...")

    for i in range(0, len(valid_tweet_ids), batch_size):
        batch_ids = valid_tweet_ids[i:i+batch_size]
        batch_tweets = []

        # Build batch with parent tweet context for replies
        for tid in batch_ids:
            tweet_info = tweets_data.get(tid, {})
            text = tweet_info.get("text", "")
            reply_to_id = tweet_info.get("reply_to")

            if reply_to_id and reply_to_id in parent_tweets:
                parent_text = parent_tweets[reply_to_id]
                batch_tweets.append((tid, text, parent_text))
            else:
                batch_tweets.append((tid, text, None))

        # Create prompt for Claude with parent tweet context
        tweets_text = ""
        for idx, (tid, text, parent_text) in enumerate(batch_tweets):
            if parent_text:
                tweets_text += f"Tweet {idx+1} (ID: {tid}):\n[Parent tweet: {parent_text}]\n[Reply: {text}]\n\n"
            else:
                tweets_text += f"Tweet {idx+1} (ID: {tid}):\n{text}\n\n"

        prompt = f"""You are evaluating if tweets belong in a topic cluster.

Cluster Topic: {topic}
Cluster Description: {description}

Here are {len(batch_tweets)} tweets from this cluster. For each tweet, determine if it genuinely belongs to this topic cluster or if it seems mismatched/off-topic.

IMPORTANT CRITERIA:
1. Is the tweet substantively about the topic? (Not just mentioning it in passing)
2. Does it add value to understanding this topic? (Not just "thanks!" or "glad you like it!")
3. For replies: Does the CONVERSATION (parent + reply together) discuss the topic meaningfully? If the parent tweet is off-topic, the reply should be removed too.

{tweets_text}

Respond with ONLY a JSON array of tweet numbers (1-{len(batch_tweets)}) for tweets that DO belong in this cluster (are relevant and substantive).
For example: [1, 3, 4, 7] would mean tweets 1, 3, 4, and 7 belong, while 2, 5, 6 should be removed.

If ALL tweets belong, return all numbers. If NONE belong, return an empty array [].

JSON array only, no explanation:"""

        try:
            message = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse response
            response_text = message.content[0].text.strip()
            # Extract JSON array from response
            json_match = re.search(r'\[[\d,\s]*\]', response_text)

            if json_match:
                kept_indices = json.loads(json_match.group())

                # Convert to tweet IDs
                for idx, (tid, text, _) in enumerate(batch_tweets):
                    if (idx + 1) in kept_indices:
                        kept_ids.append(tid)
                    else:
                        removed_ids.append(tid)
                        print(f"  Removing off-topic tweet {tid}: {text[:100]}...")
            else:
                print(f"Warning: Could not parse response for batch {i//batch_size + 1}")
                # Keep all tweets in this batch if parsing fails
                kept_ids.extend(batch_ids)

        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            # Keep all tweets in this batch if error occurs
            kept_ids.extend(batch_ids)

        # Progress indicator
        if (i // batch_size + 1) % 5 == 0:
            print(f"  Processed {i + len(batch_ids)}/{len(valid_tweet_ids)} tweets...")

    print(f"Kept: {len(kept_ids)}, Removed: {len(removed_ids)}")

    return {
        "year": year,
        "community": community,
        "cluster_id": cluster_id,
        "topic": topic,
        "description": description,
        "kept_tweet_ids": kept_ids,
        "removed_tweet_ids": removed_ids,
        "kept_count": len(kept_ids),
        "removed_count": len(removed_ids),
        "original_count": len(tweet_ids),
        "orphaned_replies_count": len(orphaned_replies),
        "low_engagement_count": len(low_engagement_tweets)
    }


@app.local_entrypoint()
def main(
    year: str = "2018",
    dry_run: bool = False,
    min_engagement: int = 0
):
    """
    Filter all clusters for a given year.
    SAVES INCREMENTALLY after each community.
    PROCESSES SEQUENTIALLY to avoid rate limits.
    """
    import json
    import time

    # Load the topic summary file
    input_file = f"frontend/public/data/topics_year_{year}_summary.json"
    output_file = f"frontend/public/data/topics_year_{year}_summary_filtered.json"
    print(f"Loading {input_file}...")
    print(f"Output will be saved to {output_file}")
    print(f"Min engagement threshold: {min_engagement} (0 = no filtering)")

    with open(input_file, "r") as f:
        data = json.load(f)

    print(f"Found {len(data['communities'])} communities")

    # Process community by community
    all_results = []

    for community_id, clusters in data["communities"].items():
        print(f"\n{'='*60}")
        print(f"Processing community {community_id} ({len(clusters)} clusters)...")
        print(f"{'='*60}")

        # Process each cluster one at a time
        for cluster in clusters:
            try:
                # Verify cluster has required fields
                if "tweet_ids" not in cluster:
                    print(f"\n⚠️  SKIPPING cluster {cluster.get('cluster_id', '?')}: Missing 'tweet_ids' field")
                    print(f"   Available keys: {list(cluster.keys())}")
                    continue

                if not cluster["tweet_ids"]:
                    print(f"\n⚠️  SKIPPING cluster {cluster['cluster_id']}: Empty tweet_ids list")
                    continue

                print(f"\nQueueing cluster {cluster['cluster_id']}: {cluster['topic']}")

                result = filter_cluster.remote(
                    year=year,
                    community=community_id,
                    cluster_id=cluster["cluster_id"],
                    topic=cluster["topic"],
                    description=cluster["description"],
                    tweet_ids=cluster["tweet_ids"],
                    min_engagement=min_engagement
                )

                all_results.append(result)

            except KeyError as e:
                print(f"\n❌ ERROR processing cluster {cluster.get('cluster_id', '?')}: Missing key {e}")
                print(f"   Available keys: {list(cluster.keys())}")
                print(f"   SKIPPING this cluster and continuing...")
                continue

            except Exception as e:
                print(f"\n❌ ERROR processing cluster {cluster.get('cluster_id', '?')}: {type(e).__name__}: {e}")
                print(f"   SKIPPING this cluster and continuing...")
                continue

        print(f"\n✓ Community {community_id} processed!")

        # SAVE INCREMENTALLY after each community
        if not dry_run:
            print(f"💾 Saving progress after community {community_id}...")

            # Update data with all results so far
            for result in all_results:
                comm_id = result["community"]
                cluster_id = result["cluster_id"]

                for cluster in data["communities"][comm_id]:
                    if cluster["cluster_id"] == cluster_id:
                        cluster["tweet_ids"] = result["kept_tweet_ids"]
                        cluster["num_tweets"] = result["kept_count"]
                        cluster["filtered"] = True
                        cluster["original_num_tweets"] = result["original_count"]
                        cluster["removed_num_tweets"] = result["removed_count"]
                        cluster["orphaned_replies_removed"] = result["orphaned_replies_count"]
                        cluster["low_engagement_removed"] = result["low_engagement_count"]
                        break

            # Save to filtered output file (preserves original)
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)

            print(f"✓ Progress saved to {output_file}! ({len(all_results)} clusters complete)")

        # Brief pause between communities to avoid rate limits
        if community_id != list(data["communities"].keys())[-1]:
            print("⏸️  Pausing 10 seconds between communities to avoid rate limits...")
            time.sleep(10)

    print(f"\n{'='*60}")
    print("✓ All communities processed!")
    print(f"{'='*60}")

    results = all_results

    # Calculate statistics
    total_original = sum(r["original_count"] for r in results)
    total_kept = sum(r["kept_count"] for r in results)
    total_removed = sum(r["removed_count"] for r in results)
    total_orphaned = sum(r["orphaned_replies_count"] for r in results)
    total_low_engagement = sum(r["low_engagement_count"] for r in results)

    print(f"\n{'='*60}")
    print(f"SUMMARY FOR YEAR {year}")
    print(f"{'='*60}")
    print(f"Total tweets processed: {total_original}")
    print(f"Total tweets kept: {total_kept} ({100*total_kept/total_original:.1f}%)")
    print(f"Total tweets removed: {total_removed} ({100*total_removed/total_original:.1f}%)")
    print(f"  - Orphaned replies: {total_orphaned}")
    print(f"  - Low engagement: {total_low_engagement}")
    print(f"  - Off-topic: {total_removed - total_orphaned - total_low_engagement}")
    print()

    # Show clusters with most removals
    results_sorted = sorted(results, key=lambda r: r["removed_count"], reverse=True)
    print("Clusters with most removals:")
    for r in results_sorted[:10]:
        if r["removed_count"] > 0:
            pct = 100 * r["removed_count"] / r["original_count"]
            print(f"  {r['topic'][:50]}: {r['removed_count']}/{r['original_count']} removed ({pct:.1f}%)")

    if not dry_run:
        # Update stats
        data["stats"]["total_tweets_before_filtering"] = total_original
        data["stats"]["total_tweets_after_filtering"] = total_kept
        data["stats"]["tweets_removed_by_filtering"] = total_removed
        data["stats"]["orphaned_replies_removed"] = total_orphaned
        data["stats"]["low_engagement_removed"] = total_low_engagement

        # Save final filtered file (preserves original)
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\n✓ Filtered data saved to: {output_file}")
        print(f"✓ Original data preserved at: {input_file}")
    else:
        print("\nDRY RUN - No files were modified.")
        print("Run with --dry-run=false to actually update the files.")
