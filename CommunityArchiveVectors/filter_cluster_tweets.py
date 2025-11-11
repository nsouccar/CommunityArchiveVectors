"""
Filter tweets in clusters that don't match the cluster topic/description.

Uses Claude API to evaluate if each tweet belongs to its cluster.
"""

import modal
import json
from typing import List, Dict
import anthropic
import os

app = modal.App("filter-cluster-tweets")

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
    tweet_ids: List[str]
) -> Dict:
    """
    Filter tweets in a cluster, keeping only those that match the topic.

    Returns dict with:
    - kept_tweet_ids: List of tweet IDs that match
    - removed_tweet_ids: List of tweet IDs that don't match
    - kept_count: Number of tweets kept
    - removed_count: Number of tweets removed
    """
    from supabase import create_client
    import anthropic
    import os

    print(f"Filtering cluster {cluster_id}: {topic}")
    print(f"Original tweet count: {len(tweet_ids)}")

    # Initialize clients
    supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Fetch tweets from Supabase in batches to avoid URL length limits
    print("Fetching tweets from Supabase...")
    tweets_data = {}
    fetch_batch_size = 100

    for i in range(0, len(tweet_ids), fetch_batch_size):
        batch = tweet_ids[i:i+fetch_batch_size]
        response = supabase.from_("tweets").select("tweet_id, full_text").in_("tweet_id", batch).execute()
        for t in response.data:
            tweets_data[t["tweet_id"]] = t["full_text"]

    print(f"Fetched {len(tweets_data)} tweets")

    # Batch evaluate tweets (do 20 at a time for efficiency)
    kept_ids = []
    removed_ids = []

    batch_size = 20
    for i in range(0, len(tweet_ids), batch_size):
        batch_ids = tweet_ids[i:i+batch_size]
        batch_tweets = [(tid, tweets_data.get(tid, "")) for tid in batch_ids]

        # Create prompt for Claude
        tweets_text = "\n\n".join([
            f"Tweet {idx+1} (ID: {tid}):\n{text}"
            for idx, (tid, text) in enumerate(batch_tweets)
        ])

        prompt = f"""You are evaluating if tweets belong in a topic cluster.

Cluster Topic: {topic}
Cluster Description: {description}

Here are {len(batch_tweets)} tweets from this cluster. For each tweet, determine if it genuinely belongs to this topic cluster or if it seems mismatched/off-topic.

{tweets_text}

Respond with ONLY a JSON array of tweet numbers (1-{len(batch_tweets)}) for tweets that DO belong in this cluster (are relevant to the topic).
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
            import re
            json_match = re.search(r'\[[\d,\s]*\]', response_text)
            if json_match:
                kept_indices = json.loads(json_match.group())

                # Convert to tweet IDs
                for idx, (tid, _) in enumerate(batch_tweets):
                    if (idx + 1) in kept_indices:
                        kept_ids.append(tid)
                    else:
                        removed_ids.append(tid)
                        print(f"  Removing tweet {tid}: {tweets_data.get(tid, '')[:100]}...")
            else:
                print(f"Warning: Could not parse response for batch {i//batch_size + 1}")
                # Keep all tweets in this batch if parsing fails
                kept_ids.extend(batch_ids)

        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            # Keep all tweets in this batch if error occurs
            kept_ids.extend(batch_ids)

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
        "original_count": len(tweet_ids)
    }


@app.local_entrypoint()
def main(year: str = "2018", dry_run: bool = False):
    """
    Filter all clusters for a given year.

    Args:
        year: Year to process (e.g., "2018")
        dry_run: If True, only show what would be removed without updating files
    """
    import json

    # Load the topic summary file
    input_file = f"frontend/public/data/topics_year_{year}_summary.json"
    print(f"Loading {input_file}...")

    with open(input_file, "r") as f:
        data = json.load(f)

    print(f"Found {len(data['communities'])} communities")

    # Process each community's clusters - collect all remote calls first
    remote_calls = []
    for community_id, clusters in data["communities"].items():
        print(f"\nQueueing community {community_id} ({len(clusters)} clusters)...")

        for cluster in clusters:
            call = filter_cluster.remote(
                year=year,
                community=community_id,
                cluster_id=cluster["cluster_id"],
                topic=cluster["topic"],
                description=cluster["description"],
                tweet_ids=cluster["tweet_ids"]
            )
            remote_calls.append(call)

    print(f"\n✓ Queued {len(remote_calls)} clusters for parallel processing on Modal!")
    print("Waiting for all results...")

    # Wait for all results at once (runs in parallel!)
    results = [call for call in remote_calls]

    print("✓ All clusters processed!")

    # Calculate statistics
    total_original = sum(r["original_count"] for r in results)
    total_kept = sum(r["kept_count"] for r in results)
    total_removed = sum(r["removed_count"] for r in results)

    print(f"\n{'='*60}")
    print(f"SUMMARY FOR YEAR {year}")
    print(f"{'='*60}")
    print(f"Total tweets processed: {total_original}")
    print(f"Total tweets kept: {total_kept} ({100*total_kept/total_original:.1f}%)")
    print(f"Total tweets removed: {total_removed} ({100*total_removed/total_original:.1f}%)")
    print()

    # Show clusters with most removals
    results_sorted = sorted(results, key=lambda r: r["removed_count"], reverse=True)
    print("Clusters with most removals:")
    for r in results_sorted[:10]:
        if r["removed_count"] > 0:
            pct = 100 * r["removed_count"] / r["original_count"]
            print(f"  {r['topic'][:50]}: {r['removed_count']}/{r['original_count']} removed ({pct:.1f}%)")

    if not dry_run:
        print(f"\nUpdating {input_file}...")

        # Update the data with filtered tweet IDs
        for result in results:
            community_id = result["community"]
            cluster_id = result["cluster_id"]

            # Find and update the cluster
            for cluster in data["communities"][community_id]:
                if cluster["cluster_id"] == cluster_id:
                    cluster["tweet_ids"] = result["kept_tweet_ids"]
                    cluster["num_tweets"] = result["kept_count"]

                    # Add metadata about filtering
                    cluster["filtered"] = True
                    cluster["original_num_tweets"] = result["original_count"]
                    cluster["removed_num_tweets"] = result["removed_count"]
                    break

        # Update stats
        data["stats"]["total_tweets_before_filtering"] = total_original
        data["stats"]["total_tweets_after_filtering"] = total_kept
        data["stats"]["tweets_removed_by_filtering"] = total_removed

        # Save updated file
        with open(input_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Updated file saved!")
    else:
        print("\nDRY RUN - No files were modified.")
        print("Run with --dry-run=false to actually update the files.")
