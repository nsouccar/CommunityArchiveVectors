"""
Generate funny/fitting community names based on topics discussed.
Downloads summaries from Modal volume and uses Claude API to generate names.
"""

import modal
import json
import os
from pathlib import Path
import anthropic

# Modal setup
app = modal.App("community-names")
volume = modal.Volume.from_name("tweet-vectors-large", create_if_missing=False)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "anthropic"
)

@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("anthropic-api-key")],
    timeout=3600,  # 1 hour
)
def generate_names_for_year(year: int):
    """Generate community names for a specific year."""

    # Load the summary JSON
    summary_path = Path(f"/data/topics_year_{year}_summary.json")

    if not summary_path.exists():
        print(f"❌ No summary found for year {year}")
        return None

    with open(summary_path, 'r') as f:
        summary = json.load(f)

    print(f"\n{'='*80}")
    print(f"Generating community names for year {year}")
    print(f"{'='*80}")
    print(f"Found {len(summary['communities'])} communities")

    # Initialize Claude client
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Process each community
    communities_with_names = []

    for community_id_str, topics in summary['communities'].items():
        community_id = int(community_id_str)
        # Calculate total number of tweets from all topics
        num_tweets = sum(t.get('num_tweets', 0) for t in topics)

        print(f"\nCommunity {community_id}: {num_tweets:,} tweets, {len(topics)} topics")

        # Skip if no topics
        if not topics:
            communities_with_names.append({
                'community_id': community_id,
                'year': year,
                'name': f"The Silent Constellation #{community_id}",
                'description': "A mysterious community with no discernible topics",
                'num_tweets': num_tweets,
                'num_topics': 0,
                'topics': []
            })
            continue

        # Prepare topics for Claude
        topics_text = "\n".join([f"- {t['topic']}" for t in topics[:15]])  # Top 15 topics

        # Ask Claude to generate a funny/fitting name
        try:
            prompt = f"""Given these topics discussed in an online Twitter community:

{topics_text}

Generate a funny, clever, and fitting name for this community. The name should:
1. Be creative and memorable (3-6 words)
2. Capture the essence of what this community discusses
3. Have a bit of humor or wit
4. Be appropriate but can be slightly cheeky

Also provide a one-sentence description (10-15 words) of what makes this community unique.

Respond in JSON format:
{{
  "name": "Community Name",
  "description": "A brief description of the community"
}}"""

            response = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=300,
                temperature=0.9,  # Higher for more creativity
                messages=[{"role": "user", "content": prompt}]
            )

            result = json.loads(response.content[0].text)

            print(f"  ✓ Generated name: {result['name']}")
            print(f"    {result['description']}")

            communities_with_names.append({
                'community_id': community_id,
                'year': year,
                'name': result['name'],
                'description': result['description'],
                'num_tweets': num_tweets,
                'num_topics': len(topics),
                'topics': topics[:10]  # Include top 10 topics
            })

        except Exception as e:
            print(f"  ❌ Error generating name: {e}")
            # Fallback name
            if topics:
                main_topic = topics[0]['topic']
                communities_with_names.append({
                    'community_id': community_id,
                    'year': year,
                    'name': f"The {main_topic} Crew",
                    'description': f"A community focused on {main_topic.lower()}",
                    'num_tweets': num_tweets,
                    'num_topics': len(topics),
                    'topics': topics[:10]
                })
            else:
                communities_with_names.append({
                    'community_id': community_id,
                    'year': year,
                    'name': f"Community #{community_id}",
                    'description': "An enigmatic gathering of Twitter users",
                    'num_tweets': num_tweets,
                    'num_topics': len(topics),
                    'topics': []
                })

    # Save results
    output_path = Path(f"/data/community_names_{year}.json")
    with open(output_path, 'w') as f:
        json.dump({
            'year': year,
            'total_communities': len(communities_with_names),
            'communities': communities_with_names
        }, f, indent=2)

    print(f"\n✓ Saved community names to {output_path}")

    # Commit to volume
    volume.commit()

    return communities_with_names


@app.local_entrypoint()
def main(years: str = "2012,2018,2019,2020,2021,2022,2023,2024,2025"):
    """Generate community names for multiple years."""

    year_list = [int(y.strip()) for y in years.split(',')]

    print(f"\n{'='*80}")
    print(f"GENERATING COMMUNITY NAMES")
    print(f"{'='*80}")
    print(f"Years to process: {year_list}")

    all_results = {}

    for year in year_list:
        result = generate_names_for_year.remote(year)
        if result:
            all_results[year] = result
            print(f"\n✓ Completed year {year}: {len(result)} communities named")

    print(f"\n{'='*80}")
    print(f"COMPLETE!")
    print(f"{'='*80}")
    print(f"Generated names for {len(all_results)} years")
    print(f"Total communities named: {sum(len(r) for r in all_results.values())}")
