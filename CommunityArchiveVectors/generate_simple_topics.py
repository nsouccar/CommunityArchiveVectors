#!/usr/bin/env python3
"""
Generate semantic topic labels for communities using LLM analysis of sample tweets
No embeddings or clustering needed - just let the LLM identify topics directly
"""

import pickle
import json
import random
from collections import defaultdict, Counter
import anthropic
import os

def load_data():
    """Load metadata"""
    print("Loading metadata...")
    with open('metadata.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['metadata']

def build_network_and_communities(metadata, min_interactions=2):
    """Build network and detect communities"""
    import networkx as nx
    from networkx.algorithms import community as nx_community

    print("\nBuilding network...")

    # Build mapping: tweet_id -> username
    tweet_to_user = {}
    for tweet_id, tweet_data in metadata.items():
        tweet_to_user[tweet_id] = tweet_data.get('username', 'unknown')

    # Count interactions
    interactions = defaultdict(int)
    for tweet_id, tweet_data in metadata.items():
        from_user = tweet_data.get('username')
        reply_to_id = tweet_data.get('reply_to_tweet_id')

        if from_user and reply_to_id and reply_to_id in tweet_to_user:
            to_user = tweet_to_user[reply_to_id]
            if from_user != to_user:
                interactions[(from_user, to_user)] += 1

    # Build graph
    G = nx.DiGraph()
    for (from_user, to_user), count in interactions.items():
        if count >= min_interactions:
            G.add_edge(from_user, to_user, weight=count)

    print(f"Network: {G.number_of_nodes():,} users, {G.number_of_edges():,} interactions")

    # Detect communities
    print("Detecting communities...")
    G_undirected = G.to_undirected()
    communities = nx_community.louvain_communities(G_undirected, seed=42)

    user_to_community = {}
    for comm_id, community in enumerate(communities):
        if len(community) >= 5:
            for user in community:
                user_to_community[user] = comm_id

    print(f"Found {len(set(user_to_community.values()))} communities (5+ users)")

    return user_to_community

def get_community_tweets(metadata, user_to_community):
    """Get all tweets for each community"""
    print("\nGrouping tweets by community...")

    community_tweets = defaultdict(list)

    for tweet_id, tweet_data in metadata.items():
        username = tweet_data.get('username')
        if username in user_to_community:
            comm_id = user_to_community[username]
            community_tweets[comm_id].append({
                'tweet_id': tweet_id,
                'text': tweet_data.get('full_text', ''),
                'username': username,
                'created_at': tweet_data.get('created_at', '')
            })

    return community_tweets

def analyze_community_with_llm(comm_id, tweets, client, n_topics=5, sample_size=150):
    """
    Use LLM to identify main topics in a community
    """
    print(f"\n{'='*80}")
    print(f"COMMUNITY {comm_id}: {len(tweets):,} tweets")
    print(f"{'='*80}")

    # Sample tweets
    if len(tweets) > sample_size:
        sampled_tweets = random.sample(tweets, sample_size)
        print(f"Sampled {sample_size} tweets for analysis")
    else:
        sampled_tweets = tweets

    # Create tweet text for LLM
    tweets_text = "\n\n".join([
        f"{i+1}. @{t['username']}: {t['text'][:300]}"
        for i, t in enumerate(sampled_tweets[:100])  # Max 100 tweets to keep prompt reasonable
    ])

    prompt = f"""You are analyzing a Twitter community. Below are {min(100, len(sampled_tweets))} sample tweets from this community of {len(tweets):,} total tweets.

{tweets_text}

Based on these tweets, identify the {n_topics} main topics or themes that this community discusses. For each topic:
1. Give it a short, descriptive label (2-5 words) like "AI and machine learning" or "Ukraine Russia war"
2. Provide 2-3 example tweet numbers that represent this topic

Respond with ONLY a JSON array in this exact format:
[
  {{"label": "Topic name", "example_indices": [1, 5, 12]}},
  {{"label": "Another topic", "example_indices": [3, 8, 15]}},
  ...
]

Be specific and concrete. Avoid generic labels like "general discussion" or "various topics"."""

    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = message.content[0].text.strip()

        # Extract JSON from response (handle code blocks)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        topics_data = json.loads(response_text)

        # Build topics with example tweets
        topics = []
        for topic in topics_data[:n_topics]:
            example_tweets = []
            for idx in topic.get('example_indices', [])[:3]:
                if 0 < idx <= len(sampled_tweets):
                    tweet = sampled_tweets[idx - 1]
                    example_tweets.append({
                        'username': tweet['username'],
                        'text': tweet['text'],
                        'created_at': tweet['created_at']
                    })

            topics.append({
                'label': topic['label'],
                'num_tweets': len(tweets) // n_topics,  # Rough estimate
                'example_tweets': example_tweets
            })

            print(f"  ✓ {topic['label']}")

        return topics

    except Exception as e:
        print(f"  Error analyzing community {comm_id}: {e}")
        return []

def main():
    """Main function to generate community topics"""

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nERROR: ANTHROPIC_API_KEY not set")
        print("Run: export ANTHROPIC_API_KEY='your-key'")
        return

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # Load data
    metadata = load_data()

    # Build network and detect communities
    user_to_community = build_network_and_communities(metadata, min_interactions=2)

    # Get tweets for each community
    community_tweets = get_community_tweets(metadata, user_to_community)

    # Sort communities by size
    sorted_communities = sorted(
        community_tweets.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )

    print("\n" + "="*80)
    print("ANALYZING COMMUNITIES WITH LLM")
    print("="*80)

    # Analyze top 10 communities
    community_topics = {}

    for comm_id, tweets in sorted_communities[:10]:
        topics = analyze_community_with_llm(comm_id, tweets, client, n_topics=5, sample_size=150)

        if topics:
            # Get unique users
            users = set(t['username'] for t in tweets)

            community_topics[str(comm_id)] = {
                'num_tweets': len(tweets),
                'num_users': len(users),
                'topics': topics,
                'top_users': [u for u, _ in Counter(t['username'] for t in tweets).most_common(10)]
            }

    # Save results
    output = {
        'num_communities': len(community_topics),
        'total_tweets_analyzed': sum(ct['num_tweets'] for ct in community_topics.values()),
        'communities': community_topics
    }

    with open('community_topics.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total communities analyzed: {len(community_topics)}")
    print(f"Total tweets analyzed: {output['total_tweets_analyzed']:,}")
    print(f"\nResults saved to community_topics.json")

    # Print summary
    print("\nTop Communities by Size:")
    for i, (comm_id, data) in enumerate(sorted(community_topics.items(), key=lambda x: x[1]['num_tweets'], reverse=True), 1):
        print(f"\n{i}. Community {comm_id}")
        print(f"   {data['num_tweets']:,} tweets from {data['num_users']} users")
        print(f"   Topics:")
        for topic in data['topics']:
            print(f"     - {topic['label']}")

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    main()
