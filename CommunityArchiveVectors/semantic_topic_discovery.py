"""
Discover semantic topics for each community using the search API

Instead of clustering embeddings directly (which CoreNN doesn't support),
we use semantic search to find which topics are most represented in each community.
"""

import pickle
import requests
from typing import Dict, List
from collections import defaultdict, Counter

# Topic keywords to test
TOPIC_KEYWORDS = [
    # Politics
    "politics", "election", "voting", "government", "policy",
    # Technology
    "technology", "software", "programming", "AI", "machine learning",
    # Social issues
    "feminism", "gender", "race", "equality", "justice",
    # Culture
    "art", "music", "books", "movies", "culture",
    # Science
    "science", "research", "climate", "space", "health",
    # Economics
    "economics", "business", "finance", "cryptocurrency", "money",
    # Social media/Online
    "twitter", "social media", "online", "internet", "memes",
    # Philosophy/Theory
    "philosophy", "theory", "ethics", "epistemology", "metaphysics",
    # Identity
    "queer", "trans", "gay", "lesbian", "LGBTQ",
    # Activism
    "activism", "protest", "resistance", "organizing", "solidarity"
]

def load_metadata():
    """Load tweet metadata"""
    print("Loading metadata.pkl...")
    with open('metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    print(f"Loaded {len(metadata['metadata']):,} tweets")
    return metadata['metadata']

def load_community_assignments():
    """Load community assignments"""
    print("Loading community assignments...")
    with open('user_communities.pkl', 'rb') as f:
        user_to_community = pickle.load(f)
    print(f"Loaded {len(user_to_community):,} users")
    return user_to_community

def search_topic(topic: str, limit: int = 50, server_url: str = "http://45.63.18.97:8000") -> List[Dict]:
    """Search for tweets matching a topic"""
    try:
        response = requests.get(
            f"{server_url}/search",
            params={"query": topic, "limit": limit},
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            return data.get('results', [])
        else:
            print(f"    Error searching '{topic}': {response.status_code}")
            return []

    except Exception as e:
        print(f"    Error searching '{topic}': {e}")
        return []

def analyze_community_topics(metadata: Dict, user_to_community: Dict[str, int], top_n_communities: int = 10):
    """Analyze which topics are most represented in each community"""

    # Group tweets by community
    print("\nGrouping tweets by community...")
    community_users = defaultdict(set)

    for username, comm_id in user_to_community.items():
        community_users[comm_id].add(username)

    # Sort communities by size
    large_communities = sorted(
        community_users.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )[:top_n_communities]

    print(f"Analyzing top {len(large_communities)} communities by size\n")

    # For each topic, search and see which communities have the most matches
    topic_community_scores = defaultdict(lambda: defaultdict(int))

    for topic in TOPIC_KEYWORDS:
        print(f"Searching for topic: '{topic}'...")
        results = search_topic(topic, limit=100)

        if not results:
            continue

        # Count how many tweets from each community
        for result in results:
            username = result.get('username')
            if username in user_to_community:
                comm_id = user_to_community[username]
                topic_community_scores[comm_id][topic] += 1

        print(f"  Found {len(results)} results across {len(set(r['username'] for r in results if r['username'] in user_to_community))} community users")

    # Display results
    print("\n" + "="*80)
    print("COMMUNITY TOPIC PROFILES")
    print("="*80)

    for comm_id, users in large_communities:
        print(f"\n{'='*80}")
        print(f"COMMUNITY {comm_id}: {len(users)} users")
        print(f"{'='*80}")

        # Get top topics for this community
        topics = topic_community_scores[comm_id]
        if not topics:
            print("  No matches found for standard topics")
            continue

        # Sort by score
        sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)

        print(f"\nTop topics (by tweet count in search results):\n")
        for i, (topic, count) in enumerate(sorted_topics[:10], 1):
            print(f"  {i}. {topic:20s} - {count:3d} tweets")

    # Save results
    with open('semantic_topic_profiles.pkl', 'wb') as f:
        pickle.dump(dict(topic_community_scores), f)

    print("\n" + "="*80)
    print("Analysis complete! Results saved to semantic_topic_profiles.pkl")

if __name__ == "__main__":
    metadata = load_metadata()
    user_to_community = load_community_assignments()
    analyze_community_topics(metadata, user_to_community)
