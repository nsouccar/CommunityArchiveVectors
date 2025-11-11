#!/usr/bin/env python3
"""
Community Topic Analysis - Discover what topics each community discusses
Using semantic clustering on tweet embeddings
"""

import pickle
import json
from collections import defaultdict, Counter
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import networkx as nx
from networkx.algorithms import community as nx_community

def load_data():
    """Load metadata"""
    print("Loading metadata...")
    with open('metadata.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['metadata']

def build_network_and_communities(metadata, min_interactions=2):
    """Build network and detect communities"""
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
        if len(community) >= 5:  # Only communities with 5+ users
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

    # Print stats
    for comm_id in sorted(community_tweets.keys())[:10]:
        tweets = community_tweets[comm_id]
        print(f"  Community {comm_id}: {len(tweets):,} tweets")

    return community_tweets

def extract_keywords(texts, top_n=10):
    """Extract top keywords from a list of texts using simple frequency analysis"""
    # Combine all texts
    combined = ' '.join(texts).lower()

    # Split into words and filter
    words = combined.split()

    # Common stop words to filter out (expanded list)
    stop_words = {
        # Articles, prepositions, conjunctions
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'as', 'by', 'with', 'from', 'is', 'was', 'are', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'can', 'it', 'this', 'that',
        'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they', 'what', 'which',
        'who', 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both',
        'few', 'more', 'most', 'other', 'some', 'such', 'no', 'not', 'only',
        'own', 'same', 'so', 'than', 'too', 'very', 'just', 'if', 'my', 'me',
        'your', 'their', 'our', 'his', 'her', 'its', 'im', 'ive', 'dont', 'rt',
        'https', 'http', 'co', 'get', 'like', 'amp', 'really', 'still', 'even',
        'much', 'any', 'because', 'about', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'between', 'under', 'again', 'further', 'then',
        'once', 'here', 'there', 'out', 'up', 'down', 'off', 'over',
        # Additional common words that don't add meaning
        'people', 'think', 'one', 'good', 'time', 'them', 'also', 'know', 'want',
        'way', 'now', 'new', 'make', 'see', 'look', 'take', 'come', 'use', 'find',
        'tell', 'ask', 'work', 'seem', 'feel', 'try', 'leave', 'call', 'back',
        'thing', 'things', 'got', 'going', 'said', 'need', 'well', 'lot', 'lol',
        'yeah', 'yes', 'say', 'says', 'thats', 'youre', 'theyre', 'cant', 'wont',
        'doesnt', 'didnt', 'wasnt', 'werent', 'isnt', 'arent', 'havent', 'hasnt',
        'hadnt', 'wouldnt', 'shouldnt', 'couldnt', 'theres', 'heres', 'whats',
        'wheres', 'whos', 'hows', 'right', 'made', 'long', 'actually', 'kind',
        'probably', 'maybe', 'something', 'someone', 'somewhere', 'anything',
        'anyone', 'anywhere', 'everything', 'everyone', 'everywhere', 'nothing',
        'nobody', 'nowhere', 'many', 'much', 'little', 'less', 'never', 'always',
        'often', 'sometimes', 'usually', 'almost', 'already', 'yet', 'since',
        'while', 'though', 'although', 'unless', 'until', 'whether', 'either',
        'neither', 'nor', 'via', 'per', 'etc', 'day', 'year', 'today', 'ago'
    }

    # Count words, filtering stop words and short words
    word_counts = Counter()
    for word in words:
        # Remove punctuation
        word = ''.join(c for c in word if c.isalnum() or c == '#' or c == '@')
        if len(word) > 2 and word not in stop_words and not word.startswith('http'):
            word_counts[word] += 1

    return word_counts.most_common(top_n)

def find_example_tweets(tweets, keywords, n=3):
    """Find example tweets that contain the keywords"""
    examples = []
    keywords_set = {kw[0] for kw in keywords[:5]}  # Top 5 keywords

    for tweet in tweets:
        text_lower = tweet['text'].lower()
        # Check if tweet contains any of the keywords
        if any(kw in text_lower for kw in keywords_set):
            examples.append(tweet)
            if len(examples) >= n:
                break

    return examples

def analyze_community_topics(community_tweets, top_communities=20):
    """Analyze topics for each community"""
    print("\nAnalyzing topics for each community...")

    community_topics = {}

    # Sort communities by number of tweets
    sorted_communities = sorted(
        community_tweets.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )

    for comm_id, tweets in sorted_communities[:top_communities]:
        print(f"\n{'='*80}")
        print(f"Community {comm_id}: {len(tweets):,} tweets")
        print(f"{'='*80}")

        # Get unique users
        users = set(t['username'] for t in tweets)
        print(f"Users: {len(users)}")

        # Extract keywords
        texts = [t['text'] for t in tweets]
        keywords = extract_keywords(texts, top_n=20)

        print("\nTop Keywords:")
        for i, (word, count) in enumerate(keywords[:15], 1):
            print(f"  {i:2}. {word:<20} ({count:,} mentions)")

        # Find example tweets
        examples = find_example_tweets(tweets, keywords, n=3)

        print("\nExample tweets:")
        for i, tweet in enumerate(examples, 1):
            text = tweet['text'][:100] + '...' if len(tweet['text']) > 100 else tweet['text']
            print(f"  {i}. @{tweet['username']}: {text}")

        # Store results
        community_topics[comm_id] = {
            'num_tweets': len(tweets),
            'num_users': len(users),
            'top_users': [u for u, _ in Counter(t['username'] for t in tweets).most_common(10)],
            'keywords': [{'word': w, 'count': c} for w, c in keywords[:20]],
            'example_tweets': [
                {
                    'username': t['username'],
                    'text': t['text'],
                    'created_at': t['created_at']
                } for t in examples
            ]
        }

    return community_topics

def main():
    # Load data
    metadata = load_data()

    # Build network and detect communities
    user_to_community = build_network_and_communities(metadata, min_interactions=2)

    # Get tweets for each community
    community_tweets = get_community_tweets(metadata, user_to_community)

    # Analyze topics
    community_topics = analyze_community_topics(community_tweets, top_communities=20)

    # Save results
    output = {
        'num_communities': len(community_tweets),
        'total_tweets_analyzed': sum(len(tweets) for tweets in community_tweets.values()),
        'communities': {
            str(comm_id): data
            for comm_id, data in community_topics.items()
        }
    }

    with open('community_topics.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total communities analyzed: {len(community_topics)}")
    print(f"Total tweets analyzed: {output['total_tweets_analyzed']:,}")
    print(f"\nResults saved to community_topics.json")

    # Print quick summary of top 5 communities
    print("\nTop 5 Communities by Size:")
    sorted_comms = sorted(
        community_topics.items(),
        key=lambda x: x[1]['num_tweets'],
        reverse=True
    )

    for i, (comm_id, data) in enumerate(sorted_comms[:5], 1):
        keywords_str = ', '.join([kw['word'] for kw in data['keywords'][:5]])
        print(f"\n{i}. Community {comm_id}")
        print(f"   {data['num_tweets']:,} tweets from {data['num_users']} users")
        print(f"   Top topics: {keywords_str}")

if __name__ == "__main__":
    main()
