#!/usr/bin/env python3
"""
Generate year-by-year network visualization data for animation
"""

import pickle
import json
from collections import defaultdict
import networkx as nx
from networkx.algorithms import community as nx_community

def load_metadata(metadata_path: str):
    """Load tweet metadata"""
    with open(metadata_path, 'rb') as f:
        data = pickle.load(f)
    return data['metadata']

def build_graph_for_period(tweets_in_period, min_interactions=2):
    """Build network graph for a specific time period"""
    # Build mapping: tweet_id -> username
    tweet_to_user = {}
    for tweet_id, tweet_data in tweets_in_period.items():
        tweet_to_user[tweet_id] = tweet_data.get('username', 'unknown')

    # Count interactions
    interactions = defaultdict(int)
    for tweet_id, tweet_data in tweets_in_period.items():
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

    return G

def detect_communities(G):
    """Detect communities using Louvain method"""
    if G.number_of_nodes() == 0:
        return {}

    G_undirected = G.to_undirected()
    communities = nx_community.louvain_communities(G_undirected, seed=42)

    user_to_community = {}
    for comm_id, community in enumerate(communities):
        for user in community:
            user_to_community[user] = comm_id

    return user_to_community

def export_network_for_viz(G, user_to_community, year):
    """Export network data in format for D3.js"""
    nodes = []
    for user in G.nodes():
        community = user_to_community.get(user, 0)
        degree = G.degree(user)

        nodes.append({
            'id': user,
            'community': community,
            'degree': degree
        })

    edges = []
    for from_user, to_user, data in G.edges(data=True):
        edges.append({
            'source': from_user,
            'target': to_user,
            'weight': data['weight']
        })

    return {
        'year': year,
        'nodes': nodes,
        'edges': edges,
        'num_communities': len(set(user_to_community.values())) if user_to_community else 0,
        'num_users': G.number_of_nodes(),
        'num_interactions': G.number_of_edges()
    }

def main():
    print("Loading metadata...")
    metadata = load_metadata("metadata.pkl")

    # Group tweets by year
    tweets_by_year = defaultdict(dict)
    for tweet_id, tweet_data in metadata.items():
        created_at = tweet_data.get('created_at')
        if created_at:
            year = created_at.split('-')[0]
            tweets_by_year[year][tweet_id] = tweet_data

    # Generate network data for each year
    years_data = []
    for year in sorted(tweets_by_year.keys()):
        print(f"\nProcessing {year}...")
        tweets = tweets_by_year[year]
        print(f"  {len(tweets):,} tweets")

        # Build graph
        G = build_graph_for_period(tweets, min_interactions=2)
        print(f"  {G.number_of_nodes()} users, {G.number_of_edges()} interactions")

        if G.number_of_nodes() > 0:
            # Detect communities
            user_to_comm = detect_communities(G)

            # Export
            year_data = export_network_for_viz(G, user_to_comm, year)
            years_data.append(year_data)
            print(f"  {year_data['num_communities']} communities")

    # Save all years data
    output = {
        'years': years_data,
        'metadata': {
            'total_tweets': len(metadata),
            'years_covered': [y['year'] for y in years_data]
        }
    }

    with open('network_animation_data.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\n✅ Generated network_animation_data.json")
    print(f"   Years covered: {', '.join([y['year'] for y in years_data])}")

if __name__ == "__main__":
    main()
