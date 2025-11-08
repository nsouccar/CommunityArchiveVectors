#!/usr/bin/env python3
"""
Network Analysis - Build and analyze user interaction networks from tweet data
Shows how communities evolve over time and how they interact with each other
"""

import pickle
from collections import defaultdict, Counter
from datetime import datetime
import json
from typing import Dict, List, Tuple, Set
import networkx as nx
from networkx.algorithms import community as nx_community

def load_metadata(metadata_path: str) -> Dict:
    """Load tweet metadata from pickle file"""
    print(f"Loading metadata from {metadata_path}...")
    with open(metadata_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {data['count']:,} tweets")
    return data['metadata']

def build_interaction_graph(metadata: Dict, min_interactions: int = 2) -> nx.DiGraph:
    """
    Build a directed graph where:
    - Nodes are users
    - Edges represent reply interactions (A -> B means A replied to B's tweet)
    - Edge weight = number of interactions
    """
    print("\nBuilding interaction graph...")

    # Build mapping: tweet_id -> username
    tweet_to_user = {}
    for tweet_id, tweet_data in metadata.items():
        tweet_to_user[tweet_id] = tweet_data.get('username', 'unknown')

    # Count interactions between users
    interactions = defaultdict(int)  # (from_user, to_user) -> count

    for tweet_id, tweet_data in metadata.items():
        from_user = tweet_data.get('username')
        reply_to_id = tweet_data.get('reply_to_tweet_id')

        if from_user and reply_to_id and reply_to_id in tweet_to_user:
            to_user = tweet_to_user[reply_to_id]
            if from_user != to_user:  # Exclude self-replies
                interactions[(from_user, to_user)] += 1

    # Build graph with weighted edges
    G = nx.DiGraph()

    for (from_user, to_user), count in interactions.items():
        if count >= min_interactions:  # Filter out weak connections
            G.add_edge(from_user, to_user, weight=count)

    print(f"Graph has {G.number_of_nodes():,} users and {G.number_of_edges():,} interactions")
    print(f"(Filtered to interactions with >= {min_interactions} replies)")

    return G

def detect_communities(G: nx.Graph, min_community_size: int = 5) -> Dict[str, int]:
    """
    Detect communities using Louvain method
    Returns: {username: community_id}
    """
    print("\nDetecting communities...")

    # Convert to undirected for community detection
    G_undirected = G.to_undirected()

    # Use Louvain method for community detection
    communities = nx_community.louvain_communities(G_undirected, seed=42)

    # Assign community IDs to users
    user_to_community = {}
    community_sizes = []

    for comm_id, community in enumerate(communities):
        if len(community) >= min_community_size:
            for user in community:
                user_to_community[user] = comm_id
            community_sizes.append(len(community))

    print(f"Found {len(communities)} communities")
    print(f"Top 10 community sizes: {sorted(community_sizes, reverse=True)[:10]}")

    return user_to_community

def analyze_cross_community_interactions(G: nx.DiGraph, user_to_community: Dict[str, int]) -> List[Dict]:
    """
    Find users who bridge between communities
    Returns list of bridge users with their cross-community interaction counts
    """
    print("\nAnalyzing cross-community connections...")

    bridge_users = []

    for user in G.nodes():
        if user not in user_to_community:
            continue

        user_comm = user_to_community[user]

        # Count interactions with other communities
        other_comm_interactions = defaultdict(int)

        for neighbor in G.neighbors(user):
            if neighbor in user_to_community:
                neighbor_comm = user_to_community[neighbor]
                if neighbor_comm != user_comm:
                    weight = G[user][neighbor]['weight']
                    other_comm_interactions[neighbor_comm] += weight

        if other_comm_interactions:
            bridge_users.append({
                'username': user,
                'community': user_comm,
                'cross_community_interactions': dict(other_comm_interactions),
                'total_cross_interactions': sum(other_comm_interactions.values())
            })

    bridge_users.sort(key=lambda x: x['total_cross_interactions'], reverse=True)

    print(f"Found {len(bridge_users)} users who bridge communities")
    print(f"Top bridge user: @{bridge_users[0]['username']} with {bridge_users[0]['total_cross_interactions']} cross-community interactions")

    return bridge_users

def temporal_analysis(metadata: Dict, time_windows: List[str]) -> Dict:
    """
    Analyze how networks change over time
    time_windows: list of time periods like ['2018', '2019', '2020', etc.]
    """
    print("\nPerforming temporal analysis...")

    # Group tweets by time period
    tweets_by_period = defaultdict(dict)

    for tweet_id, tweet_data in metadata.items():
        created_at = tweet_data.get('created_at')
        if created_at:
            # Extract year from timestamp
            year = created_at.split('-')[0]
            if year in time_windows:
                tweets_by_period[year][tweet_id] = tweet_data

    # Build network for each time period
    period_networks = {}

    for period in sorted(tweets_by_period.keys()):
        print(f"\nAnalyzing {period}...")
        period_data = tweets_by_period[period]
        print(f"  Tweets in period: {len(period_data):,}")

        # Build graph for this period
        G = build_interaction_graph(period_data, min_interactions=2)

        if G.number_of_nodes() > 0:
            # Detect communities
            user_to_comm = detect_communities(G, min_community_size=3)

            # Find bridges
            bridges = analyze_cross_community_interactions(G, user_to_comm)

            period_networks[period] = {
                'num_users': G.number_of_nodes(),
                'num_interactions': G.number_of_edges(),
                'num_communities': len(set(user_to_comm.values())),
                'num_bridges': len(bridges),
                'top_bridges': bridges[:10]  # Top 10 bridge users
            }

    return period_networks

def export_for_visualization(G: nx.DiGraph, user_to_community: Dict[str, int], output_path: str):
    """
    Export network data in format suitable for D3.js visualization
    """
    print(f"\nExporting visualization data to {output_path}...")

    # Prepare nodes
    nodes = []
    for user in G.nodes():
        community = user_to_community.get(user, -1)
        degree = G.degree(user)
        nodes.append({
            'id': user,
            'community': community,
            'degree': degree
        })

    # Prepare edges
    edges = []
    for from_user, to_user, data in G.edges(data=True):
        edges.append({
            'source': from_user,
            'target': to_user,
            'weight': data['weight']
        })

    # Save as JSON
    viz_data = {
        'nodes': nodes,
        'edges': edges,
        'num_communities': len(set(user_to_community.values()))
    }

    with open(output_path, 'w') as f:
        json.dump(viz_data, f, indent=2)

    print(f"Exported {len(nodes)} nodes and {len(edges)} edges")

if __name__ == "__main__":
    # Load metadata
    metadata = load_metadata("metadata.pkl")

    # Build full network
    print("\n" + "="*80)
    print("FULL NETWORK ANALYSIS")
    print("="*80)
    G = build_interaction_graph(metadata, min_interactions=2)

    # Detect communities
    user_to_community = detect_communities(G, min_community_size=5)

    # Find bridge users
    bridges = analyze_cross_community_interactions(G, user_to_community)

    # Print top bridge users
    print("\nTop 20 bridge users (connecting different communities):")
    for i, bridge in enumerate(bridges[:20], 1):
        print(f"{i}. @{bridge['username']}")
        print(f"   Community: {bridge['community']}")
        print(f"   Cross-community interactions: {bridge['total_cross_interactions']}")
        print(f"   Interacts with communities: {list(bridge['cross_community_interactions'].keys())}")
        print()

    # Export for visualization
    export_for_visualization(G, user_to_community, "network_viz.json")

    # Temporal analysis
    print("\n" + "="*80)
    print("TEMPORAL NETWORK EVOLUTION")
    print("="*80)
    years = ['2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025']
    period_networks = temporal_analysis(metadata, years)

    # Print temporal summary
    print("\nNetwork evolution over time:")
    print("-" * 80)
    print(f"{'Year':<8} {'Users':<10} {'Interactions':<15} {'Communities':<15} {'Bridges':<10}")
    print("-" * 80)
    for year in sorted(period_networks.keys()):
        data = period_networks[year]
        print(f"{year:<8} {data['num_users']:<10,} {data['num_interactions']:<15,} {data['num_communities']:<15} {data['num_bridges']:<10}")

    # Save temporal data
    with open('temporal_networks.json', 'w') as f:
        json.dump(period_networks, f, indent=2)

    print("\nAnalysis complete!")
    print("Generated files:")
    print("  - network_viz.json: Full network data for visualization")
    print("  - temporal_networks.json: Network evolution over time")
