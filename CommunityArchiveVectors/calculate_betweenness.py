#!/usr/bin/env python3
"""
Calculate betweenness centrality for all nodes in each year's network.
This identifies true "super connectors" who bridge different communities.
"""

import json
import networkx as nx
from pathlib import Path

def calculate_betweenness_centrality():
    """Calculate betweenness centrality for each year and add to network data."""

    # Load network data
    network_file = Path("network_animation_data.json")
    with open(network_file, 'r') as f:
        network_data = json.load(f)

    print(f"Calculating betweenness centrality for {len(network_data['years'])} years...")

    for year_data in network_data['years']:
        year = year_data['year']
        print(f"\n{year}:")

        # Build NetworkX graph
        G = nx.Graph()

        # Add nodes
        for node in year_data['nodes']:
            G.add_node(node['id'], community=node['community'])

        # Add edges
        for edge in year_data['edges']:
            G.add_edge(edge['source'], edge['target'])

        print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

        # Calculate betweenness centrality
        # This measures how often each node appears on shortest paths between other nodes
        print(f"  Calculating betweenness centrality...")
        betweenness = nx.betweenness_centrality(G, normalized=True)

        # Add betweenness to node data
        for node in year_data['nodes']:
            node['betweenness'] = betweenness.get(node['id'], 0.0)

        # Find super connectors (top 10% by betweenness)
        betweenness_values = [b for b in betweenness.values() if b > 0]
        if betweenness_values:
            threshold = sorted(betweenness_values, reverse=True)[min(len(betweenness_values)-1, len(betweenness_values)//10)]
            super_connectors = [node['id'] for node in year_data['nodes'] if node.get('betweenness', 0) >= threshold]
            print(f"  Super connectors (top 10%): {len(super_connectors)}")
            print(f"  Betweenness threshold: {threshold:.4f}")

            # Show top 5
            top_nodes = sorted(year_data['nodes'], key=lambda n: n.get('betweenness', 0), reverse=True)[:5]
            print(f"  Top 5 by betweenness:")
            for node in top_nodes:
                print(f"    {node['id']}: {node['betweenness']:.4f} (degree: {node.get('degree', 0)})")

    # Save updated network data
    output_file = Path("network_animation_data_with_betweenness.json")
    with open(output_file, 'w') as f:
        json.dump(network_data, f, indent=2)

    print(f"\n✅ Saved to {output_file}")
    print("\nNext steps:")
    print("1. Replace network_animation_data.json with this new file")
    print("2. Update frontend to use betweenness centrality for super connector detection")

if __name__ == "__main__":
    calculate_betweenness_centrality()
