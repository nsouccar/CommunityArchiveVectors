#!/usr/bin/env python3
"""
Check how well-connected users are within their own communities
"""

import json
import networkx as nx

# Load network data
with open('network_animation_data.json', 'r') as f:
    data = json.load(f)

print("Analyzing community connectivity for each year...\n")

for year_data in data['years']:
    year = year_data['year']
    print(f"\n{'='*80}")
    print(f"YEAR: {year}")
    print(f"{'='*80}")

    # Build graph
    G = nx.Graph()

    # Add edges
    for edge in year_data['edges']:
        G.add_edge(edge['source'], edge['target'], weight=edge.get('weight', 1))

    # Group nodes by community
    communities = {}
    for node in year_data['nodes']:
        comm = node['community']
        if comm not in communities:
            communities[comm] = []
        communities[comm].append(node['id'])

    print(f"\nTotal users: {len(year_data['nodes'])}")
    print(f"Total edges: {len(year_data['edges'])}")
    print(f"Total communities: {len(communities)}\n")

    # Analyze each community
    disconnected_users_total = 0

    for comm_id in sorted(communities.keys()):
        members = communities[comm_id]

        # Count connections within community
        internal_edges = 0
        users_with_no_internal_connections = []

        for user in members:
            # Check if this user has any connections to other members of their community
            has_internal_connection = False

            if user in G:
                for neighbor in G.neighbors(user):
                    if neighbor in members:
                        internal_edges += 1
                        has_internal_connection = True

            if not has_internal_connection:
                users_with_no_internal_connections.append(user)
                disconnected_users_total += 1

        # Divide by 2 because we count each edge twice
        internal_edges = internal_edges // 2

        # Calculate connectivity
        max_possible_edges = len(members) * (len(members) - 1) // 2
        connectivity = (internal_edges / max_possible_edges * 100) if max_possible_edges > 0 else 0

        print(f"Community {comm_id}:")
        print(f"  Size: {len(members)}")
        print(f"  Internal edges: {internal_edges} / {max_possible_edges} possible ({connectivity:.1f}% connected)")
        print(f"  Users with NO connections to their own community: {len(users_with_no_internal_connections)}")

        if len(users_with_no_internal_connections) > 0 and len(users_with_no_internal_connections) <= 5:
            print(f"    Disconnected users: {users_with_no_internal_connections}")
        elif len(users_with_no_internal_connections) > 5:
            print(f"    Example disconnected users: {users_with_no_internal_connections[:5]}")

    print(f"\n📊 SUMMARY for {year}:")
    print(f"  Total users with NO connections to their own community: {disconnected_users_total}/{len(year_data['nodes'])} ({disconnected_users_total/len(year_data['nodes'])*100:.1f}%)")
