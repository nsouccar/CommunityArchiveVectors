"""
Temporal Community Alignment - Track how communities evolve across years.

This script computes community centroids for each year and finds alignments between
communities across adjacent years using cosine similarity.
"""

import modal
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

app = modal.App("temporal-community-alignment")
volume = modal.Volume.from_name("tweet-vectors-large", create_if_missing=False)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy",
    "scikit-learn"
)

@app.function(
    image=image,
    volumes={"/data": volume},
    timeout=3600,
    memory=16384,  # 16GB to load pickle file
)
def compute_all_centroids():
    """
    Compute centroid embeddings for all communities across all years.

    Loads organized_by_community.pkl and computes centroids for each year/community.

    Returns:
        Dict: {year: {community_id: centroid_vector}}
    """
    import pickle

    print("Loading organized embeddings from Modal volume...")

    organized_path = Path("/data/organized_by_community.pkl")
    if not organized_path.exists():
        print(f"ERROR: {organized_path} does not exist!")
        return {}

    with open(organized_path, 'rb') as f:
        organized = pickle.load(f)

    print(f"✓ Loaded organized data for {len(organized)} years")

    all_centroids = {}

    for year_str in organized.keys():
        year = int(year_str)
        print(f"\nProcessing year {year}...")

        # Aggregate embeddings by community across all months
        community_embeddings = {}

        for month in organized[year_str]:
            for community_id in organized[year_str][month]:
                if community_id not in community_embeddings:
                    community_embeddings[community_id] = []

                emb_array = organized[year_str][month][community_id]['embeddings']
                community_embeddings[community_id].append(emb_array)

        # Compute centroids for each community
        community_centroids = {}

        for community_id, emb_list in community_embeddings.items():
            # Concatenate all embeddings for this community
            all_embeddings = np.vstack(emb_list)

            # Compute centroid (mean)
            centroid = np.mean(all_embeddings, axis=0)

            # Normalize for cosine similarity
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

            community_centroids[community_id] = centroid

            print(f"  Community {community_id}: {len(all_embeddings)} tweets")

        all_centroids[year] = community_centroids
        print(f"✓ Year {year}: {len(community_centroids)} communities")

    return all_centroids


@app.function(
    image=image,
    timeout=3600,
)
def align_years(year1_centroids: Dict, year2_centroids: Dict,
                year1: int, year2: int, threshold: float = 0.5):
    """
    Find alignments between communities from two adjacent years.

    Args:
        year1_centroids: Dict of community_id -> centroid for year 1
        year2_centroids: Dict of community_id -> centroid for year 2
        year1: First year
        year2: Second year
        threshold: Minimum cosine similarity to consider an alignment

    Returns:
        List of alignments: [(year1_comm_id, year2_comm_id, similarity)]
    """
    print(f"\nAligning {year1} -> {year2}...")

    alignments = []

    for comm1_id, centroid1 in year1_centroids.items():
        best_match = None
        best_similarity = threshold

        for comm2_id, centroid2 in year2_centroids.items():
            # Compute cosine similarity
            similarity = np.dot(centroid1, centroid2)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = comm2_id

        if best_match:
            alignments.append({
                "year1": year1,
                "year2": year2,
                "community1_id": comm1_id,
                "community2_id": best_match,
                "similarity": float(best_similarity)
            })
            print(f"  {year1} Community {comm1_id} -> {year2} Community {best_match} (similarity: {best_similarity:.3f})")

    return alignments


@app.local_entrypoint()
def main(threshold: float = 0.5):
    """
    Compute temporal alignments across all years.

    Args:
        threshold: Minimum cosine similarity to consider communities aligned
    """
    print("=" * 80)
    print("TEMPORAL COMMUNITY ALIGNMENT")
    print("=" * 80)
    print(f"Threshold: {threshold}")
    print("=" * 80)

    # Step 1: Compute centroids for all years (single function call)
    print("\n### STEP 1: Computing Community Centroids ###\n")

    year_centroids = compute_all_centroids.remote()

    if not year_centroids:
        print("✗ No centroids computed!")
        return

    years = sorted(year_centroids.keys())
    print(f"\n✓ Computed centroids for {len(years)} years: {years}")

    # Step 2: Find alignments between adjacent years
    print("\n### STEP 2: Finding Temporal Alignments ###\n")

    all_alignments = []
    for i in range(len(years) - 1):
        year1, year2 = years[i], years[i + 1]

        if year1 not in year_centroids or year2 not in year_centroids:
            print(f"Skipping {year1} -> {year2} (missing centroids)")
            continue

        alignments = align_years.remote(
            year_centroids[year1],
            year_centroids[year2],
            year1,
            year2,
            threshold
        )
        all_alignments.extend(alignments)

    # Step 3: Build community lineages
    print("\n### STEP 3: Building Community Lineages ###\n")

    # Create a graph of community evolution
    lineages = defaultdict(list)

    for alignment in all_alignments:
        key = f"{alignment['year1']}_comm_{alignment['community1_id']}"
        lineages[key].append({
            "next_year": alignment["year2"],
            "next_community": alignment["community2_id"],
            "similarity": alignment["similarity"]
        })

    # Find continuous lineages (communities that persist across multiple years)
    print("\n🔍 Continuous Community Lineages (3+ years):\n")

    visited = set()
    continuous_lineages = []

    for start_key in lineages.keys():
        if start_key in visited:
            continue

        # Trace this lineage forward
        lineage = [start_key]
        current = start_key

        while current in lineages and lineages[current]:
            visited.add(current)

            # Get best match for next year
            next_alignments = lineages[current]
            if not next_alignments:
                break

            best = max(next_alignments, key=lambda x: x["similarity"])
            next_key = f"{best['next_year']}_comm_{best['next_community']}"

            lineage.append((next_key, best["similarity"]))
            current = next_key

        if len(lineage) >= 3:  # At least 3 years
            continuous_lineages.append(lineage)

    # Sort by length (longest lineages first)
    continuous_lineages.sort(key=len, reverse=True)

    for i, lineage in enumerate(continuous_lineages[:10], 1):  # Show top 10
        print(f"\nLineage {i} ({len(lineage)} years):")
        for j, item in enumerate(lineage):
            if j == 0:
                print(f"  {item}")
            else:
                comm, sim = item
                print(f"  -> {comm} (sim: {sim:.3f})")

    # Step 4: Save results
    output_path = Path("community_temporal_alignments.json")

    with open(output_path, "w") as f:
        json.dump({
            "threshold": threshold,
            "years": years,
            "alignments": all_alignments,
            "continuous_lineages": [
                {
                    "length": len(lineage),
                    "path": [item if isinstance(item, str) else item[0] for item in lineage]
                }
                for lineage in continuous_lineages
            ],
            "statistics": {
                "total_alignments": len(all_alignments),
                "years_processed": len(year_centroids),
                "continuous_lineages": len(continuous_lineages)
            }
        }, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total alignments found: {len(all_alignments)}")
    print(f"Continuous lineages (3+ years): {len(continuous_lineages)}")
    print(f"Longest lineage: {max(len(l) for l in continuous_lineages) if continuous_lineages else 0} years")
    print("=" * 80)
