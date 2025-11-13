#!/usr/bin/env python3
"""
Public API for accessing community archive data
Anyone can access this via HTTP without needing Modal credentials
"""

import modal
import pickle
from pathlib import Path
import json

app = modal.App("community-archive-public-api")
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy", "fastapi")
volume = modal.Volume.from_name("tweet-vectors-large", create_if_missing=False)

# Cache the organized data in memory for faster access
@app.function(
    volumes={"/data": volume},
    image=image,
    keep_warm=1,  # Keep one instance warm for faster response
    memory=8192,  # 8GB RAM to hold the data
)
@modal.web_endpoint(method="GET")
def list_structure():
    """
    GET endpoint to list all available years, months, and communities

    URL: https://[your-workspace]--community-archive-public-api-list-structure.modal.run

    Returns JSON with structure of available data
    """
    with open("/data/organized_by_community.pkl", 'rb') as f:
        organized = pickle.load(f)

    structure = {}
    total_tweets = 0

    for year in sorted(organized.keys(), key=lambda x: int(x) if x.isdigit() else 0):
        structure[year] = {}
        year_tweets = 0

        for month in sorted(organized[year].keys()):
            communities = list(organized[year][month].keys())
            month_tweets = sum(
                len(organized[year][month][cid]['tweets'])
                for cid in communities
            )

            structure[year][str(month)] = {
                "num_communities": len(communities),
                "community_ids": communities,
                "num_tweets": month_tweets
            }
            year_tweets += month_tweets

        structure[year]["_total_tweets"] = year_tweets
        total_tweets += year_tweets

    return {
        "structure": structure,
        "total_tweets": total_tweets,
        "instructions": {
            "get_community": "Use /get_community?year=2022&month=1&community_id=2",
            "get_tweet_ids": "Use /get_tweet_ids?year=2022&month=1&community_id=2",
            "get_year": "Use /get_year?year=2022"
        }
    }

@app.function(
    volumes={"/data": volume},
    image=image,
    keep_warm=1,
    memory=8192,
)
@modal.web_endpoint(method="GET")
def get_community(year: str, month: int, community_id: str):
    """
    GET endpoint to retrieve tweets and embeddings for a specific community

    URL: https://[workspace]--community-archive-public-api-get-community.modal.run?year=2022&month=1&community_id=2

    Parameters:
        year: Year as string (e.g., '2022')
        month: Month as int (1-12)
        community_id: Community ID as string

    Returns: JSON with tweets and embeddings (embeddings as list of lists)
    """
    with open("/data/organized_by_community.pkl", 'rb') as f:
        organized = pickle.load(f)

    if year not in organized:
        return {"error": f"Year {year} not found"}

    if month not in organized[year]:
        return {"error": f"Month {month} not found in year {year}"}

    if community_id not in organized[year][month]:
        return {"error": f"Community {community_id} not found"}

    community_data = organized[year][month][community_id]

    # Convert numpy array to list for JSON serialization
    embeddings_list = community_data['embeddings'].tolist()

    return {
        "year": year,
        "month": month,
        "community_id": community_id,
        "tweets": community_data['tweets'],
        "embeddings": embeddings_list,
        "num_tweets": len(community_data['tweets']),
        "embedding_dimension": len(embeddings_list[0]) if embeddings_list else 0
    }

@app.function(
    volumes={"/data": volume},
    image=image,
    keep_warm=1,
    memory=8192,
)
@modal.web_endpoint(method="GET")
def get_tweet_ids(year: str, month: int, community_id: str):
    """
    GET endpoint to retrieve only tweet IDs for a specific community (faster, smaller response)

    URL: https://[workspace]--community-archive-public-api-get-tweet-ids.modal.run?year=2022&month=1&community_id=2

    Parameters:
        year: Year as string (e.g., '2022')
        month: Month as int (1-12)
        community_id: Community ID as string

    Returns: JSON with just tweet IDs and basic info
    """
    with open("/data/organized_by_community.pkl", 'rb') as f:
        organized = pickle.load(f)

    if year not in organized:
        return {"error": f"Year {year} not found"}

    if month not in organized[year]:
        return {"error": f"Month {month} not found in year {year}"}

    if community_id not in organized[year][month]:
        return {"error": f"Community {community_id} not found"}

    tweets = organized[year][month][community_id]['tweets']

    return {
        "year": year,
        "month": month,
        "community_id": community_id,
        "tweet_ids": [tweet['tweet_id'] for tweet in tweets],
        "num_tweets": len(tweets)
    }

@app.function(
    volumes={"/data": volume},
    image=image,
    memory=16384,  # 16GB for full year data
    timeout=600,  # 10 minutes
)
@modal.web_endpoint(method="GET")
def get_year(year: str):
    """
    GET endpoint to download all data for a specific year
    WARNING: This returns a LOT of data and may be slow

    URL: https://[workspace]--community-archive-public-api-get-year.modal.run?year=2022

    Parameters:
        year: Year as string (e.g., '2022')

    Returns: All communities for that year
    """
    with open("/data/organized_by_community.pkl", 'rb') as f:
        organized = pickle.load(f)

    if year not in organized:
        return {"error": f"Year {year} not found"}

    year_data = organized[year]

    # Convert all embeddings to lists
    result = {}
    for month, communities in year_data.items():
        result[str(month)] = {}
        for comm_id, comm_data in communities.items():
            result[str(month)][comm_id] = {
                'tweets': comm_data['tweets'],
                'embeddings': comm_data['embeddings'].tolist()
            }

    return {
        "year": year,
        "data": result
    }
