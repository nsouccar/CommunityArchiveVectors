#!/usr/bin/env python3
"""
Simple single-endpoint public API for accessing community archive data
"""

import modal
import pickle

app = modal.App("community-data-api")
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy", "fastapi")
volume = modal.Volume.from_name("tweet-vectors-large", create_if_missing=False)

@app.function(
    volumes={"/data": volume},
    image=image,
    min_containers=1,
    memory=16384,  # 16GB
    timeout=600,
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, Query

    web_app = FastAPI()

    # Load data once and cache in memory
    organized_data = None

    def load_data():
        nonlocal organized_data
        if organized_data is None:
            with open("/data/organized_by_community.pkl", 'rb') as f:
                organized_data = pickle.load(f)
        return organized_data

    @web_app.get("/")
    def root():
        """Root endpoint with documentation"""
        return {
            "message": "Community Archive Data API",
            "total_tweets": "5.7 million",
            "endpoints": {
                "/structure": "List all available years, months, and communities",
                "/data": "Get data with query parameters: action, year, month, community_id"
            },
            "examples": [
                "/structure",
                "/data?action=list_years",
                "/data?action=community&year=2022&month=1&community_id=2",
                "/data?action=tweet_ids&year=2022&month=1&community_id=2",
                "/data?action=year&year=2022"
            ]
        }

    @web_app.get("/structure")
    def get_structure():
        """List all available data"""
        organized = load_data()

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
                    "community_ids": communities[:20],  # First 20
                    "num_tweets": month_tweets
                }
                year_tweets += month_tweets

            structure[year]["_total_tweets"] = year_tweets
            total_tweets += year_tweets

        return {"structure": structure, "total_tweets": total_tweets}

    @web_app.get("/data")
    def get_data(
        action: str = Query(..., description="Action: list_years, community, tweet_ids, or year"),
        year: str = Query(None),
        month: int = Query(None),
        community_id: str = Query(None)
    ):
        """
        Get data based on action parameter

        Actions:
        - list_years: List all available years
        - community: Get tweets and embeddings (requires year, month, community_id)
        - tweet_ids: Get only tweet IDs (requires year, month, community_id)
        - year: Get all data for a year (requires year)
        """
        organized = load_data()

        if action == "list_years":
            years = sorted(organized.keys(), key=lambda x: int(x) if x.isdigit() else 0)
            return {"years": years}

        elif action == "community":
            if not all([year, month is not None, community_id]):
                return {"error": "year, month, and community_id required"}

            if year not in organized:
                return {"error": f"Year {year} not found"}
            if month not in organized[year]:
                return {"error": f"Month {month} not found"}
            if community_id not in organized[year][month]:
                return {"error": f"Community {community_id} not found"}

            community_data = organized[year][month][community_id]
            return {
                "year": year,
                "month": month,
                "community_id": community_id,
                "tweets": community_data['tweets'],
                "embeddings": community_data['embeddings'].tolist(),
                "num_tweets": len(community_data['tweets'])
            }

        elif action == "tweet_ids":
            if not all([year, month is not None, community_id]):
                return {"error": "year, month, and community_id required"}

            if year not in organized:
                return {"error": f"Year {year} not found"}
            if month not in organized[year]:
                return {"error": f"Month {month} not found"}
            if community_id not in organized[year][month]:
                return {"error": f"Community {community_id} not found"}

            tweets = organized[year][month][community_id]['tweets']
            return {
                "year": year,
                "month": month,
                "community_id": community_id,
                "tweet_ids": [t['tweet_id'] for t in tweets],
                "num_tweets": len(tweets)
            }

        elif action == "year":
            if not year:
                return {"error": "year parameter required"}

            if year not in organized:
                return {"error": f"Year {year} not found"}

            year_data = organized[year]
            result = {}
            for month, communities in year_data.items():
                result[str(month)] = {}
                for comm_id, comm_data in communities.items():
                    result[str(month)][comm_id] = {
                        'tweets': comm_data['tweets'],
                        'embeddings': comm_data['embeddings'].tolist()
                    }

            return {"year": year, "data": result}

        else:
            return {"error": f"Unknown action: {action}"}

    return web_app
