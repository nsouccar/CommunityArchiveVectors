#!/usr/bin/env python3
"""
Simple FastAPI server for Supabase vector search
Can be deployed to Vercel, Render, or any serverless platform
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import voyageai
import os
from supabase import create_client
from typing import List, Optional
import time

app = FastAPI(
    title="Tweet Search API (Supabase)",
    description="Semantic search over 6.4M tweets using Supabase pgvector",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients (lazy loaded)
supabase_client = None
voyage_client = None


def get_supabase():
    """Lazy load Supabase client"""
    global supabase_client
    if supabase_client is None:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        if not supabase_url or not supabase_key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set")
        supabase_client = create_client(supabase_url, supabase_key)
    return supabase_client


def get_voyage():
    """Lazy load Voyage AI client"""
    global voyage_client
    if voyage_client is None:
        voyage_key = os.getenv("VOYAGE_API_KEY")
        if not voyage_key:
            raise RuntimeError("VOYAGE_API_KEY must be set")
        voyage_client = voyageai.Client(api_key=voyage_key)
    return voyage_client


class SearchResult(BaseModel):
    tweet_id: int
    full_text: str
    username: str
    created_at: Optional[str]
    similarity: float
    retweet_count: Optional[int] = 0
    favorite_count: Optional[int] = 0


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    search_time_ms: float


@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "running",
        "message": "Tweet Search API (Supabase)",
        "version": "1.0.0"
    }


@app.get("/search", response_model=SearchResponse)
async def search(
    query: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(10, ge=1, le=100, description="Number of results")
):
    """
    Semantic search over tweets using Supabase pgvector

    Args:
        query: Search query text
        limit: Number of results to return (1-100)

    Returns:
        SearchResponse with matching tweets
    """
    try:
        start_time = time.time()

        # Get clients
        supabase = get_supabase()
        voyage = get_voyage()

        # Generate embedding
        result = voyage.embed([query], model="voyage-3", input_type="query")
        query_embedding = result.embeddings[0]

        # Search using Supabase RPC function
        response = supabase.rpc(
            "search_tweets",
            {
                "query_embedding": query_embedding,
                "match_count": limit
            }
        ).execute()

        # Format results
        search_results = []
        for row in response.data:
            search_results.append(SearchResult(
                tweet_id=row["tweet_id"],
                full_text=row["full_text"] or "",
                username=row["username"] or "unknown",
                created_at=row["created_at"],
                similarity=float(row["similarity"]),
                retweet_count=row["retweet_count"] or 0,
                favorite_count=row["favorite_count"] or 0
            ))

        search_time = (time.time() - start_time) * 1000

        return SearchResponse(
            query=query,
            results=search_results,
            search_time_ms=search_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
