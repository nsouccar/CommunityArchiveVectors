#!/usr/bin/env python3
"""
Hetzner Backend - FastAPI server for tweet semantic search
Simple, clean, always-running server that loads database once on startup
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import voyageai
import os
from corenn_py import CoreNN
import pickle
import time
from typing import List, Optional
import logging
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Tweet Search API",
    description="Semantic search over 6.4M tweets using CoreNN",
    version="1.0.0"
)

# CORS - allow all origins (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (loaded on startup)
db = None
metadata = None
vo_client = None
database_loaded = False


class SearchResult(BaseModel):
    tweet_id: int
    full_text: str
    username: str
    created_at: Optional[str]
    similarity: float
    retweet_count: Optional[int] = 0
    favorite_count: Optional[int] = 0
    reply_to_tweet_id: Optional[int] = None
    parent_tweet_text: Optional[str] = None
    parent_tweet_username: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    search_time_ms: float
    database_size: int


@app.on_event("startup")
async def load_database():
    """Load CoreNN database and metadata on server startup"""
    global db, metadata, vo_client, database_loaded

    logger.info("=" * 80)
    logger.info("🚀 STARTING TWEET SEARCH SERVER")
    logger.info("=" * 80)

    try:
        # Initialize Voyage AI
        logger.info("🤖 Initializing Voyage AI client...")
        voyage_key = os.getenv("VOYAGE_API_KEY")
        if not voyage_key:
            raise RuntimeError("VOYAGE_API_KEY environment variable not set!")
        vo_client = voyageai.Client(api_key=voyage_key)
        logger.info("✅ Voyage AI client initialized")

        # Load CoreNN database
        logger.info("📥 Loading CoreNN database from disk...")
        db_path = "/home/deploy/tweet-search/corenn_db"

        if not os.path.exists(db_path):
            # Try current directory (for local testing)
            db_path = "./corenn_db"

        if not os.path.exists(db_path):
            raise RuntimeError(f"Database not found at {db_path}")

        start_time = time.time()
        db = CoreNN.open(db_path)
        load_time = time.time() - start_time
        logger.info(f"✅ Database loaded in {load_time:.2f}s")

        # Load metadata
        logger.info("📥 Loading metadata...")
        metadata_path = "/home/deploy/tweet-search/metadata.pkl"
        if not os.path.exists(metadata_path):
            metadata_path = "./metadata.pkl"

        with open(metadata_path, "rb") as f:
            metadata_obj = pickle.load(f)

        metadata = metadata_obj["metadata"]
        count = metadata_obj["count"]
        logger.info(f"✅ Metadata loaded: {count:,} tweets")

        database_loaded = True

        logger.info("=" * 80)
        logger.info("🎉 SERVER READY!")
        logger.info(f"   Database: {count:,} tweets")
        logger.info(f"   Load time: {load_time:.2f}s")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ Failed to load database: {e}")
        raise


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "database_loaded": database_loaded,
        "database_size": len(metadata) if metadata else 0,
        "message": "Tweet Search API is running"
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    if not database_loaded:
        raise HTTPException(status_code=503, detail="Database not loaded yet")

    return {
        "status": "healthy",
        "database_loaded": True,
        "database_size": len(metadata),
        "voyage_ai_ready": vo_client is not None
    }


@app.get("/search", response_model=SearchResponse)
async def search(
    query: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(10, ge=1, le=100, description="Number of results")
):
    """
    Semantic search over tweets

    Args:
        query: Search query text
        limit: Number of results to return (1-100)

    Returns:
        SearchResponse with matching tweets
    """
    if not database_loaded:
        raise HTTPException(
            status_code=503,
            detail="Database still loading, please wait..."
        )

    try:
        start_time = time.time()

        # Generate embedding for query
        logger.info(f"🔍 Search query: '{query}'")
        result = vo_client.embed([query], model="voyage-3", input_type="query")
        query_embedding = np.array([result.embeddings[0]], dtype=np.float32)  # Add batch dimension

        # Search database
        results = db.query_f32(query_embedding, k=limit)[0]  # Get first (and only) result
        results = list(reversed(results))  # Reverse to show highest similarity first

        # Build response
        search_results = []
        for tweet_id, similarity in results:
            tweet_meta = metadata.get(tweet_id, {})

            # Get parent tweet info if this is a reply
            reply_to_id = tweet_meta.get("reply_to_tweet_id")
            parent_text = None
            parent_username = None

            if reply_to_id:
                parent_meta = metadata.get(reply_to_id, {})
                parent_text = parent_meta.get("full_text")
                parent_username = parent_meta.get("username")

            search_results.append(SearchResult(
                tweet_id=tweet_id,
                full_text=tweet_meta.get("full_text", ""),
                username=tweet_meta.get("username", "unknown"),
                created_at=tweet_meta.get("created_at"),
                similarity=float(similarity),
                retweet_count=tweet_meta.get("retweet_count", 0),
                favorite_count=tweet_meta.get("favorite_count", 0),
                reply_to_tweet_id=reply_to_id,
                parent_tweet_text=parent_text,
                parent_tweet_username=parent_username
            ))

        search_time = (time.time() - start_time) * 1000
        logger.info(f"✅ Search completed in {search_time:.2f}ms, found {len(search_results)} results")

        return SearchResponse(
            query=query,
            results=search_results,
            search_time_ms=search_time,
            database_size=len(metadata)
        )

    except Exception as e:
        logger.error(f"❌ Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
