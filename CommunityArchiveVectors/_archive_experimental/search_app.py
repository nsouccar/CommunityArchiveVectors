"""
Tweet Semantic Search App - Using 6.4M Vector CoreNN Database

This app provides a web UI for searching your 6.4M tweet database.
Uses OpenAI embeddings and the CoreNN database built with incremental_builder.py
"""

import modal
import os
from typing import List, Dict, Any

# Create Modal app
app = modal.App("tweet-search")

# Use the volume with your 6.4M vector database
vector_volume = modal.Volume.from_name("tweet-vectors-volume", create_if_missing=False)

# Python image with dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "corenn-py",
    "numpy",
    "fastapi",
    "pydantic",
    "openai"
)

# OpenAI secret for embeddings
secrets = modal.Secret.from_name("openai-secret")


# ============================================================================
# SEARCH FUNCTION
# ============================================================================

@app.function(
    image=image,
    volumes={"/data": vector_volume},
    secrets=[secrets],
    cpu=4.0,
    memory=16384,
    timeout=300,
)
def search(query: str, limit: int = 10) -> Dict[str, Any]:
    """
    Search the 6.4M tweet database using semantic search.

    Args:
        query: Search query string
        limit: Maximum number of results to return

    Returns:
        Dict with results and metadata
    """
    from corenn_py import CoreNN
    import numpy as np
    import pickle
    from openai import OpenAI

    vector_volume.reload()

    print(f"🔍 Searching for: '{query}' (limit: {limit})")

    # Step 1: Load database
    print("📂 Opening CoreNN database...")
    db = CoreNN.open("/data/corenn_db")
    print("✅ Database opened")

    # Step 2: Load metadata
    print("📝 Loading metadata...")
    with open("/data/metadata.pkl", "rb") as f:
        metadata_pkg = pickle.load(f)
        metadata_dict = metadata_pkg["metadata"]
        total_vectors = metadata_pkg["count"]
    print(f"✅ Loaded metadata for {total_vectors:,} tweets")

    # Step 3: Generate query embedding
    print("🔄 Generating query embedding...")
    client = OpenAI()
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=query,
        dimensions=1024
    )
    query_embedding = response.data[0].embedding
    # Convert to 2D array (shape: [1, 1024]) for query_f32
    query_vector = np.array([query_embedding], dtype=np.float32)

    # Normalize for cosine similarity
    norm = np.linalg.norm(query_vector)
    if norm > 0:
        query_vector = query_vector / norm
    print("✅ Query embedded")

    # Step 4: Search database
    print(f"🔎 Searching database...")
    results_list = db.query_f32(query_vector, limit)
    # query_f32 returns list of lists - get first (and only) result list
    results = results_list[0] if results_list else []
    print(f"✅ Found {len(results)} results")

    # Step 5: Format results
    formatted_results = []
    for i, (tweet_id, distance) in enumerate(results, 1):
        meta = metadata_dict.get(tweet_id, {})
        # Convert distance to cosine similarity (range: -1 to 1)
        cosine_similarity = 1 - distance
        # Normalize to 0-1 range for display (0% = opposite, 100% = identical)
        normalized_similarity = (cosine_similarity + 1) / 2

        result = {
            "rank": i,
            "tweet_id": tweet_id,
            "similarity": round(normalized_similarity, 4),
            "text": meta.get("full_text", "N/A"),
            "username": meta.get("username", "N/A"),
            "created_at": meta.get("created_at", "N/A"),
        }
        formatted_results.append(result)

    return {
        "query": query,
        "total_results": len(formatted_results),
        "database_size": total_vectors,
        "results": formatted_results
    }


# ============================================================================
# WEB UI
# ============================================================================

@app.function(image=image)
@modal.asgi_app()
def web():
    """Serve the search UI"""
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    # Define SearchRequest inside the web function where FastAPI uses it
    class SearchRequest(BaseModel):
        query: str
        limit: int = 10

    web_app = FastAPI()

    # Enable CORS
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @web_app.post("/api/search")
    async def api_search(request: SearchRequest):
        """API endpoint for search"""
        try:
            # Call the search function with separate parameters
            result = search.remote(request.query, request.limit)
            return JSONResponse(content=result)
        except Exception as e:
            return JSONResponse(
                content={"error": str(e)},
                status_code=500
            )

    @web_app.get("/", response_class=HTMLResponse)
    async def home():
        """Serve the search UI"""
        html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Tweet Semantic Search - 6.4M Vectors</title>
            <style>
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }

                body {
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    padding: 20px;
                }

                .container {
                    max-width: 900px;
                    margin: 0 auto;
                }

                .header {
                    text-align: center;
                    color: white;
                    margin-bottom: 40px;
                }

                .header h1 {
                    font-size: 2.5rem;
                    margin-bottom: 10px;
                }

                .header p {
                    font-size: 1.1rem;
                    opacity: 0.9;
                }

                .search-box {
                    background: white;
                    border-radius: 12px;
                    padding: 30px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                    margin-bottom: 30px;
                }

                .search-input-container {
                    display: flex;
                    gap: 10px;
                    margin-bottom: 15px;
                }

                input[type="text"] {
                    flex: 1;
                    padding: 15px 20px;
                    font-size: 16px;
                    border: 2px solid #e0e0e0;
                    border-radius: 8px;
                    outline: none;
                    transition: border-color 0.3s;
                }

                input[type="text"]:focus {
                    border-color: #667eea;
                }

                button {
                    padding: 15px 30px;
                    font-size: 16px;
                    font-weight: 600;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border: none;
                    border-radius: 8px;
                    cursor: pointer;
                    transition: transform 0.2s, box-shadow 0.2s;
                }

                button:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
                }

                button:active {
                    transform: translateY(0);
                }

                button:disabled {
                    opacity: 0.6;
                    cursor: not-allowed;
                }

                .controls {
                    display: flex;
                    gap: 15px;
                    align-items: center;
                }

                .controls label {
                    font-size: 14px;
                    color: #666;
                }

                .controls select {
                    padding: 8px 12px;
                    border: 2px solid #e0e0e0;
                    border-radius: 6px;
                    font-size: 14px;
                    outline: none;
                }

                .results {
                    background: white;
                    border-radius: 12px;
                    padding: 30px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                }

                .result-card {
                    padding: 20px;
                    border-bottom: 1px solid #e0e0e0;
                    transition: background 0.2s;
                }

                .result-card:last-child {
                    border-bottom: none;
                }

                .result-card:hover {
                    background: #f9f9f9;
                }

                .result-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 10px;
                }

                .result-rank {
                    font-size: 14px;
                    color: #999;
                    font-weight: 600;
                }

                .similarity-badge {
                    padding: 4px 12px;
                    border-radius: 12px;
                    font-size: 12px;
                    font-weight: 600;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                }

                .tweet-text {
                    font-size: 16px;
                    line-height: 1.6;
                    margin-bottom: 12px;
                    color: #333;
                }

                .tweet-meta {
                    font-size: 14px;
                    color: #666;
                }

                .tweet-meta strong {
                    color: #667eea;
                }

                .loading {
                    text-align: center;
                    padding: 40px;
                    color: #667eea;
                    font-size: 18px;
                }

                .error {
                    background: #fee;
                    border: 2px solid #fcc;
                    color: #c33;
                    padding: 15px;
                    border-radius: 8px;
                    margin-top: 20px;
                }

                .stats {
                    text-align: center;
                    color: #666;
                    margin-bottom: 20px;
                    font-size: 14px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔍 Tweet Semantic Search</h1>
                    <p>Search across 6.4 million tweets using AI</p>
                </div>

                <div class="search-box">
                    <div class="search-input-container">
                        <input
                            type="text"
                            id="searchInput"
                            placeholder="Enter your search query... (e.g., 'artificial intelligence')"
                            onkeypress="if(event.key === 'Enter') performSearch()"
                        >
                        <button onclick="performSearch()" id="searchBtn">Search</button>
                    </div>
                    <div class="controls">
                        <label for="limitSelect">Results per page:</label>
                        <select id="limitSelect">
                            <option value="5">5</option>
                            <option value="10" selected>10</option>
                            <option value="20">20</option>
                            <option value="50">50</option>
                        </select>
                    </div>
                </div>

                <div id="results"></div>
            </div>

            <script>
                async function performSearch() {
                    const query = document.getElementById('searchInput').value.trim();
                    const limit = parseInt(document.getElementById('limitSelect').value);
                    const resultsDiv = document.getElementById('results');
                    const searchBtn = document.getElementById('searchBtn');

                    if (!query) {
                        alert('Please enter a search query');
                        return;
                    }

                    // Show loading state
                    searchBtn.disabled = true;
                    searchBtn.textContent = 'Searching...';
                    resultsDiv.innerHTML = '<div class="results"><div class="loading">🔍 Searching 6.4M tweets...</div></div>';

                    try {
                        const response = await fetch('/api/search', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({ query, limit })
                        });

                        const data = await response.json();

                        if (!response.ok) {
                            throw new Error(data.error || 'Search failed');
                        }

                        displayResults(data);
                    } catch (error) {
                        resultsDiv.innerHTML = `
                            <div class="results">
                                <div class="error">
                                    <strong>Error:</strong> ${error.message}
                                </div>
                            </div>
                        `;
                    } finally {
                        searchBtn.disabled = false;
                        searchBtn.textContent = 'Search';
                    }
                }

                function displayResults(data) {
                    const resultsDiv = document.getElementById('results');

                    if (data.results.length === 0) {
                        resultsDiv.innerHTML = `
                            <div class="results">
                                <p style="text-align: center; color: #666;">No results found for "${data.query}"</p>
                            </div>
                        `;
                        return;
                    }

                    const resultsHTML = `
                        <div class="results">
                            <div class="stats">
                                Found ${data.total_results} results from ${data.database_size.toLocaleString()} tweets
                            </div>
                            ${data.results.map(result => `
                                <div class="result-card">
                                    <div class="result-header">
                                        <span class="result-rank">#${result.rank}</span>
                                        <span class="similarity-badge">${(result.similarity * 100).toFixed(1)}% match</span>
                                    </div>
                                    <div class="tweet-text">${escapeHtml(result.text)}</div>
                                    <div class="tweet-meta">
                                        <strong>@${escapeHtml(result.username)}</strong> • ${result.created_at}
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    `;

                    resultsDiv.innerHTML = resultsHTML;
                }

                function escapeHtml(text) {
                    const div = document.createElement('div');
                    div.textContent = text;
                    return div.innerHTML;
                }
            </script>
        </body>
        </html>
        """
        return html

    return web_app
