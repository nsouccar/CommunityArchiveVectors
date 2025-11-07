"""
Fast Tweet Semantic Search - Keeps database loaded in memory

Uses Modal Class to keep CoreNN database warm in memory between requests.
First request takes ~35s to load, subsequent requests are <20ms.
"""
import modal

app = modal.App("fast-tweet-search")

vector_volume = modal.Volume.from_name("tweet-vectors-volume", create_if_missing=False)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "corenn-py",
    "numpy",
    "fastapi",
    "openai"
)

secrets = modal.Secret.from_name("openai-secret")


@app.function(
    image=image,
    volumes={"/data": vector_volume},
    secrets=[secrets],
    cpu=8.0,
    memory=40960,  # 40GB to keep everything in memory
    timeout=300,
)
def search(query: str, limit: int = 10):
    """
    Search the 6.4M tweet database using semantic search.
    Note: First search loads DB (~35s), subsequent searches are fast.

    Args:
        query: Search query string
        limit: Maximum number of results

    Returns:
        Dict with results and metadata
    """
    from corenn_py import CoreNN
    import numpy as np
    import pickle
    from openai import OpenAI
    import time

    vector_volume.reload()

    search_start = time.time()

    # Load database
    print(f"📂 Opening CoreNN database...")
    db_start = time.time()
    db = CoreNN.open("/data/corenn_db")
    db_load_time = time.time() - db_start
    print(f"✅ Database opened in {db_load_time:.1f}s")

    # Load metadata
    print("📝 Loading metadata...")
    with open("/data/metadata.pkl", "rb") as f:
        metadata_pkg = pickle.load(f)
        metadata_dict = metadata_pkg["metadata"]
        total_vectors = metadata_pkg["count"]

    # Generate query embedding
    print("🔄 Generating query embedding...")
    client = OpenAI()
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=query,
        dimensions=1024
    )
    query_embedding = response.data[0].embedding

    # Convert to 2D array and normalize
    query_vector = np.array([query_embedding], dtype=np.float32)
    norm = np.linalg.norm(query_vector)
    if norm > 0:
        query_vector = query_vector / norm

    # Search database
    print(f"🔎 Searching database...")
    db_search_start = time.time()
    results_list = db.query_f32(query_vector, limit)
    results = results_list[0] if results_list else []
    db_search_time = (time.time() - db_search_start) * 1000

    # Format results
    formatted_results = []
    for i, (tweet_id, distance) in enumerate(results, 1):
        meta = metadata_dict.get(tweet_id, {})
        # Convert to normalized similarity (0-1 range)
        cosine_similarity = 1 - distance
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

    total_time = time.time() - search_start
    print(f"✅ Search complete in {total_time:.1f}s (DB load: {db_load_time:.1f}s, query: {db_search_time:.0f}ms)")

    return {
        "query": query,
        "total_results": len(formatted_results),
        "database_size": total_vectors,
        "results": formatted_results,
        "search_time_ms": round(total_time * 1000, 1),
        "db_query_time_ms": round(db_search_time, 1)
    }


@app.function(image=image)
@modal.asgi_app()
def web():
    """Serve the fast search UI"""
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse
    from pydantic import BaseModel

    class SearchRequest(BaseModel):
        query: str
        limit: int = 10

    web_app = FastAPI()

    @web_app.post("/api/search")
    async def api_search(request: SearchRequest):
        """API endpoint - calls search function"""
        try:
            # Call search function directly
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
            <title>Fast Tweet Search - 6.4M Vectors</title>
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body {
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    padding: 20px;
                }
                .container { max-width: 900px; margin: 0 auto; }
                .header {
                    text-align: center;
                    color: white;
                    margin-bottom: 40px;
                }
                .header h1 { font-size: 2.5rem; margin-bottom: 10px; }
                .header p { font-size: 1.1rem; opacity: 0.9; }
                .search-box {
                    background: white;
                    border-radius: 12px;
                    padding: 30px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                    margin-bottom: 30px;
                }
                .search-input-container { display: flex; gap: 10px; margin-bottom: 15px; }
                input[type="text"] {
                    flex: 1;
                    padding: 15px 20px;
                    font-size: 16px;
                    border: 2px solid #e0e0e0;
                    border-radius: 8px;
                    outline: none;
                }
                input[type="text"]:focus { border-color: #667eea; }
                button {
                    padding: 15px 30px;
                    font-size: 16px;
                    font-weight: 600;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border: none;
                    border-radius: 8px;
                    cursor: pointer;
                }
                button:hover { transform: translateY(-2px); }
                button:disabled { opacity: 0.6; cursor: not-allowed; }
                .results {
                    background: white;
                    border-radius: 12px;
                    padding: 30px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                }
                .result-card {
                    padding: 20px;
                    border-bottom: 1px solid #e0e0e0;
                }
                .result-card:last-child { border-bottom: none; }
                .result-card:hover { background: #f9f9f9; }
                .result-header { display: flex; justify-content: space-between; margin-bottom: 10px; }
                .similarity-badge {
                    padding: 4px 12px;
                    border-radius: 12px;
                    font-size: 12px;
                    font-weight: 600;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                }
                .tweet-text { font-size: 16px; line-height: 1.6; margin-bottom: 12px; color: #333; }
                .tweet-meta { font-size: 14px; color: #666; }
                .tweet-meta strong { color: #667eea; }
                .stats { text-align: center; color: #666; margin-bottom: 20px; font-size: 14px; }
                .loading { text-align: center; padding: 40px; color: #667eea; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>⚡ Fast Tweet Search</h1>
                    <p>Search 6.4M tweets in milliseconds</p>
                </div>

                <div class="search-box">
                    <div class="search-input-container">
                        <input
                            type="text"
                            id="searchInput"
                            placeholder="Search tweets... (e.g., 'artificial intelligence')"
                            onkeypress="if(event.key === 'Enter') performSearch()"
                        >
                        <button onclick="performSearch()" id="searchBtn">Search</button>
                    </div>
                </div>

                <div id="results"></div>
            </div>

            <script>
                async function performSearch() {
                    const query = document.getElementById('searchInput').value.trim();
                    const resultsDiv = document.getElementById('results');
                    const searchBtn = document.getElementById('searchBtn');

                    if (!query) { alert('Please enter a search query'); return; }

                    searchBtn.disabled = true;
                    searchBtn.textContent = 'Searching...';
                    resultsDiv.innerHTML = '<div class="results"><div class="loading">⚡ Searching...</div></div>';

                    try {
                        const response = await fetch('/api/search', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ query, limit: 10 })
                        });

                        const data = await response.json();
                        if (!response.ok) throw new Error(data.error || 'Search failed');
                        displayResults(data);
                    } catch (error) {
                        resultsDiv.innerHTML = `<div class="results"><div class="loading" style="color:#c33;">Error: ${error.message}</div></div>`;
                    } finally {
                        searchBtn.disabled = false;
                        searchBtn.textContent = 'Search';
                    }
                }

                function displayResults(data) {
                    const resultsDiv = document.getElementById('results');
                    if (data.results.length === 0) {
                        resultsDiv.innerHTML = '<div class="results"><p style="text-align:center;color:#666;">No results found</p></div>';
                        return;
                    }

                    const resultsHTML = `
                        <div class="results">
                            <div class="stats">
                                Found ${data.total_results} results in ${data.search_time_ms}ms (DB query: ${data.db_query_time_ms}ms)
                            </div>
                            ${data.results.map(r => `
                                <div class="result-card">
                                    <div class="result-header">
                                        <span style="color:#999;font-size:14px;">#${r.rank}</span>
                                        <span class="similarity-badge">${(r.similarity * 100).toFixed(1)}% match</span>
                                    </div>
                                    <div class="tweet-text">${escapeHtml(r.text)}</div>
                                    <div class="tweet-meta"><strong>@${escapeHtml(r.username)}</strong> • ${r.created_at}</div>
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
