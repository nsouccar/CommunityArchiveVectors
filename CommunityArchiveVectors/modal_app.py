"""
Tweet Vector Database on Modal
- Voyage AI embeddings (1024 dimensions)
- In-memory vector storage with FAISS
- Semantic search API
- Auto-sync from Supabase

Note: Using FAISS instead of Milvus for better Modal compatibility
"""

import modal
import os
from typing import List, Dict, Any
import pickle

# Create Modal app
app = modal.App("tweet-vectors")

# Create persistent volume for vector index
vector_volume = modal.Volume.from_name("tweet-vectors-storage", create_if_missing=True)

# Python image with all dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "voyageai",
    "supabase",
    "faiss-cpu",  # Facebook AI Similarity Search
    "numpy",
    "fastapi",
    "pydantic",
    "scikit-learn",  # For KMeans clustering
    "openai"  # For LLM-based topic naming
)

# Secrets for API keys (created via: modal secret create tweet-vectors-secrets)
secrets = modal.Secret.from_name("tweet-vectors-secrets")


# ============================================================================
# VOYAGE AI EMBEDDINGS
# ============================================================================

@app.function(
    image=image,
    secrets=[secrets],
    timeout=3600
)
def generate_voyage_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate Voyage AI embeddings for a batch of texts (documents to be indexed)"""
    import voyageai

    vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

    result = vo.embed(
        texts=texts,
        model="voyage-3",  # 1024 dimensions
        input_type="document"  # For tweets being indexed
    )

    return result.embeddings


@app.function(
    image=image,
    secrets=[secrets],
    timeout=60,
    min_containers=1  # Keep warm for faster first search
)
def generate_query_embedding(query: str) -> List[float]:
    """Generate Voyage AI embedding for a search query"""
    import voyageai

    vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

    result = vo.embed(
        texts=[query],
        model="voyage-3",  # 1024 dimensions
        input_type="query"  # For search queries
    )

    return result.embeddings[0]


# ============================================================================
# VECTOR DATABASE (FAISS)
# ============================================================================

class VectorDB:
    """Simple vector database using FAISS with HNSW for fast search"""

    def __init__(self):
        import faiss
        import numpy as np

        self.dimension = 1024  # Voyage-3 dimensions
        # Use HNSW index for much faster search (approximate nearest neighbors)
        # M=32: number of connections per layer (higher = better accuracy, more memory)
        # efSearch will be set dynamically during search
        self.index = faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = 200  # Quality of index construction
        self.metadata = []  # Store tweet metadata

    def add(self, embeddings: List[List[float]], metadata: List[Dict]):
        """Add vectors and metadata to index"""
        import numpy as np
        import faiss

        vectors = np.array(embeddings, dtype=np.float32)

        # Normalize for cosine similarity
        faiss.normalize_L2(vectors)

        self.index.add(vectors)
        self.metadata.extend(metadata)

    def search(self, query_embedding: List[float], limit: int = 10):
        """Search for similar vectors with optimized HNSW parameters"""
        import numpy as np
        import faiss

        query_vector = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vector)

        # Set efSearch for better quality (higher = better accuracy, slower search)
        # For 10k vectors, ef=100 gives good balance
        if hasattr(self.index, 'hnsw'):
            self.index.hnsw.efSearch = 100

        scores, indices = self.index.search(query_vector, limit)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.metadata):  # Check for valid index
                result = self.metadata[idx].copy()
                result["score"] = float(score)
                results.append(result)

        return results

    def save(self, path: str):
        """Save index and metadata to disk"""
        import faiss
        import pickle

        # Save FAISS index
        faiss.write_index(self.index, f"{path}/index.faiss")

        # Save metadata
        with open(f"{path}/metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, path: str):
        """Load index and metadata from disk"""
        import faiss
        import pickle
        import os

        index_path = f"{path}/index.faiss"
        metadata_path = f"{path}/metadata.pkl"

        if os.path.exists(index_path) and os.path.exists(metadata_path):
            self.index = faiss.read_index(index_path)
            with open(metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
            return True
        return False

    def count(self):
        """Get number of vectors"""
        return self.index.ntotal


# ============================================================================
# DATA SYNC FROM SUPABASE
# ============================================================================

@app.function(
    image=image,
    secrets=[secrets],
    volumes={"/data": vector_volume},
    timeout=7200,  # 2 hours
)
def sync_tweets_from_supabase(limit: int = 10000):
    """
    Fetch tweets from Supabase and generate Voyage AI embeddings

    Args:
        limit: Number of tweets to process (default 10,000)
    """
    from supabase import create_client
    from datetime import datetime
    import re

    print(f"🚀 Starting sync...")
    print(f"📅 Fetching ALL tweets from October 1, 2025 onwards (limit={limit:,} if specified)\n")

    # Connect to Supabase
    supabase = create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_KEY"]
    )

    # Initialize or load vector DB
    db = VectorDB()
    if db.load("/data"):
        print(f"📂 Loaded existing index with {db.count():,} vectors\n")
    else:
        print("📂 Creating new index\n")

    # Helper: Clean text
    def clean_text(text: str) -> str:
        cleaned = text
        cleaned = re.sub(r'https?://\S+', '', cleaned)
        cleaned = re.sub(r'www\.\S+', '', cleaned)
        cleaned = re.sub(r'@\w+', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    # Helper: Get username
    def get_username(account_id: str) -> str:
        result = supabase.table("all_account").select("username").eq("account_id", account_id).limit(1).execute()
        if result.data and len(result.data) > 0:
            return result.data[0]["username"]
        return account_id

    # Step 1: Fetch tweets
    print(f"📥 Fetching up to {limit:,} tweets from Supabase (from October 1, 2025 onwards)...")
    all_tweets = []
    offset = 0
    batch_size = 1000

    # Fetch tweets up to the limit
    while len(all_tweets) < limit:
        fetch_count = min(batch_size, limit - len(all_tweets))

        response = supabase.table("tweets").select(
            "tweet_id, full_text, reply_to_tweet_id, created_at, account_id, "
            "retweet_count, favorite_count"
        ).gte(
            "created_at", "2025-10-01"
        ).order(
            "created_at", desc=False
        ).range(
            offset, offset + fetch_count - 1
        ).execute()

        if not response.data or len(response.data) == 0:
            break

        all_tweets.extend(response.data)
        offset += len(response.data)
        print(f"   Fetched {len(all_tweets):,} tweets...")

        if len(response.data) < fetch_count:
            break

    tweets = all_tweets
    print(f"✅ Retrieved {len(tweets):,} tweets\n")

    # Step 2: Get usernames in batches (much faster than one-by-one)
    print("👤 Fetching usernames...")
    unique_account_ids = list(set(t["account_id"] for t in tweets if t.get("account_id")))
    username_map = {}

    # Fetch in batches of 1000
    batch_size = 1000
    for i in range(0, len(unique_account_ids), batch_size):
        batch_ids = unique_account_ids[i:i+batch_size]

        # Fetch all usernames in this batch with a single query
        response = supabase.table("all_account").select("account_id, username").in_("account_id", batch_ids).execute()

        for account in response.data:
            username_map[account["account_id"]] = account["username"]

        if (i + batch_size) % 10000 == 0 or (i + batch_size) >= len(unique_account_ids):
            print(f"   Fetched {min(i + batch_size, len(unique_account_ids)):,}/{len(unique_account_ids):,} usernames...")

    print(f"✅ Found {len(username_map):,} usernames\n")

    # Step 3: Prepare embeddings data (standalone tweets, no thread context)
    print("🔄 Preparing embeddings (standalone tweets)...")
    texts_to_embed = []
    metadata_list = []

    for tweet in tweets:
        # Just embed the cleaned current tweet text (no parent context)
        cleaned_current = clean_text(tweet["full_text"])

        if not cleaned_current:
            continue  # Skip empty tweets after cleaning

        texts_to_embed.append(cleaned_current)

        metadata_list.append({
            "tweet_id": tweet["tweet_id"],
            "full_text": tweet["full_text"][:5000],
            "thread_context": cleaned_current[:10000],  # Same as cleaned text for standalone
            "thread_root_id": "",  # Not available in database
            "depth": 0,  # Not available in database
            "is_root": True,  # All treated as standalone
            "account_id": tweet.get("account_id", ""),
            "account_username": username_map.get(tweet.get("account_id", ""), tweet.get("account_id", "")),
            "favorite_count": tweet.get("favorite_count", 0),
            "retweet_count": tweet.get("retweet_count", 0),
            "created_at": tweet.get("created_at", ""),
            "processed_at": datetime.now().isoformat(),
            "embedding_version": "voyage-3-standalone"
        })

    print(f"✅ Prepared {len(texts_to_embed):,} embeddings\n")

    # Step 4: Generate embeddings in batches (sequentially to avoid rate limits)
    print("🤖 Generating Voyage AI embeddings...")
    BATCH_SIZE = 128

    all_embeddings = []
    for i in range(0, len(texts_to_embed), BATCH_SIZE):
        batch_texts = texts_to_embed[i:i+BATCH_SIZE]
        batch_num = i//BATCH_SIZE + 1
        total_batches = (len(texts_to_embed) + BATCH_SIZE - 1)//BATCH_SIZE

        if batch_num % 100 == 0 or batch_num == 1:
            print(f"   Batch {batch_num}/{total_batches} ({len(batch_texts)} texts)...")

        batch_embeddings = generate_voyage_embeddings.remote(batch_texts)
        all_embeddings.extend(batch_embeddings)

    print(f"✅ Generated {len(all_embeddings):,} embeddings\n")

    # Step 5: Add to vector DB
    print("📤 Adding to vector index...")
    db.add(all_embeddings, metadata_list)

    # Step 6: Save to disk
    print("💾 Saving index to disk...")
    db.save("/data")
    vector_volume.commit()

    print(f"\n🎉 Sync complete!")
    print(f"   Total vectors: {db.count():,}")

    return {
        "status": "success",
        "processed": len(all_embeddings),
        "total_vectors": db.count()
    }


# ============================================================================
# COMBINED WEB APP (UI + SEARCH API)
# ============================================================================

@app.function(
    image=image,
    secrets=[secrets],
    volumes={"/data": vector_volume},
    timeout=60,
    min_containers=1  # Keep one instance warm for faster responses
)
@modal.asgi_app()
def web():
    """Combined web app with UI and search API"""
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel

    web_app = FastAPI()

    class SearchRequest(BaseModel):
        query: str
        limit: int = 10

    # Pre-load the vector database once at startup
    _cached_db = {"db": None}  # Use dict to avoid global/nonlocal issues
    def get_vector_db():
        """Get or load the vector database (cached)"""
        if _cached_db["db"] is None:
            print("Loading vector database into memory...")
            db = VectorDB()
            if not db.load("/data"):
                return None
            print(f"Vector database loaded with {len(db.metadata)} tweets")
            _cached_db["db"] = db
        return _cached_db["db"]

    # Cache Voyage AI client for fast embedding generation
    _voyage_client = {"client": None}
    def get_voyage_client():
        """Get or create Voyage AI client (cached)"""
        if _voyage_client["client"] is None:
            import voyageai
            print("Initializing Voyage AI client...")
            _voyage_client["client"] = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
        return _voyage_client["client"]

    @web_app.get("/users")
    async def get_users(min_tweets: int = 30):
        """Get list of all users in the index with at least min_tweets tweets and meaningful clusters"""
        # Get cached vector DB
        db = get_vector_db()
        if db is None:
            return {"error": "vector database not initialized"}

        # Count tweets per user
        from collections import Counter
        username_counts = Counter()
        for meta in db.metadata:
            username = meta.get("account_username")
            if username:
                username_counts[username] += 1

        # Filter users with enough tweets (higher threshold since LLM will filter more)
        # Using min_tweets=30 because after LLM filtering, users need more raw tweets
        # to end up with meaningful clusters
        qualified_users = [
            {"username": username, "tweet_count": count}
            for username, count in username_counts.items()
            if count >= min_tweets
        ]

        # Sort by tweet count (most tweets first)
        qualified_users.sort(key=lambda x: x["tweet_count"], reverse=True)

        return {
            "users": qualified_users,
            "total_users": len(qualified_users),
            "min_tweets": min_tweets
        }

    @web_app.get("/user/{username}/topics")
    async def get_user_topics(username: str, num_topics: int = 5):
        """Get top topics for a specific user using clustering"""
        import numpy as np
        from sklearn.cluster import KMeans
        from collections import Counter
        import re

        # Get cached vector DB
        db = get_vector_db()
        if db is None:
            return {"error": "vector database not initialized"}

        def extract_topic_name(cluster_tweets):
            """Extract topic name from cluster tweets using LLM analysis"""
            from openai import OpenAI

            # Use up to 10 sample tweets for analysis (to keep prompt size reasonable)
            sample_tweets = cluster_tweets[:10]

            # Create prompt for LLM
            tweets_text = "\n".join([f"- {tweet}" for tweet in sample_tweets])

            prompt = f"""Analyze these tweets and create a short, descriptive topic name (2-4 words max) that captures the main theme:

{tweets_text}

Respond with ONLY the topic name, nothing else. Examples of good topic names:
- "AI & Machine Learning"
- "Web Development"
- "Personal Updates"
- "Tech News & Trends"
"""

            try:
                client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

                response = client.chat.completions.create(
                    model="gpt-4o-mini",  # Fast and cheap model
                    messages=[
                        {"role": "system", "content": "You are a topic classifier that creates short, descriptive topic names from tweet clusters."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=20,
                    temperature=0.3
                )

                topic_name = response.choices[0].message.content.strip()
                # Remove any quotes that might be added
                topic_name = topic_name.strip('"\'')

                return topic_name if topic_name else "General"

            except Exception as e:
                print(f"Error generating topic name with LLM: {e}")
                # Fallback to simple keyword extraction
                from collections import Counter
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from'}
                all_text = ' '.join(cluster_tweets).lower()
                words = re.findall(r'\b[a-z]{3,}\b', all_text)
                meaningful_words = [w for w in words if w not in stop_words]
                word_counts = Counter(meaningful_words)
                top_words = [word for word, count in word_counts.most_common(3)]
                return " & ".join(word.capitalize() for word in top_words) if top_words else "General"


        # Get all tweets and embeddings for this user
        user_embeddings = []
        user_tweets = []
        user_tweet_data = []  # Store full tweet data including IDs
        user_indices = []

        for idx, meta in enumerate(db.metadata):
            if meta.get("account_username") == username:
                user_indices.append(idx)
                tweet_text = meta.get("full_text", "")
                user_tweets.append(tweet_text)
                # Store full data for later use
                user_tweet_data.append({
                    "text": tweet_text,
                    "tweet_id": meta.get("tweet_id"),
                    "username": meta.get("account_username")
                })

        if len(user_indices) == 0:
            return {"error": f"No tweets found for user {username}"}

        # Reconstruct embeddings from FAISS index
        for idx in user_indices:
            embedding = db.index.reconstruct(int(idx))
            user_embeddings.append(embedding)

        user_embeddings = np.array(user_embeddings)

        # Cluster the embeddings
        n_clusters = min(num_topics, len(user_embeddings))
        if n_clusters < 2:
            # Not enough tweets to cluster - rank by similarity to mean
            centroid = np.mean(user_embeddings, axis=0)
            centroid_norm = centroid / np.linalg.norm(centroid)

            similarities = []
            for i, (tweet, embedding) in enumerate(zip(user_tweets, user_embeddings)):
                embedding_norm = embedding / np.linalg.norm(embedding)
                similarity = np.dot(embedding_norm, centroid_norm)
                similarities.append((similarity, tweet))

            similarities.sort(reverse=True)
            ranked_tweets = [{"text": tweet, "similarity": float(sim)} for sim, tweet in similarities]

            topic_name = extract_topic_name(user_tweets)
            return {
                "username": username,
                "tweet_count": len(user_tweets),
                "topics": [{
                    "topic_id": 0,
                    "topic_name": topic_name,
                    "tweets": ranked_tweets,
                    "tweet_count": len(user_tweets)
                }]
            }

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(user_embeddings)

        def refine_cluster_with_llm(cluster_tweets):
            """Use LLM to filter out tweets that don't belong to the cluster's main topic"""
            from openai import OpenAI

            if len(cluster_tweets) < 5:
                # Too few tweets to meaningfully filter
                return cluster_tweets

            # Sample up to 20 tweets for LLM analysis
            sample_size = min(20, len(cluster_tweets))
            sample_tweets = cluster_tweets[:sample_size]

            # Create prompt asking LLM to identify which tweets belong together
            tweets_text = "\n".join([f"{i+1}. {tweet}" for i, tweet in enumerate(sample_tweets)])

            prompt = f"""Analyze these {sample_size} tweets and identify which ones share a common meaningful topic or theme.

{tweets_text}

Instructions:
1. Identify the main topic/theme that MOST tweets share
2. List the tweet numbers (1-{sample_size}) that genuinely belong to this topic
3. Exclude tweets that are just short replies, greetings, or don't fit the main theme
4. If there's no coherent topic (just random short replies), return "NONE"

Respond ONLY with comma-separated numbers (e.g., "1,3,5,7,12") or "NONE"."""

            try:
                client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a tweet clustering expert that identifies coherent topics and filters out noise."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=50,
                    temperature=0.2
                )

                result = response.choices[0].message.content.strip()

                # If LLM says there's no coherent topic, mark this cluster as low-quality
                if result.upper() == "NONE":
                    return []

                # Parse the tweet numbers
                try:
                    valid_indices = [int(x.strip()) - 1 for x in result.split(",") if x.strip().isdigit()]
                    # Only keep tweets that LLM identified as belonging to the main topic
                    filtered_tweets = [cluster_tweets[i] for i in valid_indices if i < len(cluster_tweets)]

                    # If less than 30% of tweets belong, this cluster is probably noise
                    if len(filtered_tweets) < len(cluster_tweets) * 0.3:
                        return []

                    return filtered_tweets
                except:
                    # If parsing fails, keep all tweets
                    return cluster_tweets

            except Exception as e:
                print(f"Error refining cluster with LLM: {e}")
                # On error, keep all tweets
                return cluster_tweets

        # Group tweets by cluster and rank by similarity to centroid
        topics = []
        for cluster_id in range(n_clusters):
            # Get indices of tweets in this cluster
            cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]

            # Skip empty clusters
            if len(cluster_indices) == 0:
                continue

            cluster_tweets = [user_tweets[i] for i in cluster_indices]
            cluster_embeddings = user_embeddings[cluster_indices]

            # Refine cluster using LLM to remove noise
            refined_tweets = refine_cluster_with_llm(cluster_tweets)

            # Skip clusters that are mostly noise
            if len(refined_tweets) < 3:
                continue

            # Re-index embeddings for refined tweets and get corresponding tweet data
            refined_indices = [i for i, tweet in enumerate(cluster_tweets) if tweet in refined_tweets]
            refined_embeddings = cluster_embeddings[refined_indices]

            # Map refined tweets back to original indices to get tweet IDs
            original_indices = [cluster_indices[i] for i in refined_indices]
            refined_tweet_data = [user_tweet_data[idx] for idx in original_indices]

            # Calculate centroid (mean) of cluster embeddings
            centroid = np.mean(refined_embeddings, axis=0)

            # Calculate cosine similarity of each tweet to the centroid
            # Normalize vectors for cosine similarity
            centroid_norm = centroid / np.linalg.norm(centroid)
            similarities = []
            for i, embedding in enumerate(refined_embeddings):
                embedding_norm = embedding / np.linalg.norm(embedding)
                similarity = np.dot(embedding_norm, centroid_norm)
                similarities.append((i, similarity, refined_tweet_data[i]))

            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Get ranked tweets with similarity scores and IDs
            ranked_tweets = [
                {
                    "text": tweet_data["text"],
                    "similarity": float(sim),
                    "tweet_id": tweet_data["tweet_id"],
                    "username": tweet_data["username"]
                }
                for _, sim, tweet_data in similarities
            ]

            # Extract topic name from refined cluster tweets
            topic_name = extract_topic_name(refined_tweets)

            topics.append({
                "topic_id": cluster_id,
                "topic_name": topic_name,
                "tweets": ranked_tweets,  # All tweets, ranked by similarity
                "tweet_count": len(refined_tweets)
            })

        # Sort by tweet count (most common topics first)
        topics.sort(key=lambda x: x["tweet_count"], reverse=True)

        # If no meaningful topics after LLM filtering, return error
        if len(topics) == 0:
            return {
                "error": f"No meaningful topics found for {username}. This user's tweets are mostly short replies or don't cluster into coherent themes."
            }

        return {
            "username": username,
            "tweet_count": len(user_tweets),
            "topics": topics
        }

    @web_app.post("/search")
    async def search_endpoint(request: SearchRequest):
        """Search API endpoint"""
        query_text = request.query
        limit = request.limit

        if not query_text:
            return {"error": "query parameter required"}

        # Get cached vector DB
        db = get_vector_db()
        if db is None:
            return {"error": "vector database not initialized - run sync first"}

        # Generate query embedding directly (no separate function call)
        vo = get_voyage_client()
        result = vo.embed(
            texts=[query_text],
            model="voyage-3",
            input_type="query"
        )
        query_embedding = result.embeddings[0]

        # Search
        results = db.search(query_embedding, limit=limit)

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "tweetId": result.get("tweet_id"),
                "text": result.get("full_text"),
                "threadContext": result.get("thread_context"),
                "similarity": f"{result.get('score', 0) * 100:.1f}%",
                "score": result.get("score", 0),
                "author": result.get("account_username") or result.get("account_id"),
                "likes": result.get("favorite_count"),
                "retweets": result.get("retweet_count"),
                "createdAt": result.get("created_at"),
                "depth": result.get("depth"),
                "isRoot": result.get("is_root"),
            })

        return {
            "query": query_text,
            "results": formatted_results,
            "total_vectors": db.count()
        }

    @web_app.get("/")
    async def ui_endpoint():
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Tweet Semantic Search - Modal</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 40px 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
        }
        h1 {
            color: white;
            font-size: 36px;
            margin-bottom: 10px;
            text-align: center;
        }
        .subtitle {
            color: rgba(255,255,255,0.9);
            text-align: center;
            margin-bottom: 40px;
            font-size: 16px;
        }
        .search-box {
            background: white;
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            margin-bottom: 30px;
        }
        input[type="text"] {
            width: 100%;
            padding: 16px 20px;
            font-size: 16px;
            border: 2px solid #e0e0e0;
            border-radius: 12px;
            transition: all 0.3s;
        }
        input[type="text"]:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        button {
            margin-top: 15px;
            width: 100%;
            padding: 16px;
            font-size: 16px;
            font-weight: 600;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            transition: transform 0.2s;
        }
        button:hover { transform: translateY(-2px); }
        button:active { transform: translateY(0); }
        #results {
            display: grid;
            gap: 20px;
        }
        .tweet-card {
            background: white;
            border-radius: 16px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .tweet-header {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
            gap: 12px;
        }
        .profile-icon {
            width: 48px;
            height: 48px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 20px;
        }
        .tweet-info { flex: 1; }
        .username {
            font-weight: 600;
            font-size: 16px;
            color: #333;
        }
        .similarity-badge {
            display: inline-block;
            padding: 6px 14px;
            background: #e3f2fd;
            color: #1976d2;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
        }
        .tweet-text {
            font-size: 15px;
            line-height: 1.6;
            color: #333;
            margin-bottom: 15px;
        }
        .thread-context {
            background: #f5f5f5;
            border-left: 4px solid #667eea;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 8px;
        }
        .context-label {
            font-size: 12px;
            color: #666;
            font-weight: 600;
            margin-bottom: 8px;
        }
        .context-text {
            font-size: 14px;
            color: #555;
            line-height: 1.5;
        }
        .tweet-meta {
            display: flex;
            gap: 20px;
            color: #666;
            font-size: 14px;
        }
        .tweet-link {
            display: inline-block;
            margin-top: 15px;
            padding: 10px 20px;
            background: #1DA1F2;
            color: white;
            text-decoration: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
        }
        .loading {
            text-align: center;
            color: white;
            font-size: 18px;
            padding: 40px;
        }
        .powered-by {
            text-align: center;
            color: rgba(255,255,255,0.8);
            margin-top: 30px;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Tweet Semantic Search</h1>
        <div style="text-align: center; margin-bottom: 30px; margin-top: 30px;">
            <a href="/topics" style="display: inline-block; padding: 12px 24px; background: white; color: #667eea; text-decoration: none; border-radius: 8px; font-weight: 600; transition: all 0.2s; box-shadow: 0 4px 12px rgba(0,0,0,0.15);" onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 16px rgba(0,0,0,0.2)'" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 12px rgba(0,0,0,0.15)'">
                Analyze User Topics
            </a>
        </div>

        <div class="search-box">
            <input type="text" id="query" placeholder="Search tweets semantically..." />
            <button onclick="search()">Search</button>
        </div>

        <div id="results"></div>
    </div>

    <script>
        async function search() {
            const query = document.getElementById('query').value;
            if (!query) return;

            document.getElementById('results').innerHTML = '<div class="loading">Searching...</div>';

            try {
                const response = await fetch('/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query, limit: 10 })
                });

                const data = await response.json();

                if (data.error) {
                    document.getElementById('results').innerHTML =
                        `<div class="loading">Error: ${data.error}</div>`;
                    return;
                }

                if (!data.results || data.results.length === 0) {
                    document.getElementById('results').innerHTML =
                        `<div class="loading">No results found. Try a different query.</div>`;
                    return;
                }

                let html = '';
                data.results.forEach((result, index) => {
                    const initial = result.author.charAt(0).toUpperCase();
                    // Only show thread context if it contains more than just the current tweet
                    // Check if thread context has parent tweet (indicated by double newline separator)
                    const hasParent = result.threadContext && result.threadContext.includes('\\n\\n');

                    html += `
                        <div class="tweet-card">
                            <div class="tweet-header">
                                <div class="profile-icon">${initial}</div>
                                <div class="tweet-info">
                                    <div class="username">@${result.author}</div>
                                </div>
                                <span class="similarity-badge">${result.similarity}</span>
                            </div>

                            <div class="tweet-text">${escapeHtml(result.text)}</div>

                            ${hasParent ? `
                                <div class="thread-context">
                                    <div class="context-label">🧵 Thread Context (what was embedded)</div>
                                    <div class="context-text">${escapeHtml(result.threadContext)}</div>
                                </div>
                            ` : ''}

                            <div class="tweet-meta">
                                <span>❤️ ${result.likes || 0}</span>
                                <span>🔄 ${result.retweets || 0}</span>
                                <span>📅 ${new Date(result.createdAt).toLocaleDateString()}</span>
                            </div>

                            <a href="https://twitter.com/${result.author}/status/${result.tweetId}"
                               target="_blank" class="tweet-link">
                                🔗 View on Twitter/X
                            </a>
                        </div>
                    `;
                });

                document.getElementById('results').innerHTML = html;
            } catch (error) {
                console.error('Search error:', error);
                document.getElementById('results').innerHTML =
                    `<div class="loading">Error: ${error.message}<br><br>Check browser console for details.</div>`;
            }
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        document.getElementById('query').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') search();
        });
    </script>
</body>
</html>
    """

        return HTMLResponse(content=html)

    @web_app.get("/topics")
    async def topics_ui():
        """User Topics Analysis UI"""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>User Topics Analysis</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 40px 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            color: white;
            font-size: 36px;
            margin-bottom: 10px;
            text-align: center;
        }
        .subtitle {
            color: rgba(255,255,255,0.9);
            text-align: center;
            margin-bottom: 40px;
            font-size: 16px;
        }
        .selector-box {
            background: white;
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            margin-bottom: 30px;
        }
        select {
            width: 100%;
            padding: 16px 20px;
            font-size: 16px;
            border: 2px solid #e0e0e0;
            border-radius: 12px;
            transition: all 0.3s;
            background: white;
        }
        select:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        button {
            margin-top: 15px;
            width: 100%;
            padding: 16px;
            font-size: 16px;
            font-weight: 600;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            transition: transform 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
        }
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        .loading {
            text-align: center;
            color: white;
            font-size: 18px;
            margin: 20px 0;
        }
        .results {
            margin-top: 30px;
        }
        .topic-card {
            background: white;
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .topic-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #f0f0f0;
        }
        .topic-name {
            font-size: 24px;
            font-weight: 700;
            color: #667eea;
        }
        .topic-count {
            background: #667eea;
            color: white;
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
        }
        .tweet-list {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .tweet-item {
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            transition: all 0.2s;
        }
        .tweet-link {
            text-decoration: none;
            color: inherit;
            display: block;
        }
        .tweet-item:hover {
            background: #e9ecef;
            transform: translateX(4px);
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.2);
        }
        .tweet-text {
            color: #2c3e50;
            line-height: 1.5;
            margin-bottom: 8px;
        }
        .tweet-similarity {
            font-size: 12px;
            color: #667eea;
            font-weight: 600;
        }
        .error {
            background: #ff4444;
            color: white;
            padding: 20px;
            border-radius: 12px;
            margin-top: 20px;
            text-align: center;
        }
        .user-header {
            background: white;
            border-radius: 16px;
            padding: 20px 30px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            text-align: center;
        }
        .username-display {
            font-size: 28px;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 5px;
        }
        .tweet-count-display {
            color: #667eea;
            font-size: 16px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>User Topics Analysis</h1>
        <p class="subtitle">Discover what users tweet about, ranked by semantic similarity</p>
        <div style="text-align: center; margin-bottom: 20px;">
            <a href="/" style="display: inline-block; padding: 10px 20px; background: white; color: #667eea; text-decoration: none; border-radius: 8px; font-weight: 600; transition: all 0.2s; box-shadow: 0 4px 12px rgba(0,0,0,0.15);" onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 16px rgba(0,0,0,0.2)'" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 12px rgba(0,0,0,0.15)'">
                ← Back to Search
            </a>
        </div>

        <div class="selector-box">
            <select id="userSelect">
                <option value="">Loading users...</option>
            </select>
            <button onclick="analyzeUser()" id="analyzeBtn" disabled>Analyze Topics</button>
        </div>

        <div id="loading" class="loading" style="display: none;">
            Analyzing topics... This may take a minute.
        </div>

        <div id="results" class="results"></div>
    </div>

    <script>
        let users = [];

        // Load users on page load
        async function loadUsers() {
            try {
                const response = await fetch('/users?min_tweets=20');
                const data = await response.json();
                users = data.users;

                const select = document.getElementById('userSelect');
                select.innerHTML = `<option value="">Select a user... (${data.total_users} users with ${data.min_tweets}+ tweets)</option>`;
                // Add info about filtering
                const subtitle = document.querySelector('.subtitle');
                subtitle.textContent = `Discover what users tweet about, ranked by semantic similarity (showing users with ${data.min_tweets}+ tweets)`;
                users.forEach(user => {
                    const option = document.createElement('option');
                    option.value = user.username;
                    option.textContent = `@${user.username} (${user.tweet_count} tweets)`;
                    select.appendChild(option);
                });

                document.getElementById('analyzeBtn').disabled = false;
            } catch (error) {
                document.getElementById('userSelect').innerHTML = '<option value="">Error loading users</option>';
                console.error('Error loading users:', error);
            }
        }

        async function analyzeUser() {
            const username = document.getElementById('userSelect').value;
            if (!username) {
                alert('Please select a user');
                return;
            }

            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').innerHTML = '';
            document.getElementById('analyzeBtn').disabled = true;

            try {
                const response = await fetch(`/user/${username}/topics?num_topics=5`);
                const data = await response.json();

                if (data.error) {
                    document.getElementById('results').innerHTML = `
                        <div class="error">${data.error}</div>
                    `;
                } else {
                    displayResults(data);
                }
            } catch (error) {
                document.getElementById('results').innerHTML = `
                    <div class="error">Error analyzing topics: ${error.message}</div>
                `;
            } finally {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('analyzeBtn').disabled = false;
            }
        }

        function displayResults(data) {
            let html = `
                <div class="user-header">
                    <div class="username-display">@${data.username}</div>
                    <div class="tweet-count-display">${data.tweet_count} tweets analyzed</div>
                </div>
            `;

            data.topics.forEach((topic, index) => {
                html += `
                    <div class="topic-card">
                        <div class="topic-header">
                            <div class="topic-name">${topic.topic_name}</div>
                            <div class="topic-count">${topic.tweet_count} tweets</div>
                        </div>
                        <div class="tweet-list">
                `;

                // Show only top 10 most relevant tweets
                topic.tweets.slice(0, 10).forEach(tweet => {
                    const similarityPercent = (tweet.similarity * 100).toFixed(1);
                    const tweetUrl = `https://twitter.com/${tweet.username}/status/${tweet.tweet_id}`;
                    html += `
                        <a href="${tweetUrl}" target="_blank" class="tweet-item tweet-link">
                            <div class="tweet-text">${escapeHtml(tweet.text)}</div>
                            <div class="tweet-similarity">Similarity: ${similarityPercent}% • Click to view on Twitter →</div>
                        </a>
                    `;
                });

                html += `
                        </div>
                    </div>
                `;
            });

            document.getElementById('results').innerHTML = html;
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Load users when page loads
        loadUsers();
    </script>
</body>
</html>
        """
        return HTMLResponse(content=html)

    return web_app


# ============================================================================
# LOCAL ENTRYPOINTS
# ============================================================================

@app.function(
    image=image,
    volumes={"/data": vector_volume},
    timeout=60
)
def clear_index():
    """Delete the existing index files to start fresh"""
    import os

    index_path = "/data/index.faiss"
    metadata_path = "/data/metadata.pkl"

    deleted = []
    if os.path.exists(index_path):
        os.remove(index_path)
        deleted.append("index.faiss")

    if os.path.exists(metadata_path):
        os.remove(metadata_path)
        deleted.append("metadata.pkl")

    vector_volume.commit()

    return {"status": "success", "deleted": deleted}


@app.function(
    image=image,
    secrets=[secrets],
    volumes={"/data": vector_volume},
    timeout=1800
)
def rebuild_with_hnsw():
    """Rebuild existing index with HNSW for faster search"""
    import faiss
    import pickle
    import os

    print("🔄 Rebuilding index with HNSW for faster search...")

    # Load existing flat index
    index_path = "/data/index.faiss"
    metadata_path = "/data/metadata.pkl"

    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        return {"error": "No existing index found"}

    print("📂 Loading existing index...")
    old_index = faiss.read_index(index_path)

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    print(f"✅ Loaded {old_index.ntotal:,} vectors")

    # Create new HNSW index
    print("🔨 Creating HNSW index...")
    dimension = old_index.d
    new_index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
    new_index.hnsw.efConstruction = 200

    # Copy vectors from old index to new index
    print("📥 Transferring vectors to HNSW index...")
    vectors = old_index.reconstruct_n(0, old_index.ntotal)
    new_index.add(vectors)

    print(f"✅ Added {new_index.ntotal:,} vectors to HNSW index")

    # Save new index
    print("💾 Saving HNSW index...")
    faiss.write_index(new_index, index_path)

    # Metadata stays the same
    print("✅ Index rebuilt successfully!")

    vector_volume.commit()

    return {
        "status": "success",
        "vectors": new_index.ntotal,
        "index_type": "HNSW",
        "parameters": {
            "M": 32,
            "efConstruction": 200
        }
    }


# ============================================================================
# INSPECT EMBEDDINGS
# ============================================================================

@app.function(
    image=image,
    volumes={"/data": vector_volume},
    timeout=60
)
def inspect_embeddings(num_samples: int = 5):
    """Inspect stored embeddings and their metadata"""
    import faiss
    import pickle
    import os

    print(f"🔍 Inspecting embeddings...")

    # Load index and metadata
    index_path = "/data/index.faiss"
    metadata_path = "/data/metadata.pkl"

    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        return {"error": "No index found"}

    print("📂 Loading index...")
    index = faiss.read_index(index_path)

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    print(f"✅ Loaded index with {index.ntotal:,} vectors\n")

    # Show index info
    print("📊 Index Information:")
    print(f"   Dimensions: {index.d}")
    print(f"   Total vectors: {index.ntotal:,}")
    print(f"   Index type: {type(index).__name__}\n")

    # Show sample embeddings
    print(f"📝 Sample Embeddings (first {num_samples}):")
    for i in range(min(num_samples, index.ntotal)):
        # Reconstruct vector from index
        vector = index.reconstruct(int(i))

        # Get metadata
        meta = metadata[i] if i < len(metadata) else {}

        print(f"\n--- Vector {i} ---")
        print(f"Original tweet: {meta.get('full_text', 'N/A')[:150]}...")
        print(f"Cleaned/embedded text: {meta.get('thread_context', 'N/A')[:150]}...")
        print(f"Author: {meta.get('account_username', 'N/A')}")
        print(f"Embedding (first 10 dims): {vector[:10].tolist()}")
        print(f"Embedding magnitude: {(vector ** 2).sum() ** 0.5:.4f}")
        print(f"Embedding version: {meta.get('embedding_version', 'N/A')}")

    return {
        "status": "success",
        "total_vectors": int(index.ntotal),
        "dimensions": int(index.d),
        "index_type": type(index).__name__,
        "samples_shown": min(num_samples, index.ntotal)
    }


@app.local_entrypoint()
def main(action: str = "help", limit: int = 10000):
    """
    Local CLI for managing the tweet vector database

    Usage:
        modal run modal_app.py                        # Show help
        modal run modal_app.py --action sync          # Sync 10,000 tweets
        modal run modal_app.py --action sync --limit 1000   # Sync 1,000 tweets
        modal run modal_app.py --action test          # Test search
    """

    if action == "sync":
        print(f"Syncing {limit:,} tweets from Supabase...")
        result = sync_tweets_from_supabase.remote(limit=limit)
        print(f"\nResult: {result}")

    elif action == "rebuild":
        print("Rebuilding index with HNSW...")
        result = rebuild_with_hnsw.remote()
        print(f"\nResult: {result}")

    elif action == "clear":
        print("Clearing vector index...")
        result = clear_index.remote()
        print(f"\nResult: {result}")

    elif action == "inspect":
        print("Inspecting embeddings...")
        result = inspect_embeddings.remote(num_samples=limit if limit < 100 else 10)
        print(f"\nResult: {result}")

    elif action == "test":
        print("Testing search...")
        result = search.remote({"query": "machine learning and AI", "limit": 5})
        print(f"\nFound {len(result.get('results', []))} results:")
        for i, r in enumerate(result.get('results', []), 1):
            print(f"\n{i}. @{r['author']} ({r['similarity']})")
            print(f"   {r['text'][:100]}...")

    else:
        print("""
🚀 Tweet Vector Database on Modal

Commands:
  modal run modal_app.py --action sync               # Sync 10,000 tweets
  modal run modal_app.py --action sync --limit 1000  # Sync custom amount
  modal run modal_app.py --action inspect            # Inspect embeddings (shows 10)
  modal run modal_app.py --action inspect --limit 5  # Inspect 5 embeddings
  modal run modal_app.py --action clear              # Clear the index
  modal run modal_app.py --action rebuild            # Rebuild with HNSW
  modal run modal_app.py --action test               # Test search

Deploy web app:
  modal deploy modal_app.py

  This will give you:
  - GET  /ui     - Web interface
  - POST /search - Search API

Features:
  ✅ Voyage AI embeddings (1024 dims)
  ✅ FAISS vector index with cosine similarity
  ✅ Immediate parent context
  ✅ Username display
  ✅ Persistent storage on Modal
        """)

        return HTMLResponse(content=html)
