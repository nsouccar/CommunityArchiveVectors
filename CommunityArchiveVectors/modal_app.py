"""
Tweet Vector Database on Modal
- Voyage AI embeddings (1024 dimensions)
- Persistent vector storage with CoreNN
- Semantic search API
- Auto-sync from Supabase
- Scales to billions of vectors

Note: Using CoreNN for billion-scale vector search on commodity hardware
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
    "corenn-py",  # CoreNN for billion-scale vector search
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
    import time

    vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

    # Retry logic for network failures
    max_retries = 5
    for attempt in range(max_retries):
        try:
            result = vo.embed(
                texts=texts,
                model="voyage-3",  # 1024 dimensions
                input_type="document"  # For tweets being indexed
            )
            return result.embeddings
        except (voyageai.error.APIConnectionError, Exception) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s, 8s, 16s
                print(f"⚠️  Voyage AI connection error (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"❌ Voyage AI failed after {max_retries} attempts")
                raise


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
# VECTOR DATABASE (CoreNN)
# ============================================================================

class VectorDB:
    """Billion-scale vector database using CoreNN for fast approximate nearest neighbor search"""

    def __init__(self):
        self.dimension = 1024  # Voyage-3 dimensions
        self.db = None  # CoreNN database
        self.metadata = {}  # Store tweet metadata keyed by tweet_id
        self.db_path = None
        self._count = 0  # Track number of vectors

    def add(self, embeddings: List[List[float]], metadata_list: List[Dict]):
        """Add vectors and metadata to index"""
        import numpy as np
        from corenn_py import CoreNN

        # Create database if it doesn't exist
        if self.db is None:
            raise RuntimeError("Database not initialized. Call save() first to set path.")

        # Prepare keys (use tweet_id as string key)
        keys = [meta["tweet_id"] for meta in metadata_list]

        # Prepare vectors
        vectors = np.array(embeddings, dtype=np.float32)

        # Normalize for cosine similarity (CoreNN uses L2 distance, normalized L2 = cosine)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        vectors = vectors / norms

        # Insert vectors (CoreNN handles insertion)
        self.db.insert_f32(keys, vectors)

        # Store metadata
        for meta in metadata_list:
            self.metadata[meta["tweet_id"]] = meta

        self._count += len(keys)

    def search(self, query_embedding: List[float], limit: int = 10):
        """Search for similar vectors"""
        import numpy as np

        if self.db is None:
            return []

        # Normalize query vector
        query_vector = np.array([query_embedding], dtype=np.float32)
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm

        # Query CoreNN (returns list of lists of (key, distance) tuples)
        results_list = self.db.query_f32(query_vector, limit)

        # Process results
        results = []
        if len(results_list) > 0:
            for key, distance in results_list[0]:  # First query's results
                # Convert distance to similarity score
                # For normalized vectors with L2 distance: similarity = 1 - (distance^2 / 4)
                # But simpler: similarity ≈ 1 - distance/2 (since distance is in [0, 2] for normalized vectors)
                similarity = max(0.0, 1.0 - (distance / 2.0))

                # Get metadata
                if key in self.metadata:
                    result = self.metadata[key].copy()
                    result["score"] = similarity
                    results.append(result)

        return results

    def save(self, path: str):
        """Save database and metadata to disk"""
        import pickle
        import os
        from corenn_py import CoreNN

        os.makedirs(path, exist_ok=True)

        # Create or update CoreNN database
        db_path = f"{path}/corenn_db"
        if self.db is None:
            # Create new database
            self.db = CoreNN.create(db_path, {"dim": self.dimension})
            self.db_path = db_path
        # Note: CoreNN automatically persists data, no explicit save needed

        # Save metadata separately
        with open(f"{path}/metadata.pkl", "wb") as f:
            pickle.dump({"metadata": self.metadata, "count": self._count}, f)

    def load(self, path: str):
        """Load database and metadata from disk"""
        import pickle
        import os
        from corenn_py import CoreNN

        db_path = f"{path}/corenn_db"
        metadata_path = f"{path}/metadata.pkl"

        if os.path.exists(db_path) and os.path.exists(metadata_path):
            # Open existing CoreNN database
            self.db = CoreNN.open(db_path)
            self.db_path = db_path

            # Load metadata
            with open(metadata_path, "rb") as f:
                data = pickle.load(f)
                self.metadata = data.get("metadata", {})
                self._count = data.get("count", len(self.metadata))

            return True
        return False

    def count(self):
        """Get number of vectors"""
        return self._count


class TopicsCache:
    """Cache for pre-computed user topics and user list"""

    def __init__(self):
        self.topics = {}  # username -> topics data
        self.users_list = None  # Pre-computed users list for /users endpoint

    def save(self, path: str):
        """Save topics cache to disk"""
        import pickle
        data = {
            "topics": self.topics,
            "users_list": self.users_list
        }
        with open(f"{path}/topics_cache.pkl", "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str):
        """Load topics cache from disk"""
        import pickle
        import os
        cache_path = f"{path}/topics_cache.pkl"
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
                # Handle both old format (just dict) and new format (dict with topics + users_list)
                if isinstance(data, dict) and "topics" in data:
                    self.topics = data["topics"]
                    self.users_list = data.get("users_list")
                else:
                    # Old format - just the topics dict
                    self.topics = data
                    self.users_list = None
            return True
        return False

    def get(self, username: str):
        """Get topics for a user"""
        return self.topics.get(username)

    def set(self, username: str, topics_data):
        """Set topics for a user"""
        self.topics[username] = topics_data

    def has(self, username: str):
        """Check if user has cached topics"""
        return username in self.topics

    def set_users_list(self, users_list):
        """Set the pre-computed users list"""
        self.users_list = users_list

    def get_users_list(self):
        """Get the pre-computed users list"""
        return self.users_list


# ============================================================================
# DATA SYNC FROM SUPABASE
# ============================================================================

@app.function(
    image=image,
    secrets=[secrets],
    volumes={"/data": vector_volume},
    timeout=5400,  # 90 minutes (100K batches take ~60 mins, 30 min buffer)
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
    print(f"📅 Fetching ALL tweets from entire archive (limit={limit:,} if specified)\n")

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

    # Step 1: Fetch tweets using cursor-based pagination (avoids timeout on large offsets)
    # Start from the highest tweet_id already in database to skip processed tweets
    max_existing_tweet_id = None
    if db.metadata:
        max_existing_tweet_id = max(db.metadata.keys())
        print(f"📊 Starting from tweet_id > {max_existing_tweet_id} (skipping already processed tweets)")

    print(f"📥 Fetching up to {limit:,} tweets from Supabase...")
    all_tweets = []
    last_tweet_id = max_existing_tweet_id  # Start cursor from last processed tweet
    batch_size = 1000

    # Fetch tweets up to the limit using cursor-based pagination
    while len(all_tweets) < limit:
        fetch_count = min(batch_size, limit - len(all_tweets))

        # Build query with cursor (ordered by tweet_id for consistency)
        query = supabase.table("tweets").select(
            "tweet_id, full_text, reply_to_tweet_id, created_at, account_id, "
            "retweet_count, favorite_count"
        ).order("tweet_id", desc=False).limit(fetch_count)

        # Always fetch tweets after the cursor (either max existing or last fetched)
        if last_tweet_id is not None:
            query = query.gt("tweet_id", last_tweet_id)

        response = query.execute()

        if not response.data or len(response.data) == 0:
            break

        all_tweets.extend(response.data)
        # Update cursor to the last tweet's ID
        last_tweet_id = response.data[-1]["tweet_id"]
        print(f"   Fetched {len(all_tweets):,} tweets... (cursor: {last_tweet_id})")

        if len(response.data) < fetch_count:
            break

    tweets = all_tweets
    print(f"✅ Retrieved {len(tweets):,} tweets\n")

    # Filter out tweets already in the database
    existing_tweet_ids = set(db.metadata.keys())
    tweets_before = len(tweets)
    tweets = [t for t in tweets if t["tweet_id"] not in existing_tweet_ids]
    if tweets_before > len(tweets):
        print(f"⏭️  Skipped {tweets_before - len(tweets):,} tweets already in database")
        print(f"📝 Processing {len(tweets):,} new tweets\n")

    if len(tweets) == 0:
        print("✅ No new tweets to process!")
        return

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

    # Step 3: Fetch parent tweets for replies
    print("🔗 Fetching parent tweets for replies...")
    parent_tweet_ids = [t["reply_to_tweet_id"] for t in tweets if t.get("reply_to_tweet_id")]
    unique_parent_ids = list(set(parent_tweet_ids))
    parent_tweets_map = {}

    if unique_parent_ids:
        # Fetch parent tweets in batches
        batch_size = 1000
        for i in range(0, len(unique_parent_ids), batch_size):
            batch_ids = unique_parent_ids[i:i+batch_size]
            response = supabase.table("tweets").select("tweet_id, full_text").in_("tweet_id", batch_ids).execute()

            for parent in response.data:
                parent_tweets_map[parent["tweet_id"]] = parent["full_text"]

            if (i + batch_size) % 10000 == 0 or (i + batch_size) >= len(unique_parent_ids):
                print(f"   Fetched {min(i + batch_size, len(unique_parent_ids)):,}/{len(unique_parent_ids):,} parent tweets...")

    print(f"✅ Found {len(parent_tweets_map):,} parent tweets\n")

    # Helper: Create contextual text with reply context
    def create_contextual_text(tweet_text: str, parent_text: str = None) -> str:
        """
        Create embedding text with reply context

        Examples:
        - Original tweet: "Great point!"
        - With parent: "AI is transforming coding\n\n[REPLY]: Great point!"
        """
        cleaned_tweet = clean_text(tweet_text)

        if parent_text:
            cleaned_parent = clean_text(parent_text)
            # Include parent context for better semantic understanding
            return f"{cleaned_parent}\n\n[REPLY]: {cleaned_tweet}"

        return cleaned_tweet

    # Step 4: Prepare embeddings data with reply context
    print("🔄 Preparing embeddings with reply context...")
    texts_to_embed = []
    metadata_list = []

    for tweet in tweets:
        # Create contextual text (with parent if it's a reply)
        parent_text = None
        if tweet.get("reply_to_tweet_id"):
            parent_text = parent_tweets_map.get(tweet["reply_to_tweet_id"])

        contextual_text = create_contextual_text(tweet["full_text"], parent_text)

        if not contextual_text:
            continue  # Skip empty tweets after cleaning

        texts_to_embed.append(contextual_text)

        metadata_list.append({
            "tweet_id": tweet["tweet_id"],
            "full_text": tweet["full_text"][:5000],
            "thread_context": contextual_text[:10000],  # Store the full context used for embedding
            "reply_to_tweet_id": tweet.get("reply_to_tweet_id", ""),
            "parent_text": parent_text[:5000] if parent_text else "",
            "thread_root_id": "",  # Not available in database
            "depth": 0,  # Not available in database
            "is_root": not bool(tweet.get("reply_to_tweet_id")),  # Root if not a reply
            "account_id": tweet.get("account_id", ""),
            "account_username": username_map.get(tweet.get("account_id", ""), tweet.get("account_id", "")),
            "favorite_count": tweet.get("favorite_count", 0),
            "retweet_count": tweet.get("retweet_count", 0),
            "created_at": tweet.get("created_at", ""),
            "processed_at": datetime.now().isoformat(),
            "embedding_version": "voyage-3-reply-context-v1"
        })

    print(f"✅ Prepared {len(texts_to_embed):,} embeddings with reply context\n")

    # Step 4: Generate embeddings in batches
    print("🤖 Generating Voyage AI embeddings...")
    BATCH_SIZE = 128

    all_embeddings = []
    all_metadata = []

    for i in range(0, len(texts_to_embed), BATCH_SIZE):
        batch_texts = texts_to_embed[i:i+BATCH_SIZE]
        batch_metadata = metadata_list[i:i+BATCH_SIZE]
        batch_num = i//BATCH_SIZE + 1
        total_batches = (len(texts_to_embed) + BATCH_SIZE - 1)//BATCH_SIZE

        if batch_num % 100 == 0 or batch_num == 1:
            print(f"   Batch {batch_num}/{total_batches} ({len(batch_texts)} texts)...")

        batch_embeddings = generate_voyage_embeddings.remote(batch_texts)
        all_embeddings.extend(batch_embeddings)
        all_metadata.extend(batch_metadata)

    print(f"✅ Generated {len(all_embeddings):,} embeddings\n")

    # Step 5: Add all embeddings to database
    print(f"🗄️  Adding {len(all_embeddings):,} embeddings to database...")
    db.add(all_embeddings, all_metadata)

    # Step 6: Final save and commit
    print("💾 Final save to persistent storage...")
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
# PRE-COMPUTE ALL TOPICS
# ============================================================================

@app.function(
    image=image,
    secrets=[secrets],
    volumes={"/data": vector_volume},
    timeout=7200,  # 2 hours for full computation
)
def precompute_all_topics(min_tweets: int = 30):
    """
    Pre-compute topics for ALL users with at least min_tweets
    This takes 30-60 minutes but makes all future requests instant!

    Usage: modal run modal_app.py::precompute_all_topics
    """
    import numpy as np
    from sklearn.cluster import KMeans
    from collections import Counter
    import re
    from openai import OpenAI

    print(f"🚀 Starting pre-computation of topics for all users with {min_tweets}+ tweets...\n")

    # Load vector DB
    db = VectorDB()
    if not db.load("/data"):
        print("❌ Error: No vector database found. Run sync first!")
        return {"error": "No vector database found"}

    print(f"✅ Loaded vector database with {len(db.metadata):,} tweets\n")

    # Load or create topics cache
    topics_cache = TopicsCache()
    topics_cache.load("/data")
    print(f"📂 Loaded existing cache with {len(topics_cache.topics)} users\n")

    # Get all users with at least min_tweets
    username_counts = Counter()
    for meta in db.metadata:
        username = meta.get("account_username")
        if username:
            username_counts[username] += 1

    qualified_users = [
        username for username, count in username_counts.items()
        if count >= min_tweets
    ]

    qualified_users.sort()  # Alphabetical order

    print(f"👥 Found {len(qualified_users)} users with {min_tweets}+ tweets\n")
    print(f"💰 Estimated cost: ${len(qualified_users) * 0.001:.2f} (OpenAI API calls)")
    print(f"⏱️  Estimated time: {len(qualified_users) * 10 / 60:.1f} minutes\n")
    print("="*60)

    # Import the topic computation logic (we'll call the endpoint internally)
    # For now, let me create a helper function that replicates the logic

    def compute_topics_for_user(username: str, num_topics: int = 5):
        """Compute topics for a single user (extracted from endpoint logic)"""

        def is_substantive_tweet(text: str) -> bool:
            if len(text.strip()) < 15:
                return False
            words = text.strip().split()
            if len(words) <= 2:
                return False
            low_quality_patterns = [
                'based', 'lol', 'lmao', 'haha', 'lmfao', 'true',
                'same', 'yep', 'nope', 'yeah', 'nah', 'agreed',
                'this', 'nice', 'cool', 'wow', 'omg', 'wtf'
            ]
            text_lower = text.lower().strip()
            if text_lower in low_quality_patterns:
                return False
            if text.startswith('@') and len(words) <= 3:
                return False
            return True

        def extract_topic_name(cluster_tweets):
            """Extract topic name from cluster tweets using LLM analysis with quality validation"""
            import os

            # Check if tweets have enough substance
            avg_length = sum(len(tweet) for tweet in cluster_tweets) / len(cluster_tweets)
            if avg_length < 40:
                return None  # Skip topic naming for low-quality clusters

            # Use up to 15 sample tweets for better context
            sample_size = min(15, len(cluster_tweets))
            sample_tweets = cluster_tweets[:sample_size]

            # Calculate metrics to help LLM assess quality
            unique_words = len(set(' '.join(sample_tweets).lower().split()))

            # Create prompt for LLM with quality check
            tweets_text = "\n".join([f"{i+1}. {tweet}" for i, tweet in enumerate(sample_tweets)])

            prompt = f"""Analyze these {sample_size} tweets from a cluster:

{tweets_text}

Cluster Statistics:
- Average tweet length: {avg_length:.0f} characters
- Unique words: {unique_words}

Task: Determine if this is a COHERENT topic cluster or just NOISE (random short replies).

RED FLAGS for noise:
- Mostly very short replies like "based", "lol", "true", "agreed"
- No substantive discussion or common theme
- Just reactions/mentions without content
- Tweets don't actually relate to each other

If this is NOISE (low-quality cluster), respond: SKIP

If this is a COHERENT topic cluster:
1. Create a clear, specific topic name (2-5 words)
2. Be descriptive and accurate
3. Base it on the actual content, not assumptions

Good topic examples:
- "Artificial Intelligence Research"
- "Climate Change Policy"
- "Software Engineering Best Practices"

Respond with ONLY the topic name or "SKIP"."""

            try:
                client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

                response = client.chat.completions.create(
                    model="gpt-4o-mini",  # Fast and cheap model
                    messages=[
                        {"role": "system", "content": "You are an expert at identifying meaningful topics from tweet clusters and filtering out noise. You are critical and will reject low-quality clusters that are just short replies or reactions."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=30,
                    temperature=0.1  # Very low temperature for consistent quality filtering
                )

                topic_name = response.choices[0].message.content.strip()
                # Remove any quotes that might be added
                topic_name = topic_name.strip('"\'')

                # If LLM says to skip, return None
                if topic_name.upper() == "SKIP" or topic_name.upper() == "NOISE":
                    return None

                return topic_name if topic_name else None

            except Exception as e:
                print(f"Error generating topic name with LLM: {e}")
                # Fallback to simple keyword extraction
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'this', 'that'}
                all_text = ' '.join(cluster_tweets).lower()
                words = re.findall(r'\b[a-z]{4,}\b', all_text)  # At least 4 letters
                meaningful_words = [w for w in words if w not in stop_words]
                word_counts = Counter(meaningful_words)
                if len(word_counts) == 0:
                    return None
                top_words = [word for word, count in word_counts.most_common(2)]
                return " & ".join(word.capitalize() for word in top_words) if top_words else None

        # Get user's tweets
        user_embeddings = []
        user_tweets = []
        user_tweet_data = []
        user_indices = []

        for idx, meta in enumerate(db.metadata):
            if meta.get("account_username") == username:
                tweet_text = meta.get("full_text", "")
                if not is_substantive_tweet(tweet_text):
                    continue
                user_indices.append(idx)
                user_tweets.append(tweet_text)
                user_tweet_data.append({
                    "text": tweet_text,
                    "tweet_id": meta.get("tweet_id"),
                    "username": meta.get("account_username")
                })

        if len(user_indices) < 5:
            return None  # Not enough substantive tweets

        # Reconstruct embeddings
        for idx in user_indices:
            embedding = db.index.reconstruct(int(idx))
            user_embeddings.append(embedding)

        user_embeddings = np.array(user_embeddings)

        # Cluster
        n_clusters = min(num_topics, len(user_embeddings))
        if n_clusters < 2:
            return None

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(user_embeddings)

        # Process clusters (simplified - no LLM refinement for speed)
        topics = []
        for cluster_id in range(n_clusters):
            cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]

            if len(cluster_indices) == 0:
                continue

            cluster_tweets = [user_tweets[i] for i in cluster_indices]
            cluster_embeddings = user_embeddings[cluster_indices]

            # Skip if average tweet length too short
            avg_length = sum(len(tweet) for tweet in cluster_tweets) / len(cluster_tweets)
            if avg_length < 40 or len(cluster_tweets) < 5:
                continue

            # Calculate centroid and rank tweets
            centroid = np.mean(cluster_embeddings, axis=0)
            centroid_norm = centroid / np.linalg.norm(centroid)

            similarities = []
            original_indices = [cluster_indices[i] for i in range(len(cluster_indices))]
            for i, embedding in enumerate(cluster_embeddings):
                embedding_norm = embedding / np.linalg.norm(embedding)
                similarity = np.dot(embedding_norm, centroid_norm)
                tweet_data = user_tweet_data[original_indices[i]]
                similarities.append((similarity, tweet_data))

            similarities.sort(key=lambda x: x[0], reverse=True)

            ranked_tweets = [
                {
                    "text": tweet_data["text"],
                    "similarity": float(sim),
                    "tweet_id": tweet_data["tweet_id"],
                    "username": tweet_data["username"]
                }
                for sim, tweet_data in similarities
            ]

            # Use LLM-based topic naming with quality validation
            topic_name = extract_topic_name(cluster_tweets)
            if topic_name is None:
                # Skip low-quality clusters
                continue

            topics.append({
                "topic_id": cluster_id,
                "topic_name": topic_name,
                "tweets": ranked_tweets,
                "tweet_count": len(cluster_tweets)
            })

        if len(topics) == 0:
            return None

        topics.sort(key=lambda x: x["tweet_count"], reverse=True)

        return {
            "username": username,
            "tweet_count": len(user_tweets),
            "topics": topics
        }

    # Pre-compute for all users
    success_count = 0
    skip_count = 0
    error_count = 0

    for i, username in enumerate(qualified_users):
        print(f"[{i+1}/{len(qualified_users)}] Processing {username}...", end=" ")

        # Skip if already cached
        if topics_cache.has(username):
            print("✓ Already cached")
            success_count += 1
            continue

        try:
            result = compute_topics_for_user(username)

            if result is None:
                print("⊘ Skipped (not enough substantive tweets)")
                skip_count += 1
            else:
                topics_cache.set(username, result)
                print(f"✓ Cached {len(result['topics'])} topics")
                success_count += 1

                # Save every 10 users to avoid losing progress
                if (i + 1) % 10 == 0:
                    topics_cache.save("/data")
                    vector_volume.commit()
                    print(f"   💾 Progress saved ({success_count} users cached)")

        except Exception as e:
            print(f"✗ Error: {e}")
            error_count += 1

    # Generate users list for /users endpoint
    print("\n📊 Generating users list...")
    users_list = []
    for username, topics_data in topics_cache.topics.items():
        if topics_data and "topics" in topics_data:
            users_list.append({
                "username": username,
                "tweet_count": topics_data.get("tweet_count", 0),
                "topic_count": len(topics_data.get("topics", []))
            })

    # Sort by tweet count
    users_list.sort(key=lambda x: x["tweet_count"], reverse=True)
    topics_cache.set_users_list(users_list)
    print(f"✓ Generated users list with {len(users_list)} users")

    # Final save
    topics_cache.save("/data")
    vector_volume.commit()

    print("\n" + "="*60)
    print(f"✅ Pre-computation complete!")
    print(f"   ✓ Successfully cached: {success_count} users")
    print(f"   ⊘ Skipped: {skip_count} users")
    print(f"   ✗ Errors: {error_count} users")
    print(f"   💾 Topics cache size: {len(topics_cache.topics)} users")
    print(f"   📊 Users list size: {len(users_list)} users")
    print("="*60)

    return {
        "status": "success",
        "cached_users": success_count,
        "skipped_users": skip_count,
        "errors": error_count,
        "total_in_cache": len(topics_cache.topics),
        "users_list_size": len(users_list)
    }


@app.function(
    image=image,
    secrets=[secrets],
    volumes={"/data": vector_volume},
    timeout=60,
)
def clear_topics_cache():
    """Clear the topics cache to force recomputation

    Usage: modal run modal_app.py::clear_topics_cache
    """
    import os
    cache_path = "/data/topics_cache.pkl"

    if os.path.exists(cache_path):
        os.remove(cache_path)
        vector_volume.commit()
        print("✅ Topics cache cleared!")
        return {"status": "success", "message": "Cache cleared"}
    else:
        print("⚠️  No cache file found")
        return {"status": "success", "message": "No cache file to clear"}


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
        rerank: bool = False  # Enable LLM-based re-ranking for better semantic relevance

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

    # Topics cache for lazy computation
    _topics_cache = {"cache": None, "last_check": 0}
    def get_topics_cache():
        """Get or load the topics cache (with periodic reloading)"""
        import time
        current_time = time.time()

        # Reload cache every 5 minutes to pick up new pre-computations
        if _topics_cache["cache"] is None or (current_time - _topics_cache["last_check"]) > 300:
            print(f"{'Loading' if _topics_cache['cache'] is None else 'Reloading'} topics cache...")
            # Reload volume to see changes from other containers (like precompute_all_topics)
            vector_volume.reload()
            cache = TopicsCache()
            cache.load("/data")
            print(f"Topics cache loaded with {len(cache.topics)} users")
            _topics_cache["cache"] = cache
            _topics_cache["last_check"] = current_time
        return _topics_cache["cache"]

    @web_app.get("/users")
    async def get_users(min_tweets: int = 30):
        """Get list of all users in the index with at least min_tweets tweets (cached)"""
        # Check if we have a pre-computed users list
        topics_cache = get_topics_cache()
        users_list = topics_cache.get_users_list()

        if users_list is not None:
            # Serve from cache (instant!)
            print(f"✅ Serving users list from cache ({len(users_list)} users)")
            # Filter by min_tweets if different from default
            filtered_users = [u for u in users_list if u["tweet_count"] >= min_tweets]
            return {
                "users": filtered_users,
                "total_users": len(filtered_users),
                "min_tweets": min_tweets
            }

        # Fallback: compute users list on-the-fly if cache doesn't have it yet
        print("⚠️  Users list not cached yet, computing on-the-fly...")
        db = get_vector_db()
        from collections import Counter
        username_counts = Counter()
        for meta in db.metadata:
            username = meta.get("account_username")
            if username:
                username_counts[username] += 1

        users_list = []
        for username, count in username_counts.items():
            if count >= min_tweets:
                users_list.append({
                    "username": username,
                    "tweet_count": count,
                    "topic_count": 0  # Unknown until topics are computed
                })

        users_list.sort(key=lambda x: x["tweet_count"], reverse=True)

        return {
            "users": users_list,
            "total_users": len(users_list),
            "min_tweets": min_tweets,
            "cached": False
        }

    @web_app.get("/user/{username}/topics")
    async def get_user_topics(username: str, num_topics: int = 5):
        """Get top topics for a specific user using clustering (with caching)"""
        import numpy as np
        from sklearn.cluster import KMeans
        from collections import Counter
        import re

        # Check if topics are already cached
        topics_cache = get_topics_cache()
        if topics_cache.has(username):
            print(f"✅ Serving cached topics for {username}")
            return topics_cache.get(username)

        # Get cached vector DB
        db = get_vector_db()
        if db is None:
            return {"error": "vector database not initialized"}

        def extract_topic_name(cluster_tweets):
            """Extract topic name from cluster tweets using LLM analysis with quality validation"""
            from openai import OpenAI

            # Check if tweets have enough substance
            avg_length = sum(len(tweet) for tweet in cluster_tweets) / len(cluster_tweets)
            if avg_length < 40:
                return None  # Skip topic naming for low-quality clusters

            # Use up to 15 sample tweets for better context
            sample_size = min(15, len(cluster_tweets))
            sample_tweets = cluster_tweets[:sample_size]

            # Calculate metrics to help LLM assess quality
            unique_words = len(set(' '.join(sample_tweets).lower().split()))

            # Create prompt for LLM with quality check
            tweets_text = "\n".join([f"{i+1}. {tweet}" for i, tweet in enumerate(sample_tweets)])

            prompt = f"""Analyze these {sample_size} tweets from a cluster:

{tweets_text}

Cluster Statistics:
- Average tweet length: {avg_length:.0f} characters
- Unique words: {unique_words}

Task: Determine if this is a COHERENT topic cluster or just NOISE (random short replies).

RED FLAGS for noise:
- Mostly very short replies like "based", "lol", "true", "agreed"
- No substantive discussion or common theme
- Just reactions/mentions without content
- Tweets don't actually relate to each other

If this is NOISE (low-quality cluster), respond: SKIP

If this is a COHERENT topic cluster:
1. Create a clear, specific topic name (2-5 words)
2. Be descriptive and accurate
3. Base it on the actual content, not assumptions

Good topic examples:
- "Artificial Intelligence Research"
- "Climate Change Policy"
- "Software Engineering Best Practices"

Respond with ONLY the topic name or "SKIP"."""

            try:
                client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

                response = client.chat.completions.create(
                    model="gpt-4o-mini",  # Fast and cheap model
                    messages=[
                        {"role": "system", "content": "You are an expert at identifying meaningful topics from tweet clusters and filtering out noise. You are critical and will reject low-quality clusters that are just short replies or reactions."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=30,
                    temperature=0.1  # Very low temperature for consistent quality filtering
                )

                topic_name = response.choices[0].message.content.strip()
                # Remove any quotes that might be added
                topic_name = topic_name.strip('"\'')

                # If LLM says to skip, return None
                if topic_name.upper() == "SKIP" or topic_name.upper() == "NOISE":
                    return None

                return topic_name if topic_name else None

            except Exception as e:
                print(f"Error generating topic name with LLM: {e}")
                # Fallback to simple keyword extraction
                from collections import Counter
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'this', 'that'}
                all_text = ' '.join(cluster_tweets).lower()
                words = re.findall(r'\b[a-z]{4,}\b', all_text)  # At least 4 letters
                meaningful_words = [w for w in words if w not in stop_words]
                word_counts = Counter(meaningful_words)
                if len(word_counts) == 0:
                    return None
                top_words = [word for word, count in word_counts.most_common(2)]
                return " & ".join(word.capitalize() for word in top_words) if top_words else None


        def is_substantive_tweet(text: str) -> bool:
            """Filter out low-quality tweets that shouldn't be clustered"""
            # Remove very short tweets (< 15 characters)
            if len(text.strip()) < 15:
                return False

            # Remove single-word or two-word replies
            words = text.strip().split()
            if len(words) <= 2:
                return False

            # Check for common short reply patterns
            low_quality_patterns = [
                'based', 'lol', 'lmao', 'haha', 'lmfao', 'true',
                'same', 'yep', 'nope', 'yeah', 'nah', 'agreed',
                'this', 'nice', 'cool', 'wow', 'omg', 'wtf'
            ]
            text_lower = text.lower().strip()
            if text_lower in low_quality_patterns:
                return False

            # If it's just a mention with one word, skip it
            if text.startswith('@') and len(words) <= 3:
                return False

            return True

        # Get all tweets and embeddings for this user
        user_embeddings = []
        user_tweets = []
        user_tweet_data = []  # Store full tweet data including IDs
        user_indices = []

        for idx, meta in enumerate(db.metadata):
            if meta.get("account_username") == username:
                tweet_text = meta.get("full_text", "")

                # Filter out low-quality tweets before clustering
                if not is_substantive_tweet(tweet_text):
                    continue

                user_indices.append(idx)
                user_tweets.append(tweet_text)
                # Store full data for later use
                user_tweet_data.append({
                    "text": tweet_text,
                    "tweet_id": meta.get("tweet_id"),
                    "username": meta.get("account_username")
                })

        if len(user_indices) == 0:
            return {"error": f"No substantive tweets found for user {username}"}

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

            # Calculate average tweet length in cluster
            avg_length = sum(len(tweet) for tweet in cluster_tweets) / len(cluster_tweets)

            # Skip clusters where average tweet is too short (likely just replies)
            if avg_length < 30:
                continue

            # Refine cluster using LLM to remove noise
            refined_tweets = refine_cluster_with_llm(cluster_tweets)

            # Skip clusters that are mostly noise
            if len(refined_tweets) < 5:
                continue

            # Check refined tweets average length too
            refined_avg_length = sum(len(tweet) for tweet in refined_tweets) / len(refined_tweets)
            if refined_avg_length < 40:
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

            # Skip clusters where we couldn't generate a good topic name
            if topic_name is None:
                continue

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

        # Build result
        result = {
            "username": username,
            "tweet_count": len(user_tweets),
            "topics": topics
        }

        # Cache the result for future requests
        print(f"💾 Caching topics for {username}")
        topics_cache.set(username, result)
        # Persist to disk (async write)
        try:
            topics_cache.save("/data")
        except Exception as e:
            print(f"Warning: Failed to save topics cache: {e}")

        return result

    def llm_rerank_results(query: str, results: list, top_k: int = 10):
        """Use LLM to re-rank search results by semantic relevance"""
        from openai import OpenAI
        import json

        if len(results) == 0:
            return []

        # Take top candidates for re-ranking (more than final limit for better selection)
        candidates = results[:min(20, len(results))]

        # Prepare tweets for LLM evaluation
        tweets_text = ""
        for i, r in enumerate(candidates):
            tweets_text += f"{i+1}. @{r.get('account_username', 'unknown')}: {r.get('full_text', '')}\n\n"

        prompt = f"""You are evaluating the semantic relevance of tweets to a search query.

Query: "{query}"

Tweets:
{tweets_text}

Task: Rank these tweets by TRUE semantic relevance to the query intent (not just keyword matching).

Consider:
- Does the tweet meaningfully address the query topic?
- Is it a substantive discussion vs. a short reaction?
- Does it provide useful information related to the query?
- Context and nuance matter more than exact word matches

Return ONLY a JSON array of tweet numbers (1-{len(candidates)}) in order from MOST to LEAST relevant.
Example: [3, 1, 7, 2, ...]

Return the top {top_k} most relevant tweet numbers."""

        try:
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at evaluating semantic relevance of text to search queries. You understand context and meaning beyond keyword matching."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.1
            )

            # Parse LLM response
            llm_output = response.choices[0].message.content.strip()
            # Extract JSON array from response
            import re
            json_match = re.search(r'\[[\d,\s]+\]', llm_output)
            if json_match:
                ranked_indices = json.loads(json_match.group())
                # Convert 1-indexed to 0-indexed and reorder results
                reranked = []
                for idx in ranked_indices[:top_k]:
                    if 1 <= idx <= len(candidates):
                        reranked.append(candidates[idx - 1])
                return reranked
            else:
                print(f"Failed to parse LLM ranking: {llm_output}")
                return results[:top_k]

        except Exception as e:
            print(f"LLM re-ranking error: {e}")
            # Fallback to original ranking
            return results[:top_k]

    @web_app.post("/search")
    async def search_endpoint(request: SearchRequest):
        """Search API endpoint with optional LLM re-ranking"""
        query_text = request.query
        limit = request.limit
        use_rerank = request.rerank

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

        # Search with larger limit if re-ranking (to get better candidates)
        search_limit = limit * 2 if use_rerank else limit
        results = db.search(query_embedding, limit=search_limit)

        # Apply LLM re-ranking if requested
        if use_rerank and len(results) > 0:
            print(f"🔄 Re-ranking {len(results)} results with LLM...")
            results = llm_rerank_results(query_text, results, top_k=limit)

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
            "total_vectors": db.count(),
            "reranked": use_rerank
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
    import os

    print(f"🔍 Inspecting CoreNN database...")

    # Load CoreNN database and metadata
    db = VectorDB()

    if not db.load("/data"):
        return {"error": "No CoreNN database found at /data"}

    print(f"✅ Loaded CoreNN database with {len(db.metadata):,} vectors\n")

    # Show database info
    print("📊 Database Information:")
    print(f"   Dimensions: {db.dimension}")
    print(f"   Total vectors: {len(db.metadata):,}")
    print(f"   Database type: CoreNN (billion-scale vector search)\n")

    # Show sample embeddings
    print(f"📝 Sample Embeddings (first {num_samples}):")
    sample_count = min(num_samples, len(db.metadata))

    # Get first N tweet_ids from metadata
    sample_tweet_ids = list(db.metadata.keys())[:sample_count]

    for i, tweet_id in enumerate(sample_tweet_ids):
        meta = db.metadata[tweet_id]

        print(f"\n--- Vector {i} (Tweet ID: {tweet_id}) ---")
        print(f"Original tweet: {meta.get('full_text', 'N/A')[:150]}...")
        print(f"Cleaned/embedded text: {meta.get('thread_context', 'N/A')[:150]}...")
        print(f"Author: {meta.get('account_username', 'N/A')}")
        print(f"Created: {meta.get('created_at', 'N/A')}")
        print(f"Engagement: {meta.get('retweet_count', 0)} RTs, {meta.get('favorite_count', 0)} likes")
        print(f"Embedding version: {meta.get('embedding_version', 'N/A')}")

    return {
        "status": "success",
        "total_vectors": len(db.metadata),
        "dimensions": db.dimension,
        "index_type": "CoreNN",
        "samples_shown": sample_count
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
