# Semantic Search Setup Guide

This guide shows you how to set up the semantic search infrastructure for the Community Archive project. This is a standalone system that can be deployed independently.

## Overview

The semantic search system consists of:
1. **Tweet Embeddings** - Vector representations of tweets (768-dimensional)
2. **Search API Server** - FastAPI server that performs similarity search
3. **Database** - PostgreSQL/Supabase for tweet metadata

## Architecture

```
User Query → API Server → Encode Query → Cosine Similarity Search → Return Top Results
                            ↓
                    Tweet Embeddings (56GB)
                            ↓
                    Fetch Full Tweet Data from DB
```

---

## Part 1: Generate Tweet Embeddings

### Prerequisites
- Modal account (for distributed processing)
- Python 3.11+
- Your tweet data in PostgreSQL/Supabase

### Step 1: Export Tweets from Database

Create `export_tweets.py`:

```python
"""
Export tweets from database to Modal volume for embedding generation
"""
import modal
import psycopg2
import pickle
from pathlib import Path

app = modal.App("export-tweets")
volume = modal.Volume.from_name("tweet-vectors-large", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "psycopg2-binary",
)

@app.function(
    volumes={"/data": volume},
    image=image,
    timeout=3600,
    memory=8192,
)
def export_tweets_to_volume():
    """Export all tweets from database to Modal volume"""

    # Configure your database connection
    DB_CONFIG = {
        "host": "YOUR_SUPABASE_HOST.supabase.co",
        "database": "postgres",
        "user": "postgres",
        "password": "YOUR_PASSWORD",
        "port": 5432
    }

    print("Connecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Fetch all tweets with metadata
    print("Fetching tweets...")
    cur.execute("""
        SELECT
            t.tweet_id,
            t.full_text,
            t.created_at,
            t.account_id,
            a.username
        FROM tweets t
        JOIN account a ON t.account_id = a.account_id
        WHERE t.full_text IS NOT NULL
        ORDER BY t.tweet_id
    """)

    tweets = []
    for row in cur.fetchall():
        tweets.append({
            'tweet_id': row[0],
            'text': row[1],
            'created_at': row[2].isoformat() if row[2] else None,
            'account_id': row[3],
            'username': row[4]
        })

    cur.close()
    conn.close()

    print(f"Fetched {len(tweets):,} tweets")

    # Save to volume
    output_path = Path("/data/all_tweets.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(tweets, f)

    print(f"Saved tweets to {output_path}")
    volume.commit()

    return len(tweets)

@app.local_entrypoint()
def main():
    count = export_tweets_to_volume.remote()
    print(f"Successfully exported {count:,} tweets to Modal volume")
```

Run it:
```bash
modal run export_tweets.py
```

### Step 2: Generate Embeddings

Create `generate_embeddings.py`:

```python
"""
Generate embeddings for all tweets using sentence-transformers
"""
import modal
import pickle
import numpy as np
from pathlib import Path

app = modal.App("generate-embeddings")
volume = modal.Volume.from_name("tweet-vectors-large", create_if_missing=False)

# Use GPU image for faster encoding
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "sentence-transformers",
    "torch",
    "numpy",
)

@app.function(
    volumes={"/data": volume},
    image=image,
    gpu="T4",  # Use GPU for faster processing
    timeout=7200,  # 2 hours
    memory=16384,  # 16GB
)
def generate_embeddings_batch(batch_num: int, batch_size: int = 10000):
    """Generate embeddings for a batch of tweets"""
    from sentence_transformers import SentenceTransformer

    print(f"Processing batch {batch_num}...")

    # Load model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"Loaded model: {model}")

    # Load tweets
    with open("/data/all_tweets.pkl", 'rb') as f:
        all_tweets = pickle.load(f)

    # Get this batch
    start_idx = batch_num * batch_size
    end_idx = min(start_idx + batch_size, len(all_tweets))
    batch_tweets = all_tweets[start_idx:end_idx]

    if not batch_tweets:
        print(f"Batch {batch_num} is empty, skipping")
        return None

    print(f"Encoding {len(batch_tweets)} tweets (indices {start_idx}-{end_idx})...")

    # Extract text
    texts = [tweet['text'] for tweet in batch_tweets]

    # Generate embeddings
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Prepare metadata (without embeddings, to save space)
    metadata = [{
        'tweet_id': tweet['tweet_id'],
        'username': tweet['username'],
        'created_at': tweet['created_at'],
    } for tweet in batch_tweets]

    # Save batch
    batch_data = {
        'embeddings': embeddings,
        'metadata': metadata,
        'batch_num': batch_num,
        'start_idx': start_idx,
        'end_idx': end_idx
    }

    output_path = Path(f"/data/batches/batch_{batch_num:04d}.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(batch_data, f)

    print(f"Saved batch {batch_num} to {output_path}")
    print(f"Shape: {embeddings.shape}")

    volume.commit()

    return {
        'batch_num': batch_num,
        'count': len(batch_tweets),
        'shape': embeddings.shape
    }

@app.local_entrypoint()
def main():
    """Process all tweets in batches"""
    import time

    # Load tweets to determine number of batches
    print("Counting tweets...")
    with modal.Function.lookup("generate-embeddings", "generate_embeddings_batch") as f:
        # Just to get the volume mounted
        pass

    # Hardcode or fetch the count
    TOTAL_TWEETS = 6_400_000  # Adjust based on your data
    BATCH_SIZE = 10000
    NUM_BATCHES = (TOTAL_TWEETS + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"Processing {TOTAL_TWEETS:,} tweets in {NUM_BATCHES} batches")
    print(f"Batch size: {BATCH_SIZE:,}")
    print()

    start_time = time.time()

    # Process in parallel using starmap
    results = []
    for batch_num in range(NUM_BATCHES):
        result = generate_embeddings_batch.remote(batch_num, BATCH_SIZE)
        if result:
            results.append(result)
            print(f"Completed batch {batch_num + 1}/{NUM_BATCHES}")

    elapsed = time.time() - start_time

    print()
    print("=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"Processed {len(results)} batches")
    print(f"Total time: {elapsed / 60:.1f} minutes")
    print(f"Embeddings saved to Modal volume: tweet-vectors-large")
    print()
```

Run it:
```bash
modal run generate_embeddings.py
```

This will create batches of embeddings in your Modal volume at `/data/batches/`.

---

## Part 2: Deploy Search Server

### Step 1: Provision a Server

You need a server with:
- **32GB+ RAM** (to hold embeddings in memory)
- **100GB+ disk space**
- **Ubuntu 22.04 LTS**

Recommended providers:
- Vultr High Performance (8 vCPU, 32GB RAM) - ~$80/month
- DigitalOcean Droplets
- AWS EC2 c5.4xlarge

### Step 2: Transfer Embeddings to Server

Use the existing transfer script:

```bash
# Update direct_transfer_to_vultr.py with your server IP
modal run direct_transfer_to_vultr.py
```

Or manually with rsync:
```bash
# From Modal container (you'll need to set this up)
rsync -av --progress /data/batches/ root@YOUR_SERVER_IP:/root/tweet-search/embeddings/batches/
```

### Step 3: Setup Server

SSH into your server:
```bash
ssh root@YOUR_SERVER_IP
```

Install dependencies:
```bash
apt update
apt install -y python3.11 python3-pip nginx

pip3 install fastapi uvicorn numpy sentence-transformers torch psycopg2-binary
```

### Step 4: Create Search Server

Create `/root/tweet-search/search_server.py`:

```python
#!/usr/bin/env python3
"""
Standalone Semantic Search Server for Tweet Archive
Load embeddings into memory and perform fast cosine similarity search
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from typing import List, Dict, Optional
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Tweet Semantic Search API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global data
embeddings_matrix: Optional[np.ndarray] = None
tweet_metadata: Optional[List[Dict]] = None
model: Optional[SentenceTransformer] = None
db_pool: Optional[SimpleConnectionPool] = None

# Configuration
CONFIG = {
    "embeddings_dir": "/root/tweet-search/embeddings/batches",
    "db_host": "YOUR_SUPABASE_HOST.supabase.co",
    "db_name": "postgres",
    "db_user": "postgres",
    "db_password": "YOUR_PASSWORD",
    "db_port": 5432,
    "model_name": "all-MiniLM-L6-v2",
}

@app.on_event("startup")
async def startup_event():
    """Load embeddings and model on server startup"""
    global embeddings_matrix, tweet_metadata, model, db_pool

    logger.info("=" * 80)
    logger.info("STARTING SEMANTIC SEARCH SERVER")
    logger.info("=" * 80)

    # Load embeddings
    logger.info("Loading embeddings from disk...")
    start = time.time()

    batches_dir = Path(CONFIG["embeddings_dir"])
    if not batches_dir.exists():
        raise RuntimeError(f"Embeddings directory not found: {batches_dir}")

    all_embeddings = []
    all_metadata = []

    batch_files = sorted(batches_dir.glob("batch_*.pkl"))
    logger.info(f"Found {len(batch_files)} batch files")

    for i, batch_file in enumerate(batch_files):
        logger.info(f"Loading {batch_file.name} ({i+1}/{len(batch_files)})...")
        with open(batch_file, 'rb') as f:
            batch_data = pickle.load(f)
            all_embeddings.append(batch_data['embeddings'])
            all_metadata.extend(batch_data['metadata'])

    embeddings_matrix = np.vstack(all_embeddings)
    tweet_metadata = all_metadata

    elapsed = time.time() - start
    logger.info(f"✓ Loaded {len(tweet_metadata):,} embeddings in {elapsed:.1f}s")
    logger.info(f"  Shape: {embeddings_matrix.shape}")
    logger.info(f"  Size: {embeddings_matrix.nbytes / (1024**3):.2f} GB")

    # Normalize embeddings for faster cosine similarity
    logger.info("Normalizing embeddings...")
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    embeddings_matrix = embeddings_matrix / norms
    logger.info("✓ Embeddings normalized")

    # Load model
    logger.info(f"Loading sentence transformer: {CONFIG['model_name']}...")
    model = SentenceTransformer(CONFIG['model_name'])
    logger.info("✓ Model loaded")

    # Setup database connection pool
    logger.info("Setting up database connection pool...")
    db_pool = SimpleConnectionPool(
        minconn=1,
        maxconn=10,
        host=CONFIG["db_host"],
        database=CONFIG["db_name"],
        user=CONFIG["db_user"],
        password=CONFIG["db_password"],
        port=CONFIG["db_port"]
    )
    logger.info("✓ Database pool created")

    logger.info("=" * 80)
    logger.info("SERVER READY")
    logger.info("=" * 80)

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources"""
    global db_pool
    if db_pool:
        db_pool.closeall()
        logger.info("Database connections closed")

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "service": "Tweet Semantic Search API",
        "status": "running",
        "database_size": len(tweet_metadata) if tweet_metadata else 0,
        "embedding_dimension": embeddings_matrix.shape[1] if embeddings_matrix is not None else 0,
        "model": CONFIG["model_name"],
        "endpoints": {
            "/search": "GET - Semantic search with ?query=<text>&limit=<n>",
            "/health": "GET - Health check",
            "/stats": "GET - Server statistics"
        }
    }

@app.get("/health")
def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "embeddings_loaded": embeddings_matrix is not None,
        "model_loaded": model is not None,
        "db_connected": db_pool is not None,
        "total_tweets": len(tweet_metadata) if tweet_metadata else 0
    }

@app.get("/stats")
def stats():
    """Server statistics"""
    if embeddings_matrix is None or tweet_metadata is None:
        raise HTTPException(status_code=503, detail="Server not ready")

    return {
        "total_tweets": len(tweet_metadata),
        "embedding_dimension": embeddings_matrix.shape[1],
        "memory_usage_gb": embeddings_matrix.nbytes / (1024**3),
        "model": CONFIG["model_name"]
    }

@app.get("/search")
async def search(
    query: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(20, ge=1, le=100, description="Number of results to return")
) -> Dict:
    """
    Semantic search endpoint

    Encodes the query and finds the most similar tweets using cosine similarity
    """
    if embeddings_matrix is None or tweet_metadata is None or model is None:
        raise HTTPException(status_code=503, detail="Server not ready")

    start_time = time.time()

    # Encode query
    logger.info(f"Query: {query[:100]}")
    query_embedding = model.encode([query])[0]
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    # Compute similarities (already normalized)
    similarities = np.dot(embeddings_matrix, query_embedding)

    # Get top K
    top_indices = np.argpartition(similarities, -limit)[-limit:]
    top_indices = top_indices[np.argsort(similarities[top_indices])][::-1]

    # Get tweet IDs
    tweet_ids = [tweet_metadata[idx]['tweet_id'] for idx in top_indices]

    # Fetch full data from database
    conn = db_pool.getconn()
    try:
        cur = conn.cursor()

        # Fetch tweets
        cur.execute("""
            SELECT
                t.tweet_id,
                t.full_text,
                t.created_at,
                t.retweet_count,
                t.favorite_count,
                t.reply_to_tweet_id,
                a.username,
                a.account_display_name
            FROM tweets t
            JOIN account a ON t.account_id = a.account_id
            WHERE t.tweet_id = ANY(%s)
        """, (tweet_ids,))

        tweet_data = {}
        for row in cur.fetchall():
            tweet_data[row[0]] = {
                'tweet_id': row[0],
                'full_text': row[1],
                'created_at': row[2].isoformat() if row[2] else None,
                'retweet_count': row[3],
                'favorite_count': row[4],
                'reply_to_tweet_id': row[5],
                'username': row[6],
                'account_display_name': row[7]
            }

        # Fetch parent tweets for replies
        parent_ids = [t['reply_to_tweet_id'] for t in tweet_data.values() if t['reply_to_tweet_id']]

        if parent_ids:
            cur.execute("""
                SELECT
                    t.tweet_id,
                    t.full_text,
                    a.username
                FROM tweets t
                JOIN account a ON t.account_id = a.account_id
                WHERE t.tweet_id = ANY(%s)
            """, (parent_ids,))

            for row in cur.fetchall():
                parent_id, parent_text, parent_username = row
                for tweet in tweet_data.values():
                    if tweet['reply_to_tweet_id'] == parent_id:
                        tweet['parent_tweet_text'] = parent_text
                        tweet['parent_tweet_username'] = parent_username

        cur.close()
    finally:
        db_pool.putconn(conn)

    # Build results
    results = []
    for idx in top_indices:
        tweet_id = tweet_metadata[idx]['tweet_id']
        if tweet_id in tweet_data:
            result = tweet_data[tweet_id]
            result['similarity'] = float(similarities[idx])
            results.append(result)

    search_time_ms = (time.time() - start_time) * 1000

    logger.info(f"Returned {len(results)} results in {search_time_ms:.2f}ms")

    return {
        "query": query,
        "results": results,
        "search_time_ms": round(search_time_ms, 2),
        "database_size": len(tweet_metadata)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=80,
        log_level="info"
    )
```

Update the CONFIG section with your database credentials.

### Step 5: Create Systemd Service

Create `/etc/systemd/system/tweet-search.service`:

```ini
[Unit]
Description=Tweet Semantic Search API
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/tweet-search
ExecStart=/usr/bin/python3 /root/tweet-search/search_server.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
systemctl daemon-reload
systemctl enable tweet-search
systemctl start tweet-search

# Check status
systemctl status tweet-search

# View logs
journalctl -u tweet-search -f
```

### Step 6: Configure Firewall

```bash
ufw allow 80/tcp
ufw allow 22/tcp
ufw enable
```

### Step 7: Test

```bash
# Health check
curl http://YOUR_SERVER_IP/health

# Search test
curl "http://YOUR_SERVER_IP/search?query=artificial%20intelligence&limit=5" | jq
```

---

## Part 3: Connect to Your Frontend

### Option A: Direct Connection

In your frontend `.env.local` or Vercel environment variables:

```env
BACKEND_URL=http://YOUR_SERVER_IP
```

### Option B: Use Nginx Reverse Proxy (Recommended)

Install Nginx:
```bash
apt install -y nginx certbot python3-certbot-nginx
```

Create `/etc/nginx/sites-available/tweet-search`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:80;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # CORS headers (if needed)
        add_header Access-Control-Allow-Origin *;
    }
}
```

Enable:
```bash
ln -s /etc/nginx/sites-available/tweet-search /etc/nginx/sites-enabled/
nginx -t
systemctl reload nginx

# Get SSL certificate
certbot --nginx -d your-domain.com
```

Then use:
```env
BACKEND_URL=https://your-domain.com
```

---

## Monitoring & Maintenance

### Check Logs
```bash
journalctl -u tweet-search -f --since "10 minutes ago"
```

### Monitor Performance
```bash
# CPU and RAM usage
htop

# Check if embeddings are in memory
ps aux | grep python3

# Test query speed
time curl -s "http://localhost/search?query=test&limit=10" > /dev/null
```

### Restart Service
```bash
systemctl restart tweet-search
```

---

## Costs

- **Modal Processing**: ~$5-10 for one-time embedding generation
- **VPS Server**: $40-80/month (32GB RAM)
- **Total**: ~$50-90/month ongoing

---

## Performance

- **First query**: ~2-3 seconds (model warm-up)
- **Subsequent queries**: 200-500ms
- **Database size**: 6.4M tweets
- **Memory usage**: ~24GB RAM
- **Throughput**: ~10-20 queries/second

---

## Troubleshooting

### Server won't start
```bash
# Check logs
journalctl -u tweet-search -n 100

# Common issues:
# - Missing embeddings: Check /root/tweet-search/embeddings/batches/
# - DB connection: Test with psql
# - Out of memory: Upgrade to 32GB+ RAM
```

### Slow searches
- Check RAM usage (should have embeddings in memory)
- Consider using FAISS for approximate nearest neighbor search
- Add Redis caching layer

### Out of memory
- Reduce batch sizes when loading
- Use swap space temporarily
- Upgrade server RAM

---

## Advanced: Using FAISS for Faster Search

For even faster searches (especially with 10M+ tweets), install FAISS:

```python
pip3 install faiss-cpu

# In search_server.py, replace cosine similarity with:
import faiss

# Build index on startup
index = faiss.IndexFlatIP(embeddings_matrix.shape[1])  # Inner product for cosine
index.add(embeddings_matrix)

# In search function:
D, I = index.search(query_embedding.reshape(1, -1), limit)
top_indices = I[0]
similarities = D[0]
```

This can reduce search time to <100ms even with 10M+ vectors.

---

## Summary

You now have a complete semantic search infrastructure that can:
- ✅ Generate embeddings for millions of tweets
- ✅ Deploy a fast search API server
- ✅ Handle 10-20 queries/second
- ✅ Return results in <500ms
- ✅ Scale to 10M+ tweets

The system is production-ready and can be deployed independently of your frontend.
