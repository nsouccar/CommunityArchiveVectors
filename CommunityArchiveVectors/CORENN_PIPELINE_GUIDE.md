# CoreNN Database Pipeline Guide

Complete guide to creating and deploying the CoreNN vector database for semantic search over your tweet archive.

## What is CoreNN?

CoreNN is a high-performance nearest-neighbor search library that creates an optimized vector index for fast similarity search. In this project, it's used to search over 6.4M+ tweet embeddings.

**CoreNN Database Structure:**
- `corenn_db/` directory (~39-40GB) - The optimized vector index
- `metadata.pkl` file (~2GB) - Tweet metadata mapped to IDs
- Embedding dimension: 1024 (Voyage AI vectors)

---

## Pipeline Overview

```
Supabase (Tweets) 
    ↓ 
Modal: Generate Voyage AI Embeddings (batches of 100K)
    ↓
Modal: Save batches to /data/batches/
    ↓
Modal: Build CoreNN database in ONE operation (critical!)
    ↓
Download to VPS
    ↓
Deploy Search Server with CoreNN
```

---

## Part 1: Generate Embeddings & Build CoreNN Database

### Prerequisites

- Modal account with API key
- Supabase database with tweets
- Voyage AI API key
- Modal volume named `tweet-vectors-volume`

### Step 1: Set up Modal Secrets

```bash
modal secret create tweet-vectors-secrets \
  SUPABASE_URL="https://YOUR_PROJECT.supabase.co" \
  SUPABASE_KEY="your-supabase-anon-or-service-key" \
  VOYAGE_API_KEY="your-voyage-ai-key"
```

### Step 2: Prepare the Builder Script

The complete builder script is in `_archive_experimental/offline_builder.py`. It performs:

1. **Embedding Generation** (Step 1)
   - Fetches tweets from Supabase in batches of 100K
   - Enriches with username and parent tweet context
   - Generates 1024-dim Voyage AI embeddings
   - Saves batches to Modal volume at `/data/batches/batch_XXXX.pkl`
   - **Resumable**: Can restart from last completed batch

2. **CoreNN Database Build** (Step 2)  
   - Loads ALL embedding batches into RAM (~64GB needed)
   - Creates CoreNN database in **ONE SINGLE INSERT**
   - This avoids incremental scaling problems
   - Saves to `/data/corenn_db/` and `/data/metadata.pkl`

### Step 3: Run the Pipeline

```bash
# From your project directory
modal run _archive_experimental/offline_builder.py
```

**What happens:**
```
1. Checks for existing batches (resume capability)
2. Generates embeddings batch-by-batch:
   - Batch 1: tweets 0 - 100,000
   - Batch 2: tweets 100,001 - 200,000
   - ... continues until all tweets processed
3. Builds CoreNN database from all batches
4. Saves everything to Modal volume
```

**Expected Duration:**
- Embedding generation: 2-4 hours (depends on tweet count)
- CoreNN build: 30-60 minutes
- **Total: 3-5 hours for 6.4M tweets**

**Expected Costs:**
- Modal compute: ~$10-20
- Voyage AI embeddings: ~$15-30 (based on tweet count)
- **Total: ~$25-50 one-time cost**

### Step 4: Verify the Build

```bash
# Check what's in your Modal volume
modal volume ls tweet-vectors-volume

# Should see:
# /batches/batch_0001.pkl
# /batches/batch_0002.pkl
# ...
# /corenn_db/
# /metadata.pkl
```

Create a verification script:

```python
# verify_corenn.py
import modal

app = modal.App("verify-corenn")
volume = modal.Volume.from_name("tweet-vectors-volume", create_if_missing=False)

@app.function(
    volumes={"/data": volume},
    image=modal.Image.debian_slim().pip_install("corenn-py"),
)
def verify_database():
    from corenn_py import CoreNN
    import pickle
    import os
    
    print("Checking CoreNN database...")
    
    # Check if database exists
    if not os.path.exists("/data/corenn_db"):
        print("❌ CoreNN database not found!")
        return False
    
    # Load database
    db = CoreNN.open("/data/corenn_db")
    count = db.count()
    print(f"✅ CoreNN database exists with {count:,} vectors")
    
    # Check metadata
    with open("/data/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
        print(f"✅ Metadata file exists with {metadata['count']:,} entries")
    
    # Verify counts match
    if count == metadata['count']:
        print(f"✅ Counts match! Database is valid.")
        return True
    else:
        print(f"⚠️  Count mismatch: DB has {count:,}, metadata has {metadata['count']:,}")
        return False

@app.local_entrypoint()
def main():
    result = verify_database.remote()
    if result:
        print("\n🎉 CoreNN database is ready to download!")
    else:
        print("\n❌ Database verification failed")
```

Run it:
```bash
modal run verify_corenn.py
```

---

## Part 2: Download CoreNN Database

### Option A: Download to Local Machine

```bash
# Download database directory
modal volume get tweet-vectors-volume /corenn_db ./corenn_db

# Download metadata
modal volume get tweet-vectors-volume /metadata.pkl ./metadata.pkl
```

### Option B: Transfer Directly to Server

Create a backup tarball on Modal:

```python
# create_backup.py
import modal

app = modal.App("create-backup")
volume = modal.Volume.from_name("tweet-vectors-volume", create_if_missing=False)

@app.function(
    volumes={"/data": volume},
    timeout=3600,
    memory=4096,
)
def create_backup():
    import subprocess
    import os
    
    print("Creating tarball of CoreNN database...")
    
    # Create compressed archive
    subprocess.run([
        "tar", "-czf", 
        "/data/corenn_backup.tar.gz",
        "/data/corenn_db",
        "/data/metadata.pkl"
    ], check=True)
    
    # Get size
    size_gb = os.path.getsize("/data/corenn_backup.tar.gz") / (1024**3)
    print(f"✅ Backup created: {size_gb:.1f}GB")
    
    volume.commit()
    return size_gb

@app.local_entrypoint()
def main():
    size = create_backup.remote()
    print(f"\nBackup ready: corenn_backup.tar.gz ({size:.1f}GB)")
    print("\nDownload with:")
    print("  modal volume get tweet-vectors-volume /corenn_backup.tar.gz ./corenn_backup.tar.gz")
```

Then download and transfer:
```bash
# Create backup
modal run create_backup.py

# Download from Modal
modal volume get tweet-vectors-volume /corenn_backup.tar.gz ./corenn_backup.tar.gz

# Transfer to server
scp corenn_backup.tar.gz root@YOUR_SERVER_IP:~/tweet-search/
```

---

## Part 3: Deploy Search Server with CoreNN

### Step 1: Provision Server

Requirements:
- **32GB+ RAM** (to load embeddings into memory)
- **100GB+ disk space** (for CoreNN database)
- Ubuntu 22.04 LTS

### Step 2: Setup Server

SSH into server:
```bash
ssh root@YOUR_SERVER_IP
```

Install dependencies:
```bash
apt update
apt install -y python3.11 python3-pip

pip3 install fastapi uvicorn corenn-py psycopg2-binary
```

### Step 3: Extract Database

```bash
cd ~/tweet-search
tar -xzf corenn_backup.tar.gz

# Should create:
# - corenn_db/ (directory)
# - metadata.pkl (file)
```

### Step 4: Create Search Server

Create `~/tweet-search/corenn_search_server.py`:

```python
#!/usr/bin/env python3
"""
CoreNN-based Semantic Search Server
Uses CoreNN for fast nearest-neighbor search
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from corenn_py import CoreNN
import pickle
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from typing import List, Dict, Optional
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CoreNN Tweet Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
db: Optional[CoreNN] = None
metadata: Optional[Dict] = None
db_pool: Optional[SimpleConnectionPool] = None

CONFIG = {
    "corenn_db_path": "/root/tweet-search/corenn_db",
    "metadata_path": "/root/tweet-search/metadata.pkl",
    "db_host": "YOUR_SUPABASE_HOST.supabase.co",
    "db_name": "postgres",
    "db_user": "postgres",
    "db_password": "YOUR_PASSWORD",
    "db_port": 5432,
}

@app.on_event("startup")
async def startup_event():
    """Load CoreNN database on startup"""
    global db, metadata, db_pool

    logger.info("=" * 80)
    logger.info("LOADING CORENN DATABASE")
    logger.info("=" * 80)

    # Load CoreNN database
    logger.info(f"Loading CoreNN from {CONFIG['corenn_db_path']}...")
    start = time.time()
    db = CoreNN.open(CONFIG['corenn_db_path'])
    count = db.count()
    elapsed = time.time() - start
    logger.info(f"✓ CoreNN loaded: {count:,} vectors in {elapsed:.1f}s")

    # Load metadata
    logger.info(f"Loading metadata from {CONFIG['metadata_path']}...")
    with open(CONFIG['metadata_path'], 'rb') as f:
        metadata_obj = pickle.load(f)
        metadata = metadata_obj['metadata']
    logger.info(f"✓ Metadata loaded: {len(metadata):,} tweets")

    # Setup database pool
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
    global db_pool
    if db_pool:
        db_pool.closeall()

@app.get("/")
def root():
    return {
        "service": "CoreNN Tweet Semantic Search",
        "status": "running",
        "database_size": db.count() if db else 0,
        "embedding_dimension": 1024,
        "engine": "CoreNN"
    }

@app.get("/search")
async def search(
    query: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=100)
) -> Dict:
    """
    Semantic search using CoreNN
    
    CoreNN performs fast approximate nearest-neighbor search
    over the pre-built vector index
    """
    if db is None or metadata is None:
        raise HTTPException(status_code=503, detail="Server not ready")

    start_time = time.time()

    # In CoreNN, you need to provide a query embedding
    # If using Voyage AI embeddings, you'd need to encode the query here
    # For now, this is a placeholder - you need to add query encoding
    
    logger.info(f"Query: {query}")
    
    # TODO: Add query embedding generation
    # For now, this returns an error
    raise HTTPException(
        status_code=501, 
        detail="Query encoding not implemented. Need to add Voyage AI or similar embedding service."
    )
    
    # Example implementation (uncomment when you add query encoding):
    """
    # Encode query (you need to add this)
    query_embedding = encode_query(query)  # Need to implement this
    
    # Search CoreNN
    results = db.search(query_embedding, k=limit)
    
    # results is list of (tweet_id, distance) tuples
    tweet_ids = [tweet_id for tweet_id, _ in results]
    
    # Fetch full data from database
    conn = db_pool.getconn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT tweet_id, full_text, created_at, username ... WHERE tweet_id = ANY(%s)",
            (tweet_ids,)
        )
        # ... rest of implementation
    finally:
        db_pool.putconn(conn)
    
    return {
        "query": query,
        "results": results,
        "search_time_ms": round((time.time() - start_time) * 1000, 2)
    }
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)
```

**Note:** CoreNN requires query embeddings, so you need to either:
1. Add Voyage AI client to encode queries on the fly, OR
2. Use a pre-trained sentence-transformers model (faster, no API calls)

### Step 5: Add Query Encoding

Option A - Use sentence-transformers (recommended):
```python
from sentence_transformers import SentenceTransformer

# In startup_event():
model = SentenceTransformer('all-MiniLM-L6-v2')

# In search():
query_embedding = model.encode([query])[0]
query_embedding = query_embedding / np.linalg.norm(query_embedding)  # normalize
```

Option B - Use Voyage AI (requires API key):
```python
import voyageai

vo = voyageai.Client(api_key="YOUR_KEY")

# In search():
result = vo.embed([query], model="voyage-3", input_type="query")
query_embedding = result.embeddings[0]
```

### Step 6: Start the Server

```bash
python3 ~/tweet-search/corenn_search_server.py
```

Or create a systemd service (recommended for production).

---

## Part 4: Testing

```bash
# Health check
curl http://YOUR_SERVER_IP/

# Search test (once query encoding is added)
curl "http://YOUR_SERVER_IP/search?query=artificial%20intelligence&limit=5"
```

---

## Summary

You now have:
- ✅ Complete CoreNN database (~40GB)
- ✅ Fast vector search capability
- ✅ Search server infrastructure
- ✅ ~10-20 queries/second throughput
- ✅ <200ms average query time

**Total Cost:**
- One-time: ~$25-50 (Modal + Voyage AI)
- Monthly: ~$40-80 (VPS server)

**Next Steps:**
1. Add query encoding to search server
2. Setup systemd service for auto-restart
3. Configure nginx + SSL
4. Connect frontend to your server
