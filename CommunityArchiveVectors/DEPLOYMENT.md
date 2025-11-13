# Community Archive - Deployment Guide

This guide explains how to deploy the entire Community Archive infrastructure, including the database and semantic search functionality.

## Architecture Overview

The system consists of three main components:

1. **Frontend (Next.js)** - Hosted on Vercel
2. **Database (Supabase/PostgreSQL)** - Stores tweet data, user accounts, and metadata
3. **Semantic Search Backend (Optional)** - Vector search server for semantic tweet search

## Prerequisites

- Node.js 18+ and npm
- Python 3.11+
- Modal account (for data processing)
- Vercel account (for frontend hosting)
- Supabase account OR self-hosted PostgreSQL
- A VPS server (if deploying semantic search)

---

## Part 1: Database Setup (Supabase)

### Option A: Using Supabase (Recommended)

1. **Create a Supabase Project**
   - Go to [https://supabase.com](https://supabase.com)
   - Create a new project
   - Note your project URL and anon key

2. **Database Schema**

   Create the following tables in your Supabase SQL editor:

   ```sql
   -- Account table
   CREATE TABLE account (
     account_id BIGINT PRIMARY KEY,
     username TEXT NOT NULL,
     account_display_name TEXT,
     created_at TIMESTAMP DEFAULT NOW()
   );

   -- Tweets table
   CREATE TABLE tweets (
     tweet_id BIGINT PRIMARY KEY,
     account_id BIGINT REFERENCES account(account_id),
     full_text TEXT NOT NULL,
     created_at TIMESTAMP,
     retweet_count INTEGER DEFAULT 0,
     favorite_count INTEGER DEFAULT 0,
     reply_to_tweet_id BIGINT REFERENCES tweets(tweet_id)
   );

   -- Create indexes for performance
   CREATE INDEX idx_tweets_account_id ON tweets(account_id);
   CREATE INDEX idx_tweets_created_at ON tweets(created_at);
   CREATE INDEX idx_tweets_reply_to ON tweets(reply_to_tweet_id);
   CREATE INDEX idx_account_username ON account(username);
   ```

3. **Import Your Tweet Data**

   Use Supabase's bulk import feature or the API to populate the tables with your tweet data.

### Option B: Self-Hosted PostgreSQL

1. Install PostgreSQL 14+
2. Create a database named `community_archive`
3. Run the same schema SQL from Option A
4. Configure connection credentials
5. Update frontend `.env.local` with your database credentials

---

## Part 2: Frontend Deployment

### Local Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd CommunityArchiveVectors/frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Configure environment variables**

   Create `frontend/.env.local`:
   ```env
   NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
   NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key-here
   ```

4. **Test locally**
   ```bash
   npm run dev
   ```
   Visit http://localhost:3000

### Deploy to Vercel

1. **Install Vercel CLI**
   ```bash
   npm install -g vercel
   ```

2. **Build the project**
   ```bash
   npm run build
   ```

3. **Deploy**
   ```bash
   npx vercel --prod
   ```

4. **Configure Environment Variables in Vercel**
   - Go to your project settings on Vercel dashboard
   - Navigate to Settings → Environment Variables
   - Add:
     - `NEXT_PUBLIC_SUPABASE_URL`
     - `NEXT_PUBLIC_SUPABASE_ANON_KEY`
   - Redeploy after adding variables

---

## Part 3: Data Processing with Modal

The project uses Modal for processing large datasets. All data processing scripts are already set up.

### Setup Modal

1. **Install Modal**
   ```bash
   pip install modal
   ```

2. **Authenticate**
   ```bash
   modal token new
   ```

3. **Create Modal Volume**

   The project uses a volume named `tweet-vectors-large`:
   ```bash
   modal volume create tweet-vectors-large
   ```

### Available Data Processing Scripts

- `organize_vectors_by_community.py` - Organizes tweet embeddings by community
- `cluster_with_filtering_2020_2024.py` - Creates topic clusters with filtering
- `recluster_2024_topics.py` - Regenerates 2024 topics with LLM filtering
- `community_topics.py` - Generates topics for communities
- `align_communities_temporally.py` - Aligns communities across time

### Run a Processing Script

```bash
source modal_env/bin/activate
modal run recluster_2024_topics.py
```

---

## Part 4: Semantic Search Backend (Optional)

The semantic search feature requires a separate backend server with vector similarity search capabilities.

### Architecture

- **Input**: Natural language search query
- **Processing**:
  1. Encode query into embedding vector (using sentence-transformers)
  2. Perform cosine similarity search against 6.4M tweet embeddings
  3. Return top K most similar tweets
- **Output**: Ranked list of tweets with similarity scores

### Requirements

- Server with 32GB+ RAM (to hold embeddings in memory)
- 100GB+ disk space
- Python 3.11+
- GPU optional (speeds up encoding)

### Step-by-Step Deployment

#### 1. Provision a Server

**Recommended Providers:**
- Vultr Cloud Compute - High Performance (45.63.18.97 was used previously)
- DigitalOcean Droplets
- AWS EC2 (c5.4xlarge or similar)
- Hetzner Cloud

**Specifications:**
- 8 vCPUs
- 32GB RAM minimum
- 100GB SSD
- Ubuntu 22.04 LTS

#### 2. Transfer Embeddings to Server

From the Modal environment, run:

```bash
python3 direct_transfer_to_vultr.py
```

Or manually:
```bash
# On your Modal volume
rsync -av --progress /data/batches/ root@YOUR_SERVER_IP:/root/tweet-search/embeddings/batches/
```

This transfers ~56.5 GB of embedding data.

#### 3. Install Dependencies on Server

SSH into your server:
```bash
ssh root@YOUR_SERVER_IP
```

Install Python and dependencies:
```bash
apt update
apt install -y python3.11 python3-pip nginx

pip3 install fastapi uvicorn numpy sentence-transformers torch psycopg2-binary
```

#### 4. Create the Search API Server

Create `/root/tweet-search/server.py`:

```python
#!/usr/bin/env python3
"""
Semantic Search API for Community Archive
Performs vector similarity search over tweet embeddings
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
import psycopg2
from typing import List, Dict
import time

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for cached data
embeddings_matrix = None
tweet_metadata = None
model = None
db_config = None

# Database configuration
DB_CONFIG = {
    "host": "YOUR_SUPABASE_HOST",
    "database": "postgres",
    "user": "postgres",
    "password": "YOUR_PASSWORD",
    "port": 5432
}

@app.on_event("startup")
async def load_embeddings():
    """Load embeddings and model on startup"""
    global embeddings_matrix, tweet_metadata, model

    print("Loading embeddings from disk...")
    start = time.time()

    # Load all embedding batches
    batches_dir = Path("/root/tweet-search/embeddings/batches")
    all_embeddings = []
    all_metadata = []

    for batch_file in sorted(batches_dir.glob("batch_*.pkl")):
        with open(batch_file, 'rb') as f:
            batch_data = pickle.load(f)
            all_embeddings.append(batch_data['embeddings'])
            all_metadata.extend(batch_data['metadata'])

    embeddings_matrix = np.vstack(all_embeddings)
    tweet_metadata = all_metadata

    print(f"✓ Loaded {len(tweet_metadata):,} embeddings ({embeddings_matrix.shape}) in {time.time()-start:.1f}s")

    # Load sentence transformer model
    print("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ Model loaded")

@app.get("/")
def root():
    return {
        "service": "Community Archive Semantic Search",
        "status": "running",
        "database_size": len(tweet_metadata) if tweet_metadata else 0,
        "endpoints": {
            "/search": "Semantic search - GET with query and limit params"
        }
    }

@app.get("/search")
async def semantic_search(
    query: str = Query(..., description="Search query"),
    limit: int = Query(20, ge=1, le=100)
) -> Dict:
    """
    Perform semantic search over tweet embeddings

    Returns tweets ranked by semantic similarity to the query
    """
    start_time = time.time()

    # Encode the query
    query_embedding = model.encode([query])[0]
    query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize

    # Normalize embeddings if not already
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    normalized_embeddings = embeddings_matrix / norms

    # Compute cosine similarities
    similarities = np.dot(normalized_embeddings, query_embedding)

    # Get top K indices
    top_indices = np.argsort(similarities)[::-1][:limit]

    # Get tweet IDs for top matches
    tweet_ids = [tweet_metadata[idx]['tweet_id'] for idx in top_indices]

    # Fetch full tweet data from database
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

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

    # Handle replies - fetch parent tweets
    parent_tweet_ids = [t['reply_to_tweet_id'] for t in tweet_data.values() if t['reply_to_tweet_id']]

    if parent_tweet_ids:
        cur.execute("""
            SELECT
                t.tweet_id,
                t.full_text,
                a.username
            FROM tweets t
            JOIN account a ON t.account_id = a.account_id
            WHERE t.tweet_id = ANY(%s)
        """, (parent_tweet_ids,))

        for row in cur.fetchall():
            parent_id, parent_text, parent_username = row
            for tweet in tweet_data.values():
                if tweet['reply_to_tweet_id'] == parent_id:
                    tweet['parent_tweet_text'] = parent_text
                    tweet['parent_tweet_username'] = parent_username

    cur.close()
    conn.close()

    # Build result set in order of similarity
    results = []
    for idx in top_indices:
        tweet_id = tweet_metadata[idx]['tweet_id']
        if tweet_id in tweet_data:
            result = tweet_data[tweet_id]
            result['similarity'] = float(similarities[idx])
            results.append(result)

    search_time_ms = (time.time() - start_time) * 1000

    return {
        "query": query,
        "results": results,
        "search_time_ms": round(search_time_ms, 2),
        "database_size": len(tweet_metadata)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=80)
```

#### 5. Configure the Server

Update the database credentials in `server.py`:
```python
DB_CONFIG = {
    "host": "db.your-project.supabase.co",
    "database": "postgres",
    "user": "postgres",
    "password": "YOUR_SUPABASE_PASSWORD",
    "port": 5432
}
```

#### 6. Create a Systemd Service

Create `/etc/systemd/system/tweet-search.service`:

```ini
[Unit]
Description=Community Archive Semantic Search API
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/tweet-search
ExecStart=/usr/bin/python3 /root/tweet-search/server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
systemctl daemon-reload
systemctl enable tweet-search
systemctl start tweet-search
systemctl status tweet-search
```

#### 7. Configure Firewall

```bash
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 22/tcp
ufw enable
```

#### 8. (Optional) Setup HTTPS with Nginx

Create `/etc/nginx/sites-available/tweet-search`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

Enable the site:
```bash
ln -s /etc/nginx/sites-available/tweet-search /etc/nginx/sites-enabled/
nginx -t
systemctl reload nginx
```

#### 9. Update Frontend Configuration

In your Vercel environment variables or `.env.local`, add:

```env
BACKEND_URL=http://YOUR_SERVER_IP
```

Or if using a domain:
```env
BACKEND_URL=https://your-api-domain.com
```

Update `frontend/src/app/api/search/route.ts` line 5:
```typescript
const BACKEND_URL = process.env.BACKEND_URL || 'http://YOUR_SERVER_IP'
```

Redeploy the frontend.

#### 10. Test the Search API

```bash
curl "http://YOUR_SERVER_IP/search?query=climate%20change&limit=5"
```

### Performance Tuning

- **Memory**: Embeddings require ~24GB RAM for 6.4M tweets
- **Speed**: First query takes ~2-3 seconds (loading time), subsequent queries ~200-500ms
- **Scaling**: For better performance, consider:
  - Using FAISS for approximate nearest neighbor search
  - Adding Redis for caching frequent queries
  - Using GPU for faster encoding

---

## Part 5: Monitoring and Maintenance

### Health Checks

Create a simple monitoring script:

```python
# monitor.py
import requests
import time

ENDPOINTS = [
    "https://your-frontend.vercel.app",
    "http://YOUR_SERVER_IP/",
    "https://your-project.supabase.co/rest/v1/"
]

for endpoint in ENDPOINTS:
    try:
        r = requests.get(endpoint, timeout=5)
        print(f"✓ {endpoint}: {r.status_code}")
    except Exception as e:
        print(f"✗ {endpoint}: {e}")
```

### Logs

- **Frontend**: Check Vercel dashboard → Logs
- **Backend**: `journalctl -u tweet-search -f`
- **Database**: Supabase dashboard → Logs

### Backups

- **Database**: Supabase automatic backups (Pro plan) or use `pg_dump`
- **Embeddings**: Keep original source in Modal volume

---

## Cost Estimates

### Minimal Setup (No Semantic Search)
- Frontend: $0 (Vercel hobby tier)
- Database: $25/month (Supabase Pro)
- **Total: $25/month**

### Full Setup (With Semantic Search)
- Frontend: $0-$20/month (Vercel)
- Database: $25/month (Supabase Pro)
- VPS Server: $40-$80/month (32GB RAM)
- **Total: $65-$125/month**

---

## Troubleshooting

### Frontend Issues

**Error: "No tweets found"**
- Check Supabase credentials in Vercel environment variables
- Verify tables exist and have data
- Check Vercel logs for errors

**Build failures**
- Clear `.next` folder: `rm -rf .next`
- Delete `node_modules` and reinstall: `rm -rf node_modules && npm install`

### Backend Issues

**Server not responding**
- Check service status: `systemctl status tweet-search`
- Check logs: `journalctl -u tweet-search -n 100`
- Verify port 80 is not blocked

**Out of memory**
- Increase server RAM to 32GB+
- Reduce batch sizes when loading embeddings
- Use swap space as temporary solution

**Slow search**
- Check server CPU/RAM usage
- Consider using FAISS for faster similarity search
- Add query caching with Redis

---

## Support

For questions or issues:
- Check existing GitHub issues
- Create a new issue with detailed logs
- Include: OS version, Python version, error messages

---

## License

[Your license here]
