# Supabase Deployment Guide - Tweet Search

Complete guide to deploy semantic search using Supabase pgvector.

## Why Supabase?

✅ **Your tweets are already there!**
✅ **No server management** - fully managed Postgres
✅ **Simple architecture** - API + frontend only
✅ **Cheap to start** - $25-50/month
✅ **Scales automatically** - handles 6.4M vectors easily

## Architecture

```
User → Frontend (Vercel) → API (Vercel/Render) → Supabase pgvector
                                                    ↓
                                              6.4M tweet embeddings
```

## Part 1: Setup Supabase Database (10 minutes)

### Step 1: Enable pgvector Extension

1. Go to your Supabase project: https://supabase.com/dashboard
2. Click "SQL Editor" in the left sidebar
3. Click "New Query"
4. Copy and paste the entire contents of `supabase_setup.sql`
5. Click "Run"

This will:
- Enable pgvector extension
- Add `embedding vector(1024)` column to tweets table
- Create HNSW index for fast vector search
- Create `search_tweets()` function for queries

### Step 2: Verify Setup

Run this query in SQL Editor:

```sql
SELECT
  COUNT(*) as total_tweets,
  COUNT(embedding) as tweets_with_embeddings
FROM tweets;
```

You should see:
- `total_tweets`: 6.4M
- `tweets_with_embeddings`: 0 (we'll upload them next)

## Part 2: Upload Embeddings from Modal (2-3 hours)

The embeddings are already in your Modal volume in 64 batch files. Let's upload them to Supabase:

### Step 1: List Available Batches

```bash
source modal_env/bin/activate
modal run upload_embeddings_to_supabase.py
```

This shows all 64 batch files (each ~900MB, containing ~100K embeddings).

### Step 2: Upload ALL Batches

Upload all 64 batches (this will take 2-3 hours):

```bash
modal run upload_embeddings_to_supabase.py --start 1 --end 64
```

Or if you want to test first, upload just the first batch:

```bash
modal run upload_embeddings_to_supabase.py --batch 1
```

**Progress monitoring:**
- Each batch uploads ~100K embeddings
- Takes ~2-3 minutes per batch
- Shows progress: "Uploaded 10,000/100,000 embeddings..."

### Step 3: Verify Upload

After upload completes, check in Supabase SQL Editor:

```sql
SELECT
  COUNT(*) as total_tweets,
  COUNT(embedding) as tweets_with_embeddings,
  ROUND(100.0 * COUNT(embedding) / COUNT(*), 2) as percentage_complete
FROM tweets;
```

You should see 6.4M tweets with embeddings!

### Step 4: Build Vector Index (30 minutes, one-time)

After all embeddings are uploaded, Supabase will automatically build the HNSW index. This takes ~30 minutes for 6.4M vectors but only needs to be done once.

Check index status:

```sql
SELECT
  schemaname,
  tablename,
  indexname,
  pg_size_pretty(pg_relation_size(indexrelid)) as index_size
FROM pg_stat_user_indexes
WHERE indexname = 'tweets_embedding_idx';
```

## Part 3: Deploy API (10 minutes)

You have two options for deploying the API:

### Option A: Deploy to Vercel (Easiest - Free Tier)

1. **Install Vercel CLI:**

```bash
npm install -g vercel
```

2. **Create Vercel project:**

```bash
mkdir tweet-search-api
cd tweet-search-api
cp ../supabase_api.py api/index.py
cp ../requirements_hetzner.txt requirements.txt
```

3. **Create `vercel.json`:**

```json
{
  "builds": [
    {
      "src": "api/index.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "api/index.py"
    }
  ]
}
```

4. **Deploy:**

```bash
vercel --prod
```

5. **Set environment variables in Vercel dashboard:**
   - `SUPABASE_URL`: Your Supabase URL
   - `SUPABASE_KEY`: Your Supabase anon/public key
   - `VOYAGE_API_KEY`: Your Voyage AI key

6. **Done!** Your API is live at `https://your-project.vercel.app`

### Option B: Deploy to Render (Alternative)

1. Push `supabase_api.py` to GitHub
2. Go to https://render.com
3. Create new "Web Service"
4. Connect your GitHub repo
5. Set:
   - **Build Command:** `pip install -r requirements_hetzner.txt`
   - **Start Command:** `uvicorn supabase_api:app --host 0.0.0.0 --port $PORT`
6. Add environment variables:
   - `SUPABASE_URL`
   - `SUPABASE_KEY`
   - `VOYAGE_API_KEY`
7. Deploy!

Cost: Free tier available, or $7/month for paid tier.

## Part 4: Deploy Frontend (5 minutes)

### Update Frontend to Use New API

1. **Update environment variable:**

```bash
cd frontend/
nano .env.local
```

Add:
```
NEXT_PUBLIC_API_URL=https://your-api.vercel.app
```

### Deploy to Vercel

```bash
cd frontend/
vercel --prod
```

Set environment variable in Vercel dashboard:
- `NEXT_PUBLIC_API_URL`: Your API URL from Part 3

**Done!** Your frontend is live!

## Part 5: Test End-to-End

1. Open your frontend URL (from Vercel)
2. Search for "artificial intelligence"
3. You should see results appear in 1-3 seconds!

## Performance Expectations

- **First search:** ~2-3 seconds (cold start)
- **Subsequent searches:** ~1-2 seconds
- **Index build time:** ~30 minutes (one-time, after upload)
- **Upload time:** ~2-3 hours (one-time)

## Cost Breakdown

| Service | Cost |
|---------|------|
| **Supabase Pro** | $25/month |
| **Vercel (API)** | Free tier OK |
| **Vercel (Frontend)** | Free tier OK |
| **Voyage AI** | ~$5-10/month (queries only) |
| **Total** | ~$30-40/month |

Compare to:
- Modal always-on: $267/month
- DigitalOcean 64GB: $240/month
- Hetzner dedicated: $48/month

## Monitoring & Maintenance

### Check Database Status

```sql
-- Count embeddings
SELECT COUNT(embedding) as total_embeddings FROM tweets WHERE embedding IS NOT NULL;

-- Check index size
SELECT pg_size_pretty(pg_total_relation_size('tweets_embedding_idx'));

-- Recent searches (check logs in Supabase dashboard)
```

### Upgrade Supabase If Needed

If searches are slow (>3 seconds), consider upgrading:
- **Supabase Pro:** $25/month (2 GB RAM)
- **Supabase Team:** $599/month (8 GB RAM) - for production scale

## Troubleshooting

### Slow Searches (>5 seconds)

**Solution:** Wait for index to finish building. Check:

```sql
SELECT * FROM pg_stat_progress_create_index;
```

If empty, index is built! If not, wait for it to complete.

### Upload Failing

**Solution:** Reduce batch size in `upload_embeddings_to_supabase.py`:

Change `CHUNK_SIZE = 1000` to `CHUNK_SIZE = 500`

### Out of Memory

**Solution:** Upgrade Supabase plan or reduce dimensions (not recommended for your use case).

## Next Steps

1. **Monitor usage** - Check Supabase dashboard for query stats
2. **Optimize if needed** - Adjust index parameters if searches are slow
3. **Add features** - Add filters (date range, username, etc.)
4. **Scale** - Supabase handles this automatically!

## Summary

✅ **Simple** - No server management
✅ **Fast** - 1-3 second searches
✅ **Cheap** - ~$30-40/month
✅ **Scalable** - Handles 6.4M vectors
✅ **Managed** - Supabase handles everything

You're now running a production tweet search system with minimal infrastructure!
