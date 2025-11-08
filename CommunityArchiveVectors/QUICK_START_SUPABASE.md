# QUICK START - Get Live in 3 Hours

Follow these steps to deploy your tweet search on Supabase TODAY.

## Timeline

- **Step 1:** Setup database (10 mins) ⚡
- **Step 2:** Upload embeddings (2-3 hours) 🔄 *runs automatically*
- **Step 3:** Deploy API (10 mins) ⚡
- **Step 4:** Deploy frontend (5 mins) ⚡
- **Total:** ~3 hours (mostly automated upload)

---

## Step 1: Setup Supabase Database (10 minutes)

### 1.1 Open Supabase SQL Editor

1. Go to https://supabase.com/dashboard
2. Select your project
3. Click "SQL Editor" (left sidebar)
4. Click "New Query"

### 1.2 Run Setup SQL

Copy ALL of `supabase_setup.sql` and paste it into the SQL editor, then click "Run".

This creates:
- ✅ pgvector extension
- ✅ embedding column (1024 dimensions)
- ✅ HNSW index (for fast search)
- ✅ search_tweets() function

### 1.3 Verify

Run this query to confirm:

```sql
SELECT COUNT(*) FROM tweets;
```

You should see ~6.4M tweets.

---

## Step 2: Upload Embeddings (2-3 hours)

This step runs automatically in the background. You can leave it running and move on to Step 3.

### 2.1 Start Upload

```bash
source modal_env/bin/activate
modal run upload_embeddings_to_supabase.py --start 1 --end 64
```

This will:
- Upload all 64 batch files (~100K embeddings each)
- Show progress: "Uploaded 10,000/100,000..."
- Take 2-3 hours total
- **Runs automatically - you can do other things!**

### 2.2 Monitor Progress (Optional)

In another terminal, check progress in Supabase:

```sql
SELECT
  COUNT(embedding) as embeddings_uploaded,
  ROUND(100.0 * COUNT(embedding) / COUNT(*), 1) as percent_complete
FROM tweets;
```

You'll see the count increase as batches upload!

### 2.3 Test with First Batch (Optional - 3 minutes)

Want to test before uploading all? Just upload batch 1:

```bash
modal run upload_embeddings_to_supabase.py --batch 1
```

Then test search with 100K tweets to verify everything works!

---

## Step 3: Deploy API (10 minutes)

While embeddings upload, deploy your API!

### Option A: Vercel (Recommended - Free)

```bash
# Install Vercel CLI
npm install -g vercel

# Login
vercel login

# Create project directory
mkdir tweet-search-api
cd tweet-search-api

# Copy API file
cp ../supabase_api.py main.py

# Create requirements.txt
cat > requirements.txt <<EOF
fastapi==0.104.1
voyageai==0.2.1
supabase==2.0.0
pydantic==2.5.0
EOF

# Create vercel.json
cat > vercel.json <<EOF
{
  "builds": [{
    "src": "main.py",
    "use": "@vercel/python"
  }]
}
EOF

# Deploy!
vercel --prod
```

### Add Environment Variables in Vercel

Go to your Vercel dashboard → Settings → Environment Variables:

- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_KEY`: Your Supabase anon key
- `VOYAGE_API_KEY`: Your Voyage AI key

Redeploy after adding variables:

```bash
vercel --prod
```

**Done!** Your API is live at `https://your-project.vercel.app`

---

## Step 4: Deploy Frontend (5 minutes)

### 4.1 Update Frontend Config

```bash
cd frontend/

# Create .env.local
echo "NEXT_PUBLIC_API_URL=https://your-api-url.vercel.app" > .env.local
```

### 4.2 Deploy to Vercel

```bash
vercel --prod
```

Add environment variable in Vercel dashboard:
- `NEXT_PUBLIC_API_URL`: Your API URL from Step 3

**Done!** Your frontend is live!

---

## Step 5: Test Everything (2 minutes)

### 5.1 Test API Directly

```bash
curl "https://your-api.vercel.app/search?query=artificial%20intelligence&limit=5"
```

You should see JSON results!

### 5.2 Test Frontend

1. Open your frontend URL
2. Type "artificial intelligence"
3. Click "Search"
4. See results appear!

---

## What to Expect

### Performance

- **First search:** 2-3 seconds
- **Subsequent searches:** 1-2 seconds
- **Totally acceptable!** Users won't notice the difference

### Costs

- Supabase Pro: $25/month
- Vercel (API + Frontend): Free tier
- Voyage AI queries: ~$5-10/month
- **Total: ~$30-40/month** (vs $267 on Modal!)

---

## Troubleshooting

### "No results found"

**Cause:** Embeddings still uploading (Step 2 not complete)

**Solution:** Wait for upload to finish. Check progress:

```sql
SELECT COUNT(embedding) FROM tweets;
```

When you see ~6.4M, all embeddings are uploaded!

### "Slow searches (>5 seconds)"

**Cause:** Index still building

**Solution:** Wait 30 minutes after upload completes for index to build. Check:

```sql
SELECT * FROM pg_stat_progress_create_index;
```

If empty, index is built!

### "API returns 500 error"

**Cause:** Missing environment variables

**Solution:** Check Vercel dashboard → Settings → Environment Variables. Make sure all 3 are set:
- SUPABASE_URL
- SUPABASE_KEY
- VOYAGE_API_KEY

---

## Next Steps After Going Live

1. **Monitor usage** - Check Supabase dashboard
2. **Optimize if needed** - Adjust index if searches are slow
3. **Add features:**
   - Date filtering
   - Username search
   - Tweet filtering (replies, retweets, etc.)
4. **Share with users!**

---

## Summary

You've built a production tweet search system with:

✅ **6.4M tweet embeddings** in Supabase
✅ **Fast vector search** with pgvector HNSW index
✅ **Serverless API** on Vercel
✅ **Modern frontend** on Vercel
✅ **~$30-40/month** total cost
✅ **Zero server management**

Congratulations! 🎉
