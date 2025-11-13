# Incremental Update System

Automatically keeps your tweet archive up-to-date as new tweets are added to Supabase.

## Overview

The incremental update system consists of two scripts that work together:

1. **`incremental_update_pipeline.py`** - Processes new tweets
   - Detects new tweets in Supabase
   - Generates embeddings
   - Assigns to communities
   - Updates `organized_by_community.pkl`

2. **`regenerate_topics_incremental.py`** - Updates topic clusters
   - Re-clusters updated communities
   - Regenerates topic labels with LLM
   - Updates topic JSON files

## How It Works

```
New Tweets Added to Supabase
    ↓
[Daily at 2 AM UTC] OR [Manual Run]
    ↓
1. incremental_update_pipeline.py
   - Queries tweets with ID > last_processed_tweet_id
   - Generates embeddings with sentence-transformers
   - Matches to communities via network data
   - Updates organized_by_community.pkl
   - Saves last_processed_tweet_id
    ↓
2. regenerate_topics_incremental.py (optional)
   - Re-clusters affected communities
   - Generates new topic labels
   - Updates topics_year_XXXX_summary.json files
    ↓
3. Download & Deploy
   - Copy updated JSON files to frontend
   - Redeploy
```

---

## Setup

### 1. Configure Database Credentials

The scripts need access to your Supabase database. Set environment variables in Modal:

```bash
# Method A: Via Modal secrets
modal secret create supabase-credentials \
  SUPABASE_HOST="YOUR_PROJECT.supabase.co" \
  SUPABASE_PASSWORD="your-postgres-password"

# Method B: Edit the scripts directly
# Update DB_CONFIG in incremental_update_pipeline.py
```

### 2. Ensure Network Data Exists

The system needs network animation data to assign tweets to communities:

```bash
# Check if file exists in Modal volume
modal run -c "ls -la /data/network_animation_data.json"

# If missing, upload it:
modal volume put tweet-vectors-large network_animation_data.json /network_animation_data.json
```

### 3. Initialize Tracking (First Run Only)

The system tracks the last processed tweet ID. On first run, it will automatically detect the highest tweet ID in your existing data.

---

## Usage

### Automatic (Recommended)

The incremental update runs automatically every day at 2 AM UTC via Modal's cron scheduler.

**Deploy the cron job:**

```bash
# Deploy to Modal (makes it run automatically)
modal deploy incremental_update_pipeline.py
```

That's it! The system will now:
- Check for new tweets daily at 2 AM UTC
- Process any new tweets found
- Update the organized data
- Track progress automatically

**Monitor the cron jobs:**

```bash
# View scheduled runs
modal app list

# View logs
modal app logs incremental-update-pipeline
```

### Manual Run

You can also run updates manually anytime:

```bash
# 1. Process new tweets
modal run incremental_update_pipeline.py

# 2. (Optional) Regenerate topics for updated communities
modal run regenerate_topics_incremental.py

# 3. Download updated files
modal volume get tweet-vectors-large /topics_year_2024_summary.json \
  frontend/public/data/topics_year_2024_summary.json

# 4. Redeploy frontend
cd frontend && npx vercel --prod
```

---

## What Gets Updated

### After `incremental_update_pipeline.py`:

- ✅ `organized_by_community.pkl` - Contains all tweets + embeddings organized by (year, month, community)
- ✅ `last_processed_tweet.json` - Tracking file with last processed tweet ID
- ✅ Tweets are assigned to correct communities
- ✅ Embeddings are generated and added

### After `regenerate_topics_incremental.py`:

- ✅ `topics_year_XXXX_summary.json` - Topic clusters for each year
- ✅ New topics identified by LLM
- ✅ Existing topics updated with new tweets
- ✅ Sample tweets refreshed

---

## Performance

### incremental_update_pipeline.py

**Processing speed:**
- 1,000 tweets: ~1-2 minutes
- 10,000 tweets: ~5-10 minutes
- 100,000 tweets: ~30-60 minutes

**Resource usage:**
- Memory: 16GB
- CPU: Moderate
- API calls: None (uses local model)

### regenerate_topics_incremental.py

**Processing speed:**
- Per community: ~10-20 seconds
- 100 communities: ~20-30 minutes
- Full year: ~30-60 minutes

**Resource usage:**
- Memory: 8GB
- CPU: 4 cores
- API calls: Claude Haiku (~$0.001 per community)

---

## Monitoring

### Check Last Update

```python
# check_status.py
import modal

app = modal.App("check-status")
volume = modal.Volume.from_name("tweet-vectors-large")

@app.function(volumes={"/data": volume})
def check_status():
    import json

    with open("/data/last_processed_tweet.json", 'r') as f:
        tracking = json.load(f)

    print(f"Last tweet ID: {tracking['last_tweet_id']}")
    print(f"Last update: {tracking['last_update']}")
    print(f"Tweets processed: {tracking['tweets_processed']}")

@app.local_entrypoint()
def main():
    check_status.remote()
```

```bash
modal run check_status.py
```

### View Logs

```bash
# View cron job logs
modal app logs incremental-update-pipeline --follow

# View manual run logs
modal app logs incremental-update-pipeline --since 1h
```

---

## Deployment Workflow

### For Daily Automatic Updates (Set and Forget)

```bash
# 1. Deploy the cron job once
modal deploy incremental_update_pipeline.py

# 2. (Optional) Set up weekly topic regeneration
# Edit regenerate_topics_incremental.py to add:
# @app.function(schedule=modal.Cron("0 3 * * 0"))  # Sundays at 3 AM
# Then: modal deploy regenerate_topics_incremental.py
```

### For Manual Control

```bash
# Run whenever you want updates
modal run incremental_update_pipeline.py
modal run regenerate_topics_incremental.py

# Download and deploy
python3 sync_to_frontend.py  # Create this script to automate downloads
cd frontend && npx vercel --prod
```

---

## Advanced: CoreNN Database Updates

**Note:** The current implementation does NOT update the CoreNN database incrementally due to performance issues with large incremental adds.

If you want semantic search to include new tweets, you have two options:

### Option A: Periodic Full Rebuild (Recommended)

Rebuild CoreNN database from scratch monthly:

```bash
# Once per month
modal run _archive_experimental/offline_builder.py
```

This ensures optimal performance.

### Option B: Incremental CoreNN Updates (Slower)

Add to `incremental_update_pipeline.py`:

```python
# After Step 7 (saving organized data)
# Add Step 8: Update CoreNN database

from corenn_py import CoreNN

db = CoreNN.open("/data/corenn_db")

# Add new tweet embeddings
for tweet, embedding in zip(new_tweets, embeddings):
    db.insert_f32([tweet['tweet_id']], embedding.reshape(1, -1))

# Note: This gets slower as database grows
```

---

## Troubleshooting

### "No new tweets found" but I know there are new tweets

Check:
1. Is `last_processed_tweet.json` correct?
2. Are new tweets in Supabase with higher tweet_ids?
3. Database connection working?

```bash
# Reset tracking to reprocess recent tweets
modal run -c "rm /data/last_processed_tweet.json"
```

### "Cannot assign to communities"

The `network_animation_data.json` file is missing or incomplete.

```bash
# Check if it exists
modal volume get tweet-vectors-large /network_animation_data.json ./check.json

# If missing, upload it
modal volume put tweet-vectors-large network_animation_data.json /network_animation_data.json
```

### "Out of memory" errors

Increase memory allocation in the script:

```python
@app.function(
    memory=32768,  # Increase from 16384 to 32768 (32GB)
    ...
)
```

### Embeddings quality issues

The default model is `all-MiniLM-L6-v2` (384-dim, fast, free).

To match original Voyage AI embeddings (1024-dim):
- Switch back to Voyage AI in the script
- Note: Requires API key and costs ~$0.001 per 1000 tweets

---

## Cost Estimates

### Automatic Daily Updates (Cron)

Assuming 1,000 new tweets per day:

- **Modal compute**: ~$0.01/day ($0.30/month)
- **Claude API** (if regenerating topics weekly): ~$0.10/week ($0.40/month)
- **Total**: ~$0.70/month

### Manual Weekly Updates

Assuming 7,000 new tweets per week:

- **Modal compute**: ~$0.05/week ($0.20/month)
- **Claude API**: ~$0.10/week ($0.40/month)
- **Total**: ~$0.60/month

**Very affordable!**

---

## Summary

You now have:
- ✅ Automatic daily tweet ingestion
- ✅ Embedding generation
- ✅ Community assignment
- ✅ Topic clustering
- ✅ Full system stays up-to-date
- ✅ Minimal manual intervention needed

Just deploy the cron job once and forget about it! Your archive will automatically stay current.
