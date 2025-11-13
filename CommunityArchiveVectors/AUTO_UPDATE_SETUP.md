# Auto-Update Setup (5 Minutes)

Get automatic daily updates for your tweet archive.

## One-Time Setup

### 1. Set Database Credentials

```bash
# Add your Supabase credentials to Modal
modal secret create supabase-credentials \
  SUPABASE_HOST="YOUR_PROJECT.supabase.co" \
  SUPABASE_PASSWORD="your-postgres-password"
```

### 2. Deploy the Cron Job

```bash
# Deploy the auto-update script
modal deploy incremental_update_pipeline.py
```

## Done!

The system now runs automatically every day at 2 AM UTC. New tweets will be:
- ✅ Detected in Supabase
- ✅ Given embeddings
- ✅ Assigned to communities
- ✅ Added to all cluster files

---

## Optional: Check Status

```bash
# See if it's running
modal app list | grep incremental

# View logs
modal app logs incremental-update-pipeline
```

---

## If Something Breaks

Run manually to see what's wrong:
```bash
modal run incremental_update_pipeline.py
```

Then check the output for errors.

---

That's it! The detailed guide (INCREMENTAL_UPDATE_GUIDE.md) is there if you need to troubleshoot or customize, but you probably won't need it.
