# Auto-Update Setup (5 Minutes)

Get automatic daily updates for your tweet archive.

## One-Time Setup

### 1. Set Supabase Credentials

```bash
# Add your Supabase credentials to Modal
modal secret create supabase-secrets \
  SUPABASE_URL="https://fabxmporizzqflnftavs.supabase.co" \
  SUPABASE_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZhYnhtcG9yaXp6cWZsbmZ0YXZzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjIyNDQ5MTIsImV4cCI6MjAzNzgyMDkxMn0.UIEJiUNkLsW28tBHmG-RQDW-I5JNlJLt62CSk9D_qG8"
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
- ✅ Added to organized data structure
- ✅ Automatically added to existing topics (matched by similarity)

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
