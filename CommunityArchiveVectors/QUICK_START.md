# Quick Start Guide

## Setup (First Time Only)

```bash
# 1. Start Milvus
docker-compose up -d

# 2. Initialize database with OpenAI schema
bun initializeMilvus.ts

# 3. Sync all tweets since October 1st
bun syncNow.ts
```

Done! Your vector database is now running with all tweets embedded.

---

## Daily Usage

### Manual Sync (when you want to update)
```bash
bun syncNow.ts
```

### Automatic Hourly Sync (optional)
```bash
bun src/services/cronScheduler.ts
```
Leave this running to check for new tweets every hour.

---

## What You Have Now

✅ **Milvus running** (3 Docker containers)
✅ **All tweets since Oct 1** embedded with OpenAI
✅ **HNSW index** for fast search
✅ **Incremental sync** - only processes new tweets
✅ **Thread context** - full conversation included
✅ **Rich metadata** - likes, dates, usernames

---

## Next: Build Features

Now you can:
1. Add semantic search API endpoint
2. Build topic discovery
3. Create user analytics
4. Add React frontend

See [SETUP_GUIDE.md](./SETUP_GUIDE.md) for detailed documentation.
