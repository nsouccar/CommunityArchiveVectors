# Tweet Vector Database Setup Guide

## Overview

This system automatically:
1. Fetches all tweets since October 1st from Supabase
2. Generates OpenAI embeddings (1536 dimensions)
3. Stores them in Milvus vector database
4. Checks hourly for new tweets and processes them automatically

---

## Architecture

```
Supabase (PostgreSQL)           Milvus (Vector DB)
┌─────────────────────┐        ┌──────────────────────┐
│ tweets table        │        │ tweets collection    │
│ - All raw tweets    │───────▶│ - OpenAI embeddings  │
│ - Since Oct 1, 2025 │  Sync  │ - Fast search        │
│ - Updated hourly    │◀───────│ - Auto-indexed       │
└─────────────────────┘        └──────────────────────┘
         │                              │
         │                              │
         └──────────┬───────────────────┘
                    │
            ┌───────▼────────┐
            │  Background    │
            │  Sync Job      │
            │  (Runs hourly) │
            └────────────────┘
```

---

## Initial Setup

### Step 1: Initialize Milvus with OpenAI Schema

```bash
# Start Milvus (if not running)
docker-compose up -d

# Create collection with OpenAI embedding dimensions
bun initializeMilvus.ts
```

This creates a collection with:
- **1536 dimensions** (OpenAI text-embedding-3-small)
- **HNSW index** for fast similarity search
- **Metadata fields** (likes, dates, thread info, etc.)

### Step 2: Run Initial Sync

```bash
# Process all tweets since October 1st
bun syncNow.ts
```

This will:
1. Fetch unprocessed tweets from Supabase
2. Build thread contexts
3. Generate OpenAI embeddings (respects 100 RPM limit)
4. Insert into Milvus
5. Show progress

**Time estimate:**
- 1,000 tweets: ~15 minutes
- 10,000 tweets: ~2.5 hours
- 100,000 tweets: ~25 hours

### Step 3: Start Hourly Sync (Optional for now)

```bash
# This will check every hour for new tweets
bun src/services/cronScheduler.ts
```

Keep this running in a terminal or use PM2 for production.

---

## How It Works

### Incremental Sync Process

```javascript
Every hour:
1. Query Milvus for existing tweet IDs
2. Query Supabase for tweets since Oct 1
3. Filter out already-processed tweets
4. Generate embeddings for new tweets only
5. Insert into Milvus
6. Repeat next hour
```

### Thread Context

Each tweet includes its full thread context:

```
Original Tweet: "I love coding"
  └─ Reply: "Me too!"
      └─ Reply: "What language?"

Embedding includes:
"I love coding\n\nMe too!\n\nWhat language?"
```

This ensures semantic search understands conversation context!

---

## Index Structure

### Milvus Collection: "tweets"

| Field | Type | Purpose |
|-------|------|---------|
| tweet_id | VarChar (PK) | Unique identifier |
| embedding | FloatVector(1536) | OpenAI embedding |
| full_text | VarChar(5000) | Tweet content |
| thread_context | VarChar(10000) | Full conversation |
| thread_root_id | VarChar | Root of thread |
| depth | Int64 | Position in thread |
| is_root | Bool | Is root tweet? |
| account_id | VarChar | Username |
| favorite_count | Int64 | Likes |
| retweet_count | Int64 | Retweets |
| created_at | VarChar | Timestamp |
| processed_at | VarChar | When embedded |
| embedding_version | VarChar | Model used |

### Indexes

- **HNSW on embedding** - Fast vector search
- **Scalar indexes** - Fast filtering on metadata

---

## Query Examples

### 1. Semantic Search

```typescript
import { MilvusClient } from '@zilliz/milvus2-sdk-node';
import OpenAI from 'openai';

const milvus = new MilvusClient({ address: 'localhost:19530' });
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Generate query embedding
const response = await openai.embeddings.create({
  model: 'text-embedding-3-small',
  input: 'machine learning and AI'
});

// Search
const results = await milvus.search({
  collection_name: 'tweets',
  data: [response.data[0].embedding],
  limit: 10,
  output_fields: ['tweet_id', 'full_text', 'account_id', 'favorite_count']
});
```

### 2. Time-Range Search

```typescript
// Search tweets from last 7 days
const results = await milvus.search({
  collection_name: 'tweets',
  data: [queryEmbedding],
  filter: 'created_at >= "2025-10-24"',
  limit: 10
});
```

### 3. User-Specific Search

```typescript
// Search within one user's tweets
const results = await milvus.search({
  collection_name: 'tweets',
  data: [queryEmbedding],
  filter: 'account_id == "elonmusk"',
  limit: 10
});
```

### 4. Popular Tweets Only

```typescript
// Only tweets with >100 likes
const results = await milvus.search({
  collection_name: 'tweets',
  data: [queryEmbedding],
  filter: 'favorite_count > 100',
  limit: 10
});
```

### 5. Combined Filters

```typescript
// Popular recent tweets from specific user
const results = await milvus.search({
  collection_name: 'tweets',
  data: [queryEmbedding],
  filter: 'account_id == "sama" && created_at >= "2025-10-01" && favorite_count > 50',
  limit: 10
});
```

---

## Rate Limits

### OpenAI Embedding API
- **Free tier**: 500 RPM, 10,000 TPM
- **Paid tier 1**: 500 RPM, 200,000 TPM
- **Current batch size**: 90 requests per minute (safe)

### How Sync Handles Rate Limits
```typescript
BATCH_SIZE = 90;        // Process 90 tweets
BATCH_DELAY = 60000;    // Wait 60 seconds
// = ~90 embeddings per minute (under 100 RPM limit)
```

---

## Cost Estimates

### OpenAI Embedding Costs

**Model**: text-embedding-3-small
**Price**: $0.020 per 1M tokens

| Tweets | Avg Tokens | Cost |
|--------|------------|------|
| 1,000 | ~150k | $0.003 |
| 10,000 | ~1.5M | $0.03 |
| 100,000 | ~15M | $0.30 |
| 1,000,000 | ~150M | $3.00 |

**Very affordable!**

---

## Production Deployment

### When deploying to server:

1. **Copy files to server:**
```bash
scp -r src/ docker-compose.yml package.json user@server:~/app/
```

2. **Install dependencies:**
```bash
ssh user@server
cd ~/app
bun install
```

3. **Start Milvus:**
```bash
docker-compose up -d
```

4. **Initialize database:**
```bash
bun initializeMilvus.ts
```

5. **Run initial sync:**
```bash
bun syncNow.ts
```

6. **Start cron with PM2:**
```bash
bun add -g pm2
pm2 start "bun src/services/cronScheduler.ts" --name tweet-sync
pm2 save
pm2 startup
```

Now it runs 24/7 and checks hourly!

---

## Monitoring

### Check sync status:
```typescript
import { EmbeddingSync } from './src/services/embeddingSync';

const syncer = new EmbeddingSync();
const stats = await syncer.getStats();
console.log(`Total vectors: ${stats.totalVectors}`);
```

### View Milvus stats:
```bash
# In browser
http://localhost:9091/webui/

# Or programmatically
const stats = await milvus.getCollectionStatistics({
  collection_name: 'tweets'
});
```

---

## Troubleshooting

### "No new tweets to process"
✅ This is good! Means database is up to date.

### Rate limit errors (429)
- Reduce BATCH_SIZE in embeddingSync.ts
- Increase BATCH_DELAY

### Out of memory
- Process in smaller batches
- Increase Docker memory limit

### Collection not found
```bash
bun initializeMilvus.ts
```

---

## Next Steps

Now that you have the vector database:

1. **Build Express API** - Add search endpoints
2. **Build Frontend** - React UI for search
3. **Add Features** - Topic discovery, user analytics
4. **Deploy** - Move to production server

Your foundation is ready! 🚀
