# Community Archive Vectors - Backend Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Data Pipeline](#data-pipeline)
4. [Core Components](#core-components)
5. [Modal Cloud Infrastructure](#modal-cloud-infrastructure)
6. [Topic Clustering System](#topic-clustering-system)
7. [API Endpoints](#api-endpoints)
8. [Data Storage](#data-storage)

---

## System Overview

Community Archive Vectors is a semantic archive system that transforms Twitter conversations into an explorable constellation visualization. The backend processes millions of tweets, generates embeddings, detects communities, clusters topics, and serves the data to the frontend.

### Tech Stack
- **Cloud Compute**: Modal.com (serverless Python functions)
- **Embeddings**: OpenAI `text-embedding-3-small` (1536 dimensions)
- **Storage**: Modal Volumes (persistent cloud storage)
- **Clustering**: scikit-learn K-means
- **Topic Labeling**: Claude API (Anthropic)
- **Network Analysis**: NetworkX, python-louvain
- **Frontend**: Next.js + React + D3.js

---

## Architecture & Communication

### System Components

```
┌─────────────────────┐
│  Supabase           │  Raw tweets, metadata, reply relationships
│  (Postgres DB)      │
└──────────┬──────────┘
           │
           │ HTTP API calls
           ↓
┌─────────────────────────────────────────────┐
│  Modal Cloud (Python Serverless Functions)  │
│                                              │
│  1. Generate Embeddings (OpenAI API)        │
│  2. Build Reply Network Graph               │
│  3. Detect Communities (Louvain)            │
│  4. Organize by (Year, Community)           │
│  5. Cluster Topics (K-means)                │
│  6. Label Topics (Claude API)               │
│  7. Generate Community Names                │
└────────────┬────────────────────────────────┘
             │
             │ Volume persistence
             ↓
┌────────────────────────┐
│  Modal Volume Storage  │
│  - batches/            │ (56.5 GB embeddings)
│  - organized/          │ (grouped by year/community)
│  - topics_year_*.pkl   │ (clustering results)
│  - *_summary.json      │ (human-readable summaries)
└────────────┬───────────┘
             │
             │ Direct download / API endpoints
             ↓
┌──────────────────────────────────────┐
│  Vercel (Frontend Hosting)           │
│                                      │
│  Next.js App:                        │
│  - NetworkGraph.tsx (D3.js viz)      │
│  - Search overlay (queries Modal)    │
│  - Static topic data (public/data/)  │
└──────────────────────────────────────┘
```

### Communication Flow

1. **Data ingestion**: Supabase stores raw tweets with `reply_to_tweet_id` field
2. **Processing**: Modal functions fetch from Supabase via HTTP, process, store in Modal Volumes
3. **Network building**: Uses reply relationships to build conversation graph
4. **Visualization data**: Downloaded to `frontend/public/data/` as JSON files
5. **Search**: Frontend calls Modal HTTP endpoints for semantic search (experimental: CoreNN)

**CoreNN Database**: CoreNN (Core Nearest Neighbors) is a vector database currently hosted on Vercel that provides fast semantic search capabilities. The frontend queries CoreNN for real-time semantic search across tweet embeddings.

**Search Architecture**:
- **CoreNN on Vercel**: Hosts the vector index for fast nearest-neighbor search
- **Frontend**: Calls CoreNN endpoints directly via HTTP for semantic search queries
- **Modal**: Alternative/backup search endpoints (some experimental implementations in `_archive_experimental/`)

---

## Data Pipeline

### Phase 1: Embedding Generation
**Input**: Raw tweets from Supabase
**Output**: 1536-dimensional vectors stored in Modal Volume

```python
# Batched processing: 8,000 tweets per file
batch_file = f"batches/batch_{batch_num:06d}.pkl"
{
  'ids': [...],           # Tweet IDs
  'texts': [...],         # Tweet text
  'embeddings': [...],    # OpenAI embeddings (1536-dim)
  'metadata': [...]       # year, month, community, etc.
}
```

### Phase 2: Network Analysis (Conversation Graph)
**Script**: `network_analysis.py`

**IMPORTANT**: The network is built on **reply relationships**, NOT semantic similarity!

1. **Load tweet metadata** from pickle file
2. **Build interaction graph** where:
   - **Nodes** = users
   - **Edges** = reply relationships (A → B means A replied to B's tweet)
   - **Edge weight** = number of times A replied to B
3. **Filter weak connections** (requires minimum 2 interactions)
4. **Community detection** using Louvain algorithm on reply graph
5. **Assign community IDs** to users based on conversation patterns

```python
# Build graph from reply relationships
for tweet_id, tweet_data in metadata.items():
    from_user = tweet_data.get('username')
    reply_to_id = tweet_data.get('reply_to_tweet_id')

    if from_user and reply_to_id and reply_to_id in tweet_to_user:
        to_user = tweet_to_user[reply_to_id]
        if from_user != to_user:  # Exclude self-replies
            G.add_edge(from_user, to_user, weight=interaction_count)

# Result: Each user gets a community_id based on who they talk to
metadata['community'] = detected_community_id
```

**Why reply-based?** The constellation visualizes **conversation communities** - groups of people who actually talk to each other, not groups with similar topics.

### Phase 3: Organization by (Year, Community)
**Script**: `organize_on_modal.py`

Groups tweets into manageable clusters for efficient processing:

```python
# Example grouping
organized_data = {
    (2023, 0): [...],    # Year 2023, Community 0
    (2023, 1): [...],    # Year 2023, Community 1
    (2025, 0): [...],    # Year 2025, Community 0
    ...
}
```

**Why this matters**:
- Reduces clustering from 3,127 communities → ~500-600 (year, community) pairs
- Enables incremental processing (can save after each year)
- Skip logic avoids reprocessing completed years

### Phase 4: Topic Clustering & Labeling
**Script**: `cluster_by_year_community.py`
**Status**: Currently processing year 2025 (96% complete)

For each (year, community) group:

1. **K-means clustering**
   - Automatically determine K using elbow method
   - Typical range: 1-15 topics per community
   - Skip communities with < 20 tweets

2. **Topic sampling**
   - Sample 10 representative tweets per cluster
   - Include diversity (first/middle/end of cluster)

3. **LLM Topic Labeling** (Claude API)
   ```python
   response = client.messages.create(
       model="claude-3-5-haiku-20241022",
       max_tokens=300,
       messages=[{
           "role": "user",
           "content": f"Analyze these tweets and identify the topic: {sample_tweets}"
       }]
   )
   ```

4. **Quality filtering**
   - LLM can mark clusters as "not relevant" if too noisy
   - Fallback to "Unknown Topic" if API fails

5. **Save results**
   - Pickle file: Full clustering data for analysis
   - JSON summary: Human-readable topics, examples, stats

**Output files**:
```
/data/topics_year_2023.pkl           # Full clustering data
/data/topics_year_2023_summary.json  # Human-readable summary
```

### Phase 5: Community Name Generation
**Script**: `generate_community_names.py`

Uses Claude API with high temperature (0.9) for creativity:

```python
prompt = f"""Given these topics discussed in an online Twitter community:
{topics_text}

Generate a funny, clever, and fitting name for this community.
The name should be creative and memorable (3-6 words).
"""

# Example outputs:
# "The Vasocomputation Voyagers"
# "AI Alignment Anxiety Club"
# "Supersonic Aircraft Dreamers"
```

**Output**: `community_names_2023.json`

### Phase 6: Download to Frontend
**Script**: `download_summaries.py`

Transfers data from Modal Volume to local frontend:

```
frontend/public/data/
  ├── topics_year_2012_summary.json
  ├── topics_year_2018_summary.json
  ├── ...
  ├── topics_year_2025_summary.json
  └── all_topics.json                 # Combined file
```

---

## Core Components

### 1. `cluster_by_year_community.py`
**Purpose**: Main clustering and topic labeling pipeline

**Key Functions**:
- `cluster_and_label_by_year()`: Orchestrates entire process
- `cluster_and_label_community()`: Processes one (year, community) group
- `get_optimal_k()`: Determines optimal number of clusters
- `call_claude_api()`: Labels topics using Claude

**Features**:
- **Skip logic**: Checks if results exist before processing
- **Incremental saving**: Saves after each year completes
- **Error handling**: Graceful fallback for API failures
- **Progress tracking**: Detailed console output

**Usage**:
```bash
modal run cluster_by_year_community.py
```

### 2. `organize_on_modal.py`
**Purpose**: Groups embeddings by (year, community) for efficient processing

**Key Functions**:
- `organize_embeddings()`: Loads all batches and organizes by keys
- `check_organized_data()`: Validates organized data structure

**Data Structure**:
```python
{
  (2023, 0): {
    'vectors': np.array([...]),      # Embeddings
    'tweet_ids': [...],               # IDs
    'texts': [...],                   # Full text
    'metadata': [...]                 # year, month, etc.
  }
}
```

### 3. `generate_community_names.py`
**Purpose**: Generate creative names for each community

**Configuration**:
- Model: `claude-3-5-haiku-20241022`
- Temperature: 0.9 (high creativity)
- Max tokens: 300

**Output Format**:
```json
{
  "year": 2023,
  "communities": [
    {
      "community_id": 0,
      "name": "The AI Alignment Optimizers",
      "description": "A community obsessed with AI safety and alignment",
      "num_tweets": 35533,
      "num_topics": 13,
      "topics": [...]
    }
  ]
}
```

### 4. `download_summaries.py`
**Purpose**: Transfer topic summaries from Modal to frontend

**Process**:
1. Downloads each year's summary from Modal Volume
2. Saves individual files to `frontend/public/data/`
3. Creates combined `all_topics.json` for easy access

### 5. `network_analysis.py`
**Purpose**: Community detection using graph algorithms

**Algorithm**:
1. Compute pairwise cosine similarity
2. Create edges for similarities > threshold
3. Apply Louvain community detection
4. Assign community IDs to metadata

**Output**: Updated batch files with `community` field in metadata

---

## Modal Cloud Infrastructure

### What is Modal?
Modal.com is a serverless Python platform that provides:
- **Scalable compute**: Spin up containers on-demand
- **Persistent storage**: Volumes that persist across runs
- **GPU support**: For embedding generation
- **Easy deployment**: Python decorators for cloud functions

### Key Modal Concepts

**1. Modal Apps**
```python
app = modal.App("cluster-by-year-community")
```

**2. Volumes (Persistent Storage)**
```python
volume = modal.Volume.from_name("tweet-vectors-large", create_if_missing=False)
```
- Mounts at `/data` inside functions
- Persists across runs
- Requires `volume.commit()` to save changes

**3. Functions**
```python
@app.function(
    image=image,                    # Docker-like environment
    volumes={"/data": volume},      # Mount volume
    secrets=[...],                  # API keys
    timeout=36000,                  # 10 hours
    cpu=8                           # 8 CPU cores
)
def cluster_and_label_by_year():
    # Your code here
    volume.commit()  # Save changes!
```

**4. Local Entrypoints**
```python
@app.local_entrypoint()
def main():
    # Runs on local machine, calls remote functions
    result = cluster_and_label_by_year.remote()
```

### Running Modal Scripts

```bash
# Activate environment
source modal_env/bin/activate

# Run a script
modal run cluster_by_year_community.py

# Serve an API
modal serve warm_search_service.py

# Check logs
modal app logs cluster-by-year-community
```

---

## Topic Clustering System

### Why Cluster by (Year, Community)?

**Problem**: Original approach tried to cluster all 3,127 communities at once
- Too many groups
- Long processing time
- Difficult to track progress
- Can't save incrementally

**Solution**: Group by (year, community) pairs
- Reduces to ~500-600 groups
- Process one year at a time
- Save after each year completes
- Skip already-completed years
- Much faster! (30-60 min vs. hours)

### Clustering Algorithm

**Step 1: Determine optimal K**
```python
def get_optimal_k(vectors, min_k=2, max_k=15):
    inertias = []
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(vectors)
        inertias.append(kmeans.inertia_)

    # Find elbow using differences
    diffs = np.diff(inertias)
    diffs2 = np.diff(diffs)
    optimal_k = np.argmax(diffs2) + min_k + 1
    return optimal_k
```

**Step 2: K-means clustering**
```python
kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
labels = kmeans.fit_predict(vectors)
```

**Step 3: Sample representative tweets**
```python
# Get 10 diverse tweets from cluster
indices = np.where(labels == cluster_id)[0]
sample_indices = [
    indices[0],                    # First
    indices[len(indices)//4],      # Quarter
    indices[len(indices)//2],      # Middle
    indices[3*len(indices)//4],    # Three-quarter
    indices[-1],                   # Last
    # ... plus 5 random
]
```

**Step 4: LLM topic labeling**
```python
response = client.messages.create(
    model="claude-3-5-haiku-20241022",
    max_tokens=300,
    messages=[{"role": "user", "content": prompt}]
)

result = json.loads(response.content[0].text)
# {
#   "topic": "AI Alignment and Optimization",
#   "description": "...",
#   "is_relevant": true,
#   "confidence": "high"
# }
```

### Topic Filtering

Claude can mark clusters as irrelevant:
- Too noisy / fragmented
- No coherent theme
- Spam / low-quality content

Example filtered cluster:
```
FILTERED by LLM (The tweets appear to be disconnected replies
with no clear coherent theme beyond being part of a conversational
thread. They touch on various unrelated topics...)
```

---

## Data Storage

### Modal Volume Structure

```
/data/
├── batches/                           # Original embeddings (56.5 GB)
│   ├── batch_000001.pkl
│   ├── batch_000002.pkl
│   └── ...
│
├── organized/                         # Grouped by (year, community)
│   └── organized_data.pkl             # Main organized dataset
│
├── topics_year_2012.pkl               # Clustering results (pickle)
├── topics_year_2012_summary.json      # Human-readable summary
│
├── topics_year_2018.pkl
├── topics_year_2018_summary.json
│
└── ... (all years)
```

### Summary JSON Structure

```json
{
  "year": "2023",
  "stats": {
    "total_communities": 40,
    "communities_processed": 14,
    "communities_skipped_too_small": 26,
    "total_clusters": 120,
    "clusters_filtered_llm": 23,
    "clusters_kept": 97
  },
  "communities": {
    "0": [
      {
        "cluster_id": 1,
        "topic": "Personal Autonomy and Expectations",
        "description": "A discussion about navigating social obligations...",
        "confidence": "high",
        "num_tweets": 14868,
        "sample_tweets": [...]
      }
    ]
  }
}
```

### Frontend Data

```
frontend/public/data/
├── topics_year_2012_summary.json      # 52 KB
├── topics_year_2018_summary.json      # 1.6 MB
├── topics_year_2019_summary.json      # 5.6 MB
├── topics_year_2020_summary.json      # 29 KB
├── topics_year_2021_summary.json      # 18 KB
├── topics_year_2022_summary.json      # 25 KB
├── topics_year_2023_summary.json      # 29 KB
├── topics_year_2024_summary.json      # 51 KB
└── all_topics.json                    # 7.8 MB (combined)
```

---

## API Endpoints

### Current Status
Currently, the backend is focused on data processing. Future API endpoints for semantic search are planned but not yet implemented.

### Planned Architecture
- **Modal serve**: Host search API on Modal
- **Vector search**: FAISS or similar for fast nearest-neighbor search
- **Caching**: Pre-load embeddings into memory for fast queries

---

## Performance Optimizations

### 1. Skip Logic
```python
summary_path = Path(f"/data/topics_year_{year}_summary.json")
if summary_path.exists():
    print(f"SKIPPING year {year} - Results already exist")
    return
```

### 2. Incremental Saving
Save after each year completes, not at the very end:
```python
for year in years:
    process_year(year)
    save_year_results(year)
    volume.commit()  # Persist to Modal Volume
```

### 3. Batched API Calls
Process multiple clusters in parallel when possible (future optimization)

### 4. Efficient Data Structures
- NumPy arrays for embeddings (memory-efficient)
- Pickle for fast serialization
- JSON for human-readable summaries

---

## Monitoring & Debugging

### Check Clustering Progress
```bash
# View live logs
source modal_env/bin/activate
modal app logs cluster-by-year-community

# Or use BashOutput in CLI
# Shows current community being processed
```

### Common Issues

**1. Out of Memory**
- Solution: Reduce batch size or increase CPU/RAM in Modal function

**2. API Rate Limits (Claude)**
- Solution: Add retry logic with exponential backoff
- Current: Falls back to "Unknown Topic"

**3. Volume Not Committed**
- Solution: Always call `volume.commit()` before function exits
- Wrap in try/finally block

**4. JSON Parsing Errors**
- Solution: Robust error handling, fallback values
- Log raw response for debugging

---

## Current Status

### Completed ✅
- Embedding generation (100%)
- Network analysis (100%)
- Community detection (100%)
- Organization by (year, community) (100%)
- Topic clustering for years 2012-2024 (100%)
- Download summaries to frontend (100%)

### In Progress ⏳
- Year 2025 clustering: **96% complete** (Community 344/359)
- Estimated completion: 2-3 minutes

### Next Steps 🎯
1. Download 2025 summary
2. Generate community names for all years
3. Build community legend UI component
4. Add interactive topic exploration

---

## File Reference

### Active Python Scripts
- `cluster_by_year_community.py` - Main clustering pipeline
- `organize_on_modal.py` - Group embeddings by (year, community)
- `generate_community_names.py` - Create creative community names
- `download_summaries.py` - Transfer data to frontend
- `network_analysis.py` - Community detection

### Experimental/Archive (can be removed)
- `cluster_and_label_topics.py` - Old clustering approach
- `semantic_topic_discovery.py` - Early experiments
- `generate_llm_topics.py` - Superseded by current system
- Various search/sync scripts - No longer needed

### Frontend
- `frontend/src/app/page.tsx` - Home page with constellation
- `frontend/src/components/NetworkGraph.tsx` - D3.js visualization

---

## Questions?

This documentation covers the complete backend architecture. Key concepts:
1. **Modal Cloud** runs all processing
2. **Embeddings** stored in Modal Volume
3. **Clustering** groups topics within (year, community) pairs
4. **Claude API** labels topics with human-readable names
5. **JSON summaries** feed the frontend visualization

The system is designed for scale, reliability, and incremental processing. All data persists in Modal Volumes and can be re-processed or updated as needed.
