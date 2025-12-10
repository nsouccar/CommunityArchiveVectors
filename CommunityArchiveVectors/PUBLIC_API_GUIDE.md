# Community Archive Data - Public API

**Public URL**: https://nsouccar--community-data-api-fastapi-app.modal.run

This API provides access to **5.7 million tweets** organized by communities with their 1024-dimensional embeddings.

**No authentication required** - anyone can access this data!

---

## Quick Start

### 1. View API Documentation

Open this URL in your browser:
```
https://nsouccar--community-data-api-fastapi-app.modal.run/
```

### 2. See Available Data Structure

```bash
curl "https://nsouccar--community-data-api-fastapi-app.modal.run/structure"
```

Returns JSON with all years, months, communities, and tweet counts.

### 3. List All Years

```bash
curl "https://nsouccar--community-data-api-fastapi-app.modal.run/data?action=list_years"
```

### 4. Get Specific Community Data

Get tweets and embeddings from Community 2 in January 2022:
```bash
curl "https://nsouccar--community-data-api-fastapi-app.modal.run/data?action=community&year=2022&month=1&community_id=2"
```

### 5. Get Just Tweet IDs (Faster)

```bash
curl "https://nsouccar--community-data-api-fastapi-app.modal.run/data?action=tweet_ids&year=2022&month=1&community_id=2"
```

---

## Using in Python

```python
import requests

BASE_URL = "https://nsouccar--community-data-api-fastapi-app.modal.run"

# 1. See what data is available
response = requests.get(f"{BASE_URL}/structure")
structure = response.json()
print(f"Total tweets: {structure['total_tweets']:,}")

# 2. List all years
response = requests.get(f"{BASE_URL}/data", params={"action": "list_years"})
years = response.json()['years']
print(f"Available years: {years}")

# 3. Get specific community
response = requests.get(f"{BASE_URL}/data", params={
    "action": "community",
    "year": "2022",
    "month": 1,
    "community_id": "2"
})
data = response.json()

tweets = data['tweets']
embeddings = data['embeddings']  # List of 1024-dim vectors

print(f"Got {len(tweets)} tweets")
print(f"First tweet: @{tweets[0]['username']}: {tweets[0]['text'][:80]}")
print(f"Embedding dimensions: {len(embeddings[0])}")

# 4. Get just tweet IDs (much faster and smaller response)
response = requests.get(f"{BASE_URL}/data", params={
    "action": "tweet_ids",
    "year": "2022",
    "month": 1,
    "community_id": "2"
})
tweet_ids = response.json()['tweet_ids']
print(f"Tweet IDs: {tweet_ids[:5]}")

# 5. Download full year (WARNING: Very large response, may take minutes)
response = requests.get(f"{BASE_URL}/data", params={
    "action": "year",
    "year": "2022"
})
year_data = response.json()
# Save to file
import json
with open('year_2022.json', 'w') as f:
    json.dump(year_data, f)
```

---

## Using in JavaScript

```javascript
const BASE_URL = 'https://nsouccar--community-data-api-fastapi-app.modal.run';

// 1. See what data is available
fetch(`${BASE_URL}/structure`)
  .then(res => res.json())
  .then(data => {
    console.log(`Total tweets: ${data.total_tweets.toLocaleString()}`);
  });

// 2. Get specific community
const params = new URLSearchParams({
  action: 'community',
  year: '2022',
  month: '1',
  community_id: '2'
});

fetch(`${BASE_URL}/data?${params}`)
  .then(res => res.json())
  .then(data => {
    console.log(`Got ${data.tweets.length} tweets`);
    console.log(`First tweet: @${data.tweets[0].username}`);
    console.log(`Embedding dim: ${data.embeddings[0].length}`);
  });

// 3. Get just tweet IDs (faster)
const params2 = new URLSearchParams({
  action: 'tweet_ids',
  year: '2022',
  month: '1',
  community_id: '2'
});

fetch(`${BASE_URL}/data?${params2}`)
  .then(res => res.json())
  .then(data => {
    console.log(`Tweet IDs: ${data.tweet_ids.slice(0, 5)}`);
  });
```

---

## API Endpoints

### `GET /`
Returns API documentation

### `GET /structure`
Returns complete data structure with all years, months, and communities

**Response**:
```json
{
  "structure": {
    "2022": {
      "1": {
        "num_communities": 12,
        "community_ids": ["0", "1", "2", ...],
        "num_tweets": 123456
      },
      ...
    },
    ...
  },
  "total_tweets": 5718848
}
```

### `GET /data`
Query data with different actions

**Parameters**:
- `action` (required): One of `list_years`, `community`, `tweet_ids`, `year`
- `year` (optional): Year as string (e.g., "2022")
- `month` (optional): Month as int (1-12)
- `community_id` (optional): Community ID as string

**Actions**:

#### `action=list_years`
Returns list of available years
```json
{"years": ["2012", "2018", "2019", ...]}
```

#### `action=community`
Returns tweets and embeddings for a community
Requires: `year`, `month`, `community_id`
```json
{
  "year": "2022",
  "month": 1,
  "community_id": "2",
  "tweets": [
    {
      "tweet_id": "1477067086916571136",
      "username": "example_user",
      "text": "Tweet text...",
      "timestamp": "2022-01-01T00:00:10+00:00"
    },
    ...
  ],
  "embeddings": [[0.123, 0.456, ...], ...],
  "num_tweets": 12158
}
```

#### `action=tweet_ids`
Returns only tweet IDs (faster, smaller response)
Requires: `year`, `month`, `community_id`
```json
{
  "year": "2022",
  "month": 1,
  "community_id": "2",
  "tweet_ids": ["1477067086916571136", ...],
  "num_tweets": 12158
}
```

#### `action=year`
Returns all data for a year (WARNING: Very large, 1-10 GB)
Requires: `year`

---

## Data Structure

### Tweet Object
```json
{
  "tweet_id": "1477067086916571136",
  "username": "example_user",
  "text": "Full tweet text here...",
  "timestamp": "2022-01-01T00:00:10+00:00"
}
```

### Embedding
- Array of 1024 floating-point numbers
- Each tweet has a corresponding embedding at the same index
- Embeddings are from the same model used for clustering

---

## Available Data

**Years**: 2012, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025

**Total**: 5,718,848 tweets

**Breakdown by year**:
- 2012: 1,755 tweets
- 2018: 58,753 tweets
- 2019: 211,421 tweets
- 2020: 605,338 tweets
- 2021: 795,632 tweets
- 2022: 939,464 tweets
- 2023: 1,109,592 tweets
- 2024: 1,157,877 tweets
- 2025: 839,016 tweets

---

## Performance Notes

- **First request**: May take 10-30 seconds while data loads into memory
- **Subsequent requests**: Much faster (1-3 seconds)
- **Large requests** (full year data): May take several minutes and return GB of data
- **Recommended**: Use `action=tweet_ids` first to get IDs, then fetch specific communities as needed

---

## Rate Limits

- No authentication required
- No strict rate limits, but please be reasonable
- For heavy usage, consider caching responses locally

---

## Example Use Cases

### 1. Explore communities in a specific month
```python
import requests

BASE_URL = "https://nsouccar--community-data-api-fastapi-app.modal.run"

# Get structure for 2022
response = requests.get(f"{BASE_URL}/structure")
data = response.json()

# See communities in January 2022
jan_2022 = data['structure']['2022']['1']
print(f"January 2022 has {jan_2022['num_communities']} communities")
print(f"Community IDs: {jan_2022['community_ids']}")

# Get tweets from each community
for comm_id in jan_2022['community_ids'][:3]:  # First 3 communities
    response = requests.get(f"{BASE_URL}/data", params={
        "action": "tweet_ids",
        "year": "2022",
        "month": 1,
        "community_id": comm_id
    })
    tweet_ids = response.json()['tweet_ids']
    print(f"Community {comm_id}: {len(tweet_ids)} tweets")
```

### 2. Download specific communities for analysis
```python
import requests
import json

BASE_URL = "https://nsouccar--community-data-api-fastapi-app.modal.run"

# Download specific communities you're interested in
communities_to_download = [
    ("2022", 1, "2"),
    ("2022", 6, "5"),
    ("2023", 3, "10")
]

for year, month, comm_id in communities_to_download:
    response = requests.get(f"{BASE_URL}/data", params={
        "action": "community",
        "year": year,
        "month": month,
        "community_id": comm_id
    })

    data = response.json()

    # Save to file
    filename = f"community_{year}_{month}_{comm_id}.json"
    with open(filename, 'w') as f:
        json.dump(data, f)

    print(f"Saved {len(data['tweets'])} tweets to {filename}")
```

---


