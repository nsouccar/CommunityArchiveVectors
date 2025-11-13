# Community Archive Data - Public API Access

This dataset contains **5.7 million tweets** organized by communities with their embeddings (1024-dimensional vectors).

## Quick Start

### 1. View Available Data

Open this URL in your browser or use curl:
```bash
curl https://nsouccar--community-archive-public-api-list-structure.modal.run
```

This returns JSON showing all available years, months, and communities.

### 2. Get Specific Community Data

Example: Get tweets and embeddings from Community 2 in January 2022:
```bash
curl "https://nsouccar--community-archive-public-api-get-community.modal.run?year=2022&month=1&community_id=2"
```

### 3. Get Just Tweet IDs (Faster)

If you only need tweet IDs without full text and embeddings:
```bash
curl "https://nsouccar--community-archive-public-api-get-tweet-ids.modal.run?year=2022&month=1&community_id=2"
```

### 4. Download Full Year Data

⚠️ Warning: This returns a LOT of data (multiple GB)
```bash
curl "https://nsouccar--community-archive-public-api-get-year.modal.run?year=2022" > year_2022.json
```

## Using in Python

```python
import requests

# 1. See what data is available
response = requests.get('https://nsouccar--community-archive-public-api-list-structure.modal.run')
structure = response.json()
print(f"Total tweets: {structure['total_tweets']:,}")

# 2. Get specific community
response = requests.get(
    'https://nsouccar--community-archive-public-api-get-community.modal.run',
    params={'year': '2022', 'month': 1, 'community_id': '2'}
)
data = response.json()

tweets = data['tweets']
embeddings = data['embeddings']  # List of 1024-dim vectors

print(f"Got {len(tweets)} tweets")
print(f"First tweet: @{tweets[0]['username']}: {tweets[0]['text']}")
print(f"Embedding dimensions: {len(embeddings[0])}")

# 3. Get just tweet IDs (faster)
response = requests.get(
    'https://nsouccar--community-archive-public-api-get-tweet-ids.modal.run',
    params={'year': '2022', 'month': 1, 'community_id': '2'}
)
tweet_ids = response.json()['tweet_ids']
print(f"Tweet IDs: {tweet_ids[:5]}")
```

## Using in JavaScript/Node.js

```javascript
// 1. See what data is available
fetch('https://nsouccar--community-archive-public-api-list-structure.modal.run')
  .then(res => res.json())
  .then(data => {
    console.log(`Total tweets: ${data.total_tweets.toLocaleString()}`);
  });

// 2. Get specific community
const url = new URL('https://nsouccar--community-archive-public-api-get-community.modal.run');
url.searchParams.set('year', '2022');
url.searchParams.set('month', '1');
url.searchParams.set('community_id', '2');

fetch(url)
  .then(res => res.json())
  .then(data => {
    console.log(`Got ${data.tweets.length} tweets`);
    console.log(`First tweet: @${data.tweets[0].username}`);
  });
```

## Data Structure

### Tweet Object
```json
{
  "tweet_id": "1477067086916571136",
  "username": "nopranablem",
  "text": "Full tweet text here...",
  "timestamp": "2022-01-01T00:00:10+00:00"
}
```

### Embedding
- Array of 1024 floating-point numbers
- Each tweet has a corresponding embedding at the same index

## Available Data

**Years**: 2012, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025

**Total**: ~5.7 million tweets across all years

**Breakdown by year:**
- 2012: 1,755 tweets
- 2018: 58,753 tweets
- 2019: 211,421 tweets
- 2020: 605,338 tweets
- 2021: 795,632 tweets
- 2022: 939,464 tweets
- 2023: 1,109,592 tweets
- 2024: 1,157,877 tweets
- 2025: 839,016 tweets

## Rate Limits & Performance

- First request to an endpoint may be slow (~10-30 seconds) while the data loads
- Subsequent requests are much faster
- No authentication required
- No strict rate limits, but please be reasonable

## Questions?

Contact the dataset maintainer if you have questions or need help accessing the data.
