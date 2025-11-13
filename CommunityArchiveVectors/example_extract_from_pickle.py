#!/usr/bin/env python3
"""
Example script showing how to extract data from organized_by_community.pkl

The pickle file structure:
{
  'year': {           # string like '2022', '2023'
    month: {          # int like 1, 2, 3, ..., 12
      community_id: { # int like 0, 1, 2, ...
        'tweets': [   # list of tweet dicts
          {
            'tweet_id': '1477067086916571136',
            'username': 'nopranablem',
            'text': 'Full tweet text...',
            'timestamp': '2022-01-01T00:00:10+00:00'
          },
          ...
        ],
        'embeddings': numpy.ndarray  # shape (num_tweets, 1024)
      }
    }
  }
}
"""

import modal
import pickle
from pathlib import Path

app = modal.App("extract-example")
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy")
volume = modal.Volume.from_name("tweet-vectors-large", create_if_missing=False)

@app.function(volumes={"/data": volume}, image=image)
def show_extraction_examples():
    print("\n" + "="*80)
    print("LOADING PICKLE FILE")
    print("="*80)

    # Load the pickle file
    with open("/data/organized_by_community.pkl", 'rb') as f:
        organized = pickle.load(f)

    print("\n✓ Loaded organized_by_community.pkl")

    print("\n" + "="*80)
    print("EXAMPLE 1: Get all data for year 2022")
    print("="*80)

    year_2022 = organized['2022']
    print(f"\nYear 2022 has {len(year_2022)} months")
    print(f"Months: {list(year_2022.keys())}")

    print("\n" + "="*80)
    print("EXAMPLE 2: Get all communities in January 2022")
    print("="*80)

    jan_2022 = organized['2022'][1]
    print(f"\nJanuary 2022 has {len(jan_2022)} communities")
    print(f"Community IDs: {list(jan_2022.keys())}")

    print("\n" + "="*80)
    print("EXAMPLE 3: Get tweets from a specific community")
    print("="*80)

    # Get community '2' from January 2022 (community IDs are strings)
    community_2 = organized['2022'][1]['2']
    tweets = community_2['tweets']
    embeddings = community_2['embeddings']

    print(f"\nCommunity 2 in January 2022:")
    print(f"  Number of tweets: {len(tweets)}")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"\n  First tweet:")
    print(f"    ID: {tweets[0]['tweet_id']}")
    print(f"    Username: @{tweets[0]['username']}")
    print(f"    Text: {tweets[0]['text'][:100]}...")
    print(f"    Timestamp: {tweets[0]['timestamp']}")
    print(f"\n  First tweet embedding:")
    print(f"    Shape: {embeddings[0].shape}")
    print(f"    First 10 values: {embeddings[0][:10]}")

    print("\n" + "="*80)
    print("EXAMPLE 4: Extract all tweet IDs from a community")
    print("="*80)

    tweet_ids = [tweet['tweet_id'] for tweet in tweets]
    print(f"\nExtracted {len(tweet_ids)} tweet IDs")
    print(f"First 5 IDs: {tweet_ids[:5]}")

    print("\n" + "="*80)
    print("EXAMPLE 5: Filter tweets by keyword")
    print("="*80)

    keyword = "meditation"
    filtered_tweets = [
        tweet for tweet in tweets
        if keyword.lower() in tweet['text'].lower()
    ]
    print(f"\nFound {len(filtered_tweets)} tweets containing '{keyword}'")
    if filtered_tweets:
        print(f"Example: @{filtered_tweets[0]['username']}: {filtered_tweets[0]['text'][:80]}...")

    print("\n" + "="*80)
    print("EXAMPLE 6: Get embeddings for specific tweets")
    print("="*80)

    # Get embeddings for first 3 tweets
    first_3_embeddings = embeddings[:3]
    print(f"\nFirst 3 tweet embeddings shape: {first_3_embeddings.shape}")
    print(f"Each embedding is a vector of length {first_3_embeddings.shape[1]}")

    print("\n" + "="*80)
    print("EXAMPLE 7: Iterate through all years, months, communities")
    print("="*80)

    print("\nIterating through structure:")
    total_communities = 0
    for year in sorted(organized.keys(), key=lambda x: int(x) if x.isdigit() else 0):
        year_communities = 0
        for month in organized[year]:
            year_communities += len(organized[year][month])
        total_communities += year_communities
        print(f"  {year}: {year_communities} communities")
    print(f"\nTotal communities: {total_communities}")

    print("\n" + "="*80)
    print("USAGE IN YOUR CODE")
    print("="*80)
    print("""
To use this in your own script:

import pickle

# Load the file
with open('organized_by_community.pkl', 'rb') as f:
    organized = pickle.load(f)

# Access data (note: years and community_ids are strings, months are ints)
year = '2022'
month = 1
community_id = '2'  # community IDs are strings

# Get tweets and embeddings
community = organized[year][month][community_id]
tweets = community['tweets']
embeddings = community['embeddings']  # numpy array

# Work with the data
for i, tweet in enumerate(tweets):
    tweet_id = tweet['tweet_id']
    username = tweet['username']
    text = tweet['text']
    timestamp = tweet['timestamp']
    embedding = embeddings[i]  # 1024-dimensional vector

    # Do something with tweet and embedding...
""")

@app.local_entrypoint()
def main():
    show_extraction_examples.remote()
