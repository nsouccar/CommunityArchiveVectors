-- Supabase Vector Search Setup
-- Run this in your Supabase SQL Editor

-- Step 1: Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Step 2: Add vector column to tweets table (1024 dimensions for Voyage AI)
ALTER TABLE tweets
ADD COLUMN IF NOT EXISTS embedding vector(1024);

-- Step 3: Create index for vector similarity search (HNSW is fastest)
-- This will take ~30 minutes for 6.4M rows, but only needs to be done once
CREATE INDEX IF NOT EXISTS tweets_embedding_idx
ON tweets
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Step 4: Create a function for semantic search
CREATE OR REPLACE FUNCTION search_tweets(
  query_embedding vector(1024),
  match_count int DEFAULT 10
)
RETURNS TABLE (
  tweet_id bigint,
  full_text text,
  username text,
  created_at timestamp,
  retweet_count int,
  favorite_count int,
  similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    t.tweet_id,
    t.full_text,
    a.username,
    t.created_at,
    t.retweet_count,
    t.favorite_count,
    1 - (t.embedding <=> query_embedding) AS similarity
  FROM tweets t
  LEFT JOIN all_account a ON t.account_id = a.account_id
  WHERE t.embedding IS NOT NULL
  ORDER BY t.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

-- Grant permissions
GRANT EXECUTE ON FUNCTION search_tweets TO anon, authenticated;

-- Check status
SELECT
  COUNT(*) as total_tweets,
  COUNT(embedding) as tweets_with_embeddings,
  ROUND(100.0 * COUNT(embedding) / COUNT(*), 2) as percentage_complete
FROM tweets;
