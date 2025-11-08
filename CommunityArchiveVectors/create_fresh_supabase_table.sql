-- Create Fresh Tweets Table with Embeddings
-- Run this in your current Supabase project

-- Step 1: Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Step 2: Create tweets table from scratch
CREATE TABLE IF NOT EXISTS tweets (
  tweet_id bigint PRIMARY KEY,
  full_text text,
  username text,
  account_id bigint,
  created_at timestamp,
  retweet_count int DEFAULT 0,
  favorite_count int DEFAULT 0,
  reply_to_tweet_id bigint,
  embedding vector(1024)  -- Our embeddings!
);

-- Step 3: Create index for fast vector search
CREATE INDEX IF NOT EXISTS tweets_embedding_idx
ON tweets
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Step 4: Create search function
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
    t.username,
    t.created_at,
    t.retweet_count,
    t.favorite_count,
    1 - (t.embedding <=> query_embedding) AS similarity
  FROM tweets t
  WHERE t.embedding IS NOT NULL
  ORDER BY t.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

-- Grant permissions
GRANT EXECUTE ON FUNCTION search_tweets TO anon, authenticated;

-- Done! Now we can upload all data from Modal batch files
SELECT 'Table created successfully!' as status;
