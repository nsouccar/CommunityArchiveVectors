import express from 'express';
import { MilvusClient } from '@zilliz/milvus2-sdk-node';
import OpenAI from 'openai';
import cors from 'cors';

const app = express();

// Middleware
app.use(express.json());
app.use(cors()); // Allow frontend to connect

// Initialize clients
const milvus = new MilvusClient({
  address: process.env.MILVUS_ADDRESS || 'localhost:19530',
});

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY!,
});

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', message: 'Server is running' });
});

// Semantic search endpoint
app.post('/api/search', async (req, res) => {
  try {
    const { query, limit = 10, filters } = req.body;

    if (!query) {
      return res.status(400).json({ error: 'Query is required' });
    }

    console.log(`🔍 Search query: "${query}"`);

    // 1. Generate embedding for query
    const response = await openai.embeddings.create({
      model: 'text-embedding-3-small',
      input: query,
    });

    const queryEmbedding = response.data[0].embedding;
    console.log(`✅ Generated query embedding`);

    // 2. Search Milvus with optimized HNSW parameters
    const searchParams: any = {
      collection_name: 'tweets',
      data: [queryEmbedding],
      limit: parseInt(limit),
      output_fields: [
        'tweet_id',
        'full_text',
        'thread_context',
        'account_id',
        'account_username',
        'favorite_count',
        'retweet_count',
        'created_at',
        'depth',
        'is_root',
      ],
      // HNSW search parameters for better quality
      params: {
        ef: 200,  // Higher = better recall/accuracy (range: 1-32768, default: 10)
      },
      metric_type: 'COSINE',
    };

    // Add filters if provided
    if (filters) {
      searchParams.filter = filters;
    }

    const results = await milvus.search(searchParams);
    console.log(`✅ Found ${results.results.length} results`);

    // 3. Format results
    const formattedResults = results.results.map((result: any) => ({
      tweetId: result.tweet_id,
      text: result.full_text,
      threadContext: result.thread_context,
      similarity: (result.score * 100).toFixed(1) + '%',
      score: result.score,
      author: result.account_username || result.account_id,
      likes: result.favorite_count,
      retweets: result.retweet_count,
      createdAt: result.created_at,
      depth: result.depth,
      isRoot: result.is_root,
    }));

    res.json({
      query,
      results: formattedResults,
      count: formattedResults.length,
    });
  } catch (error: any) {
    console.error('❌ Search error:', error);
    res.status(500).json({
      error: 'Search failed',
      message: error.message,
    });
  }
});

// Get collection stats
app.get('/api/stats', async (req, res) => {
  try {
    const stats = await milvus.getCollectionStatistics({
      collection_name: 'tweets',
    });

    res.json({
      totalVectors: stats.data.row_count,
      collectionName: 'tweets',
    });
  } catch (error: any) {
    res.status(500).json({
      error: 'Failed to get stats',
      message: error.message,
    });
  }
});

const PORT = process.env.PORT || 3001;

app.listen(PORT, () => {
  console.log(`\n🚀 Search API running on http://localhost:${PORT}`);
  console.log(`\nAvailable endpoints:`);
  console.log(`  GET  /api/health - Health check`);
  console.log(`  POST /api/search - Semantic search`);
  console.log(`  GET  /api/stats  - Collection statistics\n`);
});
