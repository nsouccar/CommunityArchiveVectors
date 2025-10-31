import { MilvusClient } from '@zilliz/milvus2-sdk-node';
import { createClient } from '@supabase/supabase-js';
import OpenAI from 'openai';

interface Tweet {
  tweet_id: string;
  full_text: string;
  reply_to_tweet_id: string | null;
  created_at: string;
  account_id: string;
  retweet_count: number;
  favorite_count: number;
}

export class EmbeddingSync {
  private milvus: MilvusClient;
  private supabase: ReturnType<typeof createClient>;
  private openai: OpenAI;
  private collectionName = 'tweets';

  constructor() {
    // Initialize clients
    this.milvus = new MilvusClient({
      address: process.env.MILVUS_ADDRESS || 'localhost:19530',
    });

    this.supabase = createClient(
      process.env.SUPABASE_URL!,
      process.env.SUPABASE_KEY!
    );

    this.openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY!,
    });
  }

  /**
   * Build thread context by traversing reply chain
   */
  private async buildThreadContext(
    tweetId: string,
    tweetsMap: Map<string, Tweet>
  ): Promise<string[]> {
    const thread: string[] = [];
    let currentId: string | null = tweetId;
    const visited = new Set<string>();

    while (currentId && !visited.has(currentId)) {
      visited.add(currentId);
      const tweet = tweetsMap.get(currentId);

      if (!tweet) break;

      thread.unshift(tweet.full_text);
      currentId = tweet.reply_to_tweet_id;
    }

    return thread;
  }

  /**
   * Get tweets from Supabase that haven't been embedded yet
   */
  async getUnprocessedTweets(sinceDate: string): Promise<Tweet[]> {
    console.log(`📥 Fetching tweets since ${sinceDate}...`);

    // First, get all tweet IDs already in Milvus
    const processedIds = await this.getProcessedTweetIds();
    console.log(`   Already processed: ${processedIds.size} tweets`);

    // Fetch tweets from Supabase
    const { data: tweets, error } = await this.supabase
      .from('tweets')
      .select('tweet_id, full_text, reply_to_tweet_id, created_at, account_id, retweet_count, favorite_count')
      .gte('created_at', sinceDate)
      .order('created_at', { ascending: true });

    if (error) {
      throw new Error(`Supabase error: ${error.message}`);
    }

    // Filter out already processed tweets
    const unprocessed = (tweets || []).filter(
      (tweet) => !processedIds.has(tweet.tweet_id)
    );

    console.log(`   Found ${unprocessed.length} new tweets to process`);
    return unprocessed as Tweet[];
  }

  /**
   * Get set of tweet IDs already in Milvus
   */
  private async getProcessedTweetIds(): Promise<Set<string>> {
    try {
      // Query all tweet IDs from Milvus
      const results = await this.milvus.query({
        collection_name: this.collectionName,
        output_fields: ['tweet_id'],
        limit: 10000000, // Get all
      });

      return new Set(results.data.map((r: any) => r.tweet_id));
    } catch (error) {
      // Collection might not exist yet
      console.log('   No existing tweets in Milvus');
      return new Set();
    }
  }

  /**
   * Generate OpenAI embeddings for tweets
   */
  async generateEmbeddings(tweets: Tweet[]): Promise<any[]> {
    if (tweets.length === 0) {
      console.log('✅ No new tweets to process');
      return [];
    }

    console.log(`🤖 Generating OpenAI embeddings for ${tweets.length} tweets...`);

    // Create tweets map for thread building
    const tweetsMap = new Map<string, Tweet>();
    tweets.forEach((tweet) => tweetsMap.set(tweet.tweet_id, tweet));

    // Build thread contexts
    const tweetData = [];
    for (const tweet of tweets) {
      const threadContext = await this.buildThreadContext(tweet.tweet_id, tweetsMap);
      const contextText = threadContext.join('\n\n');

      // Find root tweet ID
      let rootId = tweet.tweet_id;
      let currentTweet = tweet;
      while (currentTweet.reply_to_tweet_id && tweetsMap.has(currentTweet.reply_to_tweet_id)) {
        rootId = currentTweet.reply_to_tweet_id;
        currentTweet = tweetsMap.get(currentTweet.reply_to_tweet_id)!;
      }

      tweetData.push({
        tweet: tweet,
        contextText: contextText,
        threadRootId: rootId,
        depth: threadContext.length - 1,
      });
    }

    // Generate embeddings in batches (OpenAI: 100 RPM limit)
    const BATCH_SIZE = 90;
    const BATCH_DELAY = 60000; // 60 seconds
    const embeddingsData = [];

    for (let i = 0; i < tweetData.length; i += BATCH_SIZE) {
      const batch = tweetData.slice(i, i + BATCH_SIZE);
      console.log(`   Processing batch ${Math.floor(i / BATCH_SIZE) + 1}/${Math.ceil(tweetData.length / BATCH_SIZE)}...`);

      const batchPromises = batch.map(async (item) => {
        const response = await this.openai.embeddings.create({
          model: 'text-embedding-3-small',
          input: item.contextText,
        });

        return {
          tweet_id: item.tweet.tweet_id,
          embedding: response.data[0].embedding,
          full_text: item.tweet.full_text.slice(0, 5000),
          thread_context: item.contextText.slice(0, 10000),
          thread_root_id: item.threadRootId,
          depth: item.depth,
          is_root: item.tweet.reply_to_tweet_id === null,
          account_id: item.tweet.account_id,
          favorite_count: item.tweet.favorite_count,
          retweet_count: item.tweet.retweet_count,
          created_at: item.tweet.created_at,
          processed_at: new Date().toISOString(),
          embedding_version: 'openai-text-embedding-3-small',
        };
      });

      const batchResults = await Promise.all(batchPromises);
      embeddingsData.push(...batchResults);

      // Rate limiting
      if (i + BATCH_SIZE < tweetData.length) {
        console.log(`   Waiting 60s for rate limit...`);
        await new Promise((resolve) => setTimeout(resolve, BATCH_DELAY));
      }
    }

    console.log(`✅ Generated ${embeddingsData.length} embeddings`);
    return embeddingsData;
  }

  /**
   * Insert embeddings into Milvus
   */
  async insertEmbeddings(embeddings: any[]): Promise<void> {
    if (embeddings.length === 0) return;

    console.log(`📤 Inserting ${embeddings.length} embeddings into Milvus...`);

    // Insert in batches
    const BATCH_SIZE = 1000;
    for (let i = 0; i < embeddings.length; i += BATCH_SIZE) {
      const batch = embeddings.slice(i, i + BATCH_SIZE);

      await this.milvus.insert({
        collection_name: this.collectionName,
        data: batch,
      });

      console.log(`   Inserted ${Math.min(i + BATCH_SIZE, embeddings.length)}/${embeddings.length}`);
    }

    // Flush to persist
    await this.milvus.flush({ collection_names: [this.collectionName] });
    console.log(`✅ All embeddings inserted and flushed`);
  }

  /**
   * Main sync function - call this every hour
   */
  async syncTweets(sinceDate: string = '2025-10-01'): Promise<void> {
    const startTime = Date.now();
    console.log(`\n${'='.repeat(60)}`);
    console.log(`🔄 Starting embedding sync at ${new Date().toISOString()}`);
    console.log(`${'='.repeat(60)}\n`);

    try {
      // 1. Get unprocessed tweets
      const tweets = await this.getUnprocessedTweets(sinceDate);

      if (tweets.length === 0) {
        console.log('✅ No new tweets to process. Database is up to date!');
        return;
      }

      // 2. Generate embeddings
      const embeddings = await this.generateEmbeddings(tweets);

      // 3. Insert into Milvus
      await this.insertEmbeddings(embeddings);

      // 4. Summary
      const duration = ((Date.now() - startTime) / 1000).toFixed(1);
      console.log(`\n${'='.repeat(60)}`);
      console.log(`✅ Sync complete!`);
      console.log(`   Processed: ${tweets.length} tweets`);
      console.log(`   Duration: ${duration}s`);
      console.log(`${'='.repeat(60)}\n`);
    } catch (error) {
      console.error(`\n❌ Sync failed:`, error);
      throw error;
    }
  }

  /**
   * Get sync statistics
   */
  async getStats(): Promise<any> {
    const stats = await this.milvus.getCollectionStatistics({
      collection_name: this.collectionName,
    });

    return {
      totalVectors: stats.data.row_count,
      lastSync: new Date().toISOString(),
    };
  }
}
