import { MilvusClient } from '@zilliz/milvus2-sdk-node';
import { VoyageAIClient } from 'voyageai';

async function testSearch() {
  console.log('🔍 Testing Milvus Vector Search\n');

  // 1. Connect to Milvus
  const milvus = new MilvusClient({ address: 'localhost:19530' });
  console.log('✅ Connected to Milvus\n');

  // 2. Connect to Voyage AI
  const voyage = new VoyageAIClient({ apiKey: process.env.VOYAGE_API_KEY });
  console.log('✅ Connected to Voyage AI\n');

  // 3. Test query
  const query = 'coding and programming';
  console.log(`📝 Search query: "${query}"\n`);

  // 4. Generate embedding for query
  console.log('🔄 Generating embedding for query...');
  const result = await voyage.embed({
    input: [query],
    model: 'voyage-3',
    inputType: 'query', // Important: use 'query' for search queries
  });

  const queryEmbedding = result.data[0].embedding;
  console.log(`✅ Generated embedding (${queryEmbedding.length} dimensions)\n`);

  // 5. Search Milvus
  console.log('🔎 Searching Milvus for similar tweets...');
  const searchResults = await milvus.search({
    collection_name: 'tweets',
    data: [queryEmbedding],
    limit: 10,
    output_fields: ['tweet_id', 'full_text', 'favorite_count', 'depth', 'created_at'],
  });

  console.log(`✅ Found ${searchResults.results.length} results\n`);

  // 6. Display results
  console.log('📊 Top 10 Most Similar Tweets:\n');
  console.log('='.repeat(80));

  searchResults.results.forEach((result: any, index: number) => {
    console.log(`\n${index + 1}. Similarity: ${(result.score * 100).toFixed(1)}%`);
    console.log(`   Tweet: ${result.full_text.slice(0, 150)}${result.full_text.length > 150 ? '...' : ''}`);
    console.log(`   Likes: ${result.favorite_count} | Depth: ${result.depth}`);
    console.log(`   Created: ${result.created_at.slice(0, 10)}`);
    console.log(`   Tweet ID: ${result.tweet_id}`);
  });

  console.log('\n' + '='.repeat(80));

  // 7. Try another search with filtering
  console.log('\n\n🔍 Testing Search with Filter (tweets with >5 likes)...\n');

  const filteredResults = await milvus.search({
    collection_name: 'tweets',
    data: [queryEmbedding],
    limit: 5,
    filter: 'favorite_count > 5',
    output_fields: ['tweet_id', 'full_text', 'favorite_count'],
  });

  console.log(`✅ Found ${filteredResults.results.length} results with >5 likes\n`);

  filteredResults.results.forEach((result: any, index: number) => {
    console.log(`${index + 1}. [${result.favorite_count} likes] ${result.full_text.slice(0, 100)}...`);
  });

  console.log('\n🎉 Search test complete!\n');
}

// Run the test
testSearch().catch((error) => {
  console.error('❌ Error:', error);
  process.exit(1);
});
