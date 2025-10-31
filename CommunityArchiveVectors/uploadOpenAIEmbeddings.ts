import { MilvusClient } from '@zilliz/milvus2-sdk-node';
import { readFile } from 'fs/promises';

async function uploadOpenAIEmbeddings() {
  console.log('🚀 Uploading existing OpenAI embeddings to Milvus...\n');

  // Connect to Milvus
  const client = new MilvusClient({ address: 'localhost:19530' });
  console.log('✅ Connected to Milvus\n');

  // Read existing OpenAI embeddings
  console.log('📂 Reading embeddings_output.json...');
  const fileContent = await readFile('embeddings_output.json', 'utf-8');
  const embeddings = JSON.parse(fileContent);
  console.log(`✅ Loaded ${embeddings.length} OpenAI embeddings\n`);

  // Prepare data for Milvus
  console.log('🔄 Preparing data...');
  const data = embeddings.map((item: any) => ({
    tweet_id: item.tweet_id,
    embedding: item.embedding,
    full_text: item.full_text.slice(0, 5000),
    thread_context: item.thread_context.slice(0, 10000),
    thread_root_id: item.thread_root_id,
    depth: item.depth,
    is_root: item.is_root,
    account_id: item.metadata.account_id || '',
    favorite_count: item.metadata.favorite_count || 0,
    retweet_count: item.metadata.retweet_count || 0,
    created_at: item.metadata.created_at || '',
    processed_at: new Date().toISOString(),
    embedding_version: 'openai-text-embedding-3-small',
  }));

  console.log('✅ Data prepared\n');

  // Insert in batches
  console.log('📤 Inserting into Milvus...');
  const BATCH_SIZE = 1000;
  let inserted = 0;

  for (let i = 0; i < data.length; i += BATCH_SIZE) {
    const batch = data.slice(i, i + BATCH_SIZE);

    await client.insert({
      collection_name: 'tweets',
      data: batch,
    });

    inserted += batch.length;
    console.log(`   Inserted ${inserted}/${data.length} embeddings...`);
  }

  // Flush to persist
  console.log('\n💾 Flushing data to disk...');
  await client.flush({ collection_names: ['tweets'] });
  console.log('✅ Data flushed!\n');

  // Get stats
  const stats = await client.getCollectionStatistics({ collection_name: 'tweets' });
  console.log('📊 Collection Statistics:');
  console.log(`   Total vectors: ${stats.data.row_count}\n`);

  console.log('🎉 Upload complete! Ready for semantic search!\n');
}

uploadOpenAIEmbeddings().catch((error) => {
  console.error('❌ Error:', error);
  process.exit(1);
});
