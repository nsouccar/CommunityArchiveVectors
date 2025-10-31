import { MilvusClient, DataType } from '@zilliz/milvus2-sdk-node';

async function initializeMilvus() {
  console.log('🚀 Initializing Milvus with OpenAI schema...\n');

  const client = new MilvusClient({
    address: 'localhost:19530',
  });

  const collectionName = 'tweets';

  // Drop existing collection if it exists
  try {
    const hasCollection = await client.hasCollection({ collection_name: collectionName });
    if (hasCollection.value) {
      console.log('🗑️  Dropping existing collection...');
      await client.dropCollection({ collection_name: collectionName });
      console.log('✅ Old collection dropped\n');
    }
  } catch (error) {
    console.log('ℹ️  No existing collection\n');
  }

  // Create collection with OpenAI schema (1536 dimensions)
  console.log('📋 Creating collection with OpenAI embeddings (1536 dims)...');
  await client.createCollection({
    collection_name: collectionName,
    fields: [
      {
        name: 'tweet_id',
        data_type: DataType.VarChar,
        is_primary_key: true,
        max_length: 100,
      },
      {
        name: 'embedding',
        data_type: DataType.FloatVector,
        dim: 1536, // OpenAI text-embedding-3-small
      },
      {
        name: 'full_text',
        data_type: DataType.VarChar,
        max_length: 5000,
      },
      {
        name: 'thread_context',
        data_type: DataType.VarChar,
        max_length: 10000,
      },
      {
        name: 'thread_root_id',
        data_type: DataType.VarChar,
        max_length: 100,
      },
      {
        name: 'depth',
        data_type: DataType.Int64,
      },
      {
        name: 'is_root',
        data_type: DataType.Bool,
      },
      {
        name: 'account_id',
        data_type: DataType.VarChar,
        max_length: 100,
      },
      {
        name: 'favorite_count',
        data_type: DataType.Int64,
      },
      {
        name: 'retweet_count',
        data_type: DataType.Int64,
      },
      {
        name: 'created_at',
        data_type: DataType.VarChar,
        max_length: 100,
      },
      {
        name: 'processed_at',
        data_type: DataType.VarChar,
        max_length: 100,
      },
      {
        name: 'embedding_version',
        data_type: DataType.VarChar,
        max_length: 50,
      },
    ],
  });

  console.log('✅ Collection created!\n');

  // Create HNSW index for vector search
  console.log('🔍 Creating HNSW index...');
  await client.createIndex({
    collection_name: collectionName,
    field_name: 'embedding',
    index_type: 'HNSW',
    metric_type: 'COSINE',
    params: {
      M: 16,
      efConstruction: 200,
    },
  });

  console.log('✅ Vector index created!\n');

  // Load collection into memory
  console.log('💾 Loading collection into memory...');
  await client.loadCollection({ collection_name: collectionName });
  console.log('✅ Collection loaded!\n');

  console.log('🎉 Milvus initialized and ready for OpenAI embeddings!\n');
  console.log('Collection details:');
  console.log('  - Name: tweets');
  console.log('  - Vector dimensions: 1536 (OpenAI text-embedding-3-small)');
  console.log('  - Index: HNSW with cosine similarity');
  console.log('  - Fields: 13 total (including metadata)\n');
}

initializeMilvus().catch((error) => {
  console.error('❌ Error:', error);
  process.exit(1);
});
