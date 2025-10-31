import { MilvusClient, DataType } from '@zilliz/milvus2-sdk-node';
import { readFile } from 'fs/promises';

async function uploadToMilvus() {
  console.log('🚀 Starting Milvus upload process...\n');

  // 1. Connect to Milvus
  console.log('📡 Connecting to Milvus...');
  const client = new MilvusClient({
    address: 'localhost:19530',
  });

  console.log('✅ Connected to Milvus!\n');

  // 2. Check if collection exists and drop it if it does (for clean start)
  const collectionName = 'tweets';

  try {
    const hasCollection = await client.hasCollection({ collection_name: collectionName });
    if (hasCollection.value) {
      console.log('🗑️  Dropping existing collection...');
      await client.dropCollection({ collection_name: collectionName });
      console.log('✅ Old collection dropped\n');
    }
  } catch (error) {
    console.log('ℹ️  No existing collection to drop\n');
  }

  // 3. Create collection schema
  console.log('📋 Creating collection schema...');
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
        dim: 1024, // Voyage-3 dimensions
      },
      {
        name: 'full_text',
        data_type: DataType.VarChar,
        max_length: 2000,
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
    ],
  });

  console.log('✅ Collection created!\n');

  // 4. Create index for vector search
  console.log('🔍 Creating vector index (HNSW)...');
  await client.createIndex({
    collection_name: collectionName,
    field_name: 'embedding',
    index_type: 'HNSW',
    metric_type: 'COSINE', // Cosine similarity for semantic search
    params: {
      M: 16,              // Number of bi-directional links
      efConstruction: 200, // Size of dynamic candidate list
    },
  });

  console.log('✅ Index created!\n');

  // 5. Load collection into memory
  console.log('💾 Loading collection into memory...');
  await client.loadCollection({ collection_name: collectionName });
  console.log('✅ Collection loaded!\n');

  // 6. Read Voyage embeddings from JSON file
  console.log('📂 Reading embeddings from file...');
  const fileContent = await readFile('embeddings_output_voyage.json', 'utf-8');
  const embeddings = JSON.parse(fileContent);
  console.log(`✅ Loaded ${embeddings.length} embeddings from file\n`);

  // 7. Prepare data for insertion
  console.log('🔄 Preparing data for insertion...');
  const data = embeddings.map((item: any) => ({
    tweet_id: item.tweet_id,
    embedding: item.embedding,
    full_text: item.full_text.slice(0, 2000), // Truncate if too long
    thread_root_id: item.thread_root_id,
    depth: item.depth,
    is_root: item.is_root,
    favorite_count: item.metadata.favorite_count || 0,
    retweet_count: item.metadata.retweet_count || 0,
    created_at: item.metadata.created_at || '',
  }));

  // 8. Insert data in batches (Milvus recommends batches of 1000-5000)
  console.log('📤 Inserting data into Milvus...');
  const batchSize = 1000;
  let insertedCount = 0;

  for (let i = 0; i < data.length; i += batchSize) {
    const batch = data.slice(i, i + batchSize);

    await client.insert({
      collection_name: collectionName,
      data: batch,
    });

    insertedCount += batch.length;
    console.log(`   Inserted ${insertedCount}/${data.length} vectors...`);
  }

  console.log('✅ All data inserted!\n');

  // 9. Flush to ensure data is persisted
  console.log('💾 Flushing data to disk...');
  await client.flush({ collection_names: [collectionName] });
  console.log('✅ Data flushed!\n');

  // 10. Get collection statistics
  console.log('📊 Collection Statistics:');
  const stats = await client.getCollectionStatistics({ collection_name: collectionName });
  console.log(`   Total entities: ${stats.data.row_count}`);

  console.log('\n🎉 Upload complete! Your vectors are now in Milvus!\n');
  console.log('Next steps:');
  console.log('  1. Test a search: bun testSearch.ts');
  console.log('  2. Access Web UI: http://localhost:9091/webui/\n');
}

// Run the upload
uploadToMilvus().catch((error) => {
  console.error('❌ Error:', error);
  process.exit(1);
});
