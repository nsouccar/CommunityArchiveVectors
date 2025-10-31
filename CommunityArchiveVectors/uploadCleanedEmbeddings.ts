import { MilvusClient } from '@zilliz/milvus2-sdk-node';
import { readFile } from 'fs/promises';

async function uploadCleanedEmbeddings() {
  const client = new MilvusClient({ address: 'localhost:19530' });
  const collectionName = 'tweets';

  console.log('📖 Reading cleaned embeddings file...');

  const fileContent = await readFile('embeddings_output_cleaned.json', 'utf-8');
  const embeddings = JSON.parse(fileContent);

  console.log(`Found ${embeddings.length} embeddings to upload`);

  console.log('🗑️  Dropping existing collection...');

  // Drop the existing collection
  const hasCollection = await client.hasCollection({ collection_name: collectionName });
  if (hasCollection.value) {
    await client.dropCollection({ collection_name: collectionName });
    console.log('✅ Collection dropped');
  }

  console.log('🏗️  Recreating collection with cleaned data schema...');

  // Recreate the collection (same schema as before)
  const { DataType } = await import('@zilliz/milvus2-sdk-node');

  await client.createCollection({
    collection_name: collectionName,
    fields: [
      { name: 'tweet_id', data_type: DataType.VarChar, is_primary_key: true, max_length: 100 },
      { name: 'embedding', data_type: DataType.FloatVector, dim: 1536 },
      { name: 'full_text', data_type: DataType.VarChar, max_length: 5000 },
      { name: 'thread_context', data_type: DataType.VarChar, max_length: 10000 },
      { name: 'thread_root_id', data_type: DataType.VarChar, max_length: 100 },
      { name: 'depth', data_type: DataType.Int64 },
      { name: 'is_root', data_type: DataType.Bool },
      { name: 'account_id', data_type: DataType.VarChar, max_length: 100 },
      { name: 'account_username', data_type: DataType.VarChar, max_length: 100 },
      { name: 'favorite_count', data_type: DataType.Int64 },
      { name: 'retweet_count', data_type: DataType.Int64 },
      { name: 'created_at', data_type: DataType.VarChar, max_length: 100 },
      { name: 'processed_at', data_type: DataType.VarChar, max_length: 100 },
      { name: 'embedding_version', data_type: DataType.VarChar, max_length: 50 },
    ],
  });

  console.log('✅ Collection created');

  console.log('📊 Creating HNSW index...');

  // Create HNSW index
  await client.createIndex({
    collection_name: collectionName,
    field_name: 'embedding',
    index_type: 'HNSW',
    metric_type: 'COSINE',
    params: { M: 16, efConstruction: 200 },
  });

  console.log('✅ Index created');

  console.log('📥 Preparing data for insertion...');

  // Prepare data for insertion
  const data = embeddings.map((item: any) => ({
    tweet_id: item.tweet_id,
    embedding: item.embedding,
    full_text: item.full_text.slice(0, 5000),
    thread_context: item.thread_context.slice(0, 10000),
    thread_root_id: item.thread_root_id,
    depth: item.depth,
    is_root: item.is_root,
    account_id: item.metadata.account_id || '',
    account_username: item.metadata.account_username || '',
    favorite_count: item.metadata.favorite_count || 0,
    retweet_count: item.metadata.retweet_count || 0,
    created_at: item.metadata.created_at || '',
    processed_at: new Date().toISOString(),
    embedding_version: 'openai-text-embedding-3-small-cleaned',
  }));

  console.log('⬆️  Inserting embeddings into Milvus...');

  await client.insert({
    collection_name: collectionName,
    data: data,
  });

  console.log('💾 Flushing data to disk...');

  await client.flush({
    collection_names: [collectionName],
  });

  console.log('📂 Loading collection into memory...');

  await client.loadCollection({
    collection_name: collectionName,
  });

  console.log('✅ Done! Cleaned embeddings uploaded successfully');
  console.log(`📊 Total vectors in collection: ${embeddings.length}`);
}

uploadCleanedEmbeddings()
  .then(() => {
    console.log('🎉 Upload complete!');
    process.exit(0);
  })
  .catch((err) => {
    console.error('❌ Error:', err);
    process.exit(1);
  });
