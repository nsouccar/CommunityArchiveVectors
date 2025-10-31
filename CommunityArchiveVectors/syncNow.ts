import { EmbeddingSync } from './src/services/embeddingSync';

async function main() {
  const syncer = new EmbeddingSync();

  // Sync all tweets since October 1st
  await syncer.syncTweets('2025-10-01T00:00:00');

  // Show stats
  const stats = await syncer.getStats();
  console.log('\n📊 Current Stats:');
  console.log(`   Total vectors in Milvus: ${stats.totalVectors}`);
}

main().catch((error) => {
  console.error('Error:', error);
  process.exit(1);
});
