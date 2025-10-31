const fs = require('fs');
const data = JSON.parse(fs.readFileSync('embeddings_output_cleaned.json', 'utf-8'));

console.log('Total embeddings:', data.length);
console.log('='.repeat(100));

data.forEach((item, index) => {
  console.log('');
  console.log('[' + (index + 1) + '] Tweet ID: ' + item.tweet_id);
  console.log('Depth: ' + item.depth + ' | Root: ' + item.is_root);
  console.log('Created: ' + item.metadata.created_at);
  console.log('Tweet: ' + item.full_text);
  if (item.thread_context !== item.full_text) {
    console.log('Thread Context: ' + item.thread_context);
  }
  console.log('-'.repeat(100));
});
