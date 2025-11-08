const { createClient } = require('@supabase/supabase-js');
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY);

(async () => {
  console.log('🔬 ANALYZING CORRUPTION BY TWEET_ID RANGES\n');

  // Sample different ranges to see where corruption is concentrated
  const ranges = [
    { name: 'First 1,000', order: 'asc', limit: 1000 },
    { name: 'First 10,000', order: 'asc', limit: 10000 },
    { name: 'Latest 1,000', order: 'desc', limit: 1000 },
  ];

  for (const range of ranges) {
    console.log(`📊 Checking ${range.name} tweets (ordered by tweet_id ${range.order})...`);

    // Get tweets in this range
    const { data: tweets } = await supabase
      .from('tweets')
      .select('tweet_id, created_at')
      .order('tweet_id', { ascending: range.order === 'asc' })
      .limit(range.limit);

    if (!tweets || tweets.length === 0) continue;

    // Count how many have future dates
    const futureCount = tweets.filter(t => {
      const date = new Date(t.created_at);
      return date >= new Date('2025-01-01');
    }).length;

    const percent = ((futureCount / tweets.length) * 100).toFixed(1);

    console.log(`  Total: ${tweets.length}`);
    console.log(`  Future dates (2025+): ${futureCount} (${percent}%)`);
    console.log(`  Tweet ID range: ${tweets[0].tweet_id} to ${tweets[tweets.length-1].tweet_id}`);
    console.log();
  }
})();
