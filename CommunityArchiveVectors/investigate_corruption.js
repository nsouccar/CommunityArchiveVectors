const { createClient } = require('@supabase/supabase-js');

const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY);

(async () => {
  console.log('🔍 INVESTIGATING DATA CORRUPTION PATTERN\n');

  // 1. Check what tables exist
  console.log('📋 Checking available tables...');
  const { data: tables, error } = await supabase.rpc('get_tables');

  // Alternative: just try to query different potential table names
  console.log('\n📊 Trying different table names...');
  const tableNames = ['tweets', 'all_tweets', 'tweet', 'clean_tweets', 'original_tweets'];

  for (const tableName of tableNames) {
    try {
      const { count, error } = await supabase
        .from(tableName)
        .select('*', { count: 'exact', head: true });

      if (!error) {
        console.log(`  ✓ Table "${tableName}" exists: ${count?.toLocaleString()} rows`);
      }
    } catch (e) {
      // Table doesn't exist
    }
  }

  // 2. Check when corruption starts (by tweet_id ranges)
  console.log('\n🔬 Analyzing corruption by tweet_id ranges...');

  const ranges = [
    { name: 'Very old (< 1000000000000000000)', min: '0', max: '1000000000000000000' },
    { name: 'Old (1000000000000000000-1500000000000000000)', min: '1000000000000000000', max: '1500000000000000000' },
    { name: 'Mid (1500000000000000000-1900000000000000000)', min: '1500000000000000000', max: '1900000000000000000' },
    { name: 'Recent (1900000000000000000+)', min: '1900000000000000000', max: '9999999999999999999' },
  ];

  for (const range of ranges) {
    // Count total in range
    const { count: total } = await supabase
      .from('tweets')
      .select('*', { count: 'exact', head: true })
      .gte('tweet_id', range.min)
      .lt('tweet_id', range.max);

    // Count future dates in range
    const { count: futureCount } = await supabase
      .from('tweets')
      .select('*', { count: 'exact', head: true })
      .gte('tweet_id', range.min)
      .lt('tweet_id', range.max)
      .gte('created_at', '2025-01-01');

    const percent = total > 0 ? ((futureCount / total) * 100).toFixed(1) : '0.0';
    console.log(`  ${range.name}:`);
    console.log(`    Total: ${total?.toLocaleString()}, Future dates: ${futureCount?.toLocaleString()} (${percent}%)`);
  }

  // 3. Sample some OLD tweets to see if they have correct dates
  console.log('\n📅 Sampling OLD tweets (should have dates from ~2006-2015)...');
  const { data: oldTweets } = await supabase
    .from('tweets')
    .select('tweet_id, created_at, full_text')
    .lt('tweet_id', '1000000000000000000')
    .order('tweet_id', { ascending: true })
    .limit(5);

  oldTweets?.forEach(tweet => {
    console.log(`  Tweet ${tweet.tweet_id}: ${tweet.created_at} - "${tweet.full_text?.substring(0, 50)}..."`);
  });

  // 4. Check if there's a backup or archived table
  console.log('\n🗄️  Checking for backup/archive tables...');
  const backupNames = [
    'tweets_backup', 'tweets_original', 'tweets_archive',
    'tweets_clean', 'tweets_old', 'clean_data'
  ];

  for (const tableName of backupNames) {
    try {
      const { count, error } = await supabase
        .from(tableName)
        .select('*', { count: 'exact', head: true });

      if (!error) {
        console.log(`  ✓ Found: "${tableName}" with ${count?.toLocaleString()} rows`);
      }
    } catch (e) {
      // Doesn't exist
    }
  }

  // 5. Check the schema to see what columns exist
  console.log('\n📐 Checking tweets table structure...');
  const { data: sampleRow } = await supabase
    .from('tweets')
    .select('*')
    .limit(1);

  if (sampleRow && sampleRow[0]) {
    console.log('  Columns:', Object.keys(sampleRow[0]).join(', '));
  }
})();
