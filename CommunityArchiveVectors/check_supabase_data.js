const { createClient } = require('@supabase/supabase-js');

const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY);

(async () => {
  console.log('🔗 Connecting to Supabase:', process.env.SUPABASE_URL);
  console.log();

  // Check for tweets containing "love"
  console.log('📊 Checking tweets containing "love"...');
  const { data: loveTweets } = await supabase
    .from('tweets')
    .select('tweet_id, full_text, created_at, account_id')
    .ilike('full_text', '%love%')
    .limit(10);

  console.log(`\n✅ Found ${loveTweets?.length} tweets:\n`);
  loveTweets?.forEach(tweet => {
    console.log(`Tweet ID: ${tweet.tweet_id}`);
    console.log(`Created: ${tweet.created_at}`);
    console.log(`Text: ${tweet.full_text?.substring(0, 100)}...`);
    console.log();
  });

  // Check for tweets from 2025+
  console.log('\n🔮 Checking for tweets with dates in 2025 or later...');
  const { data: futureTweets } = await supabase
    .from('tweets')
    .select('tweet_id, full_text, created_at')
    .gte('created_at', '2025-01-01')
    .limit(10);

  if (futureTweets && futureTweets.length > 0) {
    console.log(`⚠️  Found ${futureTweets.length} tweets from 2025+:`);
    futureTweets.forEach(tweet => {
      console.log(`  - Tweet ${tweet.tweet_id}: ${tweet.created_at}`);
    });
  } else {
    console.log('✅ No tweets from 2025+ found in Supabase');
  }

  // Check date range
  console.log('\n📅 Checking date range in database...');
  const { data: earliest } = await supabase
    .from('tweets')
    .select('created_at')
    .order('created_at', { ascending: true })
    .limit(1);

  if (earliest && earliest.length > 0) {
    console.log(`Earliest tweet: ${earliest[0].created_at}`);
  }

  const { data: latest } = await supabase
    .from('tweets')
    .select('created_at')
    .order('created_at', { ascending: false })
    .limit(1);

  if (latest && latest.length > 0) {
    console.log(`Latest tweet: ${latest[0].created_at}`);
  }

  // Count total tweets
  console.log('\n📈 Total tweet count...');
  const { count } = await supabase
    .from('tweets')
    .select('*', { count: 'exact', head: true });

  console.log(`Total tweets in Supabase: ${count?.toLocaleString()}`);
})();
