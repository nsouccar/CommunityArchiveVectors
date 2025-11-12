/**
 * Export username → avatar URL mapping from Supabase
 * Uses the URLs stored in all_profile table (Twitter CDN)
 */

const { createClient } = require('@supabase/supabase-js');
const fs = require('fs');
const path = require('path');

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_KEY
);

async function main() {
  console.log('Fetching avatar URLs from Supabase...\n');

  // Get all profiles with avatars
  console.log('Step 1: Fetching profiles...');
  let allProfiles = [];
  let page = 0;
  const pageSize = 1000;

  while (true) {
    const { data, error } = await supabase
      .from('all_profile')
      .select('account_id, avatar_media_url')
      .not('avatar_media_url', 'is', null)
      .range(page * pageSize, (page + 1) * pageSize - 1);

    if (error) {
      console.error('Error:', error);
      break;
    }

    if (!data || data.length === 0) break;

    allProfiles = allProfiles.concat(data);
    console.log(`  Fetched ${allProfiles.length} profiles...`);

    if (data.length < pageSize) break;
    page++;
  }

  console.log(`\nTotal profiles with avatars: ${allProfiles.length}`);

  // Create account_id → avatar_url map
  const avatarByAccountId = {};
  allProfiles.forEach(p => {
    // Upgrade to higher quality
    avatarByAccountId[p.account_id] = p.avatar_media_url.replace('_normal', '_400x400');
  });

  // Get usernames from all_account table
  console.log('\nStep 2: Fetching usernames from all_account...');
  const accountIds = Object.keys(avatarByAccountId);
  const usernameMapping = {};
  const batchSize = 500; // Smaller batches to avoid network issues

  for (let i = 0; i < accountIds.length; i += batchSize) {
    const batch = accountIds.slice(i, i + batchSize);

    // Retry logic for network errors
    let success = false;
    let retries = 0;
    const maxRetries = 3;

    while (!success && retries < maxRetries) {
      try {
        const { data, error } = await supabase
          .from('all_account')
          .select('account_id, username')
          .in('account_id', batch);

        if (error) {
          throw error;
        }

        data.forEach(account => {
          const username = account.username;
          const accountId = account.account_id;
          if (username && avatarByAccountId[accountId]) {
            usernameMapping[username] = avatarByAccountId[accountId];
          }
        });

        success = true;
      } catch (error) {
        retries++;
        if (retries < maxRetries) {
          console.log(`  Retry ${retries}/${maxRetries} for batch ${i}...`);
          await new Promise(resolve => setTimeout(resolve, 1000 * retries)); // Wait longer each retry
        } else {
          console.error(`  Failed after ${maxRetries} retries:`, error.message);
        }
      }
    }

    if (i % 5000 === 0 || i + batchSize >= accountIds.length) {
      console.log(`  Processed ${Math.min(i + batchSize, accountIds.length)}/${accountIds.length}...`);
    }
  }

  // Save mapping
  const outputFile = path.join(__dirname, 'frontend/public/avatar_urls.json');
  fs.writeFileSync(outputFile, JSON.stringify(usernameMapping, null, 2));

  console.log(`\n✓ Created ${outputFile}`);
  console.log(`  ${Object.keys(usernameMapping).length} username → Twitter CDN URL mappings`);
  console.log(`\nFrontend usage:`);
  console.log(`  import avatarUrls from './avatar_urls.json';`);
  console.log(`  const avatarUrl = avatarUrls[username];`);
  console.log(`  <img src={avatarUrl} alt={username} />`);
}

main().catch(console.error);
