/**
 * Create username → avatar URL mapping from Supabase
 * No downloads needed - just a mapping file for the frontend
 */

const { createClient } = require('@supabase/supabase-js');
const fs = require('fs');
const path = require('path');

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_KEY
);

async function main() {
  console.log('Creating username → avatar URL mapping...\n');

  // Get all profiles with avatars
  console.log('Fetching profiles with avatars...');
  const { data: profiles, error: profileError } = await supabase
    .from('all_profile')
    .select('account_id, avatar_media_url')
    .not('avatar_media_url', 'is', null);

  if (profileError) {
    console.error('Error fetching profiles:', profileError);
    return;
  }

  console.log(`Found ${profiles.length} profiles with avatars`);

  // Create account_id → avatar_url map
  const avatarMap = {};
  profiles.forEach(p => {
    // Upgrade to higher quality (_400x400 instead of _normal)
    avatarMap[p.account_id] = p.avatar_media_url.replace('_normal', '_400x400');
  });

  // Get usernames for each account_id in batches
  console.log('Fetching usernames...');
  const accountIds = Object.keys(avatarMap);
  const usernameMapping = {};
  const batchSize = 1000;

  for (let i = 0; i < accountIds.length; i += batchSize) {
    const batch = accountIds.slice(i, i + batchSize);

    const { data: accounts, error: accountsError } = await supabase
      .from('all_account')
      .select('account_id, username')
      .in('account_id', batch);

    if (accountsError) {
      console.error('Error fetching accounts:', accountsError);
      continue;
    }

    accounts.forEach(a => {
      const username = a.username;
      const accountId = a.account_id;
      if (username && accountId in avatarMap) {
        usernameMapping[username] = avatarMap[accountId];
      }
    });

    console.log(`  Processed ${Math.min(i + batchSize, accountIds.length)}/${accountIds.length}...`);
  }

  // Save mapping
  const outputFile = path.join(__dirname, 'frontend/public/avatar_mapping.json');
  fs.writeFileSync(outputFile, JSON.stringify(usernameMapping, null, 2));

  console.log(`\n✓ Created ${outputFile}`);
  console.log(`  ${Object.keys(usernameMapping).length} username → avatar URL mappings`);
  console.log(`\nFrontend usage:`);
  console.log(`  const avatar = avatarMap[username];`);
  console.log(`  <img src={avatar} />`);
}

main().catch(console.error);
