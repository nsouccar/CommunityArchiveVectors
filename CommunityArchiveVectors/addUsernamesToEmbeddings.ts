import { createClient } from "@supabase/supabase-js";
import { readFile, writeFile } from "fs/promises";

// Initialize Supabase client
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_KEY;

if (!supabaseUrl || !supabaseKey) {
  throw new Error("Missing SUPABASE_URL or SUPABASE_KEY in environment variables");
}

const supabase = createClient(supabaseUrl, supabaseKey);

async function lookupUsername(accountId: string): Promise<string | null> {
  // Look up username from all_account table
  const { data, error } = await supabase
    .from("all_account")
    .select("username")
    .eq("account_id", accountId)
    .limit(1);

  if (error || !data || data.length === 0) {
    return null;
  }

  return data[0].username;
}

async function addUsernames() {
  console.log("📖 Reading embeddings file...");

  const fileContent = await readFile("embeddings_output_cleaned.json", "utf-8");
  const embeddings = JSON.parse(fileContent);

  console.log(`Found ${embeddings.length} embeddings`);

  // Get all unique account IDs
  const uniqueAccountIds = new Set<string>();
  embeddings.forEach((item: any) => {
    if (item.metadata.account_id) {
      uniqueAccountIds.add(item.metadata.account_id);
    }
  });

  console.log(`Found ${uniqueAccountIds.size} unique account IDs`);
  console.log("Looking up usernames from Supabase...");

  // Build a map of account_id -> username
  const usernameMap = new Map<string, string>();
  let processed = 0;

  for (const accountId of uniqueAccountIds) {
    const username = await lookupUsername(accountId);
    if (username) {
      usernameMap.set(accountId, username);
      console.log(`✓ ${accountId} -> @${username}`);
    } else {
      console.log(`✗ ${accountId} -> username not found`);
    }

    processed++;
    if (processed % 10 === 0) {
      console.log(`Progress: ${processed}/${uniqueAccountIds.size}`);
    }
  }

  console.log(`\n✅ Found ${usernameMap.size}/${uniqueAccountIds.size} usernames`);

  // Add usernames to embeddings
  console.log("Adding usernames to embeddings...");
  embeddings.forEach((item: any) => {
    const username = usernameMap.get(item.metadata.account_id);
    item.metadata.account_username = username || item.metadata.account_id;
  });

  // Save updated embeddings
  console.log("💾 Saving updated embeddings...");
  await writeFile("embeddings_output_cleaned.json", JSON.stringify(embeddings, null, 2));

  console.log("🎉 Done! Embeddings now include usernames");
}

addUsernames()
  .then(() => {
    console.log("✅ Success!");
    process.exit(0);
  })
  .catch((err) => {
    console.error("❌ Error:", err);
    process.exit(1);
  });
