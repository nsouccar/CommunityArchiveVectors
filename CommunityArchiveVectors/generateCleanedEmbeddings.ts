import { createClient } from "@supabase/supabase-js";
import OpenAI from "openai";
import { writeFile } from "fs/promises";
import { join } from "path";

// Initialize Supabase client
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_KEY;

if (!supabaseUrl || !supabaseKey) {
  throw new Error("Missing SUPABASE_URL or SUPABASE_KEY in environment variables");
}

const supabase = createClient(supabaseUrl, supabaseKey);

// Initialize OpenAI client
const openaiApiKey = process.env.OPENAI_API_KEY;

if (!openaiApiKey) {
  throw new Error("Missing OPENAI_API_KEY in environment variables");
}

const openai = new OpenAI({
  apiKey: openaiApiKey,
  project: "proj_d2VDjwr2i3cNRCIHdEBGhVbP"
});

// Type definitions
interface Tweet {
  tweet_id: string;
  full_text: string;
  reply_to_tweet_id: string | null;
  created_at: string;
  account_id: string;
  account_username: string;
  retweet_count: number;
  favorite_count: number;
}

// Function to clean text by removing URLs and mentions
function cleanText(text: string): string {
  let cleaned = text;

  // Remove URLs (http, https, and www links)
  cleaned = cleaned.replace(/https?:\/\/\S+/g, '');
  cleaned = cleaned.replace(/www\.\S+/g, '');

  // Remove @mentions
  cleaned = cleaned.replace(/@\w+/g, '');

  // Remove extra whitespace (multiple spaces, newlines, etc.)
  cleaned = cleaned.replace(/\s+/g, ' ').trim();

  return cleaned;
}

// Helper function to build context with only immediate parent tweet
async function buildThreadContext(tweetId: string, tweetsMap: Map<string, Tweet>): Promise<string[]> {
  const context: string[] = [];
  const tweet = tweetsMap.get(tweetId);

  if (!tweet) return context;

  // If this tweet is a reply, add the parent tweet first
  if (tweet.reply_to_tweet_id) {
    const parentTweet = tweetsMap.get(tweet.reply_to_tweet_id);
    if (parentTweet) {
      const cleanedParent = cleanText(parentTweet.full_text);
      if (cleanedParent.length > 0) {
        context.push(cleanedParent);
      }
    }
  }

  // Add the current tweet
  const cleanedCurrent = cleanText(tweet.full_text);
  if (cleanedCurrent.length > 0) {
    context.push(cleanedCurrent);
  }

  return context;
}

// Helper function to sleep for rate limiting
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function generateCleanedEmbeddings() {
  try {
    console.log("Fetching 1,000 tweets from after October 1, 2024...");

    // Supabase has a 1000 row limit, so we need to paginate
    const TWEETS_TARGET = 1000;
    const PAGE_SIZE = 1000;
    const tweets: Tweet[] = [];

    for (let i = 0; i < TWEETS_TARGET; i += PAGE_SIZE) {
      console.log(`Fetching page ${Math.floor(i / PAGE_SIZE) + 1}/${Math.ceil(TWEETS_TARGET / PAGE_SIZE)}...`);

      const { data, error } = await supabase
        .from("tweets")
        .select("tweet_id, full_text, reply_to_tweet_id, created_at, account_id, account_username, retweet_count, favorite_count")
        .gte("created_at", "2024-10-01T00:00:00Z")
        .order("created_at", { ascending: false })
        .range(i, i + PAGE_SIZE - 1);

      if (error) {
        console.error("Error fetching tweets:", error);
        throw error;
      }

      if (!data || data.length === 0) {
        console.log(`No more tweets found at offset ${i}`);
        break;
      }

      tweets.push(...(data as Tweet[]));
      console.log(`Fetched ${tweets.length} tweets so far...`);
    }

    if (tweets.length === 0) {
      throw new Error("No tweets found");
    }

    console.log(`✅ Fetched ${tweets.length} tweets total. Fetching parent tweets...`);

    // Create a map for quick lookup
    const tweetsMap = new Map<string, Tweet>();
    tweets.forEach((tweet) => {
      tweetsMap.set(tweet.tweet_id, tweet as Tweet);
    });

    // Find all unique parent tweet IDs that we need to fetch
    const parentIdsToFetch = new Set<string>();
    tweets.forEach((tweet) => {
      if (tweet.reply_to_tweet_id && !tweetsMap.has(tweet.reply_to_tweet_id)) {
        parentIdsToFetch.add(tweet.reply_to_tweet_id);
      }
    });

    console.log(`Need to fetch ${parentIdsToFetch.size} parent tweets from database...`);

    // Fetch parent tweets in batches
    if (parentIdsToFetch.size > 0) {
      const parentIds = Array.from(parentIdsToFetch);
      const PARENT_BATCH_SIZE = 1000;

      for (let i = 0; i < parentIds.length; i += PARENT_BATCH_SIZE) {
        const batchIds = parentIds.slice(i, i + PARENT_BATCH_SIZE);

        const { data: parentTweets, error: parentError} = await supabase
          .from("tweets")
          .select("tweet_id, full_text, reply_to_tweet_id, created_at, account_id, account_username, retweet_count, favorite_count")
          .in("tweet_id", batchIds);

        if (parentError) {
          console.error("Error fetching parent tweets:", parentError);
        } else if (parentTweets) {
          parentTweets.forEach((tweet) => {
            tweetsMap.set(tweet.tweet_id, tweet as Tweet);
          });
          console.log(`Fetched ${parentTweets.length} parent tweets...`);
        }
      }
    }

    console.log(`✅ Total tweets in map (including parents): ${tweetsMap.size}. Building cleaned thread contexts...`);

    // Build thread contexts and prepare for embedding
    const embeddingsData = [];
    let processedCount = 0;

    // Rate limiting: Process in batches to stay under 100 RPM
    const BATCH_SIZE = 90; // Process 90 at a time (leave buffer for safety)
    const BATCH_DELAY = 60000; // 60 seconds between batches = ~90/min max

    for (let i = 0; i < tweets.length; i += BATCH_SIZE) {
      const batch = tweets.slice(i, i + BATCH_SIZE);
      const batchPromises = [];

      for (const tweet of batch) {
        const threadContext = await buildThreadContext(tweet.tweet_id, tweetsMap);
        const contextText = threadContext.join("\n\n");

        // Skip if context is empty after cleaning
        if (contextText.length === 0) {
          console.log(`Skipping tweet ${tweet.tweet_id} - no text left after cleaning`);
          processedCount++;
          continue;
        }

        // Find root tweet ID
        let rootId = tweet.tweet_id;
        let currentTweet = tweet;
        while (currentTweet.reply_to_tweet_id && tweetsMap.has(currentTweet.reply_to_tweet_id)) {
          rootId = currentTweet.reply_to_tweet_id;
          currentTweet = tweetsMap.get(currentTweet.reply_to_tweet_id)!;
        }

        // Clean the full text for storage
        const cleanedFullText = cleanText(tweet.full_text);

        // Create promise for embedding generation
        const embeddingPromise = openai.embeddings
          .create({
            model: "text-embedding-3-small",
            input: contextText,
          })
          .then((embeddingResponse) => {
            const embedding = embeddingResponse.data[0].embedding;

            return {
              tweet_id: tweet.tweet_id,
              full_text: cleanedFullText,  // Store cleaned version
              thread_context: contextText,  // Already cleaned
              thread_root_id: rootId,
              depth: threadContext.length - 1,
              is_root: tweet.reply_to_tweet_id === null,
              embedding: embedding,
              metadata: {
                created_at: tweet.created_at,
                account_id: tweet.account_id,
                account_username: tweet.account_username,
                retweet_count: tweet.retweet_count,
                favorite_count: tweet.favorite_count,
              },
            };
          });

        batchPromises.push(embeddingPromise);
      }

      // Wait for all embeddings in this batch to complete
      const batchResults = await Promise.all(batchPromises);
      embeddingsData.push(...batchResults);

      processedCount += batch.length;
      console.log(`Processed ${processedCount}/${tweets.length} tweets...`);

      // Rate limiting: Wait before next batch (unless we're done)
      if (i + BATCH_SIZE < tweets.length) {
        console.log(`Waiting 60 seconds before next batch to respect rate limits...`);
        await sleep(BATCH_DELAY);
      }
    }

    console.log("Saving embeddings to file...");

    // Save to JSON file
    const outputPath = join(process.cwd(), "embeddings_output_cleaned.json");
    await writeFile(outputPath, JSON.stringify(embeddingsData, null, 2));

    console.log(`✅ Cleaned embeddings saved to ${outputPath}`);
    console.log(`📊 Total embeddings generated: ${embeddingsData.length}`);

    return {
      success: true,
      count: embeddingsData.length,
      outputPath,
    };
  } catch (err) {
    console.error("Unexpected error:", err);
    throw err;
  }
}

// Run the function
generateCleanedEmbeddings()
  .then((result) => {
    console.log("✅ Complete!", result);
    process.exit(0);
  })
  .catch((err) => {
    console.error("❌ Failed:", err);
    process.exit(1);
  });
