import express from "express";
import { createClient } from "@supabase/supabase-js";
import OpenAI from "openai";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { VoyageAIClient } from "voyageai";
import { writeFile } from "fs/promises";
import { join } from "path";

const app = express();

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

// Initialize Gemini client
const geminiApiKey = process.env.GEMINI_API_KEY;

if (!geminiApiKey) {
  console.warn("Missing GEMINI_API_KEY - Gemini embeddings route will not work");
}

const genAI = geminiApiKey ? new GoogleGenerativeAI(geminiApiKey) : null;

// Initialize Voyage AI client
const voyageApiKey = process.env.VOYAGE_API_KEY;

if (!voyageApiKey) {
  console.warn("Missing VOYAGE_API_KEY - Voyage AI embeddings route will not work");
}

const voyageClient = voyageApiKey ? new VoyageAIClient({ apiKey: voyageApiKey }) : null;

// Type definitions
interface Tweet {
  tweet_id: string;
  full_text: string;
  reply_to_tweet_id: string | null;
  created_at: string;
  account_id: string;
  retweet_count: number;
  favorite_count: number;
}

app.get("/hello", (_, res) => {
  res.send("Hello Vite + TypeScript!");
});

// Helper function to build thread context for a tweet
async function buildThreadContext(tweetId: string, tweetsMap: Map<string, Tweet>): Promise<string[]> {
  const thread: string[] = [];
  let currentId: string | null = tweetId;
  const visited = new Set<string>();

  // Traverse up the reply chain to build full context
  while (currentId && !visited.has(currentId)) {
    visited.add(currentId);
    const tweet = tweetsMap.get(currentId);

    if (!tweet) break;

    thread.unshift(tweet.full_text); // Add to beginning (root first)
    currentId = tweet.reply_to_tweet_id;
  }

  return thread;
}

// Route to fetch tweets from Supabase
app.get("/tweets", async (_, res) => {
  try {
    // Get total count
    const { count, error: countError } = await supabase
      .from("tweets")
      .select("*", { count: "exact", head: true });

    if (countError) {
      console.error("Error counting tweets:", countError);
    } else {
      console.log(`Total tweets in database: ${count}`);
    }

    // Get actual data
    const { data, error } = await supabase
      .from("tweets")
      .select("*");

    if (error) {
      console.error("Error fetching tweets:", error);
      return res.status(500).json({ error: error.message });
    }

    return res.json({ tweets: data, count: data?.length || 0, total_in_db: count });
  } catch (err) {
    console.error("Unexpected error:", err);
    return res.status(500).json({ error: "Internal server error" });
  }
});

// Helper function to sleep for rate limiting
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Route to generate embeddings for 1,000 tweets with rate limiting
app.get("/generate-embeddings", async (_, res) => {
  try {
    console.log("Fetching 1,000 tweets from database...");

    // Fetch 1,000 tweets for testing
    const { data: tweets, error } = await supabase
      .from("tweets")
      .select("tweet_id, full_text, reply_to_tweet_id, created_at, account_id, retweet_count, favorite_count")
      .limit(1000);

    if (error) {
      console.error("Error fetching tweets:", error);
      return res.status(500).json({ error: error.message });
    }

    if (!tweets || tweets.length === 0) {
      return res.status(404).json({ error: "No tweets found" });
    }

    console.log(`Fetched ${tweets.length} tweets. Building thread contexts...`);

    // Create a map for quick lookup
    const tweetsMap = new Map<string, Tweet>();
    tweets.forEach((tweet) => {
      tweetsMap.set(tweet.tweet_id, tweet as Tweet);
    });

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

        // Find root tweet ID
        let rootId = tweet.tweet_id;
        let currentTweet = tweet;
        while (currentTweet.reply_to_tweet_id && tweetsMap.has(currentTweet.reply_to_tweet_id)) {
          rootId = currentTweet.reply_to_tweet_id;
          currentTweet = tweetsMap.get(currentTweet.reply_to_tweet_id)!;
        }

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
              full_text: tweet.full_text,
              thread_context: contextText,
              thread_root_id: rootId,
              depth: threadContext.length - 1,
              is_root: tweet.reply_to_tweet_id === null,
              embedding: embedding,
              metadata: {
                created_at: tweet.created_at,
                account_id: tweet.account_id,
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
        await sleep(BATCH_DELAY);
      }
    }

    console.log("Saving embeddings to file...");

    // Save to JSON file
    const outputPath = join(process.cwd(), "embeddings_output.json");
    await writeFile(outputPath, JSON.stringify(embeddingsData, null, 2));

    console.log(`Embeddings saved to ${outputPath}`);

    return res.json({
      success: true,
      count: embeddingsData.length,
      outputPath,
      message: "Embeddings generated and saved successfully",
    });
  } catch (err) {
    console.error("Unexpected error:", err);
    return res.status(500).json({ error: "Internal server error", details: String(err) });
  }
});

// Route to generate Gemini embeddings for 1,000 tweets with rate limiting
app.get("/generate-embeddings-gemini", async (_, res) => {
  try {
    if (!genAI) {
      return res.status(500).json({ error: "Gemini API key not configured" });
    }

    console.log("Fetching 1,000 tweets from database...");

    // Fetch 1,000 tweets for testing
    const { data: tweets, error } = await supabase
      .from("tweets")
      .select("tweet_id, full_text, reply_to_tweet_id, created_at, account_id, retweet_count, favorite_count")
      .limit(1000);

    if (error) {
      console.error("Error fetching tweets:", error);
      return res.status(500).json({ error: error.message });
    }

    if (!tweets || tweets.length === 0) {
      return res.status(404).json({ error: "No tweets found" });
    }

    console.log(`Fetched ${tweets.length} tweets. Building thread contexts...`);

    // Create a map for quick lookup
    const tweetsMap = new Map<string, Tweet>();
    tweets.forEach((tweet) => {
      tweetsMap.set(tweet.tweet_id, tweet as Tweet);
    });

    // Build thread contexts and prepare for embedding
    const embeddingsData = [];
    let processedCount = 0;

    // Get the embedding model
    const model = genAI.getGenerativeModel({ model: "text-embedding-004" });

    // Gemini has different rate limits - adjust as needed
    const BATCH_SIZE = 50; // Gemini can handle larger batches
    const BATCH_DELAY = 1000; // 1 second between batches

    for (let i = 0; i < tweets.length; i += BATCH_SIZE) {
      const batch = tweets.slice(i, i + BATCH_SIZE);
      const batchPromises = [];

      for (const tweet of batch) {
        const threadContext = await buildThreadContext(tweet.tweet_id, tweetsMap);
        const contextText = threadContext.join("\n\n");

        // Find root tweet ID
        let rootId = tweet.tweet_id;
        let currentTweet = tweet;
        while (currentTweet.reply_to_tweet_id && tweetsMap.has(currentTweet.reply_to_tweet_id)) {
          rootId = currentTweet.reply_to_tweet_id;
          currentTweet = tweetsMap.get(currentTweet.reply_to_tweet_id)!;
        }

        // Create promise for embedding generation using Gemini
        const embeddingPromise = model
          .embedContent(contextText)
          .then((result) => {
            const embedding = result.embedding.values;

            return {
              tweet_id: tweet.tweet_id,
              full_text: tweet.full_text,
              thread_context: contextText,
              thread_root_id: rootId,
              depth: threadContext.length - 1,
              is_root: tweet.reply_to_tweet_id === null,
              embedding: embedding,
              metadata: {
                created_at: tweet.created_at,
                account_id: tweet.account_id,
                retweet_count: tweet.retweet_count,
                favorite_count: tweet.favorite_count,
                model: "text-embedding-004",
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
        await sleep(BATCH_DELAY);
      }
    }

    console.log("Saving embeddings to file...");

    // Save to JSON file with different name
    const outputPath = join(process.cwd(), "embeddings_output_gemini.json");
    await writeFile(outputPath, JSON.stringify(embeddingsData, null, 2));

    console.log(`Gemini embeddings saved to ${outputPath}`);

    return res.json({
      success: true,
      count: embeddingsData.length,
      outputPath,
      model: "text-embedding-004",
      message: "Gemini embeddings generated and saved successfully",
    });
  } catch (err) {
    console.error("Unexpected error:", err);
    return res.status(500).json({ error: "Internal server error", details: String(err) });
  }
});

// Route to generate Voyage AI embeddings for 10,000 tweets with rate limiting
app.get("/generate-embeddings-voyage", async (_, res) => {
  try {
    if (!voyageClient) {
      return res.status(500).json({ error: "Voyage AI API key not configured" });
    }

    console.log("Fetching 10,000 tweets from database...");

    // Fetch 10,000 tweets
    const { data: tweets, error } = await supabase
      .from("tweets")
      .select("tweet_id, full_text, reply_to_tweet_id, created_at, account_id, retweet_count, favorite_count")
      .limit(10000);

    if (error) {
      console.error("Error fetching tweets:", error);
      return res.status(500).json({ error: error.message });
    }

    if (!tweets || tweets.length === 0) {
      return res.status(404).json({ error: "No tweets found" });
    }

    console.log(`Fetched ${tweets.length} tweets. Building thread contexts...`);

    // Create a map for quick lookup
    const tweetsMap = new Map<string, Tweet>();
    tweets.forEach((tweet) => {
      tweetsMap.set(tweet.tweet_id, tweet as Tweet);
    });

    // Build thread contexts and prepare for embedding
    const embeddingsData = [];
    let processedCount = 0;

    // Voyage AI rate limits: Tier 1 = 2000 RPM
    // With 2000 RPM, we can do 1 request every 30ms
    // Using batch size 128 to maximize throughput while staying well under limits
    const BATCH_SIZE = 128; // Voyage can handle up to 128 inputs per request
    const BATCH_DELAY = 100; // 100ms between batches = 600 requests/min (well under 2000 RPM)

    for (let i = 0; i < tweets.length; i += BATCH_SIZE) {
      const batch = tweets.slice(i, i + BATCH_SIZE);

      // Build all thread contexts for this batch
      const batchInputs = [];
      const batchMetadata = [];

      for (const tweet of batch) {
        const threadContext = await buildThreadContext(tweet.tweet_id, tweetsMap);
        const contextText = threadContext.join("\n\n");

        // Find root tweet ID
        let rootId = tweet.tweet_id;
        let currentTweet = tweet;
        while (currentTweet.reply_to_tweet_id && tweetsMap.has(currentTweet.reply_to_tweet_id)) {
          rootId = currentTweet.reply_to_tweet_id;
          currentTweet = tweetsMap.get(currentTweet.reply_to_tweet_id)!;
        }

        batchInputs.push(contextText);
        batchMetadata.push({
          tweet_id: tweet.tweet_id,
          full_text: tweet.full_text,
          thread_context: contextText,
          thread_root_id: rootId,
          depth: threadContext.length - 1,
          is_root: tweet.reply_to_tweet_id === null,
          metadata: {
            created_at: tweet.created_at,
            account_id: tweet.account_id,
            retweet_count: tweet.retweet_count,
            favorite_count: tweet.favorite_count,
            model: "voyage-3",
          },
        });
      }

      // Call Voyage AI API with batch of inputs
      console.log(`Calling Voyage AI for batch ${Math.floor(i / BATCH_SIZE) + 1}...`);
      const embeddingResponse = await voyageClient.embed({
        input: batchInputs,
        model: "voyage-3", // Using voyage-3 as a good general-purpose model
        inputType: "document", // These are documents, not queries
      });

      // Combine embeddings with metadata
      if (embeddingResponse.data) {
        for (let j = 0; j < embeddingResponse.data.length; j++) {
          embeddingsData.push({
            ...batchMetadata[j],
            embedding: embeddingResponse.data[j].embedding,
          });
        }
      }

      processedCount += batch.length;
      console.log(`Processed ${processedCount}/${tweets.length} tweets...`);

      // Rate limiting: Wait before next batch (unless we're done)
      if (i + BATCH_SIZE < tweets.length) {
        await sleep(BATCH_DELAY);
      }
    }

    console.log("Saving embeddings to file...");

    // Save to JSON file with Voyage-specific name
    const outputPath = join(process.cwd(), "embeddings_output_voyage.json");
    await writeFile(outputPath, JSON.stringify(embeddingsData, null, 2));

    console.log(`Voyage AI embeddings saved to ${outputPath}`);

    return res.json({
      success: true,
      count: embeddingsData.length,
      outputPath,
      model: "voyage-3",
      message: "Voyage AI embeddings generated and saved successfully",
    });
  } catch (err) {
    console.error("Unexpected error:", err);
    return res.status(500).json({ error: "Internal server error", details: String(err) });
  }
});

app.listen(3000, () => {
  console.log("Server is listening on port 3000...");
});
