import { NextRequest, NextResponse } from 'next/server'
import fs from 'fs'
import path from 'path'

const BACKEND_URL = process.env.BACKEND_URL || 'http://45.63.18.97'

// Load avatar URLs once at module initialization
let avatarUrls: Record<string, string> = {}
try {
  const avatarPath = path.join(process.cwd(), 'public', 'avatar_urls.json')
  avatarUrls = JSON.parse(fs.readFileSync(avatarPath, 'utf-8'))
} catch (error) {
  console.error('Failed to load avatar URLs:', error)
}

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams
  const query = searchParams.get('query')
  const limit = searchParams.get('limit') || '10'

  if (!query) {
    return NextResponse.json({ error: 'Query parameter is required' }, { status: 400 })
  }

  try {
    const response = await fetch(
      `${BACKEND_URL}/search?query=${encodeURIComponent(query)}&limit=${limit}`
    )

    if (!response.ok) {
      throw new Error(`Backend returned ${response.status}`)
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error('Search proxy error:', error)
    return NextResponse.json(
      { error: 'Search failed' },
      { status: 500 }
    )
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { tweetIds } = body

    if (!tweetIds || !Array.isArray(tweetIds)) {
      return NextResponse.json({ error: 'tweetIds array is required' }, { status: 400 })
    }

    // Fetch tweets from Supabase
    const { createClient } = await import('@supabase/supabase-js')
    const supabase = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
    )

    // First get tweets with reply info
    const { data: tweetsData, error: tweetsError } = await supabase
      .from('tweets')
      .select('tweet_id, account_id, full_text, created_at, retweet_count, favorite_count, reply_to_tweet_id')
      .in('tweet_id', tweetIds)

    if (tweetsError) {
      throw tweetsError
    }

    // Get parent tweet IDs for replies
    const parentTweetIds = [...new Set(tweetsData?.filter(t => t.reply_to_tweet_id).map(t => t.reply_to_tweet_id) || [])]

    // Fetch parent tweets if any exist
    let parentTweetsMap = new Map()
    if (parentTweetIds.length > 0) {
      const { data: parentTweetsData } = await supabase
        .from('tweets')
        .select('tweet_id, account_id, full_text, created_at')
        .in('tweet_id', parentTweetIds)

      parentTweetsMap = new Map(parentTweetsData?.map(t => [t.tweet_id, t]) || [])
    }

    // Get unique account IDs (including parent tweet authors)
    const allAccountIds = [...new Set([
      ...(tweetsData?.map(t => t.account_id) || []),
      ...(Array.from(parentTweetsMap.values()).map(t => t.account_id) || [])
    ])]

    // Fetch account information
    const { data: accountsData, error: accountsError } = await supabase
      .from('account')
      .select('account_id, username, account_display_name')
      .in('account_id', allAccountIds)

    if (accountsError) {
      throw accountsError
    }

    // Create a map of account_id to account data with profile image URLs
    const accountsMap = new Map(accountsData?.map(a => [a.account_id, {
      ...a,
      profile_image_url: avatarUrls[a.username] || null
    }]) || [])

    // Merge tweets with account data and parent tweets
    const tweets = tweetsData?.map(tweet => {
      const parentTweet = tweet.reply_to_tweet_id ? parentTweetsMap.get(tweet.reply_to_tweet_id) : null
      return {
        ...tweet,
        all_account: accountsMap.get(tweet.account_id),
        parent_tweet: parentTweet ? {
          ...parentTweet,
          all_account: accountsMap.get(parentTweet.account_id)
        } : null
      }
    })

    const error = null

    if (error) {
      throw error
    }

    return NextResponse.json({ tweets: tweets || [] })
  } catch (error) {
    console.error('Tweet fetch error:', error)
    return NextResponse.json(
      { error: 'Failed to fetch tweets' },
      { status: 500 }
    )
  }
}
