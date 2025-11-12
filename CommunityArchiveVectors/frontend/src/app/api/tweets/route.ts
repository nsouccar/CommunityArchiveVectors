import { NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'
import fs from 'fs'
import path from 'path'

// Load avatar URLs once at module initialization
let avatarUrls: Record<string, string> = {}
try {
  const avatarPath = path.join(process.cwd(), 'public', 'avatar_urls.json')
  avatarUrls = JSON.parse(fs.readFileSync(avatarPath, 'utf-8'))
} catch (error) {
  console.error('Failed to load avatar URLs:', error)
}

export async function POST(request: Request) {
  try {
    const { tweetIds } = await request.json()

    if (!tweetIds || !Array.isArray(tweetIds)) {
      return NextResponse.json({ error: 'tweetIds array is required' }, { status: 400 })
    }

    const supabaseUrl = process.env.SUPABASE_URL
    const supabaseKey = process.env.SUPABASE_KEY

    if (!supabaseUrl || !supabaseKey) {
      return NextResponse.json({ error: 'Database configuration missing' }, { status: 500 })
    }

    const supabase = createClient(supabaseUrl, supabaseKey)

    // Fetch tweets with their accounts and parent tweets
    const { data: tweetsData, error: tweetsError } = await supabase
      .from('tweets')
      .select(`
        *,
        all_account:accounts!tweets_account_id_fkey (
          account_id,
          username
        ),
        parent_tweet:tweets!tweets_reply_to_tweet_id_fkey (
          tweet_id,
          full_text,
          created_at,
          all_account:accounts!tweets_account_id_fkey (
            account_id,
            username
          )
        )
      `)
      .in('tweet_id', tweetIds)

    if (tweetsError) {
      console.error('Supabase error:', tweetsError)
      return NextResponse.json({ error: 'Failed to fetch tweets' }, { status: 500 })
    }

    // Enrich tweets with profile images
    const enrichedTweets = (tweetsData || []).map((tweet: any) => {
      const username = tweet.all_account?.username
      const parentUsername = tweet.parent_tweet?.all_account?.username

      return {
        ...tweet,
        profile_image_url: username ? avatarUrls[username] || null : null,
        parent_tweet: tweet.parent_tweet ? {
          ...tweet.parent_tweet,
          profile_image_url: parentUsername ? avatarUrls[parentUsername] || null : null,
        } : null,
      }
    })

    return NextResponse.json({ tweets: enrichedTweets })
  } catch (error) {
    console.error('Error fetching tweets:', error)
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 })
  }
}
