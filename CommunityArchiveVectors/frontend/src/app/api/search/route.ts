import { NextRequest, NextResponse } from 'next/server'

const BACKEND_URL = process.env.BACKEND_URL || 'http://45.63.18.97'

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

    // First get tweets
    const { data: tweetsData, error: tweetsError } = await supabase
      .from('tweets')
      .select('tweet_id, account_id, full_text, created_at, retweet_count, favorite_count')
      .in('tweet_id', tweetIds)

    if (tweetsError) {
      throw tweetsError
    }

    // Get unique account IDs
    const accountIds = [...new Set(tweetsData?.map(t => t.account_id) || [])]

    // Fetch account information
    const { data: accountsData, error: accountsError } = await supabase
      .from('account')
      .select('account_id, username, account_display_name')
      .in('account_id', accountIds)

    if (accountsError) {
      throw accountsError
    }

    // Create a map of account_id to account data
    const accountsMap = new Map(accountsData?.map(a => [a.account_id, a]) || [])

    // Merge tweets with account data
    const tweets = tweetsData?.map(tweet => ({
      ...tweet,
      all_account: accountsMap.get(tweet.account_id)
    }))

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
