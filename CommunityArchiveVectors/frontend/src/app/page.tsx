'use client'

import { useState } from 'react'

interface SearchResult {
  tweet_id: number
  full_text: string
  username: string
  created_at?: string
  similarity: number
  retweet_count?: number
  favorite_count?: number
}

interface SearchResponse {
  query: string
  results: SearchResult[]
  search_time_ms: number
  database_size: number
}

export default function Home() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [loading, setLoading] = useState(false)
  const [searchTime, setSearchTime] = useState<number | null>(null)
  const [databaseSize, setDatabaseSize] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!query.trim()) return

    setLoading(true)
    setError(null)

    try {
      // Use proxy API to avoid HTTPS->HTTP mixed content issues
      const response = await fetch(
        `/api/search?query=${encodeURIComponent(query)}&limit=20`
      )

      if (!response.ok) {
        throw new Error(`Search failed: ${response.statusText}`)
      }

      const data: SearchResponse = await response.json()
      setResults(data.results)
      setSearchTime(data.search_time_ms)
      setDatabaseSize(data.database_size)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed')
      setResults([])
    } finally {
      setLoading(false)
    }
  }

  const formatDate = (dateString?: string) => {
    if (!dateString) return 'Unknown date'
    try {
      return new Date(dateString).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
      })
    } catch {
      return 'Unknown date'
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-4xl mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold text-gray-900">Tweet Search</h1>
          <p className="text-gray-600 mt-1">
            Semantic search over {databaseSize?.toLocaleString() || '6.4M'} tweets
          </p>
        </div>
      </div>

      {/* Search Box */}
      <div className="max-w-4xl mx-auto px-4 py-8">
        <form onSubmit={handleSearch} className="mb-8">
          <div className="relative">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search for anything... (e.g., 'artificial intelligence', 'climate change')"
              className="w-full px-6 py-4 text-lg border border-gray-300 rounded-lg shadow-sm focus:ring-2 focus:ring-indigo-500 focus:border-transparent outline-none"
              disabled={loading}
            />
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="absolute right-2 top-2 px-6 py-2 bg-indigo-600 text-white font-semibold rounded-md hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? 'Searching...' : 'Search'}
            </button>
          </div>
        </form>

        {/* Search Stats */}
        {searchTime !== null && !error && (
          <div className="mb-4 text-sm text-gray-600">
            Found {results.length} results in {searchTime.toFixed(2)}ms
          </div>
        )}

        {/* Error Message */}
        {error && (
          <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
            {error}
          </div>
        )}

        {/* Results */}
        <div className="space-y-4">
          {results.map((result) => (
            <div
              key={result.tweet_id}
              className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow"
            >
              {/* Tweet Header */}
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-2">
                  <div className="w-10 h-10 bg-indigo-500 rounded-full flex items-center justify-center text-white font-bold">
                    {result.username[0].toUpperCase()}
                  </div>
                  <div>
                    <div className="font-semibold text-gray-900">
                      @{result.username}
                    </div>
                    <div className="text-sm text-gray-500">
                      {formatDate(result.created_at)}
                    </div>
                  </div>
                </div>
                <div className="text-sm text-indigo-600 font-semibold">
                  {(result.similarity * 100).toFixed(1)}% match
                </div>
              </div>

              {/* Tweet Text */}
              <p className="text-gray-800 leading-relaxed mb-3">
                {result.full_text}
              </p>

              {/* Tweet Stats */}
              <div className="flex items-center space-x-4 text-sm text-gray-500">
                <span>🔄 {result.retweet_count?.toLocaleString() || 0} retweets</span>
                <span>❤️ {result.favorite_count?.toLocaleString() || 0} likes</span>
                <span>ID: {result.tweet_id}</span>
              </div>
            </div>
          ))}

          {/* Empty State */}
          {!loading && results.length === 0 && !error && query && (
            <div className="text-center py-12 text-gray-500">
              No results found. Try a different search query.
            </div>
          )}

          {/* Initial State */}
          {!loading && results.length === 0 && !error && !query && (
            <div className="text-center py-12 text-gray-500">
              <div className="text-6xl mb-4">🔍</div>
              <div className="text-xl font-semibold mb-2">Start searching</div>
              <div>Try searching for topics, people, or events</div>
              <div className="mt-4 space-x-2">
                <button
                  onClick={() => setQuery('artificial intelligence')}
                  className="px-4 py-2 bg-gray-100 rounded-md hover:bg-gray-200 transition-colors"
                >
                  artificial intelligence
                </button>
                <button
                  onClick={() => setQuery('climate change')}
                  className="px-4 py-2 bg-gray-100 rounded-md hover:bg-gray-200 transition-colors"
                >
                  climate change
                </button>
                <button
                  onClick={() => setQuery('bitcoin')}
                  className="px-4 py-2 bg-gray-100 rounded-md hover:bg-gray-200 transition-colors"
                >
                  bitcoin
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </main>
  )
}
