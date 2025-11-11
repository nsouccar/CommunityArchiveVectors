'use client'

import { useEffect, useRef, useState } from 'react'
import { lugrasimo } from './fonts'
import * as d3 from 'd3'

interface SearchResult {
  tweet_id: number
  full_text: string
  username: string
  created_at?: string
  similarity: number
  retweet_count?: number
  favorite_count?: number
  reply_to_tweet_id?: number
  parent_tweet_text?: string
  parent_tweet_username?: string
}

interface SearchResponse {
  query: string
  results: SearchResult[]
  search_time_ms: number
  database_size: number
}

export default function Home() {
  const svgRef = useRef<SVGSVGElement>(null)
  const [currentYear, setCurrentYear] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [networkData, setNetworkData] = useState<any>(null)
  const [communityTopics, setCommunityTopics] = useState<any>(null)
  const [stats, setStats] = useState({ year: '2012', users: 0, interactions: 0, communities: 0 })
  const [searchUsername, setSearchUsername] = useState('')
  const [selectedUser, setSelectedUser] = useState<string | null>(null)
  const [userConnections, setUserConnections] = useState<Set<string>>(new Set())
  const simulationRef = useRef<any>(null)

  // Search overlay state
  const [showSearch, setShowSearch] = useState(false)
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [loading, setLoading] = useState(false)
  const [searchTime, setSearchTime] = useState<number | null>(null)
  const [databaseSize, setDatabaseSize] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    // Load network data
    fetch('/network_animation_data.json')
      .then(res => res.json())
      .then(data => {
        console.log('Loaded network data:', data.metadata)
        setNetworkData(data)
        if (data.years && data.years.length > 0) {
          updateVisualization(0, data)
        }
      })
      .catch(err => console.error('Error loading network data:', err))

    // Load community topics
    fetch('/community_topics.json')
      .then(res => res.json())
      .then(data => {
        console.log('Loaded community topics:', data)
        setCommunityTopics(data)
      })
      .catch(err => console.error('Error loading community topics:', err))
  }, [])

  useEffect(() => {
    let interval: NodeJS.Timeout
    if (isPlaying && networkData) {
      interval = setInterval(() => {
        setCurrentYear(prev => {
          const next = prev + 1
          if (next >= networkData.years.length) {
            setIsPlaying(false)
            return prev
          }
          updateVisualization(next, networkData)
          return next
        })
      }, 2000)
    }
    return () => clearInterval(interval)
  }, [isPlaying, networkData])

  // Update connections when selectedUser changes
  useEffect(() => {
    if (!selectedUser || !networkData || !networkData.years || !networkData.years[currentYear]) {
      setUserConnections(new Set())
      return
    }

    const yearData = networkData.years[currentYear]
    const connections = new Set<string>()

    yearData.edges.forEach((edge: any) => {
      if (edge.source === selectedUser || (edge.source.id && edge.source.id === selectedUser)) {
        const targetId = edge.target.id || edge.target
        connections.add(targetId)
      }
      if (edge.target === selectedUser || (edge.target.id && edge.target.id === selectedUser)) {
        const sourceId = edge.source.id || edge.source
        connections.add(sourceId)
      }
    })

    setUserConnections(connections)

    if (networkData) {
      updateVisualization(currentYear, networkData)
    }
  }, [selectedUser, currentYear, networkData])

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!query.trim()) return

    setLoading(true)
    setError(null)

    try {
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

  const updateVisualization = (yearIndex: number, data: any) => {
    if (!data || !data.years || yearIndex >= data.years.length || !svgRef.current) return

    const yearData = data.years[yearIndex]
    setStats({
      year: yearData.year,
      users: yearData.num_users,
      interactions: yearData.num_interactions,
      communities: yearData.num_communities
    })

    renderNetwork(yearData)
  }

  const renderNetwork = (yearData: any) => {
    if (!svgRef.current) return

    const width = window.innerWidth
    const height = window.innerHeight

    d3.select(svgRef.current).selectAll('*').remove()

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height)

    const g = svg.append('g')

    const zoom = d3.zoom()
      .scaleExtent([0.1, 10])
      .on('zoom', (event) => {
        g.attr('transform', event.transform)
      })

    svg.call(zoom as any)

    const colorScale = d3.scaleOrdinal(d3.schemeTableau10)
      .domain(d3.range(0, 20).map(String))

    const nodes = yearData.nodes.map((d: any) => ({ ...d }))
    const links = yearData.edges.map((d: any) => ({ ...d }))

    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id((d: any) => d.id).distance(50))
      .force('charge', d3.forceManyBody().strength(-100))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(8))

    simulationRef.current = simulation

    const link = g.append('g')
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke', (d: any) => {
        if (selectedUser) {
          const sourceId = d.source.id || d.source
          const targetId = d.target.id || d.target
          if (sourceId === selectedUser || targetId === selectedUser) {
            return '#00d4ff'
          }
        }
        return '#4a5568'
      })
      .attr('stroke-opacity', (d: any) => {
        if (selectedUser) {
          const sourceId = d.source.id || d.source
          const targetId = d.target.id || d.target
          if (sourceId === selectedUser || targetId === selectedUser) {
            return 0.8
          }
          return 0.1
        }
        return d.weight > 5 ? 0.6 : 0.3
      })
      .attr('stroke-width', (d: any) => {
        if (selectedUser) {
          const sourceId = d.source.id || d.source
          const targetId = d.target.id || d.target
          if (sourceId === selectedUser || targetId === selectedUser) {
            return Math.min(d.weight / 2, 6)
          }
        }
        return Math.min(d.weight / 2, 4)
      })

    const node = g.append('g')
      .selectAll('circle')
      .data(nodes)
      .join('circle')
      .attr('r', (d: any) => {
        if (selectedUser) {
          if (d.id === selectedUser) return Math.max(8, Math.min(d.degree, 20))
          if (userConnections.has(d.id)) return Math.max(6, Math.min(d.degree, 17))
        }
        return Math.max(4, Math.min(d.degree, 15))
      })
      .attr('fill', (d: any) => {
        if (selectedUser && d.id === selectedUser) return '#00d4ff'
        if (selectedUser && userConnections.has(d.id)) return '#4dd0e1'
        return colorScale(String(d.community))
      })
      .attr('stroke', (d: any) => {
        if (selectedUser && d.id === selectedUser) return '#00ff00'
        if (selectedUser && userConnections.has(d.id)) return '#00d4ff'
        return '#fff'
      })
      .attr('stroke-width', (d: any) => {
        if (selectedUser && d.id === selectedUser) return 3
        if (selectedUser && userConnections.has(d.id)) return 2
        return 1.5
      })
      .attr('opacity', (d: any) => {
        if (selectedUser && d.id !== selectedUser && !userConnections.has(d.id)) {
          return 0.2
        }
        return 1
      })
      .style('cursor', 'pointer')
      .on('mouseover', function(event, d: any) {
        const connectedNodes = new Set<string>()
        connectedNodes.add(d.id)

        links.forEach((link: any) => {
          const sourceId = link.source.id || link.source
          const targetId = link.target.id || link.target

          if (sourceId === d.id) connectedNodes.add(targetId)
          if (targetId === d.id) connectedNodes.add(sourceId)
        })

        link
          .attr('stroke', (l: any) => {
            const sourceId = l.source.id || l.source
            const targetId = l.target.id || l.target
            if (sourceId === d.id || targetId === d.id) {
              return '#00d4ff'
            }
            return '#4a5568'
          })
          .attr('stroke-opacity', (l: any) => {
            const sourceId = l.source.id || l.source
            const targetId = l.target.id || l.target
            if (sourceId === d.id || targetId === d.id) {
              return 1
            }
            return 0.1
          })
          .attr('stroke-width', (l: any) => {
            const sourceId = l.source.id || l.source
            const targetId = l.target.id || l.target
            if (sourceId === d.id || targetId === d.id) {
              return 3
            }
            return Math.min(l.weight / 2, 4)
          })

        node
          .attr('opacity', (n: any) => {
            if (connectedNodes.has(n.id)) return 1
            return 0.15
          })
          .attr('r', (n: any) => {
            if (n.id === d.id) return Math.max(10, Math.min(n.degree, 20))
            if (connectedNodes.has(n.id)) return Math.max(6, Math.min(n.degree, 17))
            return Math.max(4, Math.min(n.degree, 15))
          })
          .attr('stroke', (n: any) => {
            if (n.id === d.id) return '#00d4ff'
            if (connectedNodes.has(n.id)) return '#4dd0e1'
            return '#fff'
          })
          .attr('stroke-width', (n: any) => {
            if (n.id === d.id) return 4
            if (connectedNodes.has(n.id)) return 2.5
            return 1.5
          })

        const tooltip = d3.select('body')
          .append('div')
          .attr('class', 'tooltip-network')
          .style('position', 'absolute')
          .style('background', 'rgba(26, 31, 58, 0.95)')
          .style('border', '2px solid white')
          .style('border-radius', '8px')
          .style('padding', '10px')
          .style('pointer-events', 'none')
          .style('font-size', '14px')
          .style('color', 'white')
          .style('z-index', '1000')
          .html(`
            <strong>@${d.id}</strong><br/>
            Community: ${d.community + 1}<br/>
            Connections: ${d.degree}
          `)
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 10) + 'px')
      })
      .on('mouseout', function(event, d: any) {
        link
          .attr('stroke', (l: any) => {
            if (selectedUser) {
              const sourceId = l.source.id || l.source
              const targetId = l.target.id || l.target
              if (sourceId === selectedUser || targetId === selectedUser) {
                return '#00d4ff'
              }
            }
            return '#4a5568'
          })
          .attr('stroke-opacity', (l: any) => {
            if (selectedUser) {
              const sourceId = l.source.id || l.source
              const targetId = l.target.id || l.target
              if (sourceId === selectedUser || targetId === selectedUser) {
                return 0.8
              }
              return 0.1
            }
            return l.weight > 5 ? 0.6 : 0.3
          })
          .attr('stroke-width', (l: any) => {
            if (selectedUser) {
              const sourceId = l.source.id || l.source
              const targetId = l.target.id || l.target
              if (sourceId === selectedUser || targetId === selectedUser) {
                return Math.min(l.weight / 2, 6)
              }
            }
            return Math.min(l.weight / 2, 4)
          })

        node
          .attr('opacity', (n: any) => {
            if (selectedUser && n.id !== selectedUser && !userConnections.has(n.id)) {
              return 0.2
            }
            return 1
          })
          .attr('r', (n: any) => {
            if (selectedUser) {
              if (n.id === selectedUser) return Math.max(8, Math.min(n.degree, 20))
              if (userConnections.has(n.id)) return Math.max(6, Math.min(n.degree, 17))
            }
            return Math.max(4, Math.min(n.degree, 15))
          })
          .attr('stroke', (n: any) => {
            if (selectedUser && n.id === selectedUser) return '#00ff00'
            if (selectedUser && userConnections.has(n.id)) return '#00d4ff'
            return '#fff'
          })
          .attr('stroke-width', (n: any) => {
            if (selectedUser && n.id === selectedUser) return 3
            if (selectedUser && userConnections.has(n.id)) return 2
            return 1.5
          })

        d3.selectAll('.tooltip-network').remove()
      })
      .on('click', function(event, d: any) {
        window.location.href = `/network/${d.id}`
      })
      .call(d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended) as any)

    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y)

      node
        .attr('cx', (d: any) => d.x)
        .attr('cy', (d: any) => d.y)
    })

    function dragstarted(event: any) {
      if (!event.active) simulation.alphaTarget(0.3).restart()
      event.subject.fx = event.subject.x
      event.subject.fy = event.subject.y
    }

    function dragged(event: any) {
      event.subject.fx = event.x
      event.subject.fy = event.y
    }

    function dragended(event: any) {
      if (!event.active) simulation.alphaTarget(0)
      event.subject.fx = null
      event.subject.fy = null
    }
  }

  const handlePlay = () => {
    if (currentYear >= (networkData?.years.length || 0) - 1) {
      setCurrentYear(0)
    }
    setIsPlaying(!isPlaying)
  }

  const handlePrev = () => {
    if (currentYear > 0 && networkData) {
      const newIndex = currentYear - 1
      setCurrentYear(newIndex)
      updateVisualization(newIndex, networkData)
    }
  }

  const handleNext = () => {
    if (networkData && currentYear < networkData.years.length - 1) {
      const newIndex = currentYear + 1
      setCurrentYear(newIndex)
      updateVisualization(newIndex, networkData)
    }
  }

  const handleReset = () => {
    setIsPlaying(false)
    setCurrentYear(0)
    if (networkData) {
      updateVisualization(0, networkData)
    }
  }

  return (
    <div className="fixed inset-0 text-white overflow-hidden" style={{
      backgroundImage: 'url(/stars.png)',
      backgroundSize: 'cover',
      backgroundPosition: 'center',
      backgroundRepeat: 'no-repeat'
    }}>
      {/* Full-screen SVG Canvas */}
      <div className="absolute inset-0">
        <svg ref={svgRef} className="w-full h-full" style={{ background: 'transparent' }}></svg>

        {/* Loading State */}
        {!networkData && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-white text-xl">Loading constellation data...</div>
          </div>
        )}
      </div>

      {/* Overlay UI Elements */}

      {/* Top-left: Title */}
      <div className="absolute top-6 left-6 z-20">
        <h1 className={`text-4xl font-bold ${lugrasimo.className}`} style={{ color: '#ff0000' }}>
          Constellation of People
        </h1>
        <p className="text-gray-400 text-sm mt-1 max-w-md">
          How online communities formed and connected over time (2012-2025)
        </p>
      </div>

      {/* Top-right: Stats & Search Button */}
      <div className="absolute top-6 right-6 z-20 flex gap-4 items-start">
        <button
          onClick={() => setShowSearch(true)}
          className="bg-black/60 backdrop-blur-md border border-white/30 px-6 py-3 rounded-lg font-semibold hover:bg-black/80 transition-all"
          style={{ boxShadow: '0 0 20px rgba(255,255,255,0.2)' }}
        >
          Search Through the Archive
        </button>
        <div className="flex gap-4">
          <div className="bg-black/60 backdrop-blur-md rounded-lg px-4 py-3 border border-white/30" style={{ boxShadow: '0 0 10px rgba(255,255,255,0.2)' }}>
            <div className="text-2xl font-bold text-white">{stats.year}</div>
            <div className="text-gray-400 text-xs uppercase tracking-wider">Year</div>
          </div>
          <div className="bg-black/60 backdrop-blur-md rounded-lg px-4 py-3 border border-white/30" style={{ boxShadow: '0 0 10px rgba(255,255,255,0.2)' }}>
            <div className="text-2xl font-bold text-white">{stats.users.toLocaleString()}</div>
            <div className="text-gray-400 text-xs uppercase tracking-wider">Users</div>
          </div>
          <div className="bg-black/60 backdrop-blur-md rounded-lg px-4 py-3 border border-white/30" style={{ boxShadow: '0 0 10px rgba(255,255,255,0.2)' }}>
            <div className="text-2xl font-bold text-white">{stats.interactions.toLocaleString()}</div>
            <div className="text-gray-400 text-xs uppercase tracking-wider">Interactions</div>
          </div>
          <div className="bg-black/60 backdrop-blur-md rounded-lg px-4 py-3 border border-white/30" style={{ boxShadow: '0 0 10px rgba(255,255,255,0.2)' }}>
            <div className="text-2xl font-bold text-white">{stats.communities}</div>
            <div className="text-gray-400 text-xs uppercase tracking-wider">Communities</div>
          </div>
        </div>
      </div>

      {/* Bottom-left: Username Search */}
      <div className="absolute bottom-6 left-6 z-20 max-w-md">
        <div className="bg-black/60 backdrop-blur-md rounded-lg p-4 border border-white/30" style={{ boxShadow: '0 0 10px rgba(255,255,255,0.2)' }}>
          <h3 className="text-sm font-bold text-white mb-3">Search User</h3>
          <div className="flex gap-2">
            <input
              type="text"
              value={searchUsername}
              onChange={(e) => setSearchUsername(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && searchUsername.trim()) {
                  setSelectedUser(searchUsername.trim())
                }
              }}
              placeholder="Enter username..."
              className="flex-1 px-3 py-2 bg-black/50 border border-white/30 rounded text-white placeholder-gray-500 focus:outline-none transition-colors text-sm"
            />
            <button
              onClick={() => {
                if (searchUsername.trim()) {
                  setSelectedUser(searchUsername.trim())
                }
              }}
              disabled={!searchUsername.trim()}
              className="px-4 py-2 bg-purple-500 hover:bg-purple-600 disabled:bg-gray-600 disabled:cursor-not-allowed rounded font-semibold transition-all text-sm"
            >
              Search
            </button>
            {selectedUser && (
              <button
                onClick={() => {
                  setSelectedUser(null)
                  setSearchUsername('')
                  setUserConnections(new Set())
                }}
                className="px-4 py-2 bg-slate-600 hover:bg-slate-700 rounded font-semibold transition-all text-sm"
              >
                Clear
              </button>
            )}
          </div>
          {selectedUser && (
            <div className="mt-3 p-3 bg-black/50 rounded border border-white/30">
              <p className="text-white text-sm">
                Tracking: <span className="font-bold">@{selectedUser}</span>
                {userConnections.size > 0 && (
                  <span className="text-gray-400 ml-2">
                    ({userConnections.size} connections)
                  </span>
                )}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Bottom-center: Controls */}
      <div className="absolute bottom-6 left-1/2 transform -translate-x-1/2 z-20">
        <div className="bg-black/60 backdrop-blur-md rounded-lg p-3 border border-white/30 flex gap-3" style={{ boxShadow: '0 0 10px rgba(255,255,255,0.2)' }}>
          <button
            onClick={handlePlay}
            disabled={!networkData}
            className="px-5 py-2 bg-white hover:bg-gray-200 text-black disabled:bg-gray-600 disabled:cursor-not-allowed rounded font-semibold transition-all text-sm"
          >
            {isPlaying ? 'Pause' : 'Play'}
          </button>
          <button
            onClick={handlePrev}
            disabled={!networkData || currentYear === 0}
            className="px-5 py-2 bg-purple-500 hover:bg-purple-600 disabled:bg-gray-600 disabled:cursor-not-allowed rounded font-semibold transition-all text-sm"
          >
            ← Prev
          </button>
          <button
            onClick={handleNext}
            disabled={!networkData || currentYear >= (networkData?.years.length || 0) - 1}
            className="px-5 py-2 bg-purple-500 hover:bg-purple-600 disabled:bg-gray-600 disabled:cursor-not-allowed rounded font-semibold transition-all text-sm"
          >
            Next →
          </button>
          <button
            onClick={handleReset}
            disabled={!networkData}
            className="px-5 py-2 bg-slate-600 hover:bg-slate-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded font-semibold transition-all text-sm"
          >
            Reset
          </button>
        </div>
      </div>

      {/* Bottom-right: Quick Guide */}
      <div className="absolute bottom-6 right-6 z-20 max-w-xs">
        <div className="bg-black/60 backdrop-blur-md rounded-lg p-4 border border-white/30" style={{ boxShadow: '0 0 10px rgba(255,255,255,0.2)' }}>
          <h3 className="text-sm font-bold text-white mb-2">Guide</h3>
          <div className="text-xs text-gray-300 space-y-1">
            <div><strong className="text-white">Drag:</strong> Move nodes</div>
            <div><strong className="text-white">Scroll:</strong> Zoom</div>
            <div><strong className="text-white">Hover:</strong> See details</div>
            <div><strong className="text-white">Colors:</strong> Different communities</div>
          </div>
        </div>
      </div>

      {/* Search Overlay Modal */}
      {showSearch && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-6">
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/50 backdrop-blur-sm"
            onClick={() => setShowSearch(false)}
          />

          {/* Glass Modal */}
          <div className="relative w-full max-w-4xl max-h-[90vh] overflow-y-auto bg-white/10 backdrop-blur-xl border border-white/30 rounded-2xl shadow-2xl" style={{ boxShadow: '0 0 40px rgba(255,255,255,0.1), inset 0 0 40px rgba(255,255,255,0.05)' }}>
            {/* Header */}
            <div className="sticky top-0 bg-black/40 backdrop-blur-md border-b border-white/20 p-6 z-10">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-2xl font-bold text-white">Search Through the Archive</h2>
                <button
                  onClick={() => setShowSearch(false)}
                  className="text-white/70 hover:text-white text-3xl leading-none transition-colors"
                >
                  ×
                </button>
              </div>
              <p className="text-gray-300 text-sm mb-4">
                Semantic search over {databaseSize?.toLocaleString() || '6.4M'} tweets
              </p>

              {/* Search Form */}
              <form onSubmit={handleSearch} className="relative">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Search for anything... (e.g., 'artificial intelligence', 'climate change')"
                  className="w-full px-6 py-4 text-lg bg-black/30 border border-white/30 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-white/50"
                  disabled={loading}
                />
                <button
                  type="submit"
                  disabled={loading || !query.trim()}
                  className="absolute right-2 top-2 px-6 py-2 bg-purple-500 text-white font-semibold rounded-md hover:bg-purple-600 disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors"
                >
                  {loading ? 'Searching...' : 'Search'}
                </button>
              </form>
            </div>

            {/* Results */}
            <div className="p-6">
              {/* Search Stats */}
              {searchTime !== null && !error && (
                <div className="mb-4 text-sm text-gray-300">
                  Found {results.length} results in {searchTime.toFixed(2)}ms
                </div>
              )}

              {/* Error Message */}
              {error && (
                <div className="mb-4 p-4 bg-red-500/20 border border-red-500/50 rounded-lg text-red-200">
                  {error}
                </div>
              )}

              {/* Results */}
              <div className="space-y-4">
                {results.map((result) => (
                  <div
                    key={result.tweet_id}
                    className="bg-black/30 backdrop-blur-md border border-white/20 rounded-lg p-6 hover:bg-black/40 transition-all"
                  >
                    {/* Parent Tweet (if this is a reply) */}
                    {result.parent_tweet_text && (
                      <div className="mb-4 pl-4 border-l-2 border-white/30 bg-black/20 p-3 rounded">
                        <div className="text-xs text-gray-400 mb-1">Replying to:</div>
                        <div className="flex items-center space-x-2 mb-2">
                          <div className="w-6 h-6 bg-gray-500 rounded-full flex items-center justify-center text-white text-xs font-bold">
                            {result.parent_tweet_username?.[0].toUpperCase() || '?'}
                          </div>
                          <div className="text-sm font-semibold text-gray-300">
                            @{result.parent_tweet_username || 'unknown'}
                          </div>
                        </div>
                        <p className="text-sm text-gray-400 italic">
                          {result.parent_tweet_text}
                        </p>
                      </div>
                    )}

                    {/* Tweet Header */}
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-2">
                        <div className="w-10 h-10 bg-purple-500 rounded-full flex items-center justify-center text-white font-bold">
                          {result.username[0].toUpperCase()}
                        </div>
                        <div>
                          <div className="font-semibold text-white">
                            @{result.username}
                          </div>
                          <div className="text-sm text-gray-400">
                            {formatDate(result.created_at)}
                          </div>
                        </div>
                      </div>
                      <div className="text-sm text-purple-300 font-semibold">
                        {(result.similarity * 100).toFixed(1)}% match
                      </div>
                    </div>

                    {/* Tweet Text */}
                    <p className="text-white leading-relaxed mb-3">
                      {result.full_text}
                    </p>

                    {/* Tweet Stats and Actions */}
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-4 text-sm text-gray-400">
                        <span>🔄 {result.retweet_count?.toLocaleString() || 0} retweets</span>
                        <span>❤️ {result.favorite_count?.toLocaleString() || 0} likes</span>
                      </div>
                      <a
                        href={`https://x.com/${result.username}/status/${result.tweet_id}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-colors text-sm font-medium"
                      >
                        View on Twitter
                      </a>
                    </div>
                  </div>
                ))}

                {/* Empty State */}
                {!loading && results.length === 0 && !error && query && (
                  <div className="text-center py-12 text-gray-400">
                    No results found. Try a different search query.
                  </div>
                )}

                {/* Initial State */}
                {!loading && results.length === 0 && !error && !query && (
                  <div className="text-center py-12 text-gray-300">
                    <div className="text-xl font-semibold mb-2">Start searching</div>
                    <div>Try searching for topics, people, or events</div>
                    <div className="mt-4 space-x-2">
                      <button
                        onClick={() => setQuery('artificial intelligence')}
                        className="px-4 py-2 bg-white/10 border border-white/20 rounded-md hover:bg-white/20 transition-colors"
                      >
                        artificial intelligence
                      </button>
                      <button
                        onClick={() => setQuery('climate change')}
                        className="px-4 py-2 bg-white/10 border border-white/20 rounded-md hover:bg-white/20 transition-colors"
                      >
                        climate change
                      </button>
                      <button
                        onClick={() => setQuery('bitcoin')}
                        className="px-4 py-2 bg-white/10 border border-white/20 rounded-md hover:bg-white/20 transition-colors"
                      >
                        bitcoin
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

    </div>
  )
}
