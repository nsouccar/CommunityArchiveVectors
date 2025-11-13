'use client'

import { useEffect, useRef, useState } from 'react'
import Link from 'next/link'
import { lugrasimo } from '../fonts'
import * as d3 from 'd3'

export default function NetworkPage() {
  const svgRef = useRef<SVGSVGElement>(null)
  const [currentYear, setCurrentYear] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [networkData, setNetworkData] = useState<any>(null)
  const [communityTopics, setCommunityTopics] = useState<any>(null)
  const [allTopics, setAllTopics] = useState<any>(null)
  const [loadedYears, setLoadedYears] = useState<Set<string>>(new Set())
  const [communityNames, setCommunityNames] = useState<any>(null)
  const [stats, setStats] = useState({ year: '2012', users: 0, interactions: 0, communities: 0 })
  const [searchUsername, setSearchUsername] = useState('')
  const [selectedUser, setSelectedUser] = useState<string | null>(null)
  const [selectedCommunity, setSelectedCommunity] = useState<number | null>(null)
  const [selectedTopic, setSelectedTopic] = useState<any>(null)
  const [topicTweets, setTopicTweets] = useState<any[]>([])
  const [loadingTweets, setLoadingTweets] = useState(false)
  const [userConnections, setUserConnections] = useState<Set<string>>(new Set())
  const [showCommunitySidebar, setShowCommunitySidebar] = useState(true)
  const [hoveredCommunity, setHoveredCommunity] = useState<number | null>(null)
  const [viewMode, setViewMode] = useState<'independent' | 'lineage'>('independent')
  const [temporalAlignments, setTemporalAlignments] = useState<any>(null)
  const [lineageNames, setLineageNames] = useState<Record<number, string>>({})
  const [inactiveLineageMessage, setInactiveLineageMessage] = useState<string | null>(null)
  const [avatarUrls, setAvatarUrls] = useState<Record<string, string>>({})
  const simulationRef = useRef<any>(null)
  const zoomToCommunityRef = useRef<((communityId: number) => void) | null>(null)
  const resetZoomRef = useRef<(() => void) | null>(null)

  const fetchTopicTweets = async (tweetIds: string[], sampleTweets?: any[]) => {
    setLoadingTweets(true)
    console.log('fetchTopicTweets called with:', { tweetIds: tweetIds.length, sampleTweets: sampleTweets?.length })
    console.log('Sample tweet data:', sampleTweets?.[0])

    // Use sample tweets if provided (2024 approach)
    if (sampleTweets && sampleTweets.length > 0) {
      console.log('Using sample tweets:', sampleTweets)
      setTopicTweets(sampleTweets)
      setLoadingTweets(false)
      return
    }

    // Otherwise, fetch from API (old approach for other years)
    console.log('Fetching tweets from API for IDs:', tweetIds.slice(0, 50))
    try {
      const response = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tweetIds: tweetIds.slice(0, 50) })
      })

      if (!response.ok) {
        throw new Error('Failed to fetch tweets')
      }

      const data = await response.json()
      console.log('Fetched tweets from API:', data.tweets?.length)
      setTopicTweets(data.tweets || [])
    } catch (error) {
      console.error('Error fetching tweets:', error)
      setTopicTweets([])
    }

    setLoadingTweets(false)
  }

  // Lazy load topics for a specific year
  const loadYearTopics = async (year: string) => {
    if (loadedYears.has(year)) {
      console.log(`Topics for year ${year} already loaded`)
      return
    }

    console.log(`Loading topics for year ${year}...`)
    try {
      const response = await fetch(`/data/topics_year_${year}_summary.json`)
      const yearData = await response.json()

      setAllTopics((prev: any) => ({
        ...prev,
        [year]: yearData
      }))

      setLoadedYears((prev) => new Set(Array.from(prev).concat(year)))
      console.log(`✓ Loaded topics for year ${year}`)
    } catch (err) {
      console.warn(`Failed to load topics for year ${year}:`, err)
    }
  }

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

    // Load community topics (old format - keywords)
    fetch('/community_topics.json')
      .then(res => res.json())
      .then(data => {
        console.log('Loaded community topics:', data)
        setCommunityTopics(data)
      })
      .catch(err => console.error('Error loading community topics:', err))

    // Initialize allTopics as empty object - will be populated on-demand
    setAllTopics({})

    // Load community names
    fetch('/data/all_community_names.json')
      .then(res => res.json())
      .then(data => {
        console.log('Loaded community names:', data)
        setCommunityNames(data)
      })
      .catch(err => console.error('Error loading community names:', err))

    // Load avatar URLs
    fetch('/avatar_urls.json')
      .then(res => res.json())
      .then(data => {
        console.log('Loaded avatar URLs:', Object.keys(data).length)
        setAvatarUrls(data)
      })
      .catch(err => console.error('Error loading avatar URLs:', err))

    // Load temporal alignments
    fetch('/community_temporal_alignments.json')
      .then(res => res.json())
      .then(data => {
        console.log('Loaded temporal alignments:', data)
        setTemporalAlignments(data)
      })
      .catch(err => console.error('Error loading temporal alignments:', err))
  }, [])

  // Lazy load topics when year changes
  useEffect(() => {
    if (stats.year && allTopics !== null) {
      loadYearTopics(stats.year)
    }
  }, [stats.year, allTopics])

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
      }, 2000) // Change year every 2 seconds
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

    // Find all nodes connected to the selected user
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

    // Re-render the visualization to update highlighting
    if (networkData) {
      updateVisualization(currentYear, networkData)
    }
  }, [selectedUser, currentYear, networkData])

  // Zoom to community when hovered or selected
  useEffect(() => {
    if (hoveredCommunity !== null && zoomToCommunityRef.current) {
      zoomToCommunityRef.current(hoveredCommunity)
    } else if (selectedCommunity !== null && zoomToCommunityRef.current) {
      zoomToCommunityRef.current(selectedCommunity)
    } else if (hoveredCommunity === null && selectedCommunity === null && resetZoomRef.current) {
      resetZoomRef.current()
    }
  }, [hoveredCommunity, selectedCommunity])

  // Re-render when view mode changes and build lineage names
  useEffect(() => {
    if (viewMode === 'lineage' && temporalAlignments && communityNames) {
      // Build lineage ID mapping
      const lineageMapping = buildLineageMapping()
      if (lineageMapping) {
        // Build lineage names: map each lineage ID to the name from its earliest year
        const lineageToName: Record<number, string> = {}
        const lineageToEarliestYear: Record<number, number> = {}

        // Find earliest year for each lineage
        Object.entries(lineageMapping).forEach(([key, lineageId]) => {
          const [yearStr, commStr] = key.split('_')
          const year = parseInt(yearStr)
          const commId = parseInt(commStr)

          if (!lineageToEarliestYear[lineageId] || year < lineageToEarliestYear[lineageId]) {
            lineageToEarliestYear[lineageId] = year

            // Get community name from this year
            const yearNames = communityNames?.[year]?.communities
            const communityInfo = yearNames?.find((c: any) => c.community_id === commId)
            if (communityInfo?.name) {
              lineageToName[lineageId] = communityInfo.name
            }
          }
        })

        console.log('📝 Built lineage names:', Object.keys(lineageToName).length, 'lineages named')
        setLineageNames(lineageToName)
      }
    }

    if (networkData) {
      updateVisualization(currentYear, networkData)
    }
  }, [viewMode, temporalAlignments, communityNames])

  // Update node highlighting when community hover/selection changes
  useEffect(() => {
    if (!svgRef.current) return

    const svg = d3.select(svgRef.current)
    const nodeGroups = svg.selectAll('.node-group')

    // Get the color scale (must match the one used in renderNetwork)
    const colorScale = d3.scaleOrdinal(d3.schemeTableau10)
      .domain(d3.range(0, 20).map(String))

    const highlightCommunity = hoveredCommunity !== null ? hoveredCommunity : selectedCommunity

    // Cancel any existing blinking animations
    nodeGroups.interrupt()

    // Update images
    nodeGroups.selectAll('image')
      .transition()
      .duration(200)
      .attr('opacity', function(d: any) {
        // Preserve user selection dimming
        if (selectedUser && d.id !== selectedUser && !userConnections.has(d.id)) {
          return 0.2
        }

        // Dim nodes not in the hovered/selected community
        if (highlightCommunity !== null && d.community !== highlightCommunity) {
          return 0.15
        }

        return 1
      })

    // Update border circles
    nodeGroups.selectAll('circle')
      .transition()
      .duration(200)
      .attr('stroke', function(d: any) {
        // Preserve user selection highlighting
        if (selectedUser && d.id === selectedUser) return '#00ff00'
        if (selectedUser && userConnections.has(d.id)) return '#00d4ff'

        // Use community color
        return colorScale(String(d.community))
      })
      .attr('opacity', function(d: any) {
        // Preserve user selection dimming
        if (selectedUser && d.id !== selectedUser && !userConnections.has(d.id)) {
          return 0.4
        }

        // Dim nodes not in the hovered/selected community
        if (highlightCommunity !== null && d.community !== highlightCommunity) {
          return 0.3
        }

        return 1
      })
      .attr('r', function(d: any) {
        // Preserve user selection sizing
        if (selectedUser) {
          if (d.id === selectedUser) return Math.max(8, Math.min(d.degree, 20))
          if (userConnections.has(d.id)) return Math.max(6, Math.min(d.degree, 17))
        }

        // Make highlighted community nodes bigger
        if (highlightCommunity !== null && d.community === highlightCommunity) {
          return Math.max(8, Math.min(d.degree * 1.8, 25))
        }

        return Math.max(4, Math.min(d.degree, 15))
      })
      .attr('stroke-width', function(d: any) {
        // Preserve user selection stroke
        if (selectedUser && d.id === selectedUser) return 3
        if (selectedUser && userConnections.has(d.id)) return 2

        // Thicker stroke for highlighted community
        if (highlightCommunity !== null && d.community === highlightCommunity) {
          return 3
        }

        return 2
      })

    // Add blinking animation for highlighted community
    if (highlightCommunity !== null) {
      const highlightedImages = nodeGroups
        .filter((d: any) => d.community === highlightCommunity)
        .selectAll('image')

      const blink = () => {
        highlightedImages
          .transition()
          .duration(800)
          .attr('opacity', 0.6)
          .transition()
          .duration(800)
          .attr('opacity', 1)
          .on('end', function(d: any) {
            // Continue blinking if still highlighted
            if ((hoveredCommunity !== null && d.community === hoveredCommunity) ||
                (selectedCommunity !== null && d.community === selectedCommunity)) {
              d3.select(this).call(blink as any)
            }
          })
      }

      blink()
    }
  }, [hoveredCommunity, selectedCommunity, selectedUser, userConnections])

  // Helper function to build lineage mapping from temporal alignments
  const buildLineageMapping = () => {
    if (!temporalAlignments || !temporalAlignments.alignments) {
      console.log('⚠️ Cannot build lineage mapping:', {
        temporalAlignments: temporalAlignments ? 'exists' : 'null',
        alignments: temporalAlignments?.alignments ? `${temporalAlignments.alignments.length} alignments` : 'null'
      })
      return null
    }

    console.log('✅ Building lineage mapping from', temporalAlignments.alignments.length, 'alignments')

    const mapping: Record<string, number> = {} // key: "year_comm" -> lineage ID
    let lineageCounter = 0

    // Build a graph of connections
    const connections: Record<string, string[]> = {}
    temporalAlignments.alignments.forEach((alignment: any) => {
      const key1 = `${alignment.year1}_${alignment.community1_id}`
      const key2 = `${alignment.year2}_${alignment.community2_id}`
      if (!connections[key1]) connections[key1] = []
      connections[key1].push(key2)
    })

    console.log('📊 Built connection graph with', Object.keys(connections).length, 'nodes')

    // Assign lineage IDs using DFS
    const visited = new Set<string>()
    const assignLineage = (key: string, lineageId: number) => {
      if (visited.has(key)) return
      visited.add(key)
      mapping[key] = lineageId

      // Follow connections
      if (connections[key]) {
        connections[key].forEach(nextKey => {
          assignLineage(nextKey, lineageId)
        })
      }
    }

    // Process all alignments
    Object.keys(connections).forEach(key => {
      if (!visited.has(key)) {
        assignLineage(key, lineageCounter++)
      }
    })

    console.log('✨ Created', lineageCounter, 'lineages from', Object.keys(mapping).length, 'community-year pairs')
    console.log('Sample lineage mappings:', Object.entries(mapping).slice(0, 5))

    return mapping
  }

  // Helper function to get community name based on view mode
  const getCommunityName = (year: number, communityId: number): string => {
    if (viewMode === 'lineage' && temporalAlignments) {
      // In lineage mode, use simple numeric labels
      const lineageMapping = buildLineageMapping()
      if (lineageMapping) {
        const key = `${year}_${communityId}`
        const lineageId = lineageMapping[key]
        if (lineageId !== undefined) {
          return `Lineage ${lineageId}`
        }
      }
    }

    // Independent mode: return the community's topic-based name
    const yearNames = communityNames?.[year]?.communities
    const communityInfo = yearNames?.find((c: any) => c.community_id === communityId)
    return communityInfo?.name || `Community ${communityId}`
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

    // Use full window dimensions for spacey full-screen effect
    const width = window.innerWidth
    const height = window.innerHeight

    // Clear previous visualization
    d3.select(svgRef.current).selectAll('*').remove()

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height)
      // Transparent background to show stars.png

    const g = svg.append('g')

    // Add zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.1, 10])
      .on('zoom', (event) => {
        g.attr('transform', event.transform)
      })

    svg.call(zoom as any)

    // Function to zoom to a specific community
    const zoomToCommunity = (communityId: number, duration: number = 750) => {
      const communityNodes = yearData.nodes.filter((n: any) => n.community === communityId)
      if (communityNodes.length === 0) return

      // Check if nodes have valid coordinates (simulation has positioned them)
      const hasValidCoords = communityNodes.every((n: any) =>
        typeof n.x === 'number' && !isNaN(n.x) &&
        typeof n.y === 'number' && !isNaN(n.y)
      )
      if (!hasValidCoords) return

      // Calculate bounding box
      const xs = communityNodes.map((n: any) => n.x)
      const ys = communityNodes.map((n: any) => n.y)
      const minX = Math.min(...xs)
      const maxX = Math.max(...xs)
      const minY = Math.min(...ys)
      const maxY = Math.max(...ys)

      // Add padding
      const padding = 100
      const boxWidth = maxX - minX + padding * 2
      const boxHeight = maxY - minY + padding * 2

      // Calculate scale and translation
      const scale = Math.min(8, 0.9 / Math.max(boxWidth / width, boxHeight / height))
      const centerX = (minX + maxX) / 2
      const centerY = (minY + maxY) / 2
      const translateX = width / 2 - scale * centerX
      const translateY = height / 2 - scale * centerY

      // Animate zoom
      svg.transition()
        .duration(duration)
        .call(
          zoom.transform as any,
          d3.zoomIdentity.translate(translateX, translateY).scale(scale)
        )
    }

    // Function to reset zoom
    const resetZoom = (duration: number = 750) => {
      svg.transition()
        .duration(duration)
        .call(
          zoom.transform as any,
          d3.zoomIdentity
        )
    }

    // Store zoom functions in refs so they can be called from event handlers
    zoomToCommunityRef.current = zoomToCommunity
    resetZoomRef.current = resetZoom

    // Color scale (same for both modes now - only names change)
    const colorScale = d3.scaleOrdinal(d3.schemeTableau10)
        .domain(d3.range(0, 20).map(String))

    // Prepare data - limit nodes for large graphs to prevent crashes
    const MAX_NODES = 1500  // Render max 1500 nodes for performance
    let nodes = yearData.nodes.map((d: any) => ({ ...d }))
    let links = yearData.edges.map((d: any) => ({ ...d }))

    // If graph is too large, filter to most connected nodes
    if (nodes.length > MAX_NODES) {
      console.log(`Large graph detected (${nodes.length} nodes). Limiting to top ${MAX_NODES} most connected nodes...`)

      // Sort by degree (connection count) and take top N
      const topNodes = nodes
        .sort((a: any, b: any) => (b.degree || 0) - (a.degree || 0))
        .slice(0, MAX_NODES)

      const topNodeIds = new Set(topNodes.map((n: any) => n.id))

      // Filter links to only include those between top nodes
      links = links.filter((l: any) => {
        const sourceId = l.source.id || l.source
        const targetId = l.target.id || l.target
        return topNodeIds.has(sourceId) && topNodeIds.has(targetId)
      })

      nodes = topNodes
      console.log(`Reduced to ${nodes.length} nodes and ${links.length} edges`)
    }

    // Create force simulation
    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id((d: any) => d.id).distance(50))
      .force('charge', d3.forceManyBody().strength(-100))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(8))

    simulationRef.current = simulation

    // Create links
    const link = g.append('g')
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke', (d: any) => {
        // Highlight connections to the selected user
        if (selectedUser) {
          const sourceId = d.source.id || d.source
          const targetId = d.target.id || d.target
          if (sourceId === selectedUser || targetId === selectedUser) {
            return '#00d4ff' // Cyan for connections to selected user
          }
        }
        return '#4a5568' // Default gray
      })
      .attr('stroke-opacity', (d: any) => {
        if (selectedUser) {
          const sourceId = d.source.id || d.source
          const targetId = d.target.id || d.target
          if (sourceId === selectedUser || targetId === selectedUser) {
            return 0.8 // More visible for selected connections
          }
          return 0.1 // Dim other connections
        }
        return d.weight > 5 ? 0.6 : 0.3
      })
      .attr('stroke-width', (d: any) => {
        if (selectedUser) {
          const sourceId = d.source.id || d.source
          const targetId = d.target.id || d.target
          if (sourceId === selectedUser || targetId === selectedUser) {
            return Math.min(d.weight / 2, 6) // Thicker for selected connections
          }
        }
        return Math.min(d.weight / 2, 4)
      })

    // Create circular clip paths for profile images
    const defs = svg.select('defs').empty() ? svg.append('defs') : svg.select('defs')

    // Add a circular clipPath for each unique user
    nodes.forEach((d: any) => {
      if (!defs.select(`#clip-${d.id.replace(/[^a-zA-Z0-9]/g, '_')}`).empty()) return

      defs.append('clipPath')
        .attr('id', `clip-${d.id.replace(/[^a-zA-Z0-9]/g, '_')}`)
        .append('circle')
        .attr('cx', 0)
        .attr('cy', 0)
        .attr('r', (selectedUser && d.id === selectedUser) ? Math.max(8, Math.min(d.degree, 20)) : Math.max(4, Math.min(d.degree, 15)))
    })

    // Create node groups (image + border circle)
    const nodeGroup = g.append('g')
      .attr('class', 'nodes')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .attr('class', 'node-group')
      .style('cursor', 'pointer')
      .on('click', function(event, d: any) {
        window.location.href = `/network/${d.id}`
      })
      .on('mouseover', function(event, d: any) {
        // Find all connected nodes
        const connectedNodes = new Set<string>()
        connectedNodes.add(d.id)

        links.forEach((link: any) => {
          const sourceId = link.source.id || link.source
          const targetId = link.target.id || link.target

          if (sourceId === d.id) connectedNodes.add(targetId)
          if (targetId === d.id) connectedNodes.add(sourceId)
        })

        // Highlight connected edges
        link
          .attr('stroke', (l: any) => {
            const sourceId = l.source.id || l.source
            const targetId = l.target.id || l.target
            if (sourceId === d.id || targetId === d.id) {
              return '#00d4ff' // Bright cyan for connected edges
            }
            return '#4a5568'
          })
          .attr('stroke-opacity', (l: any) => {
            const sourceId = l.source.id || l.source
            const targetId = l.target.id || l.target
            if (sourceId === d.id || targetId === d.id) {
              return 1 // Full opacity for connected edges
            }
            return 0.1 // Dim other edges
          })
          .attr('stroke-width', (l: any) => {
            const sourceId = l.source.id || l.source
            const targetId = l.target.id || l.target
            if (sourceId === d.id || targetId === d.id) {
              return 3 // Thicker for connected edges
            }
            return Math.min(l.weight / 2, 4)
          })

        // Highlight connected nodes - update both images and circles
        nodeGroup.selectAll('image')
          .attr('opacity', (n: any) => {
            if (connectedNodes.has(n.id)) return 1
            return 0.15 // Dim non-connected nodes
          })

        nodeGroup.selectAll('circle')
          .attr('opacity', (n: any) => {
            if (connectedNodes.has(n.id)) return 1
            return 0.3
          })
          .attr('r', (n: any) => {
            if (n.id === d.id) return Math.max(10, Math.min(n.degree, 20)) // Larger for hovered
            if (connectedNodes.has(n.id)) return Math.max(6, Math.min(n.degree, 17)) // Medium for connected
            return Math.max(4, Math.min(n.degree, 15))
          })
          .attr('stroke', (n: any) => {
            if (n.id === d.id) return '#00d4ff' // Cyan ring for hovered
            if (connectedNodes.has(n.id)) return '#4dd0e1' // Light cyan for connected
            // Use community color
            return colorScale(String(n.community))
          })
          .attr('stroke-width', (n: any) => {
            if (n.id === d.id) return 4
            if (connectedNodes.has(n.id)) return 2.5
            return 2
          })

        // Show tooltip
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
            Community: ${d.community}<br/>
            Connections: ${d.degree}
          `)
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 10) + 'px')
      })
      .on('mouseout', function(event, d: any) {
        // Reset edges to normal
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

        // Reset nodes - update both images and circles
        nodeGroup.selectAll('image')
          .attr('opacity', (n: any) => {
            if (selectedUser && n.id !== selectedUser && !userConnections.has(n.id)) {
              return 0.2
            }
            const highlightCommunity = hoveredCommunity !== null ? hoveredCommunity : selectedCommunity
            if (highlightCommunity !== null && n.community !== highlightCommunity) {
              return 0.2
            }
            return 1
          })

        nodeGroup.selectAll('circle')
          .attr('opacity', (n: any) => {
            if (selectedUser && n.id !== selectedUser && !userConnections.has(n.id)) {
              return 0.4
            }
            const highlightCommunity = hoveredCommunity !== null ? hoveredCommunity : selectedCommunity
            if (highlightCommunity !== null && n.community !== highlightCommunity) {
              return 0.3
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
            return colorScale(String(n.community))
          })
          .attr('stroke-width', (n: any) => {
            if (selectedUser && n.id === selectedUser) return 3
            if (selectedUser && userConnections.has(n.id)) return 2
            const highlightCommunity = hoveredCommunity !== null ? hoveredCommunity : selectedCommunity
            if (highlightCommunity !== null && n.community === highlightCommunity) {
              return 3
            }
            return 2
          })

        // Remove tooltip
        d3.selectAll('.tooltip-network').remove()
      })

    // Add profile images from avatar URLs
    nodeGroup.append('image')
      .attr('xlink:href', (d: any) => avatarUrls[d.id] || '')
      .attr('width', (d: any) => {
        const r = selectedUser && d.id === selectedUser
          ? Math.max(8, Math.min(d.degree, 20))
          : Math.max(4, Math.min(d.degree, 15))
        return r * 2
      })
      .attr('height', (d: any) => {
        const r = selectedUser && d.id === selectedUser
          ? Math.max(8, Math.min(d.degree, 20))
          : Math.max(4, Math.min(d.degree, 15))
        return r * 2
      })
      .attr('x', (d: any) => {
        const r = selectedUser && d.id === selectedUser
          ? Math.max(8, Math.min(d.degree, 20))
          : Math.max(4, Math.min(d.degree, 15))
        return -r
      })
      .attr('y', (d: any) => {
        const r = selectedUser && d.id === selectedUser
          ? Math.max(8, Math.min(d.degree, 20))
          : Math.max(4, Math.min(d.degree, 15))
        return -r
      })
      .attr('clip-path', (d: any) => `url(#clip-${d.id.replace(/[^a-zA-Z0-9]/g, '_')})`)
      .attr('opacity', (d: any) => {
        // Dim nodes that are not selected or connected
        if (selectedUser && d.id !== selectedUser && !userConnections.has(d.id)) {
          return 0.2
        }

        // Dim nodes not in the hovered/selected community
        const highlightCommunity = hoveredCommunity !== null ? hoveredCommunity : selectedCommunity
        if (highlightCommunity !== null && d.community !== highlightCommunity) {
          return 0.2
        }

        return 1
      })
      .on('error', function(this: any) {
        // Fallback to colored circle if image fails to load
        d3.select(this.parentNode).append('circle')
          .attr('r', 8)
          .attr('fill', '#666')
          .attr('stroke', '#fff')
          .attr('stroke-width', 2)
      })

    // Add community-colored border circles
    const node = nodeGroup.append('circle')
      .attr('r', (d: any) => {
        // Make selected user and their connections larger
        if (selectedUser) {
          if (d.id === selectedUser) return Math.max(8, Math.min(d.degree, 20))
          if (userConnections.has(d.id)) return Math.max(6, Math.min(d.degree, 17))
        }
        return Math.max(4, Math.min(d.degree, 15))
      })
      .attr('fill', 'none')
      .attr('stroke', (d: any) => {
        // Highlight selected user in bright cyan
        if (selectedUser && d.id === selectedUser) return '#00ff00' // Green ring for selected user
        // Highlight connections in lighter cyan
        if (selectedUser && userConnections.has(d.id)) return '#00d4ff' // Cyan ring for connections

        // Use community color for border
        const colorId = String(d.community)
        return colorScale(colorId)
      })
      .attr('stroke-width', (d: any) => {
        if (selectedUser && d.id === selectedUser) return 3
        if (selectedUser && userConnections.has(d.id)) return 2

        // Thicker border for highlighted community
        const highlightCommunity = hoveredCommunity !== null ? hoveredCommunity : selectedCommunity
        if (highlightCommunity !== null && d.community === highlightCommunity) {
          return 3
        }

        return 2
      })
      .attr('opacity', (d: any) => {
        // Dim nodes that are not selected or connected
        if (selectedUser && d.id !== selectedUser && !userConnections.has(d.id)) {
          return 0.4
        }

        // Dim nodes not in the hovered/selected community
        const highlightCommunity = hoveredCommunity !== null ? hoveredCommunity : selectedCommunity
        if (highlightCommunity !== null && d.community !== highlightCommunity) {
          return 0.3
        }

        return 1
      })
      .on('click', function(event, d: any) {
        // Navigate to user profile page
        window.location.href = `/network/${d.id}`
      })
      .call(d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended) as any)

    // Update positions on tick
    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y)

      nodeGroup
        .attr('transform', (d: any) => `translate(${d.x},${d.y})`)
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

      {/* Top-left: Navigation & Title */}
      <div className="absolute top-6 left-6 z-20">
        <Link href="/" className="text-white hover:text-gray-200 transition-colors text-sm mb-4 inline-block">
          ← Back to Search
        </Link>
        <h1 className={`text-4xl font-bold ${lugrasimo.className}`} style={{ color: '#ff0000' }}>
          Constellation of People
        </h1>
        <p className="text-gray-400 text-sm mt-1 max-w-md">
          How online communities formed and connected over time (2012-2025)
        </p>
      </div>

      {/* Top-right: Stats */}
      <div className="absolute top-6 right-6 z-20 flex gap-4">
        <div className="bg-black/80 backdrop-blur rounded-lg px-4 py-3 border border-white" style={{ boxShadow: '0 0 10px rgba(255,255,255,0.5), 0 0 20px rgba(255,255,255,0.3)' }}>
          <div className="text-2xl font-bold text-white">{stats.year}</div>
          <div className="text-gray-400 text-xs uppercase tracking-wider">Year</div>
        </div>
        <div className="bg-black/80 backdrop-blur rounded-lg px-4 py-3 border border-white" style={{ boxShadow: '0 0 10px rgba(255,255,255,0.5), 0 0 20px rgba(255,255,255,0.3)' }}>
          <div className="text-2xl font-bold text-white">{stats.users.toLocaleString()}</div>
          <div className="text-gray-400 text-xs uppercase tracking-wider">Users</div>
        </div>
        <div className="bg-black/80 backdrop-blur rounded-lg px-4 py-3 border border-white" style={{ boxShadow: '0 0 10px rgba(255,255,255,0.5), 0 0 20px rgba(255,255,255,0.3)' }}>
          <div className="text-2xl font-bold text-white">{stats.interactions.toLocaleString()}</div>
          <div className="text-gray-400 text-xs uppercase tracking-wider">Interactions</div>
        </div>
        <div className="bg-black/80 backdrop-blur rounded-lg px-4 py-3 border border-white" style={{ boxShadow: '0 0 10px rgba(255,255,255,0.5), 0 0 20px rgba(255,255,255,0.3)' }}>
          <div className="text-2xl font-bold text-white">{stats.communities}</div>
          <div className="text-gray-400 text-xs uppercase tracking-wider">Communities</div>
        </div>
      </div>

      {/* Bottom-left: Username Search */}
      <div className="absolute bottom-6 left-6 z-20 max-w-md">
        <div className="bg-black/80 backdrop-blur rounded-lg p-4 border border-white" style={{ boxShadow: '0 0 10px rgba(255,255,255,0.5), 0 0 20px rgba(255,255,255,0.3)' }}>
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
              className="flex-1 px-3 py-2 bg-black/50 border border-white rounded text-white placeholder-gray-500 focus:outline-none transition-colors text-sm"
              style={{ boxShadow: '0 0 5px rgba(255,255,255,0.3)' }}
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
            <div className="mt-3 p-3 bg-black/50 rounded border border-white" style={{ boxShadow: '0 0 10px rgba(255,255,255,0.4)' }}>
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
        <div className="bg-black/80 backdrop-blur rounded-lg p-3 border border-white flex gap-3" style={{ boxShadow: '0 0 10px rgba(255,255,255,0.5), 0 0 20px rgba(255,255,255,0.3)' }}>
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

      {/* Left Sidebar: Community List */}
      {showCommunitySidebar && networkData && allTopics && (
        <div className="absolute top-24 left-6 z-20 w-64 max-h-[calc(100vh-200px)] overflow-y-auto">
          <div className="bg-black/90 backdrop-blur rounded-lg p-4 border border-white" style={{ boxShadow: '0 0 10px rgba(255,255,255,0.5), 0 0 20px rgba(255,255,255,0.3)' }}>
            <div className="flex justify-between items-center mb-3">
              <h3 className="text-sm font-bold text-white">Communities ({stats.communities})</h3>
              <button
                onClick={() => setShowCommunitySidebar(false)}
                className="text-gray-400 hover:text-white text-xs"
              >
                ✕
              </button>
            </div>
            <div className="space-y-2">
              {(() => {
                const yearData = networkData.years[currentYear]

                if (viewMode === 'lineage' && temporalAlignments) {
                  // In lineage mode, show ALL lineages
                  const lineageMapping = buildLineageMapping()
                  if (!lineageMapping) return null

                  // Get all unique lineage IDs
                  const allLineageIds = new Set<number>(Object.values(lineageMapping))

                  // Get lineages that exist in the current year
                  const activeLineages = new Set<number>()
                  yearData.nodes.forEach((node: any) => {
                    const key = `${stats.year}_${node.community}`
                    const lineageId = lineageMapping[key]
                    if (lineageId !== undefined) {
                      activeLineages.add(lineageId)
                    }
                  })

                  return Array.from(allLineageIds).sort((a, b) => a - b).map((lineageId: number) => {
                    const isActive = activeLineages.has(lineageId)
                    const lineageName = `Lineage ${lineageId}`

                    return (
                      <button
                        key={lineageId}
                        onClick={() => {
                          if (isActive) {
                            // Find a community ID from this lineage in the current year
                            const communityEntry = Object.entries(lineageMapping).find(
                              ([key, lid]) => lid === lineageId && key.startsWith(`${stats.year}_`)
                            )
                            if (communityEntry) {
                              const commId = parseInt(communityEntry[0].split('_')[1])
                              setSelectedCommunity(commId)
                            }
                          } else {
                            setInactiveLineageMessage(`${lineageName} does not exist in ${stats.year}`)
                            setTimeout(() => setInactiveLineageMessage(null), 3000)
                          }
                        }}
                        onMouseEnter={() => {
                          if (isActive) {
                            const communityEntry = Object.entries(lineageMapping).find(
                              ([key, lid]) => lid === lineageId && key.startsWith(`${stats.year}_`)
                            )
                            if (communityEntry) {
                              const commId = parseInt(communityEntry[0].split('_')[1])
                              setHoveredCommunity(commId)
                            }
                          }
                        }}
                        onMouseLeave={() => setHoveredCommunity(null)}
                        className={`w-full text-left px-3 py-2 rounded transition-all text-xs ${
                          isActive
                            ? selectedCommunity !== null && Object.entries(lineageMapping).some(
                                ([key, lid]) => lid === lineageId && key.startsWith(`${stats.year}_`) && parseInt(key.split('_')[1]) === selectedCommunity
                              )
                              ? 'bg-purple-500 text-white'
                              : 'bg-black/50 text-gray-300 hover:bg-purple-500/30'
                            : 'bg-black/20 text-gray-600 cursor-not-allowed'
                        }`}
                      >
                        <div className="font-semibold">{lineageName}</div>
                        {!isActive && (
                          <div className="text-xs opacity-50">Inactive</div>
                        )}
                      </button>
                    )
                  })
                } else {
                  // Independent mode: show only current year's communities
                  const communityIds = new Set<number>()
                  yearData.nodes.forEach((node: any) => communityIds.add(node.community))
                  const sortedCommunities = Array.from(communityIds).sort((a, b) => a - b)

                  return sortedCommunities.map((communityId: number) => {
                    const hasTopics = allTopics?.[stats.year]?.communities?.[String(communityId)]
                    const highConfTopics = hasTopics?.filter((t: any) => t.confidence === 'high') || []
                    const topicCount = highConfTopics.length
                    const communityName = getCommunityName(parseInt(stats.year), communityId)

                    return (
                      <button
                        key={communityId}
                        onClick={() => setSelectedCommunity(communityId)}
                        onMouseEnter={() => setHoveredCommunity(communityId)}
                        onMouseLeave={() => setHoveredCommunity(null)}
                        className={`w-full text-left px-3 py-2 rounded transition-all text-xs ${
                          selectedCommunity === communityId
                            ? 'bg-purple-500 text-white'
                            : 'bg-black/50 text-gray-300 hover:bg-purple-500/30'
                        }`}
                      >
                        <div className="font-semibold">{communityName}</div>
                        {topicCount > 0 && (
                          <div className="text-xs opacity-70">{topicCount} topics</div>
                        )}
                      </button>
                    )
                  })
                }
              })()}
            </div>
          </div>
        </div>
      )}

      {/* Topics Modal Overlay */}
      {selectedCommunity !== null && allTopics && (
        <div
          className="fixed inset-0 bg-black/70 backdrop-blur-sm z-30 flex items-center justify-center p-6"
          onClick={() => setSelectedCommunity(null)}
        >
          <div
            className="bg-[#1a1f3a] rounded-lg border-2 border-white max-w-2xl w-full max-h-[80vh] overflow-hidden"
            style={{ boxShadow: '0 0 20px rgba(255,255,255,0.3), 0 0 40px rgba(138,43,226,0.2)' }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="p-6 border-b border-white/20">
              <div className="flex justify-between items-center">
                <h2 className="text-2xl font-bold text-white">
                  {getCommunityName(parseInt(stats.year), selectedCommunity)}
                </h2>
                <button
                  onClick={() => setSelectedCommunity(null)}
                  className="text-gray-400 hover:text-white text-2xl"
                >
                  ✕
                </button>
              </div>
              <p className="text-gray-400 text-sm mt-1">Year: {stats.year}</p>
            </div>

            {/* Topics List */}
            <div className="p-6 overflow-y-auto max-h-[calc(80vh-140px)]">
              {(() => {
                const communityTopics = allTopics?.[stats.year]?.communities?.[String(selectedCommunity)]
                // Filter to only high confidence topics
                const highConfTopics = communityTopics?.filter((t: any) => t.confidence === 'high') || []

                if (highConfTopics.length === 0) {
                  return (
                    <div className="text-center text-gray-400 py-8">
                      No high-confidence topics available for this community
                    </div>
                  )
                }

                return (
                  <div className="space-y-4">
                    {highConfTopics.map((topic: any, idx: number) => (
                      <button
                        key={idx}
                        onClick={() => {
                          setSelectedTopic(topic)
                          fetchTopicTweets(topic.tweet_ids?.slice(0, 50) || [], topic.sample_tweets)
                        }}
                        className="w-full text-left bg-black/30 rounded-lg p-4 border-l-4 border-purple-500 hover:bg-black/50 transition-colors cursor-pointer"
                      >
                        <h3 className="text-lg font-semibold text-white mb-2">{topic.topic}</h3>
                        <p className="text-gray-300 text-sm mb-3">{topic.description}</p>
                        <div className="flex gap-4 text-xs">
                          <span className="text-gray-400">
                            📊 {topic.num_tweets?.toLocaleString()} tweets
                          </span>
                          <span className="font-semibold text-green-400">
                            ✓ high confidence
                          </span>
                        </div>
                      </button>
                    ))}
                  </div>
                )
              })()}
            </div>
          </div>
        </div>
      )}

      {/* Bottom-right: Quick Guide & Toggle */}
      <div className="absolute bottom-6 right-6 z-20 max-w-xs space-y-3">
        <div className="bg-black/80 backdrop-blur rounded-lg p-4 border border-white" style={{ boxShadow: '0 0 10px rgba(255,255,255,0.5), 0 0 20px rgba(255,255,255,0.3)' }}>
          <h3 className="text-sm font-bold text-white mb-2">Guide</h3>
          <div className="text-xs text-gray-300 space-y-1">
            <div><strong className="text-white">Drag:</strong> Move nodes</div>
            <div><strong className="text-white">Scroll:</strong> Zoom</div>
            <div><strong className="text-white">Hover:</strong> See details</div>
            <div><strong className="text-white">Colors:</strong> {viewMode === 'independent' ? 'Communities within this year' : 'Lineages across multiple years'}</div>
          </div>
        </div>

        {/* View Mode Toggle */}
        <div className="bg-black/80 backdrop-blur rounded-lg p-3 border border-white" style={{ boxShadow: '0 0 10px rgba(255,255,255,0.5), 0 0 20px rgba(255,255,255,0.3)' }}>
          <h3 className="text-sm font-bold text-white mb-2">View Mode</h3>
          <div className="flex gap-2">
            <button
              onClick={() => setViewMode('independent')}
              className={`flex-1 px-3 py-2 rounded text-xs font-semibold transition-all ${
                viewMode === 'independent'
                  ? 'bg-purple-500 text-white'
                  : 'bg-black/30 text-gray-400 hover:bg-black/50'
              }`}
            >
              Independent
            </button>
            <button
              onClick={() => setViewMode('lineage')}
              className={`flex-1 px-3 py-2 rounded text-xs font-semibold transition-all ${
                viewMode === 'lineage'
                  ? 'bg-blue-500 text-white'
                  : 'bg-black/30 text-gray-400 hover:bg-black/50'
              }`}
            >
              Lineage
            </button>
          </div>
          <p className="text-xs text-gray-400 mt-2">
            {viewMode === 'independent'
              ? 'Each year\'s communities are colored independently. The same color may represent different groups in different years.'
              : 'Communities that persist across multiple years maintain the same color, making it easy to track their evolution over time.'}
          </p>
        </div>

        {!showCommunitySidebar && (
          <button
            onClick={() => setShowCommunitySidebar(true)}
            className="w-full px-4 py-2 bg-purple-500 hover:bg-purple-600 rounded font-semibold transition-all text-sm"
          >
            Show Communities
          </button>
        )}
      </div>

      {/* Tweet Viewer Modal */}
      {selectedTopic && (
        <div
          className="fixed inset-0 bg-black/70 backdrop-blur-sm z-40 flex items-center justify-center p-6"
          onClick={() => {
            setSelectedTopic(null)
            setTopicTweets([])
          }}
        >
          <div
            className="bg-[#1a1f3a] rounded-lg border-2 border-white max-w-3xl w-full max-h-[85vh] overflow-hidden"
            style={{ boxShadow: '0 0 20px rgba(255,255,255,0.3), 0 0 40px rgba(138,43,226,0.2)' }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="p-6 border-b border-white/20">
              <div className="flex justify-between items-start">
                <div>
                  <h2 className="text-2xl font-bold text-white">{selectedTopic.topic}</h2>
                  <p className="text-gray-400 text-sm mt-1">{selectedTopic.description}</p>
                  <p className="text-gray-500 text-xs mt-2">{selectedTopic.num_tweets?.toLocaleString()} tweets total (showing first 50)</p>
                </div>
                <button
                  onClick={() => {
                    setSelectedTopic(null)
                    setTopicTweets([])
                  }}
                  className="text-gray-400 hover:text-white text-2xl"
                >
                  ✕
                </button>
              </div>
            </div>

            <div className="p-6 overflow-y-auto max-h-[calc(85vh-180px)]">
              {loadingTweets ? (
                <div className="text-center text-gray-400 py-8">Loading tweets...</div>
              ) : topicTweets.length > 0 ? (
                <div className="space-y-4">
                  {topicTweets.map((tweet: any, idx: number) => (
                    <div
                      key={idx}
                      className="bg-black/30 backdrop-blur-md border border-white/20 rounded-lg p-6 hover:bg-black/40 transition-all"
                    >
                      {/* Parent Tweet (if this is a reply) */}
                      {tweet.parent_tweet && (
                        <div className="mb-4 pb-4 border-b border-white/10">
                          <div className="flex items-center space-x-2 mb-2">
                            <span className="text-xs text-gray-400">↪ Replying to:</span>
                          </div>
                          <div className="bg-black/20 rounded-lg p-4 border border-white/10">
                            <div className="flex items-center space-x-2 mb-2">
                              {tweet.parent_tweet.all_account?.username && (
                                <img
                                  src={tweet.parent_tweet.all_account.profile_image_url || `https://unavatar.io/x/${tweet.parent_tweet.all_account.username}`}
                                  alt={`@${tweet.parent_tweet.all_account?.username || 'unknown'}`}
                                  className="w-8 h-8 rounded-full"
                                  onError={(e) => {
                                    e.currentTarget.style.display = 'none';
                                    const nextSibling = e.currentTarget.nextElementSibling as HTMLElement;
                                    if (nextSibling) nextSibling.classList.remove('hidden');
                                  }}
                                />
                              )}
                              <div className={`w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-white text-sm font-bold ${tweet.parent_tweet.all_account?.username ? 'hidden' : ''}`}>
                                {tweet.parent_tweet.all_account?.username?.[0]?.toUpperCase() || '?'}
                              </div>
                              <div>
                                <div className="font-semibold text-white text-sm">
                                  @{tweet.parent_tweet.all_account?.username || 'unknown'}
                                </div>
                                <div className="text-xs text-gray-400">
                                  {tweet.parent_tweet.created_at ? new Date(tweet.parent_tweet.created_at).toLocaleDateString('en-US', {
                                    year: 'numeric',
                                    month: 'short',
                                    day: 'numeric',
                                  }) : 'Unknown date'}
                                </div>
                              </div>
                            </div>
                            <p className="text-gray-300 text-sm leading-relaxed">
                              {tweet.parent_tweet.full_text || 'Tweet text unavailable'}
                            </p>
                          </div>
                        </div>
                      )}

                      {/* Tweet Header */}
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center space-x-2">
                          {(tweet.all_account?.username || tweet.username) && (
                            <img
                              src={tweet.all_account?.profile_image_url || `https://unavatar.io/x/${tweet.all_account?.username || tweet.username}`}
                              alt={`@${tweet.all_account?.username || tweet.username || 'unknown'}`}
                              className="w-10 h-10 rounded-full"
                              onError={(e) => {
                                e.currentTarget.style.display = 'none';
                                const nextSibling = e.currentTarget.nextElementSibling as HTMLElement;
                                if (nextSibling) nextSibling.classList.remove('hidden');
                              }}
                            />
                          )}
                          <div className={`w-10 h-10 bg-purple-500 rounded-full flex items-center justify-center text-white font-bold ${(tweet.all_account?.username || tweet.username) ? 'hidden' : ''}`}>
                            {(tweet.all_account?.username || tweet.username)?.[0]?.toUpperCase() || '?'}
                          </div>
                          <div>
                            <div className="font-semibold text-white">
                              @{tweet.all_account?.username || tweet.username || 'unknown'}
                            </div>
                            <div className="text-sm text-gray-400">
                              {(tweet.created_at || tweet.timestamp) ? new Date(tweet.created_at || tweet.timestamp).toLocaleDateString('en-US', {
                                year: 'numeric',
                                month: 'short',
                                day: 'numeric',
                              }) : 'Unknown date'}
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* Tweet Text */}
                      <p className="text-white leading-relaxed mb-3">
                        {tweet.full_text || tweet.text || 'Tweet text unavailable'}
                      </p>

                      {/* Tweet Stats and Actions */}
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-4 text-sm text-gray-400">
                          <span>🔄 {tweet.retweet_count?.toLocaleString() || 0} retweets</span>
                          <span>❤️ {tweet.favorite_count?.toLocaleString() || 0} likes</span>
                        </div>
                        <a
                          href={`https://x.com/${tweet.all_account?.username || tweet.username || 'twitter'}/status/${tweet.tweet_id || tweet.id}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="px-4 py-2 bg-blue-500 text-white rounded-none hover:bg-blue-600 transition-colors text-sm font-medium"
                        >
                          View on Twitter
                        </a>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center text-gray-400 py-8">No tweets found</div>
              )}
            </div>
          </div>
        </div>
      )}

    </div>
  )
}
