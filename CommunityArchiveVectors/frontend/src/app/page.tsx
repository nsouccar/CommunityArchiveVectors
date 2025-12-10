'use client'

import { useEffect, useRef, useState } from 'react'
import { useSearchParams } from 'next/navigation'
import { lugrasimo } from './fonts'
import * as d3 from 'd3'
import TutorialModal from '@/components/TutorialModal'
import LoadingScreen from '@/components/LoadingScreen'
import ProfileAvatar from '@/components/ProfileAvatar'
import { useDataLoader } from '@/hooks/useDataLoader'
import './retro.css'

export default function Home() {
  const searchParams = useSearchParams()
  const returnYearParam = searchParams.get('year')

  const svgRef = useRef<SVGSVGElement>(null)
  const [currentYear, setCurrentYear] = useState(0)
  const [initialYearSet, setInitialYearSet] = useState(false)
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

  // Tutorial modal state
  const [showTutorial, setShowTutorial] = useState(false)

  // Load all data with caching
  const { isLoading: isDataLoading, progress: loadingProgress, data: loadedData, error: loadError } = useDataLoader()

  // Initialize state from loaded data
  useEffect(() => {
    if (loadedData) {
      console.log('[Page] Initializing from loaded data...')
      setNetworkData(loadedData.networkData)
      setAvatarUrls(loadedData.avatarUrls)
      setCommunityNames(loadedData.communityNames)
      setCommunityTopics(loadedData.communityTopics)
      setTemporalAlignments(loadedData.temporalAlignments)
      setAllTopics(loadedData.allTopics)

      // Mark all years as loaded since we pre-loaded everything
      const years = Object.keys(loadedData.allTopics)
      setLoadedYears(new Set(years))

      // Initialize visualization - check for return year param first
      if (loadedData.networkData?.years?.length > 0) {
        let initialYearIndex = 0

        // Check if we're returning from a profile page with a specific year
        if (returnYearParam) {
          const yearIndex = loadedData.networkData.years.findIndex((y: any) => y.year === returnYearParam)
          if (yearIndex !== -1) {
            initialYearIndex = yearIndex
            setInitialYearSet(true)
          }
        }

        setCurrentYear(initialYearIndex)
        updateVisualization(initialYearIndex, loadedData.networkData, loadedData.avatarUrls)
      }
    }
  }, [loadedData, returnYearParam])

  // Auto-open tutorial on first visit
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const hasSeenTutorial = localStorage.getItem('hasSeenTutorial')
      if (!hasSeenTutorial) {
        setShowTutorial(true)
        localStorage.setItem('hasSeenTutorial', 'true')
      }
    }
  }, [])

  const fetchTopicTweets = async (sampleTweets?: any[]) => {
    setLoadingTweets(true)

    // Tweets are now embedded in topic JSON files as sample_tweets
    if (sampleTweets && sampleTweets.length > 0) {
      console.log('Using embedded sample tweets:', sampleTweets.length)
      setTopicTweets(sampleTweets)
      setLoadingTweets(false)
      return
    }

    // No sample tweets available for this topic
    console.log('No sample tweets available for this topic')
    setTopicTweets([])
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
      console.log(`Loaded topics for year ${year}`)
    } catch (err) {
      console.warn(`Failed to load topics for year ${year}:`, err)
    }
  }


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

  // Set initial year from URL parameter if present
  useEffect(() => {
    if (returnYearParam && networkData && !initialYearSet) {
      const yearIndex = networkData.years.findIndex((y: any) => y.year === returnYearParam)
      if (yearIndex !== -1) {
        setCurrentYear(yearIndex)
        updateVisualization(yearIndex, networkData)
        setInitialYearSet(true)
      }
    }
  }, [returnYearParam, networkData, initialYearSet])

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

        console.log('Built lineage names:', Object.keys(lineageToName).length, 'lineages named')
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
      .attr('opacity', function (d: any) {
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
      .attr('stroke', function (d: any) {
        // Preserve user selection highlighting
        if (selectedUser && d.id === selectedUser) return '#00ff00'
        if (selectedUser && userConnections.has(d.id)) return '#00d4ff'

        // Use community color
        return colorScale(String(d.community))
      })
      .attr('opacity', function (d: any) {
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
      .attr('r', function (d: any) {
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
      .attr('stroke-width', function (d: any) {
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
          .on('end', function (d: any) {
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
      console.log('Cannot build lineage mapping:', {
        temporalAlignments: temporalAlignments ? 'exists' : 'null',
        alignments: temporalAlignments?.alignments ? `${temporalAlignments.alignments.length} alignments` : 'null'
      })
      return null
    }

    console.log('Building lineage mapping from', temporalAlignments.alignments.length, 'alignments')

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

    console.log('Built connection graph with', Object.keys(connections).length, 'nodes')

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

    console.log('Created', lineageCounter, 'lineages from', Object.keys(mapping).length, 'community-year pairs')
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

  const updateVisualization = (yearIndex: number, data: any, avatarUrlsParam?: Record<string, string>) => {
    if (!data || !data.years || yearIndex >= data.years.length || !svgRef.current) return

    const yearData = data.years[yearIndex]
    setStats({
      year: yearData.year,
      users: yearData.num_users,
      interactions: yearData.num_interactions,
      communities: yearData.num_communities
    })

    // Pass avatarUrls to renderNetwork (use param if provided, otherwise state)
    renderNetwork(yearData, avatarUrlsParam || avatarUrls)
  }

  const renderNetwork = (yearData: any, avatarUrlsToUse: Record<string, string>) => {
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
      .on('click', function (event, d: any) {
        const currentYearString = yearData.year || '2012'
        window.location.href = `/network/${d.id}?returnYear=${currentYearString}`
      })
      .on('mouseover', function (event, d: any) {
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
      .on('mouseout', function (event, d: any) {
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

    // Add profile images from avatar URLs (fall back to unavatar.io)
    nodeGroup.append('image')
      .attr('href', (d: any) => avatarUrlsToUse[d.id] || `https://unavatar.io/x/${d.id}`)
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
      // Use native onerror handler - D3's .on('error') doesn't work reliably for SVG images
      .each(function(this: SVGImageElement, d: any) {
        const img = this
        img.onerror = () => {
          const currentHref = img.getAttribute('href') || ''
          // If we haven't tried unavatar.io yet, try it as fallback
          if (!currentHref.includes('unavatar.io')) {
            img.setAttribute('href', `https://unavatar.io/x/${d.id}`)
          } else {
            // unavatar.io also failed - hide image and we'll rely on the border circle
            img.style.opacity = '0'
          }
        }
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

  // Show loading screen while data is being fetched and cached
  if (isDataLoading) {
    return <LoadingScreen progress={loadingProgress} />
  }

  // Show error state
  if (loadError) {
    return (
      <div
        className="fixed inset-0 flex flex-col items-center justify-center text-[#e8dcc8]"
        style={{
          backgroundImage: 'url(/stars.png)',
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          backgroundRepeat: 'no-repeat'
        }}
      >
        <h1 className="text-2xl font-bold mb-4" style={{ color: '#ff66ff' }}>
          Error Loading Data
        </h1>
        <p className="text-sm opacity-80 mb-4" style={{ fontFamily: 'monospace' }}>
          {loadError}
        </p>
        <button
          onClick={() => window.location.reload()}
          className="px-4 py-2 border border-[#ff66ff] text-[#ff66ff] hover:bg-[#ff66ff]/20"
          style={{ fontFamily: 'monospace' }}
        >
          Retry
        </button>
      </div>
    )
  }

  return (
    <div className="fixed inset-0 text-[#e8dcc8] overflow-hidden" style={{
      backgroundImage: 'url(/stars.png)',
      backgroundSize: 'cover',
      backgroundPosition: 'center',
      backgroundRepeat: 'no-repeat'
    }}>
      {/* Tutorial Modal */}
      <TutorialModal
        isOpen={showTutorial}
        onClose={() => setShowTutorial(false)}
        onOpen={() => setShowTutorial(true)}
      />

      {/* Full-screen SVG Canvas */}
      <div className="absolute inset-0">
        <svg ref={svgRef} className="w-full h-full" style={{ background: 'transparent' }}></svg>
      </div>

      {/* Overlay UI Elements */}

      {/* Top-left: Title */}
      <div className="absolute top-6 left-6 z-20">
        <h1 className={`text-4xl font-bold ${lugrasimo.className}`} style={{
          color: '#ff66ff'
        }}>
          CONSTELLATION OF PEOPLE
        </h1>
        <p className="retro-text-secondary text-sm mt-1 max-w-md" style={{ fontFamily: 'monospace', letterSpacing: '0.05em' }}>
          HOW ONLINE COMMUNITIES FORMED AND CONNECTED OVER TIME (2012-2025)
        </p>
      </div>

      {/* Top-right: Stats */}
      <div className="absolute top-6 right-6 z-20 flex gap-4 items-start">
        <div className="flex gap-4">
          <div className="retro-box px-4 py-3 scanlines">
            <div className="text-2xl font-bold retro-stat">{stats.year}</div>
            <div className="retro-text-secondary text-xs uppercase tracking-wider">YEAR</div>
          </div>
          <div className="retro-box px-4 py-3 scanlines">
            <div className="text-2xl font-bold retro-stat">{stats.users.toLocaleString()}</div>
            <div className="retro-text-secondary text-xs uppercase tracking-wider">USERS</div>
          </div>
          <div className="retro-box px-4 py-3 scanlines">
            <div className="text-2xl font-bold retro-stat">{stats.interactions.toLocaleString()}</div>
            <div className="retro-text-secondary text-xs uppercase tracking-wider">INTERACTIONS</div>
          </div>
          <button
            onClick={() => setShowTutorial(true)}
            className={`${lugrasimo.className} retro-button px-4 py-3 text-xs whitespace-nowrap`}
            style={{ fontFamily: 'monospace', letterSpacing: '0.05em' }}
          >
            LEARN HOW THIS WORKS
          </button>
        </div>
      </div>

      {/* Bottom-left: Username Search */}
      <div className="absolute bottom-6 left-6 z-20 max-w-md">
        <div className={`${lugrasimo.className} retro-box p-4`}>
          <h3 className="retro-text-secondary text-sm font-bold mb-3" style={{ fontFamily: 'monospace', letterSpacing: '0.05em' }}>SEARCH USER</h3>
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
              placeholder="ENTER USERNAME..."
              className="flex-1 px-3 py-2 retro-input text-sm"
            />
            <button
              onClick={() => {
                if (searchUsername.trim()) {
                  setSelectedUser(searchUsername.trim())
                }
              }}
              disabled={!searchUsername.trim()}
              className="retro-button px-4 py-2 text-sm"
            >
              SEARCH
            </button>
            {selectedUser && (
              <button
                onClick={() => {
                  setSelectedUser(null)
                  setSearchUsername('')
                  setUserConnections(new Set())
                }}
                className="retro-button px-4 py-2 text-sm"
              >
                CLEAR
              </button>
            )}
          </div>
          {selectedUser && (
            <div className="mt-3 p-3 retro-card">
              <p className="retro-text text-sm">
                TRACKING: <span className="font-bold">@{selectedUser}</span>
                {userConnections.size > 0 && (
                  <span className="retro-text-secondary ml-2">
                    ({userConnections.size} CONNECTIONS)
                  </span>
                )}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Bottom-center: Controls */}
      <div className="absolute bottom-6 left-1/2 transform -translate-x-1/2 z-20">
        <div className={`${lugrasimo.className} retro-box p-3 flex gap-3`}>
          <button
            onClick={handlePlay}
            disabled={!networkData}
            className={`retro-button px-5 py-2 text-sm ${isPlaying ? 'retro-button-active' : ''}`}
          >
            {isPlaying ? '|| PAUSE' : '> PLAY'}
          </button>
          <button
            onClick={handlePrev}
            disabled={!networkData || currentYear === 0}
            className="retro-button px-5 py-2 text-sm"
          >
            PREV
          </button>
          <button
            onClick={handleNext}
            disabled={!networkData || currentYear >= (networkData?.years.length || 0) - 1}
            className="retro-button px-5 py-2 text-sm"
          >
            NEXT
          </button>
          <button
            onClick={handleReset}
            disabled={!networkData}
            className="retro-button px-5 py-2 text-sm"
          >
            RESET
          </button>
        </div>
      </div>

      {/* Left Sidebar: Community List */}
      {showCommunitySidebar && networkData && allTopics && (
        <div className="absolute top-32 left-6 z-20 w-64 max-h-[calc(100vh-350px)] overflow-y-auto border-b-2 border-[#6b9080]">
          <div className={`${lugrasimo.className} bg-[#6b9080]/10 border-2 border-[#6b9080] rounded-none p-4`}>
            {(() => {
              const yearData = networkData.years[currentYear]

              // Count communities with topics for the header
              const communityIds = new Set<number>()
              yearData.nodes.forEach((node: any) => communityIds.add(node.community))
              const communitiesWithTopicsCount = Array.from(communityIds).filter((communityId: number) => {
                const topics = allTopics?.[stats.year]?.communities?.[String(communityId)]
                const highConfidenceTopics = topics?.filter((t: any) => t.confidence === 'high') || []
                return highConfidenceTopics.length > 0
              }).length

              return (
                <>
                  <div className="flex justify-between items-center mb-3">
                    <h3 className="retro-text-secondary text-sm font-bold" style={{ fontFamily: 'monospace', letterSpacing: '0.05em' }}>Communities ({communitiesWithTopicsCount})</h3>
                    <button
                      onClick={() => setShowCommunitySidebar(false)}
                      className="text-gray-400 hover:text-[#d4a574] text-xs"
                    >
                      ✕
                    </button>
                  </div>
                  <div className="space-y-2">
                    {(() => {

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
                        className={`w-full text-left px-4 py-2 transition-colors text-xs rounded-none ${isActive
                            ? selectedCommunity !== null && Object.entries(lineageMapping).some(
                              ([key, lid]) => lid === lineageId && key.startsWith(`${stats.year}_`) && parseInt(key.split('_')[1]) === selectedCommunity
                            )
                              ? 'bg-[#d4a574]/30 border-2 border-[#6b9080]'
                              : 'bg-[#6b9080]/10 border-2 border-[#6b9080] hover:bg-[#6b9080]/20'
                            : 'bg-black/20 border-2 border-gray-600 cursor-not-allowed'
                          }`}
                      >
                        <div className={`font-semibold ${isActive ? 'text-[#d4a574]' : 'text-gray-600'}`}>{lineageName}</div>
                        {!isActive && (
                          <div className="text-xs opacity-50">Inactive</div>
                        )}
                      </button>
                    )
                  })
                } else {
                  // Independent mode: show only current year's communities that have topics
                  const communityIds = new Set<number>()
                  yearData.nodes.forEach((node: any) => communityIds.add(node.community))

                  // Filter to only communities with high-confidence topics
                  const communitiesWithTopics = Array.from(communityIds).filter((communityId: number) => {
                    const topics = allTopics?.[stats.year]?.communities?.[String(communityId)]
                    const highConfidenceTopics = topics?.filter((t: any) => t.confidence === 'high') || []
                    return highConfidenceTopics.length > 0
                  })
                  const sortedCommunities = communitiesWithTopics.sort((a, b) => a - b)

                  return sortedCommunities.map((communityId: number) => {
                    const hasTopics = allTopics?.[stats.year]?.communities?.[String(communityId)]
                    const displayTopics = hasTopics?.filter((t: any) => t.confidence === 'high') || []
                    const topicCount = displayTopics.length
                    const communityName = getCommunityName(parseInt(stats.year), communityId)

                    return (
                      <button
                        key={communityId}
                        onClick={() => setSelectedCommunity(communityId)}
                        onMouseEnter={() => setHoveredCommunity(communityId)}
                        onMouseLeave={() => setHoveredCommunity(null)}
                        className={`w-full text-left px-4 py-2 transition-colors text-xs rounded-none ${selectedCommunity === communityId
                            ? 'bg-[#d4a574]/30 border-2 border-[#6b9080]'
                            : 'bg-[#6b9080]/10 border-2 border-[#6b9080] hover:bg-[#6b9080]/20'
                          }`}
                      >
                        <div className="font-semibold text-[#d4a574]">{communityName}</div>
                        {topicCount > 0 && (
                          <div className="text-xs opacity-70">{topicCount} topics</div>
                        )}
                      </button>
                    )
                  })
                }
                    })()}
                  </div>
                </>
              )
            })()}
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
            className={`${lugrasimo.className} bg-[#6b9080]/10 border-2 border-[#6b9080] rounded-none max-w-2xl w-full max-h-[80vh] flex flex-col`}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="p-6 border-b-2 border-[#6b9080] flex-shrink-0">
              <div className="flex justify-between items-center">
                <h2 className="text-2xl font-bold text-[#d4a574]">
                  {getCommunityName(parseInt(stats.year), selectedCommunity)}
                </h2>
                <button
                  onClick={() => setSelectedCommunity(null)}
                  className="text-gray-400 hover:text-[#d4a574] text-2xl"
                >
                  ✕
                </button>
              </div>
              <p className="retro-text-secondary text-sm mt-1" style={{ fontFamily: 'monospace', letterSpacing: '0.05em' }}>Year: {stats.year}</p>
            </div>

            {/* Topics List */}
            <div className="overflow-y-auto flex-1 p-6">
              {(() => {
                const communityTopics = allTopics?.[stats.year]?.communities?.[String(selectedCommunity)]
                const displayTopics = communityTopics?.filter((t: any) => t.confidence === 'high') || []

                if (displayTopics.length === 0) {
                  return (
                    <div className="text-center text-gray-400 py-8">
                      No topics available for this community
                    </div>
                  )
                }

                return (
                  <div className="space-y-4">
                    {displayTopics.map((topic: any, idx: number) => (
                      <button
                        key={idx}
                        onClick={() => {
                          setSelectedTopic(topic)
                          fetchTopicTweets(topic.sample_tweets)
                        }}
                        className="w-full text-left bg-[#6b9080]/10 border-2 border-[#6b89a8] rounded-none p-4 hover:bg-[#6b9080]/20 transition-colors cursor-pointer"
                      >
                        <h3 className="text-lg font-semibold text-[#d4a574] mb-2">{topic.topic}</h3>
                        <p className="retro-text-secondary text-sm mb-3" style={{ fontFamily: 'monospace', letterSpacing: '0.05em' }}>{topic.description}</p>
                        <div className="flex gap-4 text-xs">
                          <span className="retro-text-secondary" style={{ fontFamily: 'monospace', letterSpacing: '0.05em' }}>
                            {topic.num_tweets?.toLocaleString()} tweets
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
      <div className={`${lugrasimo.className} absolute bottom-6 right-6 z-20 max-w-xs space-y-3`}>
        <div className="bg-[#6b9080]/10 border-2 border-[#6b9080] rounded-none p-4">
          <h3 className="retro-text-secondary text-sm font-bold mb-2" style={{ fontFamily: 'monospace', letterSpacing: '0.05em' }}>Guide</h3>
          <div className="text-xs text-gray-300 space-y-1">
            <div><strong className="text-[#d4a574]">Drag:</strong> Move nodes</div>
            <div><strong className="text-[#d4a574]">Scroll:</strong> Zoom</div>
            <div><strong className="text-[#d4a574]">Hover:</strong> See details</div>
            <div><strong className="text-[#d4a574]">Click:</strong> View profile</div>
            <div><strong className="text-[#d4a574]">Colors:</strong> {viewMode === 'independent' ? 'Communities within this year' : 'Lineages across multiple years'}</div>
          </div>
        </div>

        {/* View Mode Toggle */}
        <div className="bg-[#6b9080]/10 border-2 border-[#6b9080] rounded-none p-3">
          <h3 className="retro-text-secondary text-sm font-bold mb-2" style={{ fontFamily: 'monospace', letterSpacing: '0.05em' }}>View Mode</h3>
          <div className="flex gap-2">
            <button
              onClick={() => setViewMode('independent')}
              className={`flex-1 px-3 py-2 text-xs font-semibold transition-colors rounded-none ${viewMode === 'independent'
                  ? 'bg-[#d4a574]/30 border-2 border-[#6b9080] text-[#d4a574]'
                  : 'bg-[#6b9080]/10 border-2 border-[#6b9080] text-gray-300 hover:bg-[#6b9080]/20'
                }`}
            >
              Independent
            </button>
            <button
              onClick={() => setViewMode('lineage')}
              className={`flex-1 px-3 py-2 text-xs font-semibold transition-colors rounded-none ${viewMode === 'lineage'
                  ? 'bg-[#d4a574]/30 border-2 border-[#6b9080] text-[#d4a574]'
                  : 'bg-[#6b9080]/10 border-2 border-[#6b9080] text-gray-300 hover:bg-[#6b9080]/20'
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
            className="w-full px-4 py-2 bg-[#6b9080]/10 border-2 border-[#6b9080] rounded-none hover:bg-[#6b9080]/20 transition-colors text-sm text-[#d4a574]"
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
            className={`${lugrasimo.className} retro-box max-w-3xl w-full max-h-[85vh] flex flex-col`}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="p-6 border-b-2 border-[#6b9080]/50 flex-shrink-0">
              <div className="flex justify-between items-start">
                <div>
                  <h2 className="text-2xl font-bold retro-text uppercase">{selectedTopic.topic}</h2>
                  <p className="retro-text-secondary text-sm mt-1" style={{ fontFamily: 'monospace', letterSpacing: '0.05em' }}>{selectedTopic.description}</p>
                  <p className="retro-text-secondary text-xs mt-2" style={{ fontFamily: 'monospace', letterSpacing: '0.05em' }}>{selectedTopic.num_tweets?.toLocaleString()} TWEETS TOTAL (SHOWING FIRST 50)</p>
                </div>
                <button
                  onClick={() => {
                    setSelectedTopic(null)
                    setTopicTweets([])
                  }}
                  className="retro-text hover:text-gray-300 text-2xl"
                >
                  ✕
                </button>
              </div>
            </div>

            <div className="overflow-y-auto flex-1 relative">
              <div className="absolute inset-0 scanlines pointer-events-none" />
              <div className="p-6 relative z-10">
                {loadingTweets ? (
                  <div className="text-center retro-text-secondary py-8" style={{ fontFamily: 'monospace', letterSpacing: '0.05em' }}>LOADING TWEETS...</div>
                ) : topicTweets.length > 0 ? (
                  <div className="space-y-4">
                    {topicTweets.map((tweet: any, idx: number) => (
                      <div
                        key={idx}
                        className="retro-card p-6 hover:border-[#6b9080] transition-all"
                      >
                        {/* Parent Tweet (if this is a reply) */}
                        {tweet.parent_tweet && (
                          <div className="mb-4 pb-4 border-b-2 border-gray-700/30">
                            <div className="flex items-center space-x-2 mb-2">
                              <span className="text-xs retro-text-secondary" style={{ fontFamily: 'monospace', letterSpacing: '0.05em' }}>REPLYING TO:</span>
                            </div>
                            <div className="bg-black/40 p-4 border-2 border-gray-700/50">
                              <div className="flex items-center space-x-2 mb-2">
                                <ProfileAvatar
                                  username={tweet.parent_tweet.all_account?.username}
                                  displayName={tweet.parent_tweet.all_account?.account_display_name}
                                  imageUrl={tweet.parent_tweet.all_account?.profile_image_url}
                                  size="sm"
                                />
                                <div>
                                  <div className="font-semibold retro-text text-sm" style={{ fontFamily: 'monospace' }}>
                                    @{tweet.parent_tweet.all_account?.username || 'unknown'}
                                  </div>
                                  <div className="text-xs retro-text-secondary" style={{ fontFamily: 'monospace' }}>
                                    {tweet.parent_tweet.created_at ? new Date(tweet.parent_tweet.created_at).toLocaleDateString('en-US', {
                                      year: 'numeric',
                                      month: 'short',
                                      day: 'numeric',
                                    }) : 'Unknown date'}
                                  </div>
                                </div>
                              </div>
                              <p className="text-[#e8dcc8] text-sm leading-relaxed" style={{ fontFamily: 'monospace' }}>
                                {tweet.parent_tweet.full_text || 'Tweet text unavailable'}
                              </p>
                            </div>
                          </div>
                        )}

                        {/* Tweet Header */}
                        <div className="flex items-center justify-between mb-3">
                          <div className="flex items-center space-x-2">
                            <ProfileAvatar
                              username={tweet.all_account?.username || tweet.username}
                              displayName={tweet.all_account?.account_display_name}
                              imageUrl={tweet.all_account?.profile_image_url}
                              size="md"
                            />
                            <div>
                              <div className="font-semibold retro-text" style={{ fontFamily: 'monospace' }}>
                                @{tweet.all_account?.username || tweet.username || 'unknown'}
                              </div>
                              <div className="text-sm retro-text-secondary" style={{ fontFamily: 'monospace' }}>
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
                        <p className="text-[#e8dcc8] leading-relaxed mb-3" style={{ fontFamily: 'monospace' }}>
                          {tweet.full_text || tweet.text || 'Tweet text unavailable'}
                        </p>

                        {/* Tweet Stats and Actions */}
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-4 text-sm retro-text-secondary" style={{ fontFamily: 'monospace' }}>
                            <span>RT {tweet.retweet_count?.toLocaleString() || 0}</span>
                            <span>Likes {tweet.favorite_count?.toLocaleString() || 0}</span>
                          </div>
                          <a
                            href={`https://x.com/${tweet.all_account?.username || tweet.username || 'twitter'}/status/${tweet.tweet_id || tweet.id}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="retro-button px-4 py-2 text-sm"
                          >
                            VIEW ON X
                          </a>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center retro-text-secondary py-8" style={{ fontFamily: 'monospace', letterSpacing: '0.05em' }}>NO TWEETS FOUND</div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

    </div>
  )
}
