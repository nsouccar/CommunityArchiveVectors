'use client'

import { useEffect, useRef, useState } from 'react'
import Link from 'next/link'
import * as d3 from 'd3'

export default function NetworkPage() {
  const svgRef = useRef<SVGSVGElement>(null)
  const [currentYear, setCurrentYear] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [networkData, setNetworkData] = useState<any>(null)
  const [stats, setStats] = useState({ year: '2012', users: 0, interactions: 0, communities: 0 })
  const simulationRef = useRef<any>(null)

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
      }, 2000) // Change year every 2 seconds
    }
    return () => clearInterval(interval)
  }, [isPlaying, networkData])

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

    const width = 1200
    const height = 700

    // Clear previous visualization
    d3.select(svgRef.current).selectAll('*').remove()

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height)

    const g = svg.append('g')

    // Add zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.1, 10])
      .on('zoom', (event) => {
        g.attr('transform', event.transform)
      })

    svg.call(zoom as any)

    // Color scale for communities
    const colorScale = d3.scaleOrdinal(d3.schemeTableau10)
      .domain(d3.range(0, 20).map(String))

    // Prepare data
    const nodes = yearData.nodes.map((d: any) => ({ ...d }))
    const links = yearData.edges.map((d: any) => ({ ...d }))

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
      .attr('stroke', '#4a5568')
      .attr('stroke-opacity', (d: any) => d.weight > 5 ? 0.6 : 0.3)
      .attr('stroke-width', (d: any) => Math.min(d.weight / 2, 4))

    // Create nodes
    const node = g.append('g')
      .selectAll('circle')
      .data(nodes)
      .join('circle')
      .attr('r', (d: any) => Math.max(4, Math.min(d.degree, 15)))
      .attr('fill', (d: any) => colorScale(String(d.community)))
      .attr('stroke', '#fff')
      .attr('stroke-width', 1.5)
      .style('cursor', 'pointer')
      .on('mouseover', function(event, d: any) {
        d3.select(this)
          .attr('stroke-width', 3)
          .attr('stroke', '#00d4ff')

        // Show tooltip
        const tooltip = d3.select('body')
          .append('div')
          .attr('class', 'tooltip-network')
          .style('position', 'absolute')
          .style('background', 'rgba(26, 31, 58, 0.95)')
          .style('border', '2px solid #00d4ff')
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
      .on('mouseout', function() {
        d3.select(this)
          .attr('stroke-width', 1.5)
          .attr('stroke', '#fff')

        d3.selectAll('.tooltip-network').remove()
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
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        {/* Navigation */}
        <div className="mb-8">
          <Link href="/" className="text-cyan-400 hover:text-cyan-300 transition-colors">
            ← Back to Search
          </Link>
        </div>

        {/* Header */}
        <h1 className="text-5xl font-bold text-center mb-4 bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
          Twitter Network Evolution
        </h1>
        <p className="text-center text-gray-400 text-xl mb-8">
          How online communities formed and connected over time (2012-2025)
        </p>

        {/* Stats */}
        <div className="grid grid-cols-4 gap-6 mb-8">
          <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 text-center border border-slate-700">
            <div className="text-4xl font-bold text-cyan-400">{stats.year}</div>
            <div className="text-gray-400 text-sm uppercase tracking-wider mt-2">Year</div>
          </div>
          <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 text-center border border-slate-700">
            <div className="text-4xl font-bold text-cyan-400">{stats.users.toLocaleString()}</div>
            <div className="text-gray-400 text-sm uppercase tracking-wider mt-2">Users</div>
          </div>
          <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 text-center border border-slate-700">
            <div className="text-4xl font-bold text-cyan-400">{stats.interactions.toLocaleString()}</div>
            <div className="text-gray-400 text-sm uppercase tracking-wider mt-2">Interactions</div>
          </div>
          <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 text-center border border-slate-700">
            <div className="text-4xl font-bold text-cyan-400">{stats.communities}</div>
            <div className="text-gray-400 text-sm uppercase tracking-wider mt-2">Communities</div>
          </div>
        </div>

        {/* Visualization Container */}
        <div className="bg-slate-800/50 backdrop-blur rounded-xl p-8 border border-slate-700 mb-8 relative">
          <div className="absolute top-8 left-8 text-8xl font-bold text-cyan-400/20 pointer-events-none z-10">
            {stats.year}
          </div>

          {/* SVG Canvas */}
          <div className="bg-slate-900/50 rounded-lg overflow-hidden flex items-center justify-center" style={{ height: '700px' }}>
            <svg ref={svgRef} style={{ maxWidth: '100%', maxHeight: '100%' }}></svg>

            {/* Loading State */}
            {!networkData && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-cyan-400 text-xl">Loading network data...</div>
              </div>
            )}
          </div>
        </div>

        {/* Controls */}
        <div className="flex justify-center gap-4 mb-8">
          <button
            onClick={handlePlay}
            disabled={!networkData}
            className="px-6 py-3 bg-cyan-500 hover:bg-cyan-600 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg font-semibold transition-all hover:scale-105"
          >
            {isPlaying ? 'Pause' : 'Play Animation'}
          </button>
          <button
            onClick={handlePrev}
            disabled={!networkData || currentYear === 0}
            className="px-6 py-3 bg-purple-500 hover:bg-purple-600 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg font-semibold transition-all hover:scale-105"
          >
            Previous Year
          </button>
          <button
            onClick={handleNext}
            disabled={!networkData || currentYear >= (networkData?.years.length || 0) - 1}
            className="px-6 py-3 bg-purple-500 hover:bg-purple-600 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg font-semibold transition-all hover:scale-105"
          >
            Next Year
          </button>
          <button
            onClick={handleReset}
            disabled={!networkData}
            className="px-6 py-3 bg-slate-600 hover:bg-slate-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg font-semibold transition-all hover:scale-105"
          >
            Reset
          </button>
        </div>

        {/* Legend */}
        <div className="bg-slate-800/50 backdrop-blur rounded-xl p-6 border border-slate-700">
          <h3 className="text-2xl font-bold text-cyan-400 mb-4">Network Visualization Guide</h3>
          <div className="grid grid-cols-2 gap-4 text-gray-300">
            <div>
              <strong className="text-white">Circles (Nodes):</strong> Each circle represents a user
            </div>
            <div>
              <strong className="text-white">Lines (Edges):</strong> Reply interactions between users
            </div>
            <div>
              <strong className="text-white">Colors:</strong> Different communities (groups who interact frequently)
            </div>
            <div>
              <strong className="text-white">Size:</strong> Number of connections (bigger = more active)
            </div>
            <div>
              <strong className="text-white">Thick lines:</strong> Many interactions (5+  replies)
            </div>
            <div>
              <strong className="text-white">Bridge users:</strong> Connect multiple colored communities
            </div>
          </div>
          <p className="text-gray-400 mt-4">
            <strong>Interaction:</strong> Drag nodes to move them around, scroll to zoom, hover over nodes to see details
          </p>
        </div>
      </div>
    </div>
  )
}
