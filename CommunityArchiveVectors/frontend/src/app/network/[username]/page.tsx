'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import Image from 'next/image'
import { useParams } from 'next/navigation'

export default function UserProfilePage() {
  const params = useParams()
  const username = params.username as string
  const [profileData, setProfileData] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [imageError, setImageError] = useState(false)
  const [temporalAlignments, setTemporalAlignments] = useState<any>(null)
  const [communityNames, setCommunityNames] = useState<any>(null)

  useEffect(() => {
    // Load temporal alignments and community names
    Promise.all([
      fetch('/community_temporal_alignments.json').then(res => res.json()),
      fetch('/data/all_community_names.json').then(res => res.json())
    ]).then(([alignments, names]) => {
      setTemporalAlignments(alignments)
      setCommunityNames(names)
    }).catch(err => console.error('Error loading community data:', err))
  }, [])

  useEffect(() => {
    // Load network data to get user info
    fetch('/network_animation_data.json')
      .then(res => res.json())
      .then(data => {
        // Find this user across all years
        const userInfo: any = {
          username,
          yearsActive: [],
          communities: {},
          totalConnections: 0,
          totalBetweenness: 0,
          betweennessValues: [],
          isSuperConnector: false,
        }

        // Collect info from each year
        data.years.forEach((yearData: any) => {
          const userNode = yearData.nodes.find((n: any) => n.id === username)
          if (userNode) {
            userInfo.yearsActive.push(yearData.year)
            userInfo.communities[yearData.year] = userNode.community
            userInfo.totalConnections += userNode.degree || 0
            const betweenness = userNode.betweenness || 0
            userInfo.totalBetweenness += betweenness
            userInfo.betweennessValues.push(betweenness)
          }
        })

        // Store lineages info for later use
        userInfo.lineages = {}
        userInfo.communityNamesList = {}

        // Calculate average connections and betweenness
        userInfo.avgConnections = userInfo.yearsActive.length > 0
          ? Math.round(userInfo.totalConnections / userInfo.yearsActive.length)
          : 0

        userInfo.avgBetweenness = userInfo.yearsActive.length > 0
          ? userInfo.totalBetweenness / userInfo.yearsActive.length
          : 0

        // Super connector based on betweenness centrality (top ~10% have avg > 0.02)
        userInfo.isSuperConnector = userInfo.avgBetweenness > 0.02

        setProfileData(userInfo)
        setLoading(false)
      })
      .catch(err => {
        console.error('Error loading profile:', err)
        setLoading(false)
      })
  }, [username])

  // Compute lineage information when data is ready
  useEffect(() => {
    if (!profileData || !temporalAlignments || !communityNames) return

    // Build lineage mapping
    const lineageMapping: Record<string, number> = {}
    if (temporalAlignments) {
      Object.entries(temporalAlignments).forEach(([fromKey, alignments]: [string, any]) => {
        const [fromYear, fromComm] = fromKey.split('_').map(Number)

        if (lineageMapping[fromKey] === undefined) {
          const lineageId = Object.keys(lineageMapping).length
          lineageMapping[fromKey] = lineageId
        }

        if (alignments && alignments.length > 0) {
          const bestMatch = alignments[0]
          const toKey = `${bestMatch.to_year}_${bestMatch.to_community}`

          if (lineageMapping[toKey] === undefined) {
            lineageMapping[toKey] = lineageMapping[fromKey]
          }
        }
      })
    }

    // Update profileData with lineages and community names
    const updatedLineages: Record<string, number | null> = {}
    const updatedNames: Record<string, string> = {}

    profileData.yearsActive.forEach((year: number) => {
      const communityId = profileData.communities[year]
      const key = `${year}_${communityId}`
      updatedLineages[year] = lineageMapping[key] !== undefined ? lineageMapping[key] : null

      // Get community name
      const yearNames = communityNames?.[year]?.communities
      const communityInfo = yearNames?.find((c: any) => c.community_id === communityId)
      updatedNames[year] = communityInfo?.name || `Community ${communityId}`
    })

    setProfileData((prev: any) => ({
      ...prev,
      lineages: updatedLineages,
      communityNamesList: updatedNames
    }))
  }, [profileData?.username, temporalAlignments, communityNames])

  if (loading) {
    return (
      <div className="min-h-screen text-white flex items-center justify-center" style={{
        backgroundImage: 'url(/stars.png)',
        backgroundSize: 'cover',
        backgroundPosition: 'center'
      }}>
        <div className="text-white text-xl">Loading profile...</div>
      </div>
    )
  }

  if (!profileData || profileData.yearsActive.length === 0) {
    return (
      <div className="min-h-screen text-white flex items-center justify-center" style={{
        backgroundImage: 'url(/stars.png)',
        backgroundSize: 'cover',
        backgroundPosition: 'center'
      }}>
        <div className="text-center">
          <h1 className="text-4xl font-bold text-red-400 mb-4">User Not Found</h1>
          <p className="text-gray-400 mb-6">@{username} is not in the constellation network</p>
          <Link href="/network" className="text-white hover:text-white/80">
            ← Back to Constellation
          </Link>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen text-white p-8" style={{
      backgroundImage: 'url(/stars.png)',
      backgroundSize: 'cover',
      backgroundPosition: 'center'
    }}>
      <div className="max-w-4xl mx-auto">
        {/* Navigation */}
        <div className="mb-8">
          <Link href="/network" className="text-white hover:text-white/80 transition-colors">
            ← Back to Constellation
          </Link>
        </div>

        {/* Profile Header */}
        <div className="bg-black/80 backdrop-blur rounded-xl p-8 border border-white mb-8" style={{ boxShadow: '0 0 10px rgba(255,255,255,0.5), 0 0 20px rgba(255,255,255,0.3)' }}>
          <div className="flex items-start gap-6">
            {/* Profile Photo */}
            <div className="w-32 h-32 rounded-full overflow-hidden border-4 border-white flex-shrink-0 bg-gradient-to-br from-cyan-400 to-purple-400 flex items-center justify-center relative" style={{ boxShadow: '0 0 10px rgba(255,255,255,0.5), 0 0 20px rgba(255,255,255,0.3)' }}>
              {!imageError ? (
                <Image
                  src={`https://unavatar.io/x/${username}`}
                  alt={`@${username}`}
                  fill
                  className="object-cover"
                  onError={() => setImageError(true)}
                  priority
                  sizes="128px"
                />
              ) : (
                <div className="text-6xl font-bold text-white">
                  {username.charAt(0).toUpperCase()}
                </div>
              )}
            </div>

            {/* Info */}
            <div className="flex-1">
              <h1 className="text-4xl font-bold mb-2">
                <span className="text-white">@{username}</span>
                {profileData.isSuperConnector && (
                  <span className="ml-4 text-2xl px-4 py-1 bg-purple-500/30 border border-purple-400 rounded-full text-purple-300">
                    Super Connector
                  </span>
                )}
              </h1>

              <p className="text-gray-400 mb-4">
                Active from {profileData.yearsActive[0]} to {profileData.yearsActive[profileData.yearsActive.length - 1]}
              </p>

              <div className="grid grid-cols-3 gap-6">
                <div className="bg-slate-900/50 rounded-lg p-4 border border-white" style={{ boxShadow: '0 0 10px rgba(255,255,255,0.5), 0 0 20px rgba(255,255,255,0.3)' }}>
                  <div className="text-3xl font-bold text-white">{profileData.yearsActive.length}</div>
                  <div className="text-gray-400 text-sm">Years Active</div>
                </div>
                <div className="bg-slate-900/50 rounded-lg p-4 border border-white" style={{ boxShadow: '0 0 10px rgba(255,255,255,0.5), 0 0 20px rgba(255,255,255,0.3)' }}>
                  <div className="text-3xl font-bold text-white">{profileData.avgConnections}</div>
                  <div className="text-gray-400 text-sm">Avg Connections/Year</div>
                </div>
                <div className="bg-slate-900/50 rounded-lg p-4 border border-white" style={{ boxShadow: '0 0 10px rgba(255,255,255,0.5), 0 0 20px rgba(255,255,255,0.3)' }}>
                  <div className="text-3xl font-bold text-white">{profileData.avgBetweenness.toFixed(4)}</div>
                  <div className="text-gray-400 text-sm">Betweenness Score</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Super Connector Info */}
        {profileData.isSuperConnector && (
          <div className="bg-purple-900/30 backdrop-blur rounded-xl p-6 border border-purple-400/30 mb-8">
            <h2 className="text-2xl font-bold text-purple-300 mb-3">Super Connector Status</h2>
            <p className="text-gray-300 mb-4">
              This user has high <strong>betweenness centrality</strong> (score: {profileData.avgBetweenness.toFixed(4)}),
              meaning they frequently appear on the shortest paths between other people in the network.
            </p>
            <div className="text-sm text-gray-400">
              <strong>What this means:</strong> Super connectors act as bridges between different communities.
              They help information flow across the network and connect people who might not otherwise interact.
              This makes them crucial nodes for network cohesion and cross-community communication.
            </div>
            <div className="mt-3 text-xs text-gray-500">
              <strong>Technical note:</strong> Betweenness centrality measures how often a node lies on the shortest
              path between pairs of other nodes. Higher scores indicate greater importance as a network bridge.
            </div>
          </div>
        )}

        {/* Community Evolution */}
        <div className="bg-black/80 backdrop-blur rounded-xl p-6 border border-white mb-8" style={{ boxShadow: '0 0 10px rgba(255,255,255,0.5), 0 0 20px rgba(255,255,255,0.3)' }}>
          <h2 className="text-2xl font-bold text-white mb-4">Community Evolution</h2>
          <p className="text-gray-400 mb-4">How this user's community membership changed over time:</p>

          <div className="space-y-3">
            {profileData.yearsActive.map((year: string) => {
              const lineageId = profileData.lineages?.[year]
              const communityName = profileData.communityNamesList?.[year]
              const communityId = profileData.communities[year]

              return (
                <div key={year} className="flex items-center gap-4 bg-slate-900/50 rounded-lg p-4 border border-white" style={{ boxShadow: '0 0 5px rgba(255,255,255,0.3)' }}>
                  <div className="text-2xl font-bold text-white w-20">{year}</div>
                  <div className="flex-1">
                    {lineageId !== null && lineageId !== undefined ? (
                      <>
                        <div className="text-white font-semibold">Lineage {lineageId}</div>
                        <div className="text-gray-400 text-sm">Community {communityId} - {communityName}</div>
                      </>
                    ) : (
                      <div className="text-white">{communityName || `Community ${communityId}`}</div>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {/* Tweet Topics - Coming Soon */}
        <div className="bg-black/80 backdrop-blur rounded-xl p-6 border border-white" style={{ boxShadow: '0 0 10px rgba(255,255,255,0.5), 0 0 20px rgba(255,255,255,0.3)' }}>
          <h2 className="text-2xl font-bold text-white mb-4">Tweet Topics</h2>
          <div className="text-center py-8 text-gray-400">
            <div className="text-4xl mb-3">🚀</div>
            <p>Coming soon: Topics this user discusses, clustered by semantic similarity</p>
          </div>
        </div>
      </div>
    </div>
  )
}
