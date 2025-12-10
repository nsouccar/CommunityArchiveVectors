'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import Image from 'next/image'
import { useParams, useSearchParams } from 'next/navigation'
import { lugrasimo } from '../../fonts'
import '../../retro.css'

export default function UserProfilePage() {
  const params = useParams()
  const searchParams = useSearchParams()
  const username = params.username as string
  const returnYear = searchParams.get('returnYear')
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
      <div className={`${lugrasimo.className} min-h-screen text-white flex items-center justify-center`} style={{
        backgroundImage: 'url(/stars.png)',
        backgroundSize: 'cover',
        backgroundPosition: 'center'
      }}>
        <div className="retro-text text-xl" style={{ fontFamily: 'monospace' }}>LOADING PROFILE...</div>
      </div>
    )
  }

  if (!profileData || profileData.yearsActive.length === 0) {
    return (
      <div className={`${lugrasimo.className} min-h-screen text-white flex items-center justify-center`} style={{
        backgroundImage: 'url(/stars.png)',
        backgroundSize: 'cover',
        backgroundPosition: 'center'
      }}>
        <div className="text-center">
          <h1 className="text-4xl font-bold mb-4" style={{ color: '#ff0000' }}>USER NOT FOUND</h1>
          <p className="retro-text-secondary mb-6" style={{ fontFamily: 'monospace' }}>@{username} IS NOT IN THE CONSTELLATION NETWORK</p>
          <Link href={returnYear ? `/?year=${returnYear}` : "/"} className="retro-button px-6 py-3">
            BACK TO CONSTELLATION
          </Link>
        </div>
      </div>
    )
  }

  return (
    <div className={`${lugrasimo.className} min-h-screen text-white p-8`} style={{
      backgroundImage: 'url(/stars.png)',
      backgroundSize: 'cover',
      backgroundPosition: 'center'
    }}>
      <div className="max-w-4xl mx-auto">
        {/* Navigation */}
        <div className="mb-8">
          <Link href={returnYear ? `/?year=${returnYear}` : "/"} className="retro-button px-6 py-3 inline-block">
            BACK TO CONSTELLATION
          </Link>
        </div>

        {/* Profile Header */}
        <div className="retro-box p-8 mb-8">
          <div className="flex items-start gap-6">
            {/* Profile Photo */}
            <div className="w-32 h-32 overflow-hidden border-2 border-[#6b9080] flex-shrink-0 bg-gray-700 flex items-center justify-center relative">
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
                <div className="text-6xl font-bold text-black">
                  {username.charAt(0).toUpperCase()}
                </div>
              )}
            </div>

            {/* Info */}
            <div className="flex-1">
              <h1 className="text-4xl font-bold mb-2">
                <span className="retro-text" style={{ fontFamily: 'monospace' }}>@{username}</span>
                {profileData.isSuperConnector && (
                  <span className="ml-4 text-lg px-4 py-1 bg-[#6b9080]/30 border-2 border-[#6b9080] retro-text" style={{ fontFamily: 'monospace' }}>
                    SUPER CONNECTOR
                  </span>
                )}
              </h1>

              <p className="retro-text-secondary mb-4" style={{ fontFamily: 'monospace' }}>
                ACTIVE FROM {profileData.yearsActive[0]} TO {profileData.yearsActive[profileData.yearsActive.length - 1]}
              </p>

              <div className="grid grid-cols-3 gap-6">
                <div className="retro-card p-4">
                  <div className="text-3xl font-bold retro-stat">{profileData.yearsActive.length}</div>
                  <div className="retro-text-secondary text-sm" style={{ fontFamily: 'monospace' }}>YEARS ACTIVE</div>
                </div>
                <div className="retro-card p-4">
                  <div className="text-3xl font-bold retro-stat">{profileData.avgConnections}</div>
                  <div className="retro-text-secondary text-sm" style={{ fontFamily: 'monospace' }}>AVG CONNECTIONS</div>
                </div>
                <div className="retro-card p-4">
                  <div className="text-3xl font-bold retro-stat">{profileData.avgBetweenness.toFixed(4)}</div>
                  <div className="retro-text-secondary text-sm" style={{ fontFamily: 'monospace' }}>BETWEENNESS</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Super Connector Info */}
        {profileData.isSuperConnector && (
          <div className="retro-box p-6 mb-8 bg-[#6b9080]/10">
            <h2 className="text-2xl font-bold retro-text mb-3" style={{ fontFamily: 'monospace' }}>SUPER CONNECTOR STATUS</h2>
            <p className="text-[#e8dcc8] mb-4" style={{ fontFamily: 'monospace' }}>
              THIS USER HAS HIGH <strong>BETWEENNESS CENTRALITY</strong> (SCORE: {profileData.avgBetweenness.toFixed(4)}),
              MEANING THEY FREQUENTLY APPEAR ON THE SHORTEST PATHS BETWEEN OTHER PEOPLE IN THE NETWORK.
            </p>
            <div className="text-sm retro-text-secondary" style={{ fontFamily: 'monospace' }}>
              <strong>WHAT THIS MEANS:</strong> SUPER CONNECTORS ACT AS BRIDGES BETWEEN DIFFERENT COMMUNITIES.
              THEY HELP INFORMATION FLOW ACROSS THE NETWORK AND CONNECT PEOPLE WHO MIGHT NOT OTHERWISE INTERACT.
              THIS MAKES THEM CRUCIAL NODES FOR NETWORK COHESION AND CROSS-COMMUNITY COMMUNICATION.
            </div>
            <div className="mt-3 text-xs retro-text-secondary" style={{ fontFamily: 'monospace' }}>
              <strong>TECHNICAL NOTE:</strong> BETWEENNESS CENTRALITY MEASURES HOW OFTEN A NODE LIES ON THE SHORTEST
              PATH BETWEEN PAIRS OF OTHER NODES. HIGHER SCORES INDICATE GREATER IMPORTANCE AS A NETWORK BRIDGE.
            </div>
          </div>
        )}

        {/* Community Evolution */}
        <div className="retro-box p-6 mb-8">
          <h2 className="text-2xl font-bold retro-text mb-4" style={{ fontFamily: 'monospace' }}>COMMUNITY EVOLUTION</h2>
          <p className="retro-text-secondary mb-4" style={{ fontFamily: 'monospace' }}>HOW THIS USER'S COMMUNITY MEMBERSHIP CHANGED OVER TIME:</p>

          <div className="space-y-3">
            {profileData.yearsActive.map((year: string) => {
              const lineageId = profileData.lineages?.[year]
              const communityName = profileData.communityNamesList?.[year]
              const communityId = profileData.communities[year]

              return (
                <div key={year} className="flex items-center gap-4 retro-card p-4">
                  <div className="text-2xl font-bold retro-text w-20" style={{ fontFamily: 'monospace' }}>{year}</div>
                  <div className="flex-1">
                    {lineageId !== null && lineageId !== undefined ? (
                      <>
                        <div className="retro-text font-semibold" style={{ fontFamily: 'monospace' }}>LINEAGE {lineageId}</div>
                        <div className="retro-text-secondary text-sm" style={{ fontFamily: 'monospace' }}>COMMUNITY {communityId} - {communityName.toUpperCase()}</div>
                      </>
                    ) : (
                      <div className="retro-text" style={{ fontFamily: 'monospace' }}>{communityName?.toUpperCase() || `COMMUNITY ${communityId}`}</div>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}
