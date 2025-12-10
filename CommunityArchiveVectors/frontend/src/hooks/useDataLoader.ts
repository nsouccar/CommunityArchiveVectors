'use client'

import { useState, useEffect } from 'react'

const DB_NAME = 'constellation-cache'
const DB_VERSION = 5 // Force cache refresh with local avatar images
const STORE_NAME = 'data-files'

// Files to load and cache
const DATA_FILES = [
  { key: 'networkData', url: '/network_animation_data.json', weight: 15 },
  { key: 'avatarUrls', url: '/avatar_urls.json', weight: 30 },
  { key: 'communityNames', url: '/data/all_community_names.json', weight: 20 },
  { key: 'communityTopics', url: '/community_topics.json', weight: 1 },
  { key: 'temporalAlignments', url: '/community_temporal_alignments.json', weight: 1 },
  { key: 'topics_2012', url: '/data/topics_year_2012_summary.json', weight: 1 },
  { key: 'topics_2018', url: '/data/topics_year_2018_summary.json', weight: 2 },
  { key: 'topics_2019', url: '/data/topics_year_2019_summary.json', weight: 3 },
  { key: 'topics_2020', url: '/data/topics_year_2020_summary.json', weight: 4 },
  { key: 'topics_2021', url: '/data/topics_year_2021_summary.json', weight: 4 },
  { key: 'topics_2022', url: '/data/topics_year_2022_summary.json', weight: 5 },
  { key: 'topics_2023', url: '/data/topics_year_2023_summary.json', weight: 6 },
  { key: 'topics_2024', url: '/data/topics_year_2024_summary.json', weight: 1 },
  { key: 'topics_2025', url: '/data/topics_year_2025_summary.json', weight: 5 },
]

const TOTAL_WEIGHT = DATA_FILES.reduce((sum, f) => sum + f.weight, 0)

export interface LoadedData {
  networkData: any
  avatarUrls: Record<string, string>
  communityNames: any
  communityTopics: any
  temporalAlignments: any
  allTopics: Record<string, any>
}

// Open IndexedDB
function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION)

    request.onerror = () => {
      console.error('[DataLoader] IndexedDB error:', request.error)
      reject(request.error)
    }
    request.onsuccess = () => {
      console.log('[DataLoader] IndexedDB opened successfully')
      resolve(request.result)
    }

    // Handle blocked - happens if another tab has old version open
    request.onblocked = () => {
      console.warn('[DataLoader] IndexedDB blocked - close other tabs and refresh')
    }

    request.onupgradeneeded = (event) => {
      console.log('[DataLoader] IndexedDB upgrade needed - clearing old cache')
      const db = (event.target as IDBOpenDBRequest).result

      // Delete old store if it exists (clears stale cache)
      if (db.objectStoreNames.contains(STORE_NAME)) {
        db.deleteObjectStore(STORE_NAME)
      }

      // Create fresh store
      db.createObjectStore(STORE_NAME)
    }
  })
}

// Get item from IndexedDB
async function getFromDB(db: IDBDatabase, key: string): Promise<any> {
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, 'readonly')
    const store = transaction.objectStore(STORE_NAME)
    const request = store.get(key)

    request.onerror = () => reject(request.error)
    request.onsuccess = () => resolve(request.result)
  })
}

// Save item to IndexedDB
async function saveToDB(db: IDBDatabase, key: string, value: any): Promise<void> {
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE_NAME, 'readwrite')
    const store = transaction.objectStore(STORE_NAME)
    const request = store.put(value, key)

    request.onerror = () => reject(request.error)
    request.onsuccess = () => resolve()
  })
}

// Check if cache is valid (has all required keys)
async function isCacheValid(db: IDBDatabase): Promise<boolean> {
  try {
    for (const file of DATA_FILES) {
      const data = await getFromDB(db, file.key)
      if (!data) return false
    }
    return true
  } catch {
    return false
  }
}

export function useDataLoader() {
  const [isLoading, setIsLoading] = useState(true)
  const [progress, setProgress] = useState(0)
  const [data, setData] = useState<LoadedData | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let isMounted = true

    async function loadData() {
      try {
        const db = await openDB()

        // Check if we have valid cached data
        const cacheValid = await isCacheValid(db)

        let cachedOrFetchedData: any = {}
        let allTopics: Record<string, any> = {}

        if (cacheValid) {
          console.log('[DataLoader] Loading from cache...')

          // Load all from cache
          let loadedWeight = 0

          for (const file of DATA_FILES) {
            cachedOrFetchedData[file.key] = await getFromDB(db, file.key)
            loadedWeight += file.weight
            if (isMounted) {
              setProgress(Math.round((loadedWeight / TOTAL_WEIGHT) * 100))
            }
          }

          // Build the data structure
          for (const file of DATA_FILES) {
            if (file.key.startsWith('topics_')) {
              const year = file.key.replace('topics_', '')
              allTopics[year] = cachedOrFetchedData[file.key]
            }
          }
        } else {
          console.log('[DataLoader] Fetching from network...')

          // Fetch all files from network
          let loadedWeight = 0

          for (const file of DATA_FILES) {
            console.log(`[DataLoader] Fetching ${file.url}...`)
            const response = await fetch(file.url)
            if (!response.ok) {
              throw new Error(`Failed to fetch ${file.url}`)
            }
            const json = await response.json()
            cachedOrFetchedData[file.key] = json

            // Save to IndexedDB
            await saveToDB(db, file.key, json)

            loadedWeight += file.weight
            if (isMounted) {
              setProgress(Math.round((loadedWeight / TOTAL_WEIGHT) * 100))
            }
          }

          // Build the data structure
          for (const file of DATA_FILES) {
            if (file.key.startsWith('topics_')) {
              const year = file.key.replace('topics_', '')
              allTopics[year] = cachedOrFetchedData[file.key]
            }
          }
        }

        // Tweets are now embedded in the topic JSON files - no API calls needed!
        // Profile images are pre-downloaded to /avatars/ directory
        console.log('[DataLoader] Data loaded - tweets embedded, images pre-downloaded')

        if (isMounted) {
          setProgress(100)
          setData({
            networkData: cachedOrFetchedData.networkData,
            avatarUrls: cachedOrFetchedData.avatarUrls,
            communityNames: cachedOrFetchedData.communityNames,
            communityTopics: cachedOrFetchedData.communityTopics,
            temporalAlignments: cachedOrFetchedData.temporalAlignments,
            allTopics,
          })
          setIsLoading(false)
        }
      } catch (err) {
        console.error('[DataLoader] Error:', err)
        if (isMounted) {
          setError(err instanceof Error ? err.message : 'Failed to load data')
          setIsLoading(false)
        }
      }
    }

    loadData()

    return () => {
      isMounted = false
    }
  }, [])

  return { isLoading, progress, data, error }
}
