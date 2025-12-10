'use client'

import { useState } from 'react'

interface ProfileAvatarProps {
  username?: string
  displayName?: string
  imageUrl?: string | null
  size?: 'sm' | 'md' | 'lg'
  className?: string
}

// Generate initials from display name or username
function getInitials(displayName?: string, username?: string): string {
  // Try display name first
  if (displayName && displayName.trim()) {
    const words = displayName.trim().split(/\s+/)
    if (words.length >= 2) {
      return (words[0][0] + words[1][0]).toUpperCase()
    }
    return displayName.slice(0, 2).toUpperCase()
  }

  // Fall back to username
  if (username && username.trim()) {
    return username.slice(0, 2).toUpperCase()
  }

  return '??'
}

// Generate a consistent color based on the name
function getAvatarColor(name?: string): string {
  const colors = [
    'bg-purple-500',
    'bg-blue-500',
    'bg-green-500',
    'bg-yellow-500',
    'bg-red-500',
    'bg-pink-500',
    'bg-indigo-500',
    'bg-teal-500',
    'bg-orange-500',
    'bg-cyan-500',
  ]

  if (!name) return colors[0]

  // Simple hash to pick a color
  let hash = 0
  for (let i = 0; i < name.length; i++) {
    hash = name.charCodeAt(i) + ((hash << 5) - hash)
  }

  return colors[Math.abs(hash) % colors.length]
}

const sizeClasses = {
  sm: 'w-8 h-8 text-xs',
  md: 'w-10 h-10 text-sm',
  lg: 'w-12 h-12 text-base',
}

export default function ProfileAvatar({
  username,
  displayName,
  imageUrl,
  size = 'md',
  className = ''
}: ProfileAvatarProps) {
  const [imageError, setImageError] = useState(false)

  const initials = getInitials(displayName, username)
  const bgColor = getAvatarColor(displayName || username)
  const sizeClass = sizeClasses[size]

  // Determine if we should show the image or fallback
  const hasValidImage = imageUrl && !imageError

  // Build fallback URL using unavatar.io if we have a username
  const fallbackUrl = username ? `https://unavatar.io/x/${username}` : null
  const finalImageUrl = imageUrl || fallbackUrl

  if (hasValidImage || (finalImageUrl && !imageError)) {
    return (
      <img
        src={finalImageUrl!}
        alt={`@${username || 'user'}`}
        className={`${sizeClass} rounded-full object-cover ${className}`}
        onError={() => setImageError(true)}
      />
    )
  }

  // Show initials fallback
  return (
    <div
      className={`${sizeClass} ${bgColor} rounded-full flex items-center justify-center text-white font-bold ${className}`}
    >
      {initials}
    </div>
  )
}
