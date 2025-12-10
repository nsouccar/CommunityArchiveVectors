'use client'

import { lugrasimo } from '../app/fonts'

interface LoadingScreenProps {
  progress: number // 0-100
}

export default function LoadingScreen({ progress }: LoadingScreenProps) {
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
      <h1
        className={`text-4xl font-bold mb-2 ${lugrasimo.className}`}
        style={{ color: '#ff66ff' }}
      >
        Constellation of People
      </h1>

      <p
        className={`text-sm mb-12 opacity-80 ${lugrasimo.className}`}
      >
        a visualization of twitter communities over time
      </p>

      <div className="w-64">
        <p
          className="text-sm mb-3 text-center"
          style={{ fontFamily: 'monospace' }}
        >
          Loading
        </p>

        {/* Progress bar */}
        <div
          className="h-2 rounded-full overflow-hidden"
          style={{ backgroundColor: 'rgba(255, 255, 255, 0.2)' }}
        >
          <div
            className="h-full rounded-full transition-all duration-300 ease-out"
            style={{
              width: `${progress}%`,
              backgroundColor: '#ff66ff'
            }}
          />
        </div>
      </div>
    </div>
  )
}
