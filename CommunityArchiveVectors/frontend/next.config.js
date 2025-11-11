/** @type {import('next').NextConfig} */
const nextConfig = {
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  },
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'unavatar.io',
        pathname: '/x/**',
      },
    ],
    // Cache optimized images for 7 days
    minimumCacheTTL: 604800,
  },
}

module.exports = nextConfig
