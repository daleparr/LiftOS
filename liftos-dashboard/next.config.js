/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    appDir: true,
  },
  env: {
    LIFTOS_API_URL: process.env.LIFTOS_API_URL || 'http://localhost:8000',
    LIFTOS_WS_URL: process.env.LIFTOS_WS_URL || 'ws://localhost:8000',
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${process.env.LIFTOS_API_URL || 'http://localhost:8000'}/api/:path*`,
      },
    ]
  },
}

module.exports = nextConfig