/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  poweredByHeader: false,
  compress: true,
  
  // Performance optimizations
  experimental: {
    serverComponentsExternalPackages: ['plotly.js', 'd3'],
    optimizeCss: true,
    optimizePackageImports: ['@heroicons/react', 'recharts', 'lucide-react'],
  },
  
  // Environment variables
  env: {
    LIFTOS_API_URL: process.env.LIFTOS_API_URL || 'http://localhost:8000',
    LIFTOS_WS_URL: process.env.LIFTOS_WS_URL || 'ws://localhost:8000',
  },
  
  // Improved output configuration
  output: 'standalone',
  
  // Bundle analyzer configuration
  webpack: (config, { dev, isServer }) => {
    // Bundle analyzer
    if (process.env.ANALYZE === 'true') {
      const { BundleAnalyzerPlugin } = require('webpack-bundle-analyzer');
      config.plugins.push(
        new BundleAnalyzerPlugin({
          analyzerMode: 'static',
          openAnalyzer: false,
          reportFilename: isServer ? '../analyze/server.html' : '../analyze/client.html',
        })
      );
    }

    // Optimize heavy libraries
    if (!dev && !isServer) {
      config.optimization.splitChunks = {
        chunks: 'all',
        cacheGroups: {
          plotly: {
            test: /[\\/]node_modules[\\/]plotly\.js[\\/]/,
            name: 'plotly',
            priority: 20,
            reuseExistingChunk: true,
          },
          d3: {
            test: /[\\/]node_modules[\\/]d3[\\/]/,
            name: 'd3',
            priority: 15,
            reuseExistingChunk: true,
          },
          vendor: {
            test: /[\\/]node_modules[\\/]/,
            name: 'vendors',
            priority: 10,
            reuseExistingChunk: true,
          },
          common: {
            name: 'common',
            minChunks: 2,
            priority: 5,
            reuseExistingChunk: true,
          },
        },
      };
    }

    return config;
  },
  
  // Rewrites with proper environment variable handling
  async rewrites() {
    const apiUrl = process.env.LIFTOS_API_URL || 'http://localhost:8000';
    return [
      {
        source: '/api/:path*',
        destination: `${apiUrl}/api/:path*`,
      },
    ];
  },
  
  // Security and performance headers
  async headers() {
    return [
      {
        source: '/:path*',
        headers: [
          {
            key: 'X-DNS-Prefetch-Control',
            value: 'on'
          },
          {
            key: 'X-XSS-Protection',
            value: '1; mode=block'
          },
          {
            key: 'X-Frame-Options',
            value: 'SAMEORIGIN'
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff'
          },
          {
            key: 'Referrer-Policy',
            value: 'origin-when-cross-origin'
          },
        ],
      },
    ];
  },
  
  // Image optimization
  images: {
    formats: ['image/avif', 'image/webp'],
    deviceSizes: [640, 750, 828, 1080, 1200, 1920, 2048, 3840],
    imageSizes: [16, 32, 48, 64, 96, 128, 256, 384],
  },
};

module.exports = nextConfig;