'use client'

import { ReactNode } from 'react'
import { motion } from 'framer-motion'

interface MetricCardProps {
  title: string
  value: string | number
  target?: string | number
  confidence?: number
  trend?: 'up' | 'down' | 'stable'
  isLive?: boolean
  status?: 'success' | 'warning' | 'error' | 'info'
  children?: ReactNode
  onClick?: () => void
}

export function MetricCard({
  title,
  value,
  target,
  confidence,
  trend,
  isLive = false,
  status = 'info',
  children,
  onClick
}: MetricCardProps) {
  const getStatusColor = () => {
    switch (status) {
      case 'success': return 'border-confidence-green bg-green-50'
      case 'warning': return 'border-warning-orange bg-orange-50'
      case 'error': return 'border-alert-red bg-red-50'
      default: return 'border-gray-200 bg-white'
    }
  }

  const getTrendIcon = () => {
    switch (trend) {
      case 'up': return '↗️'
      case 'down': return '↘️'
      case 'stable': return '→'
      default: return null
    }
  }

  const getConfidenceColor = () => {
    if (!confidence) return 'text-gray-500'
    if (confidence >= 0.9) return 'text-confidence-green'
    if (confidence >= 0.7) return 'text-warning-orange'
    return 'text-alert-red'
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={`
        metric-card ${getStatusColor()} 
        ${onClick ? 'cursor-pointer hover:shadow-lg' : ''}
        relative overflow-hidden
      `}
      onClick={onClick}
    >
      {/* Live indicator */}
      {isLive && (
        <div className="absolute top-3 right-3">
          <div className="live-indicator">
            LIVE
          </div>
        </div>
      )}

      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <h3 className="metric-label">{title}</h3>
        {trend && (
          <span className="text-lg">{getTrendIcon()}</span>
        )}
      </div>

      {/* Main value */}
      <div className="mb-2">
        <div className="metric-value">{value}</div>
        {target && (
          <div className="text-sm text-gray-500">
            Target: {target}
          </div>
        )}
      </div>

      {/* Confidence indicator */}
      {confidence && (
        <div className="mb-3">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600">Confidence</span>
            <span className={getConfidenceColor()}>
              {(confidence * 100).toFixed(1)}%
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
            <div
              className={`h-2 rounded-full transition-all duration-300 ${
                confidence >= 0.9 ? 'bg-confidence-green' :
                confidence >= 0.7 ? 'bg-warning-orange' : 'bg-alert-red'
              }`}
              style={{ width: `${confidence * 100}%` }}
            />
          </div>
        </div>
      )}

      {/* Additional content */}
      {children}
    </motion.div>
  )
}

interface SkeletonMetricCardProps {
  count?: number
}

export function SkeletonMetricCard({ count = 1 }: SkeletonMetricCardProps) {
  return (
    <>
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="metric-card animate-pulse">
          <div className="skeleton-text w-24 mb-3" />
          <div className="skeleton-metric w-32 mb-2" />
          <div className="skeleton-text w-16" />
        </div>
      ))}
    </>
  )
}