'use client'

import { useState, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { MetricCard } from '../ui/metric-card'
import { apiClient, PerformanceMetrics } from '../../lib/api-client'

export function PerformanceMonitorDashboard() {
  const [realTimeMetrics, setRealTimeMetrics] = useState<PerformanceMetrics | null>(null)

  // Fetch performance metrics
  const { data: metrics, isLoading, error } = useQuery({
    queryKey: ['performance', 'metrics'],
    queryFn: () => apiClient.getPerformanceMetrics(),
    refetchInterval: 1000, // Refresh every second for real-time monitoring
  })

  // Real-time WebSocket updates
  useEffect(() => {
    const handleWebSocketMessage = (data: any) => {
      if (data.type === 'performance_update') {
        setRealTimeMetrics(data.metrics)
      }
    }

    apiClient.connectWebSocket(handleWebSocketMessage)

    return () => {
      apiClient.disconnectWebSocket()
    }
  }, [])

  // Use real-time metrics if available, otherwise fall back to query data
  const currentMetrics = realTimeMetrics || metrics

  if (isLoading && !currentMetrics) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="skeleton h-32 rounded-lg" />
          <div className="skeleton h-32 rounded-lg" />
          <div className="skeleton h-32 rounded-lg" />
          <div className="skeleton h-32 rounded-lg" />
        </div>
        <div className="skeleton h-64 rounded-lg" />
      </div>
    )
  }

  if (error && !currentMetrics) {
    return (
      <div className="alert-critical">
        <h3 className="text-lg font-semibold text-red-900 mb-2">
          Performance Monitoring Error
        </h3>
        <p className="text-red-700">
          Failed to load performance metrics. Please check your connection and try again.
        </p>
      </div>
    )
  }

  if (!currentMetrics) {
    return (
      <div className="alert-warning">
        <h3 className="text-lg font-semibold text-orange-900 mb-2">
          No Performance Data
        </h3>
        <p className="text-orange-700">
          No performance data available. Please ensure your services are running.
        </p>
      </div>
    )
  }

  // Performance status calculations
  const executionTimeStatus = currentMetrics.execution_time <= 0.034 ? 'success' : 'warning'
  const accuracyStatus = currentMetrics.accuracy >= 93.8 ? 'success' : 'warning'
  const speedupStatus = currentMetrics.speedup >= 241 ? 'success' : 'warning'
  const healthStatus = currentMetrics.service_health === 'healthy' ? 'success' : 
                      currentMetrics.service_health === 'degraded' ? 'warning' : 'error'

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">
              🚀 LiftOS Performance Monitor
            </h2>
            <p className="text-gray-600 mt-1">
              Real-time validation of 5 Core Policy Messages performance claims
            </p>
          </div>
          <div className="text-right">
            <div className="live-indicator">
              LIVE MONITORING
            </div>
            <div className="text-sm text-gray-500 mt-1">
              Last updated: {new Date(currentMetrics.timestamp).toLocaleTimeString()}
            </div>
          </div>
        </div>
      </div>

      {/* Core Performance Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <MetricCard
          title="Execution Time"
          value={`${currentMetrics.execution_time.toFixed(3)}s`}
          target="<0.034s"
          confidence={currentMetrics.confidence}
          status={executionTimeStatus}
          trend={currentMetrics.execution_time <= 0.034 ? 'up' : 'down'}
          isLive={true}
        >
          <div className="mt-3 text-xs text-gray-600">
            {currentMetrics.execution_time <= 0.034 ? 
              `✅ ${((0.034 - currentMetrics.execution_time) / 0.034 * 100).toFixed(1)}% faster than target` :
              `⚠️ ${((currentMetrics.execution_time - 0.034) / 0.034 * 100).toFixed(1)}% slower than target`
            }
          </div>
        </MetricCard>

        <MetricCard
          title="Accuracy"
          value={`${currentMetrics.accuracy.toFixed(1)}%`}
          target=">93.8%"
          confidence={currentMetrics.confidence}
          status={accuracyStatus}
          trend={currentMetrics.accuracy >= 93.8 ? 'up' : 'down'}
          isLive={true}
        >
          <div className="mt-3 text-xs text-gray-600">
            {currentMetrics.accuracy >= 93.8 ? 
              `✅ ${(currentMetrics.accuracy - 93.8).toFixed(1)}% above target` :
              `⚠️ ${(93.8 - currentMetrics.accuracy).toFixed(1)}% below target`
            }
          </div>
        </MetricCard>

        <MetricCard
          title="Speedup vs Legacy"
          value={`${currentMetrics.speedup.toFixed(0)}x`}
          target=">241x"
          confidence={currentMetrics.confidence}
          status={speedupStatus}
          trend={currentMetrics.speedup >= 241 ? 'up' : 'down'}
          isLive={true}
        >
          <div className="mt-3 text-xs text-gray-600">
            {currentMetrics.speedup >= 241 ? 
              `✅ ${(currentMetrics.speedup - 241).toFixed(0)}x faster than target` :
              `⚠️ ${(241 - currentMetrics.speedup).toFixed(0)}x slower than target`
            }
          </div>
        </MetricCard>

        <MetricCard
          title="System Health"
          value={currentMetrics.service_health.toUpperCase()}
          confidence={currentMetrics.confidence}
          status={healthStatus}
          isLive={true}
        >
          <div className="mt-3 text-xs text-gray-600">
            {currentMetrics.service_health === 'healthy' ? 
              '✅ All systems operational' :
              currentMetrics.service_health === 'degraded' ?
              '⚠️ Some services degraded' :
              '❌ Critical issues detected'
            }
          </div>
        </MetricCard>
      </div>

      {/* Performance Claims Validation */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          5 Core Policy Messages Validation
        </h3>
        
        <div className="space-y-4">
          {/* Policy 1: End Attribution Theatre */}
          <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
            <div>
              <div className="font-medium text-gray-900">
                Policy 1: End Attribution Theatre
              </div>
              <div className="text-sm text-gray-600">
                93.8% accuracy with full transparency
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                currentMetrics.accuracy >= 93.8 ? 
                'bg-green-100 text-green-800' : 
                'bg-orange-100 text-orange-800'
              }`}>
                {currentMetrics.accuracy.toFixed(1)}% Accuracy
              </div>
              <div className={`text-2xl ${
                currentMetrics.accuracy >= 93.8 ? 'text-green-500' : 'text-orange-500'
              }`}>
                {currentMetrics.accuracy >= 93.8 ? '✅' : '⚠️'}
              </div>
            </div>
          </div>

          {/* Policy 2: Democratize Speed and Intelligence */}
          <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
            <div>
              <div className="font-medium text-gray-900">
                Policy 2: Democratize Speed and Intelligence
              </div>
              <div className="text-sm text-gray-600">
                Real-time insights (0.034s execution, 241x faster)
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                currentMetrics.execution_time <= 0.034 && currentMetrics.speedup >= 241 ? 
                'bg-green-100 text-green-800' : 
                'bg-orange-100 text-orange-800'
              }`}>
                {currentMetrics.execution_time.toFixed(3)}s / {currentMetrics.speedup.toFixed(0)}x
              </div>
              <div className={`text-2xl ${
                currentMetrics.execution_time <= 0.034 && currentMetrics.speedup >= 241 ? 
                'text-green-500' : 'text-orange-500'
              }`}>
                {currentMetrics.execution_time <= 0.034 && currentMetrics.speedup >= 241 ? '✅' : '⚠️'}
              </div>
            </div>
          </div>

          {/* Policy 4: Complete Observability Standard */}
          <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
            <div>
              <div className="font-medium text-gray-900">
                Policy 4: Complete Observability Standard
              </div>
              <div className="text-sm text-gray-600">
                Full explainability with &lt;0.1% performance overhead
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                currentMetrics.service_health === 'healthy' ? 
                'bg-green-100 text-green-800' : 
                'bg-orange-100 text-orange-800'
              }`}>
                {currentMetrics.service_health.toUpperCase()}
              </div>
              <div className={`text-2xl ${
                currentMetrics.service_health === 'healthy' ? 'text-green-500' : 'text-orange-500'
              }`}>
                {currentMetrics.service_health === 'healthy' ? '✅' : '⚠️'}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Real-Time Performance Chart Placeholder */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Real-Time Performance Trends
        </h3>
        <div className="h-64 bg-gray-50 rounded-lg flex items-center justify-center">
          <div className="text-center text-gray-500">
            <div className="text-4xl mb-2">📊</div>
            <div>Real-time performance charts</div>
            <div className="text-sm">(Chart implementation coming in Phase 2)</div>
          </div>
        </div>
      </div>
    </div>
  )
}