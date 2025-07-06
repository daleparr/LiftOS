'use client'

import { useState, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { MetricCard } from '../ui/metric-card'
import { apiClient, AttributionData } from '../../lib/api-client'

export function AttributionTruthDashboard() {
  const [selectedChannel, setSelectedChannel] = useState<string | null>(null)

  // Fetch attribution data
  const { data: attribution, isLoading, error } = useQuery({
    queryKey: ['attribution', 'analysis'],
    queryFn: () => apiClient.getAttributionAnalysis(),
    refetchInterval: 30000, // Refresh every 30 seconds
  })

  // Real-time WebSocket updates
  useEffect(() => {
    const handleWebSocketMessage = (data: any) => {
      if (data.type === 'attribution_update') {
        // Update attribution data in real-time
        console.log('Real-time attribution update:', data)
      }
    }

    apiClient.connectWebSocket(handleWebSocketMessage)

    return () => {
      apiClient.disconnectWebSocket()
    }
  }, [])

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="skeleton h-32 rounded-lg" />
          <div className="skeleton h-32 rounded-lg" />
          <div className="skeleton h-32 rounded-lg" />
        </div>
        <div className="skeleton h-64 rounded-lg" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="alert-critical">
        <h3 className="text-lg font-semibold text-red-900 mb-2">
          Attribution Analysis Error
        </h3>
        <p className="text-red-700">
          Failed to load attribution data. Please check your connection and try again.
        </p>
      </div>
    )
  }

  if (!attribution) {
    return (
      <div className="alert-warning">
        <h3 className="text-lg font-semibold text-orange-900 mb-2">
          No Attribution Data
        </h3>
        <p className="text-orange-700">
          No attribution data available. Please ensure your data sources are connected.
        </p>
      </div>
    )
  }

  const overCreditingPercentage = attribution.over_crediting_percentage
  const totalClaimed = attribution.channels.reduce((sum, channel) => sum + channel.claimed_attribution, 0)
  const totalCausal = attribution.channels.reduce((sum, channel) => sum + channel.causal_attribution, 0)
  const wastedBudget = totalClaimed - totalCausal

  return (
    <div className="space-y-6">
      {/* Header Alert */}
      <div className="alert-critical">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-red-900 mb-2">
              🚨 Attribution Fraud Detected
            </h2>
            <p className="text-red-700">
              Your attribution system is over-crediting by{' '}
              <span className="font-bold">{overCreditingPercentage.toFixed(1)}%</span>
            </p>
          </div>
          <div className="text-right">
            <div className="text-3xl font-bold text-red-900">
              {overCreditingPercentage.toFixed(0)}%
            </div>
            <div className="text-sm text-red-700">Over-crediting</div>
          </div>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <MetricCard
          title="Total Claimed Attribution"
          value={`$${totalClaimed.toLocaleString()}`}
          status="error"
          isLive={true}
        />
        <MetricCard
          title="Actual Causal Impact"
          value={`$${totalCausal.toLocaleString()}`}
          status="success"
          isLive={true}
        />
        <MetricCard
          title="Wasted Budget"
          value={`$${wastedBudget.toLocaleString()}`}
          status="error"
          isLive={true}
        />
      </div>

      {/* Channel Attribution Table */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900">
            Causal Truth Table
          </h3>
          <p className="text-sm text-gray-600">
            Real incremental impact vs claimed attribution
          </p>
        </div>
        
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Channel
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Claimed Attribution
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Causal Truth
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Difference
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Confidence
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {attribution.channels.map((channel, index) => {
                const difference = channel.claimed_attribution - channel.causal_attribution
                const differencePercent = (difference / channel.claimed_attribution) * 100
                
                return (
                  <tr
                    key={channel.name}
                    className={`hover:bg-gray-50 cursor-pointer ${
                      selectedChannel === channel.name ? 'bg-blue-50' : ''
                    }`}
                    onClick={() => setSelectedChannel(channel.name)}
                  >
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm font-medium text-gray-900">
                        {channel.name}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm text-gray-900">
                        ${channel.claimed_attribution.toLocaleString()}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm text-gray-900">
                        ${channel.causal_attribution.toLocaleString()}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className={`text-sm font-medium ${
                        difference > 0 ? 'text-red-600' : 'text-green-600'
                      }`}>
                        {difference > 0 ? '-' : '+'}${Math.abs(difference).toLocaleString()}
                        <div className="text-xs text-gray-500">
                          ({differencePercent > 0 ? '-' : '+'}{Math.abs(differencePercent).toFixed(1)}%)
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <div className={`text-sm font-medium ${
                          channel.confidence_score >= 0.9 ? 'text-green-600' :
                          channel.confidence_score >= 0.7 ? 'text-orange-600' : 'text-red-600'
                        }`}>
                          {(channel.confidence_score * 100).toFixed(1)}%
                        </div>
                        <div className="ml-2 w-16 bg-gray-200 rounded-full h-2">
                          <div
                            className={`h-2 rounded-full ${
                              channel.confidence_score >= 0.9 ? 'bg-green-500' :
                              channel.confidence_score >= 0.7 ? 'bg-orange-500' : 'bg-red-500'
                            }`}
                            style={{ width: `${channel.confidence_score * 100}%` }}
                          />
                        </div>
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Budget Reallocation Recommendations */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Recommended Budget Reallocation
        </h3>
        
        <div className="space-y-4">
          {Object.entries(attribution.recommended_reallocation).map(([channel, change]) => (
            <div key={channel} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <div>
                <div className="font-medium text-gray-900">{channel}</div>
                <div className="text-sm text-gray-600">
                  {change > 0 ? 'Increase' : 'Decrease'} budget allocation
                </div>
              </div>
              <div className={`text-lg font-bold ${
                change > 0 ? 'text-green-600' : 'text-red-600'
              }`}>
                {change > 0 ? '+' : ''}${change.toLocaleString()}
              </div>
            </div>
          ))}
        </div>

        <button className="w-full mt-4 bg-lift-blue text-white py-3 px-4 rounded-lg font-medium hover:bg-blue-700 transition-colors">
          Apply Recommended Reallocation
        </button>
      </div>
    </div>
  )
}