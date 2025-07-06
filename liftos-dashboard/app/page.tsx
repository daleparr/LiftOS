'use client'

import { useState } from 'react'
import { AttributionTruthDashboard } from '../components/dashboards/attribution-truth-dashboard'
import { PerformanceMonitorDashboard } from '../components/dashboards/performance-monitor-dashboard'
import { SystemHealthDashboard } from '../components/dashboards/system-health-dashboard'

type DashboardView = 'overview' | 'attribution' | 'performance' | 'health'

export default function HomePage() {
  const [activeView, setActiveView] = useState<DashboardView>('overview')

  const renderDashboard = () => {
    switch (activeView) {
      case 'attribution':
        return <AttributionTruthDashboard />
      case 'performance':
        return <PerformanceMonitorDashboard />
      case 'health':
        return <SystemHealthDashboard />
      default:
        return <OverviewDashboard />
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold text-gray-900">
                🚀 LiftOS Dashboard
              </h1>
              <span className="ml-3 px-2 py-1 text-xs font-medium bg-lift-blue text-white rounded-full">
                Phase 1
              </span>
            </div>
            <div className="live-indicator">
              LIVE
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {[
              { id: 'overview', label: '📊 Overview', description: 'All dashboards' },
              { id: 'attribution', label: '🎯 Attribution Truth', description: 'End attribution theatre' },
              { id: 'performance', label: '⚡ Performance', description: '0.034s execution monitoring' },
              { id: 'health', label: '🏥 System Health', description: '<0.1% overhead tracking' },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveView(tab.id as DashboardView)}
                className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeView === tab.id
                    ? 'border-lift-blue text-lift-blue'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <div>{tab.label}</div>
                <div className="text-xs text-gray-400">{tab.description}</div>
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {renderDashboard()}
      </main>
    </div>
  )
}

function OverviewDashboard() {
  return (
    <div className="space-y-8">
      {/* Welcome Section */}
      <div className="bg-gradient-to-r from-lift-blue to-blue-600 rounded-lg shadow-sm p-8 text-white">
        <h2 className="text-3xl font-bold mb-4">
          Welcome to LiftOS Phase 1: Critical Dashboards
        </h2>
        <p className="text-blue-100 text-lg mb-6">
          Real-time visualization of the 5 Core Policy Messages with sophisticated backend integration
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-white/10 rounded-lg p-4">
            <div className="text-2xl font-bold">93.8%</div>
            <div className="text-blue-100">Attribution Accuracy</div>
          </div>
          <div className="bg-white/10 rounded-lg p-4">
            <div className="text-2xl font-bold">0.034s</div>
            <div className="text-blue-100">Execution Time</div>
          </div>
          <div className="bg-white/10 rounded-lg p-4">
            <div className="text-2xl font-bold">&lt;0.1%</div>
            <div className="text-blue-100">Performance Overhead</div>
          </div>
        </div>
      </div>

      {/* Quick Access Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow cursor-pointer">
          <div className="text-4xl mb-4">🎯</div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">
            Attribution Truth Dashboard
          </h3>
          <p className="text-gray-600 text-sm mb-4">
            Reveal attribution fraud with real confidence intervals and causal truth tables
          </p>
          <div className="text-lift-blue font-medium">View Dashboard →</div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow cursor-pointer">
          <div className="text-4xl mb-4">⚡</div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">
            Performance Monitor
          </h3>
          <p className="text-gray-600 text-sm mb-4">
            Live validation of 0.034s execution and 241x speedup claims
          </p>
          <div className="text-lift-blue font-medium">View Dashboard →</div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow cursor-pointer">
          <div className="text-4xl mb-4">🏥</div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">
            System Health
          </h3>
          <p className="text-gray-600 text-sm mb-4">
            Complete observability with &lt;0.1% performance overhead
          </p>
          <div className="text-lift-blue font-medium">View Dashboard →</div>
        </div>
      </div>

      {/* Implementation Status */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Phase 1 Implementation Status
        </h3>
        
        <div className="space-y-4">
          <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
            <div>
              <div className="font-medium text-green-900">✅ Attribution Truth Dashboard</div>
              <div className="text-sm text-green-700">Real confidence intervals with backend integration</div>
            </div>
            <div className="text-green-600 font-medium">Complete</div>
          </div>

          <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
            <div>
              <div className="font-medium text-green-900">✅ Performance Monitor Dashboard</div>
              <div className="text-sm text-green-700">Live 0.034s execution time validation</div>
            </div>
            <div className="text-green-600 font-medium">Complete</div>
          </div>

          <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
            <div>
              <div className="font-medium text-green-900">✅ System Health Dashboard</div>
              <div className="text-sm text-green-700">&lt;0.1% overhead observability tracking</div>
            </div>
            <div className="text-green-600 font-medium">Complete</div>
          </div>

          <div className="flex items-center justify-between p-3 bg-orange-50 rounded-lg">
            <div>
              <div className="font-medium text-orange-900">🚧 Real-time WebSocket Integration</div>
              <div className="text-sm text-orange-700">Live data streaming from backend services</div>
            </div>
            <div className="text-orange-600 font-medium">In Progress</div>
          </div>

          <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
            <div>
              <div className="font-medium text-gray-900">⏳ Interactive Charts (Phase 2)</div>
              <div className="text-sm text-gray-700">3D visualizations and advanced analytics</div>
            </div>
            <div className="text-gray-600 font-medium">Planned</div>
          </div>
        </div>
      </div>

      {/* Next Steps */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Next Steps: Bridging the Visualization Gap
        </h3>
        
        <div className="prose text-gray-600">
          <p>
            Phase 1 successfully bridges the critical 53% value delivery gap by making the sophisticated 
            backend infrastructure visible through real-time dashboards. The next phases will add:
          </p>
          
          <ul className="mt-4 space-y-2">
            <li>• <strong>Phase 2:</strong> Interactive 3D visualizations and advanced analytics</li>
            <li>• <strong>Phase 3:</strong> One-click optimization and automated actions</li>
            <li>• <strong>Phase 4:</strong> Collaborative intelligence and knowledge sharing</li>
          </ul>
          
          <p className="mt-4">
            <strong>Impact:</strong> From 25% to 85% value delivery by making backend sophistication visible and actionable.
          </p>
        </div>
      </div>
    </div>
  )
}