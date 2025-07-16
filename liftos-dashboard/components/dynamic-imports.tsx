import React, { Suspense } from 'react';
import { createDynamicImport } from '../lib/performance-utils';

// Loading fallback component
const LoadingSpinner = () => (
  <div className="flex items-center justify-center p-8">
    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
    <span className="ml-2 text-gray-600">Loading...</span>
  </div>
);

// Chart loading fallback
const ChartLoader = () => (
  <div className="flex items-center justify-center h-64 bg-gray-50 rounded-lg">
    <div className="text-center">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-2"></div>
      <p className="text-gray-600">Loading chart...</p>
    </div>
  </div>
);

// Dynamic imports for heavy components
export const DynamicPlotlyChart = createDynamicImport(
  () => import('./charts/plotly-chart'),
  ChartLoader
);

export const DynamicD3Visualization = createDynamicImport(
  () => import('./charts/d3-visualization'),
  ChartLoader
);

export const DynamicRechartsComponent = createDynamicImport(
  () => import('./charts/recharts-component'),
  ChartLoader
);

export const DynamicAttributionTruthDashboard = createDynamicImport(
  () => import('./dashboards/attribution-truth-dashboard'),
  LoadingSpinner
);

export const DynamicPerformanceMonitorDashboard = createDynamicImport(
  () => import('./dashboards/performance-monitor-dashboard'),
  LoadingSpinner
);

export const DynamicSystemHealthDashboard = createDynamicImport(
  () => import('./dashboards/system-health-dashboard'),
  LoadingSpinner
);

// Wrapper component for dynamic imports with error boundary
interface DynamicComponentProps {
  component: React.ComponentType;
  fallback?: React.ComponentType;
  props?: any;
}

export const DynamicComponentWrapper: React.FC<DynamicComponentProps> = ({
  component: Component,
  fallback: Fallback = LoadingSpinner,
  props = {}
}) => {
  return (
    <Suspense fallback={<Fallback />}>
      <Component {...props} />
    </Suspense>
  );
};

// Error boundary for dynamic components
class DynamicComponentErrorBoundary extends React.Component<
  { children: React.ReactNode; fallback?: React.ComponentType },
  { hasError: boolean }
> {
  constructor(props: { children: React.ReactNode; fallback?: React.ComponentType }) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('Dynamic component error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      const Fallback = this.props.fallback || (() => (
        <div className="flex items-center justify-center p-8 bg-red-50 rounded-lg">
          <div className="text-center">
            <div className="text-red-600 mb-2">⚠️</div>
            <p className="text-red-600">Error loading component</p>
            <button 
              onClick={() => this.setState({ hasError: false })}
              className="mt-2 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
            >
              Retry
            </button>
          </div>
        </div>
      ));
      
      return <Fallback />;
    }

    return this.props.children;
  }
}

// HOC for wrapping dynamic components with error boundary
export function withDynamicImport<P extends object>(
  Component: React.ComponentType<P>,
  fallback?: React.ComponentType
) {
  return function DynamicComponent(props: P) {
    return (
      <DynamicComponentErrorBoundary fallback={fallback}>
        <Suspense fallback={<LoadingSpinner />}>
          <Component {...props} />
        </Suspense>
      </DynamicComponentErrorBoundary>
    );
  };
}

// Preload function for critical components
export function preloadComponent(importFn: () => Promise<any>) {
  if (typeof window !== 'undefined') {
    // Preload on idle
    if ('requestIdleCallback' in window) {
      requestIdleCallback(() => importFn());
    } else {
      setTimeout(() => importFn(), 1);
    }
  }
}

// Preload critical components
export function preloadCriticalComponents() {
  preloadComponent(() => import('./dashboards/attribution-truth-dashboard'));
  preloadComponent(() => import('./dashboards/performance-monitor-dashboard'));
  preloadComponent(() => import('./dashboards/system-health-dashboard'));
}