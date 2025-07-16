# Performance Tools Reference Guide

## 🚀 **Quick Start Commands**

### Bundle Analysis
```bash
# Analyze UI Shell bundle
cd ui-shell && npm run analyze

# Analyze Dashboard bundle
cd liftos-dashboard && npm run analyze

# View analysis results
open analyze/client.html
open analyze/server.html
```

### Security Audit
```bash
# Check for vulnerabilities
npm audit

# Fix vulnerabilities (if any)
npm audit fix --force
```

### Performance Testing
```bash
# Run optimized builds
npm run build

# Start production server
npm run start

# Run with performance monitoring
NODE_ENV=production npm run start
```

## 📊 **Performance Monitoring**

### Using Performance Hooks
```typescript
import { usePerformanceMonitor, useDebounce, useThrottle } from './lib/performance-utils';

// Monitor component performance
function MyComponent() {
  const { renderCount, renderTime } = usePerformanceMonitor('MyComponent');
  
  // Debounce search input
  const debouncedSearch = useDebounce(searchTerm, 300);
  
  // Throttle scroll events
  const throttledScroll = useThrottle(scrollY, 100);
  
  return <div>Component content</div>;
}
```

### API Client with Caching
```typescript
import { apiClient } from './lib/api-client';

// Cached API calls (5-minute cache)
const data = await apiClient.get('/api/data');

// Disable cache for specific calls
const freshData = await apiClient.get('/api/data', { cache: false });

// Clear cache
apiClient.clearCache();
```

### Query Client Optimization
```typescript
import { queryClient, queryKeys, prefetchQueries } from './lib/query-client';

// Prefetch critical data
await prefetchQueries.prefetchDashboard('dashboard-id');

// Invalidate specific queries
queryClient.invalidateQueries({ queryKey: queryKeys.users.all });

// Monitor query performance
const metrics = performanceMonitor.getMetrics();
```

## 🎯 **Dynamic Imports**

### Using Dynamic Components
```typescript
import { DynamicPlotlyChart, DynamicD3Visualization } from './components/dynamic-imports';

// Lazy-loaded components
function Dashboard() {
  return (
    <div>
      <DynamicPlotlyChart.Component data={chartData} />
      <DynamicD3Visualization.Component config={d3Config} />
    </div>
  );
}
```

### Creating Custom Dynamic Imports
```typescript
import { createDynamicImport, withDynamicImport } from './lib/performance-utils';

// Create dynamic import
const DynamicComponent = createDynamicImport(
  () => import('./HeavyComponent'),
  LoadingSpinner
);

// Use as HOC
const OptimizedComponent = withDynamicImport(HeavyComponent);
```

## 📈 **Performance Metrics**

### Memory Monitoring
```typescript
import { useMemoryMonitor } from './lib/performance-utils';

function App() {
  const memoryInfo = useMemoryMonitor();
  
  if (memoryInfo) {
    console.log('Memory usage:', memoryInfo.usedJSHeapSize / 1024 / 1024, 'MB');
  }
}
```

### Bundle Size Tracking
```typescript
import { trackBundleSize } from './lib/performance-utils';

// Track bundle performance
trackBundleSize(); // Logs performance metrics to console
```

## 🐳 **Docker Optimizations**

### Build Optimized Images
```bash
# Build UI Shell
docker build -f ui-shell/Dockerfile.optimized -t ui-shell:optimized .

# Build Dashboard
docker build -f liftos-dashboard/Dockerfile.optimized -t dashboard:optimized .

# Run with health checks
docker run --health-interval=30s ui-shell:optimized
```

### Production Deployment
```bash
# Use optimized docker-compose
docker-compose -f docker-compose.prod.yml up -d

# Monitor container performance
docker stats
```

## 🔧 **Development Tools**

### Performance Debugging
```typescript
// Enable performance monitoring in development
if (process.env.NODE_ENV === 'development') {
  import('./lib/performance-utils').then(({ trackBundleSize }) => {
    trackBundleSize();
  });
}
```

### Bundle Analysis in CI/CD
```yaml
# .github/workflows/performance.yml
- name: Analyze Bundle
  run: |
    npm run analyze
    # Upload results to artifact storage
```

## 📋 **Best Practices**

### Component Optimization
```typescript
// Use React.memo for expensive components
const OptimizedComponent = React.memo(ExpensiveComponent);

// Use useMemo for expensive calculations
const expensiveValue = useMemo(() => 
  heavyCalculation(data), [data]
);

// Use useCallback for event handlers
const handleClick = useCallback(() => {
  // handler logic
}, [dependency]);
```

### Query Optimization
```typescript
// Use query keys consistently
const { data } = useQuery({
  queryKey: queryKeys.users.detail(userId),
  queryFn: () => fetchUser(userId),
  staleTime: 5 * 60 * 1000, // 5 minutes
});
```

### Image Optimization
```typescript
import { useLazyImage } from './lib/performance-utils';

function LazyImage({ src, alt }) {
  const { imageSrc, isLoading, imgRef } = useLazyImage(src);
  
  return (
    <img 
      ref={imgRef}
      src={imageSrc}
      alt={alt}
      loading="lazy"
    />
  );
}
```

## 🎯 **Performance Checklist**

### Before Production
- [ ] Run bundle analysis
- [ ] Check for security vulnerabilities
- [ ] Test optimized Docker builds
- [ ] Verify performance metrics
- [ ] Enable monitoring tools

### Regular Maintenance
- [ ] Update dependencies monthly
- [ ] Review bundle size reports
- [ ] Monitor performance metrics
- [ ] Update security patches
- [ ] Optimize heavy components

## 📱 **Mobile Optimization**

### Responsive Performance
```typescript
// Use virtual scrolling for mobile
const { visibleItems, totalHeight } = useVirtualScroll(
  items, 
  window.innerHeight, 
  itemHeight
);
```

### Touch Optimization
```typescript
// Optimize touch events
const handleTouch = useThrottle(onTouchMove, 16); // 60fps
```

## 🔍 **Troubleshooting**

### Common Issues
1. **Large bundle size**: Use dynamic imports
2. **Slow API calls**: Enable response caching
3. **Memory leaks**: Use memory monitoring
4. **Slow renders**: Use performance monitoring

### Debug Commands
```bash
# Check bundle composition
npm run analyze

# Profile React components
npm run dev -- --profile

# Memory profiling
node --inspect npm run dev
```

---

*Use these tools regularly to maintain optimal performance*