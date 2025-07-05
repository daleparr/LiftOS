# Module Integration Guide with KSE Best Practices

## Overview

This guide provides comprehensive instructions for integrating modules with the Lift OS Core platform, including best practices for leveraging the Knowledge Storage Engine (KSE) and memory operations.

## Table of Contents

1. [Module Architecture](#module-architecture)
2. [Integration Requirements](#integration-requirements)
3. [KSE Integration Patterns](#kse-integration-patterns)
4. [Memory Operations Best Practices](#memory-operations-best-practices)
5. [Authentication & Security](#authentication--security)
6. [Module Registration](#module-registration)
7. [Health Checks & Monitoring](#health-checks--monitoring)
8. [Development Workflow](#development-workflow)
9. [Testing Strategies](#testing-strategies)
10. [Deployment Guidelines](#deployment-guidelines)

## Module Architecture

### Core Components

Every Lift OS module should implement the following core components:

```python
# module_template.py
from fastapi import FastAPI, HTTPException, Depends
from shared.health.health_checks import HealthChecker
from shared.security.security_manager import SecurityManager
from shared.logging.structured_logger import setup_service_logging

class LiftOSModule:
    def __init__(self, module_name: str, version: str):
        self.app = FastAPI(title=f"Lift OS {module_name}", version=version)
        self.health_checker = HealthChecker(module_name)
        self.security_manager = SecurityManager()
        self.logger = setup_service_logging(module_name)
        
        # Register core endpoints
        self._register_health_endpoints()
        self._register_module_endpoints()
    
    def _register_health_endpoints(self):
        @self.app.get("/health")
        async def health_check():
            return await self.health_checker.get_health_status()
        
        @self.app.get("/ready")
        async def readiness_check():
            return await self.health_checker.get_readiness_status()
    
    def _register_module_endpoints(self):
        # Implement module-specific endpoints here
        pass
```

### Module Structure

```
your-module/
├── app.py                 # Main application entry point
├── module.json           # Module metadata and configuration
├── requirements.txt      # Python dependencies
├── Dockerfile           # Container configuration
├── src/
│   ├── handlers/        # Request handlers
│   ├── services/        # Business logic
│   ├── models/          # Data models
│   └── utils/           # Utility functions
├── tests/
│   ├── unit/           # Unit tests
│   ├── integration/    # Integration tests
│   └── fixtures/       # Test data
└── docs/
    └── README.md       # Module documentation
```

## Integration Requirements

### 1. Module Metadata (module.json)

```json
{
  "name": "analytics-module",
  "version": "1.2.0",
  "description": "Advanced analytics and reporting module",
  "author": "Lift OS Team",
  "license": "MIT",
  "endpoints": {
    "health": "/health",
    "ready": "/ready",
    "main": "/analytics"
  },
  "capabilities": [
    "read",
    "write", 
    "analytics",
    "memory_integration"
  ],
  "dependencies": {
    "memory_service": ">=1.0.0",
    "auth_service": ">=1.0.0"
  },
  "kse_integration": {
    "enabled": true,
    "memory_types": ["analytics_data", "user_insights", "reports"],
    "search_capabilities": ["semantic", "temporal", "contextual"]
  },
  "configuration": {
    "port": 9001,
    "memory_service_url": "http://localhost:8003",
    "auth_service_url": "http://localhost:8001"
  }
}
```

### 2. Environment Configuration

```bash
# .env
MODULE_NAME=analytics-module
MODULE_VERSION=1.2.0
MODULE_PORT=9001

# Service URLs
GATEWAY_URL=http://localhost:8000
AUTH_SERVICE_URL=http://localhost:8001
MEMORY_SERVICE_URL=http://localhost:8003
REGISTRY_SERVICE_URL=http://localhost:8005

# KSE Configuration
KSE_ENABLED=true
KSE_INDEX_PREFIX=analytics_
KSE_BATCH_SIZE=100

# Security
JWT_SECRET_KEY=your-secret-key
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
```

## KSE Integration Patterns

### 1. Memory Storage Pattern

```python
import aiohttp
from typing import Dict, Any, Optional

class KSEIntegration:
    def __init__(self, memory_service_url: str, auth_token: str):
        self.memory_service_url = memory_service_url
        self.auth_token = auth_token
    
    async def store_memory(
        self, 
        key: str, 
        value: Dict[str, Any], 
        context: Dict[str, str],
        memory_type: str = "module_data"
    ) -> Optional[str]:
        """Store data in KSE with proper indexing"""
        
        memory_data = {
            "key": key,
            "value": value,
            "metadata": {
                "type": memory_type,
                "module": "analytics-module",
                "version": "1.2.0",
                "indexed_at": datetime.utcnow().isoformat(),
                "tags": self._generate_tags(value)
            },
            "context": context
        }
        
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.memory_service_url}/memory/store",
                json=memory_data,
                headers=headers
            ) as response:
                if response.status == 201:
                    result = await response.json()
                    return result.get("memory_id")
                return None
    
    def _generate_tags(self, value: Dict[str, Any]) -> List[str]:
        """Generate semantic tags for better searchability"""
        tags = []
        
        # Extract entity types
        if "user_id" in value:
            tags.append("user_data")
        if "report_type" in value:
            tags.append(f"report_{value['report_type']}")
        if "date_range" in value:
            tags.append("temporal_data")
        
        return tags
```

### 2. Semantic Search Pattern

```python
async def search_memories(
    self,
    query: str,
    context: Dict[str, str],
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Perform semantic search with KSE"""
    
    search_data = {
        "query": query,
        "context": context,
        "filters": {
            "type": "analytics_data",
            "module": "analytics-module",
            **(filters or {})
        },
        "limit": limit,
        "include_highlights": True
    }
    
    headers = {"Authorization": f"Bearer {self.auth_token}"}
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{self.memory_service_url}/memory/search",
            json=search_data,
            headers=headers
        ) as response:
            if response.status == 200:
                result = await response.json()
                return result.get("results", [])
            return []

async def get_contextual_insights(
    self,
    user_id: str,
    org_id: str,
    insight_type: str = "user_behavior"
) -> Dict[str, Any]:
    """Get contextual insights using KSE"""
    
    # Build contextual query
    query = f"user insights {insight_type} patterns behavior"
    
    context = {
        "user_id": user_id,
        "org_id": org_id
    }
    
    filters = {
        "type": "user_insights",
        "date_range": {
            "start": (datetime.utcnow() - timedelta(days=30)).isoformat(),
            "end": datetime.utcnow().isoformat()
        }
    }
    
    results = await self.search_memories(query, context, filters, limit=50)
    
    # Process and aggregate insights
    insights = self._aggregate_insights(results)
    
    return {
        "user_id": user_id,
        "insight_type": insight_type,
        "insights": insights,
        "confidence_score": self._calculate_confidence(results),
        "generated_at": datetime.utcnow().isoformat()
    }
```

### 3. Hybrid Search Pattern

```python
async def hybrid_search(
    self,
    semantic_query: str,
    keyword_filters: Dict[str, Any],
    context: Dict[str, str],
    weights: Dict[str, float] = None
) -> List[Dict[str, Any]]:
    """Combine semantic and keyword search for optimal results"""
    
    if weights is None:
        weights = {"semantic": 0.7, "keyword": 0.3}
    
    # Semantic search
    semantic_results = await self.search_memories(
        semantic_query, context, limit=20
    )
    
    # Keyword search
    keyword_query = self._build_keyword_query(keyword_filters)
    keyword_results = await self.search_memories(
        keyword_query, context, keyword_filters, limit=20
    )
    
    # Combine and rank results
    combined_results = self._combine_search_results(
        semantic_results, keyword_results, weights
    )
    
    return combined_results[:10]  # Return top 10
```

## Memory Operations Best Practices

### 1. Data Modeling

```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class AnalyticsMemory(BaseModel):
    """Structured model for analytics data storage"""
    
    user_id: str = Field(..., description="User identifier")
    org_id: str = Field(..., description="Organization identifier")
    event_type: str = Field(..., description="Type of analytics event")
    event_data: Dict[str, Any] = Field(..., description="Event payload")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    session_id: Optional[str] = Field(None, description="Session identifier")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class UserInsight(BaseModel):
    """Model for user behavioral insights"""
    
    user_id: str
    insight_type: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    insights: Dict[str, Any]
    evidence: List[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
```

### 2. Batch Operations

```python
async def batch_store_analytics(
    self,
    analytics_data: List[AnalyticsMemory],
    batch_size: int = 50
) -> List[str]:
    """Store analytics data in batches for better performance"""
    
    memory_ids = []
    
    for i in range(0, len(analytics_data), batch_size):
        batch = analytics_data[i:i + batch_size]
        batch_tasks = []
        
        for item in batch:
            task = self.store_memory(
                key=f"analytics_{item.user_id}_{item.timestamp.isoformat()}",
                value=item.dict(),
                context={
                    "user_id": item.user_id,
                    "org_id": item.org_id
                },
                memory_type="analytics_data"
            )
            batch_tasks.append(task)
        
        # Execute batch concurrently
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Collect successful memory IDs
        for result in batch_results:
            if isinstance(result, str):  # Success case
                memory_ids.append(result)
            else:  # Exception case
                self.logger.error(f"Batch storage error: {result}")
    
    return memory_ids
```

### 3. Memory Lifecycle Management

```python
async def cleanup_expired_memories(self, org_id: str) -> int:
    """Clean up expired memories for an organization"""
    
    # Search for expired memories
    query = "expired memories cleanup"
    context = {"org_id": org_id}
    filters = {
        "type": "analytics_data",
        "expires_before": datetime.utcnow().isoformat()
    }
    
    expired_memories = await self.search_memories(query, context, filters, limit=1000)
    
    cleanup_count = 0
    for memory in expired_memories:
        success = await self._delete_memory(memory["memory_id"])
        if success:
            cleanup_count += 1
    
    self.logger.info(f"Cleaned up {cleanup_count} expired memories for org {org_id}")
    return cleanup_count

async def archive_old_memories(self, org_id: str, days_old: int = 365) -> int:
    """Archive old memories to cold storage"""
    
    cutoff_date = datetime.utcnow() - timedelta(days=days_old)
    
    query = f"old memories archive before {cutoff_date.isoformat()}"
    context = {"org_id": org_id}
    filters = {
        "type": "analytics_data",
        "created_before": cutoff_date.isoformat()
    }
    
    old_memories = await self.search_memories(query, context, filters, limit=5000)
    
    # Archive to cold storage (implementation depends on storage backend)
    archived_count = await self._archive_memories_to_cold_storage(old_memories)
    
    return archived_count
```

## Authentication & Security

### 1. JWT Token Handling

```python
from shared.security.security_manager import SecurityManager

class ModuleAuthHandler:
    def __init__(self):
        self.security_manager = SecurityManager()
    
    async def verify_request_token(self, authorization: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token from request header"""
        
        if not authorization or not authorization.startswith("Bearer "):
            return None
        
        token = authorization.split(" ")[1]
        user_context = self.security_manager.verify_jwt_token(token)
        
        if not user_context:
            return None
        
        return user_context
    
    def get_auth_dependency(self):
        """FastAPI dependency for authentication"""
        
        async def auth_dependency(authorization: str = Header(None)):
            user_context = await self.verify_request_token(authorization)
            if not user_context:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid or missing authentication token"
                )
            return user_context
        
        return auth_dependency
```

### 2. Rate Limiting

```python
from shared.security.security_manager import SecurityManager

async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware for module endpoints"""
    
    client_ip = request.client.host
    security_manager = SecurityManager()
    
    if not security_manager.check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )
    
    response = await call_next(request)
    return response
```

## Module Registration

### 1. Automatic Registration

```python
async def register_module_with_registry():
    """Register module with the Registry service"""
    
    module_info = {
        "name": os.getenv("MODULE_NAME"),
        "version": os.getenv("MODULE_VERSION"),
        "endpoint": f"http://localhost:{os.getenv('MODULE_PORT')}",
        "health_check": "/health",
        "capabilities": ["read", "write", "analytics", "memory_integration"],
        "metadata": {
            "description": "Advanced analytics and reporting module",
            "author": "Lift OS Team",
            "tags": ["analytics", "reporting", "insights"]
        }
    }
    
    registry_url = os.getenv("REGISTRY_SERVICE_URL")
    auth_token = await get_service_auth_token()
    
    headers = {"Authorization": f"Bearer {auth_token}"}
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{registry_url}/registry/modules",
            json=module_info,
            headers=headers
        ) as response:
            if response.status == 201:
                result = await response.json()
                logger.info(f"Module registered successfully: {result['module_id']}")
                return result["module_id"]
            else:
                logger.error(f"Module registration failed: {response.status}")
                return None
```

### 2. Health Check Registration

```python
async def register_health_checks():
    """Register health check endpoints with monitoring"""
    
    health_endpoints = {
        "liveness": "/health",
        "readiness": "/ready",
        "metrics": "/metrics"  # If implementing custom metrics
    }
    
    # Register with monitoring system
    for check_type, endpoint in health_endpoints.items():
        await register_health_endpoint(check_type, endpoint)
```

## Health Checks & Monitoring

### 1. Custom Health Checks

```python
from shared.health.health_checks import HealthChecker

class AnalyticsHealthChecker(HealthChecker):
    def __init__(self):
        super().__init__("analytics-module")
        self.kse_integration = KSEIntegration(
            memory_service_url=os.getenv("MEMORY_SERVICE_URL"),
            auth_token=os.getenv("SERVICE_AUTH_TOKEN")
        )
    
    async def check_kse_connectivity(self) -> Dict[str, Any]:
        """Check KSE connectivity and performance"""
        
        try:
            start_time = time.time()
            
            # Test search operation
            test_results = await self.kse_integration.search_memories(
                query="health check test",
                context={"user_id": "health_check", "org_id": "system"},
                limit=1
            )
            
            response_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "response_time": response_time,
                "kse_accessible": True,
                "test_query_successful": True
            }
        
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "kse_accessible": False
            }
    
    async def get_readiness_status(self, external_checks=None):
        """Enhanced readiness check including KSE"""
        
        if external_checks is None:
            external_checks = []
        
        # Add KSE connectivity check
        external_checks.append(self.check_kse_connectivity)
        
        return await super().get_readiness_status(external_checks)
```

### 2. Custom Metrics

```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
REQUEST_COUNT = Counter('module_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('module_request_duration_seconds', 'Request duration')
MEMORY_OPERATIONS = Counter('module_memory_operations_total', 'Memory operations', ['operation_type'])
KSE_SEARCH_DURATION = Histogram('module_kse_search_duration_seconds', 'KSE search duration')

class MetricsMiddleware:
    async def __call__(self, request: Request, call_next):
        start_time = time.time()
        
        # Track request
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path
        ).inc()
        
        response = await call_next(request)
        
        # Track duration
        duration = time.time() - start_time
        REQUEST_DURATION.observe(duration)
        
        return response
```

## Development Workflow

### 1. Local Development Setup

```bash
# 1. Clone module template
git clone https://github.com/liftos/module-template.git your-module
cd your-module

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment
cp .env.example .env
# Edit .env with your configuration

# 4. Start local development
python app.py

# 5. Register with local registry
curl -X POST http://localhost:8005/registry/modules \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d @module.json
```

### 2. Testing Integration

```python
# tests/test_kse_integration.py
import pytest
from unittest.mock import AsyncMock, patch
from src.kse_integration import KSEIntegration

@pytest.mark.asyncio
async def test_memory_storage():
    """Test memory storage functionality"""
    
    kse = KSEIntegration("http://localhost:8003", "test_token")
    
    with patch('aiohttp.ClientSession.post') as mock_post:
        mock_response = AsyncMock()
        mock_response.status = 201
        mock_response.json.return_value = {"memory_id": "test_123"}
        mock_post.return_value.__aenter__.return_value = mock_response
        
        memory_id = await kse.store_memory(
            key="test_key",
            value={"test": "data"},
            context={"user_id": "test_user", "org_id": "test_org"}
        )
        
        assert memory_id == "test_123"

@pytest.mark.asyncio
async def test_semantic_search():
    """Test semantic search functionality"""
    
    kse = KSEIntegration("http://localhost:8003", "test_token")
    
    with patch('aiohttp.ClientSession.post') as mock_post:
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "results": [{"memory_id": "mem_1", "score": 0.95}],
            "total": 1,
            "query_time": 0.05
        }
        mock_post.return_value.__aenter__.return_value = mock_response
        
        results = await kse.search_memories(
            query="test search",
            context={"user_id": "test_user", "org_id": "test_org"}
        )
        
        assert len(results) == 1
        assert results[0]["score"] == 0.95
```

## Testing Strategies

### 1. Unit Testing

```python
# tests/unit/test_analytics_service.py
import pytest
from unittest.mock import Mock, AsyncMock
from src.services.analytics_service import AnalyticsService

class TestAnalyticsService:
    
    @pytest.fixture
    def analytics_service(self):
        return AnalyticsService()
    
    @pytest.mark.asyncio
    async def test_generate_user_insights(self, analytics_service):
        """Test user insights generation"""
        
        # Mock KSE integration
        analytics_service.kse_integration.search_memories = AsyncMock(
            return_value=[
                {"memory_id": "mem_1", "score": 0.9, "value": {"event": "login"}},
                {"memory_id": "mem_2", "score": 0.8, "value": {"event": "page_view"}}
            ]
        )
        
        insights = await analytics_service.generate_user_insights(
            user_id="test_user",
            org_id="test_org"
        )
        
        assert insights["user_id"] == "test_user"
        assert "insights" in insights
        assert insights["confidence_score"] > 0
```

### 2. Integration Testing

```python
# tests/integration/test_module_integration.py
import pytest
import aiohttp
from testcontainers import DockerCompose

@pytest.mark.integration
class TestModuleIntegration:
    
    @pytest.fixture(scope="class")
    def docker_services(self):
        """Start required services for integration testing"""
        
        with DockerCompose(".", compose_file_name="docker-compose.test.yml") as compose:
            # Wait for services to be ready
            compose.wait_for("http://localhost:8000/health")  # Gateway
            compose.wait_for("http://localhost:8003/health")  # Memory
            yield compose
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, docker_services):
        """Test complete module workflow"""
        
        # 1. Register module
        async with aiohttp.ClientSession() as session:
            # Register with registry
            async with session.post(
                "http://localhost:8005/registry/modules",
                json={"name": "test-module", "version": "1.0.0", "endpoint": "http://localhost:9001"}
            ) as response:
                assert response.status == 201
            
            # 2. Store memory via module
            async with session.post(
                "http://localhost:9001/analytics/store",
                json={"user_id": "test_user", "event": "test_event"}
            ) as response:
                assert response.status == 201
            
            # 3. Search memories via module
            async with session.post(
                "http://localhost:9001/analytics/search",
                json={"query": "test event", "user_id": "test_user"}
            ) as response:
                assert response.status == 200
                results = await response.json()
                assert len(results["results"]) > 0
```

## Deployment Guidelines

### 1. Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 moduleuser && chown -R moduleuser:moduleuser /app
USER moduleuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${MODULE_PORT}/health || exit 1

# Expose port
EXPOSE ${MODULE_PORT}

# Start application
CMD ["python", "app.py"]
```

### 2. Docker Compose Integration

```yaml
# docker-compose.yml
version: '3.8'

services:
  analytics-module:
    build: .
    ports:
      - "9001:9001"
    environment:
      - MODULE_NAME=analytics-module
      - MODULE_VERSION=1.2.0
      - MODULE_PORT=9001
      - GATEWAY_URL=http://gateway:8000
      - MEMORY_SERVICE_URL=http://memory:8003
      - REGISTRY_SERVICE_URL=http://registry:8005
    depends_on:
      - gateway
      - memory
      - registry
    networks:
      - liftos-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

networks:
  liftos-network:
    external: true
```

### 3. Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: analytics-module
  labels:
    app: analytics-module
    version: v1.2.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: analytics-module
  template:
    metadata:
      labels:
        app: analytics-module
        version: v1.2.0
    spec:
      containers:
      - name: analytics-module
        image: liftos/analytics-module:1.2.0
        ports:
        - containerPort: 9001
        env:
        - name: MODULE_NAME
          value: "analytics-module"
        - name: MODULE_VERSION
          value: "1.2.0"
        - name: MEMORY_SERVICE_URL
          value: "http://memory-service:8003"
        livenessProbe:
          httpGet:
            path: /health
            port: 9001
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 9001
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: analytics-module-service
spec:
  selector:
    app: analytics-module
  ports:
  - protocol: TCP
    port: 9001
    targetPort: 9001
  type: ClusterIP
```

## Best Practices Summary

### 1. KSE Integration
- Use structured data models for consistent memory storage
- Implement proper tagging and metadata for searchability
- Leverage semantic search for intelligent data retrieval
- Use batch operations for performance optimization
- Implement memory lifecycle management

### 2. Security
- Always validate JWT tokens for authenticated endpoints
- Implement rate limiting to prevent abuse
- Use HTTPS in production environments
- Sanitize and validate all input data
- Follow principle of least privilege

### 3. Performance
- Use async/await patterns for non-blocking operations
- Implement connection pooling for external service calls
- Cache frequently accessed data appropriately
- Monitor and optimize memory usage
- Use batch operations for bulk data processing

### 4. Monitoring
- Implement comprehensive health checks
- Use structured logging with correlation IDs
- Expose custom metrics for monitoring
- Set up alerting for critical failures
- Monitor KSE performance and usage

### 5. Development
- Follow consistent code structure and naming conventions
- Write comprehensive tests (unit, integration, e2e)
- Use type hints and documentation
- Implement proper error handling and logging
- Follow semantic versioning for releases

This guide provides a comprehensive foundation for building robust, scalable modules that integrate seamlessly with the Lift OS Core platform and leverage the full power of the Knowledge Storage Engine.