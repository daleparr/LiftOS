# Lift OS Core - Developer Guide

Welcome to Lift OS Core development! This guide will help you get started with contributing to and extending the platform.

## 🚀 Quick Start

### Prerequisites

Before you begin, ensure you have the following installed:

- **Docker & Docker Compose** (v20.10+)
- **Node.js** (v18+) and **npm** (v8+)
- **Python** (v3.11+) and **pip**
- **Git** (v2.30+)
- **Make** (optional, for convenience commands)

### Development Environment Setup

1. **Clone the Repository**
```bash
git clone https://github.com/your-org/lift-os-core.git
cd lift-os-core
```

2. **Environment Configuration**
```bash
# Copy environment template
cp .env.example .env

# Edit environment variables
nano .env
```

3. **Start Development Environment**
```bash
# Start all services
make dev-up

# Or manually with Docker Compose
docker-compose up -d

# Wait for services to be ready
make health-check
```

4. **Verify Installation**
```bash
# Check service health
curl http://localhost:8000/health

# Access UI
open http://localhost:3000
```

## 🏗️ Architecture Overview

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   UI Shell      │    │    Gateway      │    │   Auth Service  │
│   (Next.js)     │◄──►│   (FastAPI)     │◄──►│   (FastAPI)     │
│   Port: 3000    │    │   Port: 8000    │    │   Port: 8001    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
        ┌───────▼──────┐ ┌──────▼──────┐ ┌─────▼──────┐
        │ Memory       │ │ Registry    │ │ Billing    │
        │ (FastAPI)    │ │ (FastAPI)   │ │ (FastAPI)  │
        │ Port: 8002   │ │ Port: 8003  │ │ Port: 8004 │
        └──────────────┘ └─────────────┘ └────────────┘
                │
        ┌───────▼──────┐ ┌─────────────┐ ┌─────────────┐
        │Observability │ │ PostgreSQL  │ │   Redis     │
        │ (FastAPI)    │ │ Port: 5432  │ │ Port: 6379  │
        │ Port: 8005   │ └─────────────┘ └─────────────┘
        └──────────────┘
```

### Service Responsibilities

- **Gateway**: API routing, authentication, rate limiting
- **Auth**: User management, JWT tokens, OAuth integration
- **Memory**: KSE integration, context storage, retrieval
- **Registry**: Module management, deployment, discovery
- **Billing**: Subscription management, usage tracking
- **Observability**: Metrics, logging, health monitoring
- **UI Shell**: Frontend interface, user dashboard

## 🛠️ Development Workflow

### Code Organization

```
lift-os-core/
├── services/           # Backend microservices
│   ├── gateway/       # API gateway service
│   ├── auth/          # Authentication service
│   ├── memory/        # Memory management service
│   ├── registry/      # Module registry service
│   ├── billing/       # Billing service
│   └── observability/ # Monitoring service
├── ui-shell/          # Frontend Next.js application
├── modules/           # Example modules
│   ├── _template/     # Module template
│   ├── lift-causal/   # Causal modeling module
│   └── lift-eval/     # AI evaluation module
├── shared/            # Shared libraries
├── tests/             # Test suites
├── k8s/              # Kubernetes manifests
├── docs/             # Documentation
└── scripts/          # Utility scripts
```

### Development Commands

```bash
# Start development environment
make dev-up

# Stop development environment
make dev-down

# View logs
make logs

# Run tests
make test

# Run specific test types
make test-unit
make test-integration
make test-coverage

# Access service shells
make shell-gateway
make shell-auth
make shell-memory

# Health checks
make health-check

# Clean up
make clean
```

### Service Development

#### Creating a New Service

1. **Create Service Directory**
```bash
mkdir services/my-service
cd services/my-service
```

2. **Service Structure**
```
services/my-service/
├── app.py              # Main FastAPI application
├── requirements.txt    # Python dependencies
├── Dockerfile         # Container configuration
├── models/            # Data models
├── routes/            # API route handlers
├── services/          # Business logic
└── tests/             # Service-specific tests
```

3. **Basic FastAPI App** ([`app.py`](services/my-service/app.py))
```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="My Service",
    description="Custom service for Lift OS",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "my-service"}

@app.get("/")
async def root():
    return {"message": "My Service API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)
```

4. **Add to Docker Compose**
```yaml
# docker-compose.yml
services:
  my-service:
    build: ./services/my-service
    ports:
      - "8006:8006"
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
    depends_on:
      - postgres
      - redis
    networks:
      - lift-network
```

#### Frontend Development

The UI Shell is built with Next.js and provides the main user interface.

**Key Components:**
- [`Layout.tsx`](ui-shell/src/components/Layout.tsx): Main layout with navigation
- [`Dashboard.tsx`](ui-shell/src/components/Dashboard.tsx): User dashboard
- [`Login.tsx`](ui-shell/src/components/Login.tsx): Authentication forms

**Development Server:**
```bash
cd ui-shell
npm install
npm run dev
```

**Adding New Pages:**
```typescript
// ui-shell/src/pages/my-page.tsx
import Layout from '../components/Layout'

export default function MyPage() {
  return (
    <Layout>
      <div className="p-6">
        <h1 className="text-2xl font-bold">My Custom Page</h1>
        {/* Your content here */}
      </div>
    </Layout>
  )
}
```

### Module Development

#### Creating a Custom Module

1. **Use Module Template**
```bash
cp -r modules/_template modules/my-module
cd modules/my-module
```

2. **Update Module Configuration** ([`module.json`](modules/my-module/module.json))
```json
{
  "name": "my-module",
  "version": "1.0.0",
  "description": "My custom module",
  "author": "Your Name",
  "category": "custom",
  "tags": ["utility", "custom"],
  "config": {
    "port": 8080,
    "environment": {
      "MODULE_ENV": "development"
    }
  },
  "ui_components": [
    {
      "name": "Dashboard",
      "path": "/dashboard",
      "icon": "chart-bar",
      "description": "Module dashboard"
    }
  ],
  "api_endpoints": [
    {
      "path": "/process",
      "method": "POST",
      "description": "Process data"
    }
  ]
}
```

3. **Implement Module Logic** ([`app.py`](modules/my-module/app.py))
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os

app = FastAPI(title="My Module", version="1.0.0")

class ProcessRequest(BaseModel):
    data: str
    options: dict = {}

@app.post("/process")
async def process_data(request: ProcessRequest):
    """Process incoming data"""
    try:
        # Your custom logic here
        result = {
            "processed_data": request.data.upper(),
            "options_used": request.options,
            "status": "success"
        }
        
        # Store result in memory service
        memory_response = requests.post(
            f"{os.getenv('MEMORY_SERVICE_URL', 'http://memory:8002')}/store",
            json={
                "content": f"Processed: {result['processed_data']}",
                "context": "my-module",
                "metadata": {"module": "my-module", "version": "1.0.0"}
            },
            headers={"Authorization": f"Bearer {os.getenv('SERVICE_TOKEN')}"}
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dashboard")
async def get_dashboard():
    """Return dashboard data"""
    return {
        "title": "My Module Dashboard",
        "stats": {
            "processed_items": 42,
            "success_rate": 0.95
        }
    }
```

4. **Register Module**
```bash
# Build module
docker build -t my-module:1.0.0 .

# Register with registry service
curl -X POST http://localhost:8003/modules \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d @module.json
```

## 🧪 Testing

### Test Structure

```
tests/
├── unit/              # Unit tests for individual components
├── integration/       # Integration tests across services
├── performance/       # Performance and load tests
├── conftest.py       # Pytest configuration and fixtures
└── pytest.ini       # Test settings
```

### Running Tests

```bash
# All tests
make test

# Unit tests only
make test-unit

# Integration tests only
make test-integration

# With coverage
make test-coverage

# Performance tests
make test-performance

# Specific test file
pytest tests/unit/test_auth.py -v

# Specific test function
pytest tests/unit/test_auth.py::test_login -v
```

### Writing Tests

#### Unit Test Example
```python
# tests/unit/test_auth.py
import pytest
from fastapi.testclient import TestClient
from services.auth.app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_login_valid_credentials():
    response = client.post("/login", json={
        "email": "test@example.com",
        "password": "password123"
    })
    assert response.status_code == 200
    assert "access_token" in response.json()

def test_login_invalid_credentials():
    response = client.post("/login", json={
        "email": "test@example.com",
        "password": "wrongpassword"
    })
    assert response.status_code == 401
```

#### Integration Test Example
```python
# tests/integration/test_memory_flow.py
import pytest
import requests

@pytest.mark.integration
def test_memory_storage_and_retrieval(auth_token):
    # Store memory
    store_response = requests.post(
        "http://localhost:8002/store",
        json={
            "content": "Test memory content",
            "context": "test"
        },
        headers={"Authorization": f"Bearer {auth_token}"}
    )
    assert store_response.status_code == 200
    memory_id = store_response.json()["memory_id"]
    
    # Retrieve memory
    retrieve_response = requests.get(
        f"http://localhost:8002/retrieve?query=test&context=test",
        headers={"Authorization": f"Bearer {auth_token}"}
    )
    assert retrieve_response.status_code == 200
    memories = retrieve_response.json()["memories"]
    assert len(memories) > 0
    assert any(m["memory_id"] == memory_id for m in memories)
```

### Test Fixtures

```python
# tests/conftest.py
import pytest
import requests
from typing import Generator

@pytest.fixture(scope="session")
def auth_token() -> str:
    """Get authentication token for testing"""
    response = requests.post(
        "http://localhost:8001/login",
        json={
            "email": "test@example.com",
            "password": "testpassword"
        }
    )
    return response.json()["access_token"]

@pytest.fixture
def test_user_data():
    """Test user data"""
    return {
        "email": "testuser@example.com",
        "password": "testpassword123",
        "name": "Test User"
    }
```

## 📊 Monitoring and Debugging

### Logging

All services use structured logging:

```python
import logging
import json

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Log with context
logger.info("User login", extra={
    "user_id": "user_123",
    "ip_address": "192.168.1.1",
    "user_agent": "Mozilla/5.0..."
})
```

### Metrics

Services expose Prometheus metrics:

```python
from prometheus_client import Counter, Histogram, generate_latest

# Define metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
    REQUEST_DURATION.observe(duration)
    
    return response

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### Health Checks

Implement comprehensive health checks:

```python
@app.get("/health")
async def health_check():
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "my-service",
        "version": "1.0.0"
    }
    
    # Check database connection
    try:
        await database.execute("SELECT 1")
        health_status["database"] = "healthy"
    except Exception as e:
        health_status["database"] = "unhealthy"
        health_status["status"] = "unhealthy"
        health_status["errors"] = [str(e)]
    
    # Check external dependencies
    try:
        response = requests.get("http://external-api/health", timeout=5)
        health_status["external_api"] = "healthy" if response.status_code == 200 else "unhealthy"
    except Exception as e:
        health_status["external_api"] = "unhealthy"
        health_status["status"] = "unhealthy"
    
    status_code = 200 if health_status["status"] == "healthy" else 503
    return JSONResponse(content=health_status, status_code=status_code)
```

## 🔧 Configuration Management

### Environment Variables

Use environment variables for configuration:

```python
import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    database_url: str
    redis_url: str
    jwt_secret: str
    log_level: str = "INFO"
    debug: bool = False
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### Configuration Files

For complex configuration, use YAML files:

```yaml
# config/development.yml
database:
  url: postgresql://user:pass@localhost:5432/lift_os_dev
  pool_size: 10
  
redis:
  url: redis://localhost:6379/0
  
logging:
  level: DEBUG
  format: detailed
  
features:
  oauth_enabled: true
  billing_enabled: false
```

## 🚀 Deployment

### Local Development

```bash
# Start development environment
make dev-up

# Hot reload for services
docker-compose up --build gateway auth memory
```

### Staging Deployment

```bash
# Build and tag images
docker build -t lift-os/gateway:staging ./services/gateway
docker build -t lift-os/auth:staging ./services/auth

# Deploy to staging
docker-compose -f docker-compose.yml -f docker-compose.staging.yml up -d
```

### Production Deployment

```bash
# Build production images
make build-prod

# Deploy with production configuration
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Or deploy to Kubernetes
kubectl apply -f k8s/
```

## 📚 Best Practices

### Code Style

- **Python**: Follow PEP 8, use Black for formatting
- **TypeScript**: Use Prettier and ESLint
- **Documentation**: Write docstrings and comments
- **Type Hints**: Use type annotations in Python

### API Design

- **RESTful**: Follow REST principles
- **Versioning**: Use API versioning (`/api/v1/`)
- **Error Handling**: Return consistent error responses
- **Validation**: Use Pydantic models for request/response validation

### Security

- **Authentication**: Always validate JWT tokens
- **Authorization**: Implement proper RBAC
- **Input Validation**: Sanitize all inputs
- **Rate Limiting**: Implement rate limiting
- **HTTPS**: Use HTTPS in production

### Performance

- **Caching**: Use Redis for caching
- **Database**: Optimize queries and use indexes
- **Async**: Use async/await for I/O operations
- **Connection Pooling**: Use connection pools

## 🤝 Contributing

### Pull Request Process

1. **Fork the Repository**
2. **Create Feature Branch**
```bash
git checkout -b feature/my-new-feature
```

3. **Make Changes**
4. **Write Tests**
5. **Run Tests**
```bash
make test
```

6. **Submit Pull Request**

### Code Review Guidelines

- **Functionality**: Does the code work as intended?
- **Tests**: Are there adequate tests?
- **Documentation**: Is the code well-documented?
- **Performance**: Are there any performance concerns?
- **Security**: Are there any security issues?

### Commit Message Format

```
type(scope): description

[optional body]

[optional footer]
```

Examples:
```
feat(auth): add OAuth2 integration
fix(memory): resolve memory leak in retrieval
docs(api): update authentication documentation
```

## 📞 Getting Help

### Resources

- **Documentation**: [`docs/`](docs/)
- **API Reference**: [`docs/API.md`](docs/API.md)
- **Architecture**: [`LIFT_OS_CORE_ARCHITECTURE.md`](LIFT_OS_CORE_ARCHITECTURE.md)

### Community

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share ideas
- **Discord**: Join our developer community
- **Email**: dev-support@lift-os.com

### Debugging Tips

1. **Check Logs**
```bash
make logs
docker-compose logs -f service-name
```

2. **Health Checks**
```bash
make health-check
curl http://localhost:8000/health
```

3. **Database Access**
```bash
make shell-postgres
psql -U postgres -d lift_os_dev
```

4. **Redis Access**
```bash
make shell-redis
redis-cli
```

Happy coding! 🚀