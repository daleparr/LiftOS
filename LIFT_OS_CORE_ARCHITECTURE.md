# 🚀 Lift OS Core - Comprehensive Architecture & Implementation Plan

## 📋 Executive Summary

This document outlines the development of Lift OS Core as a microservices-based platform that serves as the unified backbone for the Lift ecosystem. The architecture emphasizes modularity, scalability, and developer experience while maintaining simplicity for local development with Docker Compose.

**Key Technologies:**
- **Backend**: Python/FastAPI for core services
- **Frontend**: React/Next.js for UI shell
- **Memory**: KSE Memory SDK integration (Hybrid AI: Neural Embeddings + Conceptual Spaces + Knowledge Graphs)
- **Deployment**: Docker Compose for local development
- **Database**: PostgreSQL + Redis

## 🎯 Architecture Overview

```mermaid
graph TB
    subgraph "External Access"
        U[Users/Clients]
        D[Developers]
    end
    
    subgraph "Lift OS Core Platform"
        subgraph "Frontend Layer"
            UI[Next.js UI Shell<br/>Port 3000]
        end
        
        subgraph "Gateway Layer"
            GW[API Gateway<br/>Port 8000]
        end
        
        subgraph "Core Services"
            AUTH[Auth Service<br/>Port 8001]
            BILL[Billing Service<br/>Port 8002]
            MEM[Memory Service<br/>Port 8003]
            OBS[Observability<br/>Port 8004]
            REG[Registry Service<br/>Port 8005]
        end
        
        subgraph "Modular Apps"
            CAUSAL[Lift Causal<br/>Port 9001]
            EVAL[Lift Eval<br/>Port 9002]
            AGENT[Lift Agentic<br/>Port 9003]
            LLM[Lift LLM<br/>Port 9004]
            SURF[Surfacbility<br/>Port 9005]
            SENT[Sentiment<br/>Port 9006]
        end
        
        subgraph "Data Layer"
            KSE[KSE Memory SDK<br/>Hybrid AI Engine]
            DB[(PostgreSQL)]
            REDIS[(Redis Cache)]
        end
    end
    
    U --> UI
    U --> GW
    D --> GW
    
    UI --> GW
    GW --> AUTH
    GW --> BILL
    GW --> MEM
    GW --> OBS
    GW --> REG
    
    GW --> CAUSAL
    GW --> EVAL
    GW --> AGENT
    GW --> LLM
    GW --> SURF
    GW --> SENT
    
    AUTH --> DB
    BILL --> DB
    REG --> DB
    MEM --> KSE
    OBS --> REDIS
    
    CAUSAL --> KSE
    EVAL --> KSE
    AGENT --> KSE
```

## 🗂️ Project Structure

```
lift-os-core/
├── 📁 services/                    # Core microservices
│   ├── 📁 gateway/                 # API Gateway & Routing
│   │   ├── app.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── auth.py
│   │   │   ├── modules.py
│   │   │   └── health.py
│   │   ├── middleware/
│   │   │   ├── __init__.py
│   │   │   ├── auth_middleware.py
│   │   │   ├── rate_limiter.py
│   │   │   └── cors_middleware.py
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   ├── 📁 auth/                    # Authentication & Authorization
│   │   ├── app.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── user.py
│   │   │   └── permissions.py
│   │   ├── jwt_handler.py
│   │   ├── oauth_providers.py
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   ├── 📁 billing/                 # Billing & Subscription Management
│   │   ├── app.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── subscription.py
│   │   │   └── usage.py
│   │   ├── stripe_integration.py
│   │   ├── metering.py
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   ├── 📁 memory/                  # KSE Integration Service
│   │   ├── app.py
│   │   ├── kse_client.py
│   │   ├── memory_manager.py
│   │   ├── context_handler.py
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   ├── 📁 observability/           # Logging, Metrics, Health
│   │   ├── app.py
│   │   ├── metrics.py
│   │   ├── health_checker.py
│   │   ├── log_aggregator.py
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   └── 📁 registry/                # Module Registration & Discovery
│       ├── app.py
│       ├── models/
│       │   ├── __init__.py
│       │   └── module.py
│       ├── module_manager.py
│       ├── health_monitor.py
│       ├── requirements.txt
│       └── Dockerfile
├── 📁 ui-shell/                    # Next.js Frontend
│   ├── package.json
│   ├── next.config.js
│   ├── tailwind.config.js
│   ├── pages/
│   │   ├── _app.tsx
│   │   ├── _document.tsx
│   │   ├── index.tsx
│   │   ├── login.tsx
│   │   ├── dashboard.tsx
│   │   └── api/
│   ├── components/
│   │   ├── Layout/
│   │   ├── Auth/
│   │   ├── Dashboard/
│   │   └── Modules/
│   ├── hooks/
│   │   ├── useAuth.ts
│   │   ├── useModules.ts
│   │   └── useMemory.ts
│   ├── lib/
│   │   ├── api.ts
│   │   ├── auth.ts
│   │   └── constants.ts
│   ├── styles/
│   └── Dockerfile
├── 📁 modules/                     # Modular App Templates
│   ├── 📁 causal/
│   │   ├── app.py
│   │   ├── module.json
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   ├── 📁 eval/
│   ├── 📁 agentic/
│   ├── 📁 llm/
│   ├── 📁 surfacbility/
│   └── 📁 sentiment/
├── 📁 shared/                      # Shared Libraries
│   ├── 📁 auth/
│   │   ├── __init__.py
│   │   ├── jwt_utils.py
│   │   └── permissions.py
│   ├── 📁 models/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── common.py
│   ├── 📁 utils/
│   │   ├── __init__.py
│   │   ├── logging.py
│   │   ├── config.py
│   │   └── validators.py
│   └── 📁 kse_sdk/
│       ├── __init__.py
│       ├── client.py
│       ├── memory_context.py
│       └── hybrid_search.py
├── 📁 infrastructure/              # DevOps & Deployment
│   ├── docker-compose.yml
│   ├── docker-compose.dev.yml
│   ├── docker-compose.prod.yml
│   ├── .env.example
│   ├── nginx.conf
│   └── scripts/
│       ├── setup.sh
│       ├── dev-start.sh
│       └── test.sh
├── 📁 docs/                        # Documentation
│   ├── api/
│   │   ├── gateway.md
│   │   ├── auth.md
│   │   ├── billing.md
│   │   ├── memory.md
│   │   └── modules.md
│   ├── architecture/
│   │   ├── overview.md
│   │   ├── security.md
│   │   └── scalability.md
│   ├── integration-guide.md
│   └── deployment.md
├── 📁 tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 Makefile
└── 📄 .gitignore
```

## 🧠 KSE Memory SDK Integration

### KSE Architecture Integration

The KSE Memory SDK provides hybrid knowledge retrieval combining:
- **Neural Embeddings**: Deep learning semantic similarity
- **Conceptual Spaces**: Multi-dimensional similarity across concepts  
- **Knowledge Graphs**: Structured connections and context

```mermaid
graph LR
    subgraph "Lift OS Memory Service"
        MS[Memory Service<br/>Port 8003]
        KC[KSE Client]
        MH[Memory Handler]
        CH[Context Handler]
    end
    
    subgraph "KSE Memory SDK"
        NE[Neural Embeddings]
        CS[Conceptual Spaces]
        KG[Knowledge Graphs]
        HF[Hybrid Fusion]
    end
    
    subgraph "Module Integration"
        CAUSAL[Causal Module]
        EVAL[Eval Module]
        AGENT[Agentic Module]
    end
    
    MS --> KC
    KC --> NE
    KC --> CS
    KC --> KG
    NE --> HF
    CS --> HF
    KG --> HF
    
    CAUSAL --> MS
    EVAL --> MS
    AGENT --> MS
    
    MH --> CH
    CH --> KC
```

### Memory Service Implementation

```python
# services/memory/kse_client.py
from kse_memory import KSEMemory
from typing import Dict, List, Any, Optional
import asyncio

class LiftKSEClient:
    def __init__(self):
        self.kse = KSEMemory()
        self.org_contexts = {}
    
    async def initialize_org_memory(self, org_id: str, domain: str = None):
        """Initialize memory context for organization"""
        if org_id not in self.org_contexts:
            # Create org-specific memory space
            context = await self.kse.create_context(
                context_id=f"org_{org_id}",
                domain=domain or "general"
            )
            self.org_contexts[org_id] = context
        return self.org_contexts[org_id]
    
    async def hybrid_search(self, 
                          org_id: str, 
                          query: str, 
                          search_type: str = "hybrid",
                          limit: int = 10) -> List[Dict]:
        """Perform hybrid search using KSE capabilities"""
        context = await self.initialize_org_memory(org_id)
        
        if search_type == "neural":
            results = await self.kse.neural_search(query, context, limit=limit)
        elif search_type == "conceptual":
            results = await self.kse.conceptual_search(query, context, limit=limit)
        elif search_type == "knowledge":
            results = await self.kse.knowledge_search(query, context, limit=limit)
        else:  # hybrid
            results = await self.kse.hybrid_search(query, context, limit=limit)
        
        return results
    
    async def store_memory(self, 
                          org_id: str, 
                          content: str, 
                          metadata: Dict = None,
                          memory_type: str = "general") -> str:
        """Store content in org memory with metadata"""
        context = await self.initialize_org_memory(org_id)
        
        memory_id = await self.kse.store(
            content=content,
            context=context,
            metadata={
                "org_id": org_id,
                "memory_type": memory_type,
                "timestamp": datetime.utcnow().isoformat(),
                **(metadata or {})
            }
        )
        return memory_id
    
    async def get_memory_insights(self, org_id: str) -> Dict:
        """Get memory analytics and insights"""
        context = await self.initialize_org_memory(org_id)
        
        insights = await self.kse.analyze_context(context)
        return {
            "total_memories": insights.get("count", 0),
            "dominant_concepts": insights.get("concepts", []),
            "knowledge_density": insights.get("density", 0),
            "temporal_patterns": insights.get("temporal", {}),
            "semantic_clusters": insights.get("clusters", [])
        }
```

### Memory API Endpoints

```python
# services/memory/app.py
from fastapi import FastAPI, HTTPException, Depends
from .kse_client import LiftKSEClient
from shared.auth import get_current_user

app = FastAPI(title="Lift Memory Service")
kse_client = LiftKSEClient()

@app.post("/memory/search")
async def search_memory(
    query: str,
    search_type: str = "hybrid",
    limit: int = 10,
    user = Depends(get_current_user)
):
    """Search organization memory using KSE hybrid capabilities"""
    try:
        results = await kse_client.hybrid_search(
            org_id=user.org_id,
            query=query,
            search_type=search_type,
            limit=limit
        )
        return {"results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/store")
async def store_memory(
    content: str,
    metadata: dict = None,
    memory_type: str = "general",
    user = Depends(get_current_user)
):
    """Store content in organization memory"""
    try:
        memory_id = await kse_client.store_memory(
            org_id=user.org_id,
            content=content,
            metadata=metadata,
            memory_type=memory_type
        )
        return {"memory_id": memory_id, "status": "stored"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/insights")
async def get_insights(user = Depends(get_current_user)):
    """Get memory analytics and insights for organization"""
    try:
        insights = await kse_client.get_memory_insights(user.org_id)
        return insights
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## 🔧 Implementation Phases

### Phase 1: Foundation & Core Services (Week 1-2)

```mermaid
gantt
    title Lift OS Core Development Timeline
    dateFormat  YYYY-MM-DD
    section Phase 1: Foundation
    Project Setup           :p1-1, 2025-01-01, 2d
    Core Services          :p1-2, after p1-1, 5d
    Gateway Implementation :p1-3, after p1-2, 3d
    
    section Phase 2: Integration
    KSE Memory Integration :p2-1, after p1-3, 3d
    Auth & Billing        :p2-2, after p2-1, 4d
    
    section Phase 3: UI & Modules
    Next.js UI Shell      :p3-1, after p2-2, 5d
    Module Templates      :p3-2, after p3-1, 3d
    
    section Phase 4: Testing & Docs
    Integration Testing   :p4-1, after p3-2, 3d
    Documentation        :p4-2, after p4-1, 2d
```

#### 1.1 Project Initialization
- Set up repository structure
- Configure Docker Compose environment with all services
- Create shared libraries and utilities
- Set up development tooling (linting, formatting, pre-commit hooks)
- Install and configure KSE Memory SDK

#### 1.2 Core Service Development
- **Gateway Service**: FastAPI-based API gateway with routing, authentication middleware, and rate limiting
- **Auth Service**: JWT-based authentication with RBAC support and OAuth integration
- **Registry Service**: Module registration and discovery system
- **Observability Service**: Health checks, metrics collection, and logging aggregation
- **Memory Service**: KSE SDK wrapper with org-specific contexts

#### 1.3 Database Setup
- PostgreSQL for persistent data (users, billing, module registry)
- Redis for caching and session management
- Database migrations and seeding scripts
- KSE memory initialization per organization

### Phase 2: Integration & Business Logic (Week 3-4)

#### 2.1 KSE Memory Integration
- Implement LiftKSEClient with hybrid search capabilities
- Create org-specific memory contexts and initialization
- Build memory analytics and insights endpoints
- Integrate memory service with gateway routing
- Add memory-based user onboarding flow

#### 2.2 Billing & Subscription Management
- Stripe integration for payment processing
- Tiered access control per module with KSE usage tracking
- Usage metering and billing events (including memory operations)
- Subscription management APIs with memory quotas

#### 2.3 Advanced Gateway Features
- Token propagation to modules with memory context
- Request/response transformation with memory enrichment
- Circuit breaker patterns for KSE operations
- API versioning support

### Phase 3: Frontend & Module System (Week 5-6)

#### 3.1 Next.js UI Shell
- Authentication-aware routing with memory context
- Modular dashboard system with KSE insights
- Memory search interface and analytics visualization
- Component library for consistent UI
- Server-side rendering with API integration

#### 3.2 Module Templates & Integration
- Standardized module interface with memory access
- Health check and registration protocols
- Example implementations for each Lift app with KSE integration
- Module-specific UI component mounting
- Memory-aware module communication

#### 3.3 Developer Experience
- Hot reload for local development
- Comprehensive API documentation including memory endpoints
- Module development SDK with KSE utilities
- Integration testing framework with memory mocking

### Phase 4: Testing, Documentation & Deployment (Week 7)

#### 4.1 Testing Strategy
- Unit tests for all core services including memory operations
- Integration tests for module communication and KSE interactions
- End-to-end testing with Playwright including memory workflows
- Load testing for gateway performance and KSE scalability

#### 4.2 Documentation
- API documentation with OpenAPI/Swagger including memory endpoints
- Module integration guide with KSE best practices
- Deployment and operations manual
- Developer onboarding documentation with memory setup

## 🔌 Module Integration Pattern

```mermaid
sequenceDiagram
    participant Client
    participant Gateway
    participant Auth
    participant Memory
    participant Registry
    participant Module
    participant KSE
    
    Client->>Gateway: Request with JWT
    Gateway->>Auth: Validate Token
    Auth-->>Gateway: User Context + Org ID
    Gateway->>Memory: Get Memory Context
    Memory->>KSE: Initialize Org Context
    KSE-->>Memory: Memory Context
    Memory-->>Gateway: Context Info
    Gateway->>Registry: Get Module Info
    Registry-->>Gateway: Module Endpoint
    Gateway->>Module: Forward Request + Context + Memory
    Module->>Memory: Access/Store Memory
    Memory->>KSE: Hybrid Search/Store
    KSE-->>Memory: Results/Confirmation
    Memory-->>Module: Memory Response
    Module-->>Gateway: Response
    Gateway-->>Client: Final Response
```

### Module Registration Contract

```json
{
  "module_id": "lift_causal",
  "name": "Lift Causal",
  "version": "1.0.0",
  "base_url": "http://localhost:9001",
  "health_endpoint": "/health",
  "api_prefix": "/api/v1",
  "features": [
    "causal_modeling",
    "data_visualization", 
    "experiment_design"
  ],
  "memory_requirements": {
    "read_access": true,
    "write_access": true,
    "search_types": ["hybrid", "conceptual", "knowledge"],
    "memory_types": ["causal_models", "experiments", "insights"]
  },
  "ui_components": [
    {
      "name": "CausalDashboard",
      "path": "/dashboard",
      "permissions": ["causal:read"],
      "memory_context": "causal_models"
    }
  ],
  "permissions": [
    "causal:read",
    "causal:write", 
    "causal:admin",
    "memory:read",
    "memory:write"
  ]
}
```

### Module Memory Integration Example

```python
# modules/causal/memory_integration.py
import httpx
from typing import List, Dict, Any

class CausalMemoryClient:
    def __init__(self, memory_service_url: str, auth_token: str):
        self.base_url = memory_service_url
        self.headers = {"Authorization": f"Bearer {auth_token}"}
    
    async def store_causal_model(self, model_data: Dict, experiment_id: str):
        """Store causal model in memory with metadata"""
        payload = {
            "content": model_data["description"],
            "metadata": {
                "model_type": "causal",
                "experiment_id": experiment_id,
                "variables": model_data["variables"],
                "relationships": model_data["relationships"],
                "performance_metrics": model_data["metrics"]
            },
            "memory_type": "causal_models"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/memory/store",
                json=payload,
                headers=self.headers
            )
            return response.json()
    
    async def search_similar_models(self, query: str, limit: int = 5):
        """Search for similar causal models using hybrid search"""
        params = {
            "query": query,
            "search_type": "hybrid",
            "limit": limit
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/memory/search",
                params=params,
                headers=self.headers
            )
            return response.json()["results"]
    
    async def get_causal_insights(self):
        """Get memory insights specific to causal modeling"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/memory/insights",
                headers=self.headers
            )
            insights = response.json()
            
            # Filter for causal-specific insights
            return {
                "total_models": len([m for m in insights.get("semantic_clusters", []) 
                                   if "causal" in m.get("type", "")]),
                "common_variables": insights.get("dominant_concepts", []),
                "model_evolution": insights.get("temporal_patterns", {})
            }
```

## 🛡️ Security & Authentication Flow

```mermaid
flowchart TD
    A[User Login] --> B{Auth Method}
    B -->|OAuth| C[OAuth Provider]
    B -->|Email/Password| D[Local Auth]
    C --> E[JWT Generation]
    D --> E
    E --> F[Token with Claims + Org Context]
    F --> G[Gateway Middleware]
    G --> H{Token Valid?}
    H -->|Yes| I[Extract User + Org Context]
    H -->|No| J[401 Unauthorized]
    I --> K[Initialize Memory Context]
    K --> L[Check Permissions]
    L --> M{Has Access?}
    M -->|Yes| N[Forward to Module + Memory]
    M -->|No| O[403 Forbidden]
    N --> P[Module Response]
```

### JWT Claims Structure

```json
{
  "sub": "user_id_123",
  "org_id": "org_456", 
  "email": "user@example.com",
  "roles": ["user", "causal_analyst"],
  "permissions": [
    "causal:read",
    "causal:write",
    "memory:read",
    "memory:write"
  ],
  "memory_context": "org_456_context",
  "subscription_tier": "pro",
  "exp": 1640995200,
  "iat": 1640908800
}
```

## 📊 Observability & Monitoring

### Metrics Collection
- Request latency and throughput per service
- Module health and availability
- User authentication success/failure rates
- Billing events and subscription changes
- **KSE Memory Operations**: search latency, storage volume, context switches
- **Memory Usage Patterns**: search types, popular concepts, temporal access
- **Hybrid Search Performance**: neural vs conceptual vs knowledge graph effectiveness

### Memory-Specific Monitoring

```python
# services/observability/memory_metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Memory operation metrics
memory_searches = Counter('memory_searches_total', 'Total memory searches', ['org_id', 'search_type'])
memory_search_duration = Histogram('memory_search_duration_seconds', 'Memory search duration')
memory_storage_operations = Counter('memory_storage_total', 'Total memory storage operations', ['org_id', 'memory_type'])
active_memory_contexts = Gauge('active_memory_contexts', 'Number of active memory contexts')

class MemoryMetrics:
    @staticmethod
    def record_search(org_id: str, search_type: str, duration: float):
        memory_searches.labels(org_id=org_id, search_type=search_type).inc()
        memory_search_duration.observe(duration)
    
    @staticmethod
    def record_storage(org_id: str, memory_type: str):
        memory_storage_operations.labels(org_id=org_id, memory_type=memory_type).inc()
    
    @staticmethod
    def update_active_contexts(count: int):
        active_memory_contexts.set(count)
```

### Logging Strategy
- Structured JSON logging across all services
- Correlation IDs for request tracing including memory operations
- Centralized log aggregation in observability service
- Memory operation audit trails for compliance
- Different log levels for development vs production

### Health Checks
- Service-level health endpoints
- Module availability monitoring
- Database connection health
- **KSE SDK connectivity and performance**
- **Memory context health per organization**
- External dependency status (Stripe, OAuth providers, etc.)

## 🚀 Development Workflow

### Local Development Setup
```bash
# Clone and setup
git clone <repo-url> lift-os-core
cd lift-os-core

# Environment setup
cp .env.example .env
# Edit .env with your configuration including KSE settings

# Install KSE Memory SDK
pip install kse-memory-sdk

# Start all services
make dev-up

# Initialize sample memory contexts
make init-memory

# Access points:
# - UI Shell: http://localhost:3000
# - API Gateway: http://localhost:8000
# - API Docs: http://localhost:8000/docs
# - Memory Service: http://localhost:8003
# - Memory Docs: http://localhost:8003/docs
```

### Environment Variables (.env.example)
```bash
# Core Configuration
ENVIRONMENT=development
SECRET_KEY=your-secret-key-here
DATABASE_URL=postgresql://user:password@localhost:5432/lift_os
REDIS_URL=redis://localhost:6379

# KSE Memory Configuration
KSE_API_KEY=your-kse-api-key
KSE_ENVIRONMENT=development
KSE_DEFAULT_DOMAIN=general
KSE_MAX_CONTEXTS=100

# Authentication
JWT_SECRET=your-jwt-secret
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# OAuth Providers
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret

# Billing
STRIPE_SECRET_KEY=your-stripe-secret-key
STRIPE_WEBHOOK_SECRET=your-stripe-webhook-secret

# Observability
PROMETHEUS_PORT=9090
GRAFANA_PORT=3001
LOG_LEVEL=INFO
```

### Module Development
```bash
# Create new module from template
make create-module name=my-new-module

# Register module with core (includes memory access)
curl -X POST http://localhost:8005/modules \
  -H "Content-Type: application/json" \
  -d @modules/my-new-module/module.json

# Test module integration including memory
make test-module name=my-new-module

# Test memory integration specifically
make test-memory module=my-new-module
```

### Docker Compose Configuration

```yaml
# docker-compose.yml (key services)
version: '3.8'

services:
  gateway:
    build: ./services/gateway
    ports:
      - "8000:8000"
    environment:
      - MEMORY_SERVICE_URL=http://memory:8003
    depends_on:
      - auth
      - memory
      - registry

  memory:
    build: ./services/memory
    ports:
      - "8003:8003"
    environment:
      - KSE_API_KEY=${KSE_API_KEY}
      - DATABASE_URL=${DATABASE_URL}
    depends_on:
      - postgres
      - redis

  ui-shell:
    build: ./ui-shell
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
      - NEXT_PUBLIC_MEMORY_URL=http://localhost:8003

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: lift_os
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

## 🔄 Future Scalability Considerations

### Kubernetes Migration Path
- All services designed as stateless containers
- ConfigMaps for environment-specific configuration including KSE settings
- Helm charts for deployment orchestration
- Horizontal Pod Autoscaling based on metrics including memory operations
- **KSE context management across pods with Redis coordination**

### Event-Driven Architecture
- Redis Streams for inter-service communication
- Event sourcing for audit trails including memory operations
- Pub/sub patterns for module notifications and memory updates
- Webhook support for external integrations
- **Memory event streaming for real-time insights**

### Multi-tenancy Support
- Org-level data isolation including memory contexts
- Per-tenant module configurations and memory quotas
- Billing and usage tracking per organization including memory costs
- Custom domain support for white-labeling
- **Isolated KSE contexts per organization with shared infrastructure**

### Advanced Memory Features
- **Federated learning across organizations** (privacy-preserving)
- **Temporal reasoning** for time-aware knowledge evolution
- **Cross-module memory sharing** with permission controls
- **Memory compression and archival** for cost optimization
- **Real-time memory insights** and recommendation engines

## 📚 API Documentation Structure

###