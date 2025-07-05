# Developer Onboarding Guide

## Welcome to Lift OS Core Development

This guide will help you get started with developing on the Lift OS Core platform, including setting up your development environment, understanding the architecture, and contributing effectively.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Development Environment Setup](#development-environment-setup)
3. [Architecture Overview](#architecture-overview)
4. [Memory System & KSE Setup](#memory-system--kse-setup)
5. [Development Workflow](#development-workflow)
6. [Code Standards & Guidelines](#code-standards--guidelines)
7. [Testing Guidelines](#testing-guidelines)
8. [Debugging & Troubleshooting](#debugging--troubleshooting)
9. [Contributing Guidelines](#contributing-guidelines)
10. [Resources & Support](#resources--support)

## Prerequisites

### Required Knowledge
- **Python 3.11+**: Advanced proficiency
- **FastAPI**: Web framework experience
- **Docker & Docker Compose**: Container orchestration
- **PostgreSQL**: Database design and optimization
- **Redis**: Caching and session management
- **Git**: Version control and collaboration
- **REST APIs**: Design and implementation
- **Async/Await**: Asynchronous programming patterns

### Recommended Knowledge
- **Kubernetes**: Container orchestration (for production)
- **Vector Databases**: Semantic search and embeddings
- **JWT**: Authentication and authorization
- **Prometheus/Grafana**: Monitoring and observability
- **pytest**: Testing frameworks
- **OpenAPI/Swagger**: API documentation

## Development Environment Setup

### 1. System Requirements

```bash
# Minimum requirements for development
CPU: 4 cores, 2.5GHz
RAM: 8GB (16GB recommended)
Storage: 50GB SSD
OS: Linux, macOS, or Windows with WSL2
```

### 2. Install Required Tools

```bash
# Install Python 3.11+
# Ubuntu/Debian
sudo apt update
sudo apt install python3.11 python3.11-pip python3.11-venv

# macOS (using Homebrew)
brew install python@3.11

# Windows (using chocolatey)
choco install python311

# Install Docker and Docker Compose
# Follow official Docker installation guide for your OS
# https://docs.docker.com/get-docker/

# Install Git
sudo apt install git  # Ubuntu/Debian
brew install git      # macOS
choco install git     # Windows

# Install additional tools
pip install poetry    # Python dependency management
npm install -g @apidevtools/swagger-cli  # API documentation
```

### 3. Clone and Setup Repository

```bash
# Clone the repository
git clone https://github.com/liftos/lift-os-core.git
cd lift-os-core

# Create Python virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Copy environment configuration
cp .env.example .env
# Edit .env with your local configuration
```

### 4. Local Development Setup

```bash
# Start local development environment
docker-compose -f docker-compose.dev.yml up -d

# Verify services are running
docker-compose ps

# Check service health
curl http://localhost:8000/health  # Gateway
curl http://localhost:8001/health  # Auth
curl http://localhost:8003/health  # Memory
curl http://localhost:8005/health  # Registry

# Run database migrations
python scripts/migrate_database.py

# Seed development data
python scripts/seed_dev_data.py
```

### 5. IDE Configuration

#### VS Code Setup

```json
// .vscode/settings.json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true,
  "python.linting.mypyEnabled": true,
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length", "88"],
  "python.sortImports.args": ["--profile", "black"],
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    ".pytest_cache": true,
    ".mypy_cache": true
  }
}
```

```json
// .vscode/extensions.json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.flake8",
    "ms-python.mypy-type-checker",
    "ms-python.black-formatter",
    "ms-python.isort",
    "ms-vscode.docker",
    "redhat.vscode-yaml",
    "ms-vscode.vscode-json",
    "humao.rest-client"
  ]
}
```

#### PyCharm Setup

```python
# PyCharm configuration
# 1. Open project in PyCharm
# 2. Configure Python interpreter: Settings > Project > Python Interpreter
# 3. Select existing virtual environment: ./venv/bin/python
# 4. Configure code style: Settings > Editor > Code Style > Python
#    - Set line length to 88
#    - Enable "Black" formatter
# 5. Enable type checking: Settings > Editor > Inspections > Python > Type checker
```

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Applications                      │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP/HTTPS
┌─────────────────────┼───────────────────────────────────────┐
│                 Gateway Service                             │
│  ┌─────────────────┼─────────────────────────────────────┐  │
│  │     Routing     │    Auth Middleware    │   Proxy    │  │
│  └─────────────────┼─────────────────────────────────────┘  │
└─────────────────────┼───────────────────────────────────────┘
                      │ Internal HTTP
┌─────────────────────┼───────────────────────────────────────┐
│                Core Services                                │
│  ┌─────────────┐   │   ┌─────────────┐   ┌─────────────┐   │
│  │    Auth     │   │   │   Memory    │   │  Registry   │   │
│  │  Service    │   │   │  Service    │   │  Service    │   │
│  │             │   │   │    (KSE)    │   │             │   │
│  └─────────────┘   │   └─────────────┘   └─────────────┘   │
└─────────────────────┼───────────────────────────────────────┘
                      │
┌─────────────────────┼───────────────────────────────────────┐
│                Data Layer                                   │
│  ┌─────────────┐   │   ┌─────────────┐   ┌─────────────┐   │
│  │ PostgreSQL  │   │   │    Redis    │   │  Vector DB  │   │
│  │ (Primary)   │   │   │   (Cache)   │   │    (KSE)    │   │
│  └─────────────┘   │   └─────────────┘   └─────────────┘   │
└─────────────────────┼───────────────────────────────────────┘
```

### Core Components

#### 1. Gateway Service (`services/gateway/`)
- **Purpose**: API gateway and request routing
- **Key Features**: Authentication middleware, rate limiting, service discovery
- **Technologies**: FastAPI, aiohttp, Redis

#### 2. Auth Service (`services/auth/`)
- **Purpose**: User authentication and authorization
- **Key Features**: JWT tokens, user management, organization context
- **Technologies**: FastAPI, PostgreSQL, bcrypt, JWT

#### 3. Memory Service (`services/memory/`)
- **Purpose**: Knowledge Storage Engine (KSE) and memory operations
- **Key Features**: Semantic search, vector storage, memory context
- **Technologies**: FastAPI, Vector DB (Qdrant), Redis, PostgreSQL

#### 4. Registry Service (`services/registry/`)
- **Purpose**: Module registration and discovery
- **Key Features**: Module lifecycle, health monitoring, service mesh
- **Technologies**: FastAPI, PostgreSQL, Redis

### Shared Libraries (`shared/`)

```python
# shared/health/health_checks.py
class HealthChecker:
    """Standardized health checking across all services"""
    
    async def get_health_status(self) -> Dict[str, Any]
    async def get_readiness_status(self, external_checks=None) -> Dict[str, Any]

# shared/security/security_manager.py
class SecurityManager:
    """Centralized security operations"""
    
    def create_jwt_token(self, payload: Dict[str, Any]) -> str
    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]
    def check_rate_limit(self, client_ip: str) -> bool

# shared/logging/structured_logger.py
class LiftOSLogger:
    """Structured logging with correlation IDs"""
    
    def info(self, message: str, **kwargs)
    def error(self, message: str, **kwargs)
    def warning(self, message: str, **kwargs)

# shared/config/secrets_manager.py
class SecretsManager:
    """Multi-backend secrets management"""
    
    def get_secret(self, key: str) -> Optional[str]
    def set_secret(self, key: str, value: str) -> bool
```

## Memory System & KSE Setup

### Understanding the Knowledge Storage Engine (KSE)

The KSE is the core intelligence layer of Lift OS, providing:

1. **Semantic Memory Storage**: Store and retrieve contextual information
2. **Vector Search**: Find semantically similar content
3. **Memory Context**: Maintain user and organization-specific memory spaces
4. **Hybrid Search**: Combine semantic and keyword search

### Memory Data Flow

```python
# Example: Storing user interaction memory
memory_data = {
    "key": "user_interaction_2025_01_07",
    "value": {
        "action": "document_created",
        "document_type": "report",
        "user_feedback": "positive",
        "context": "quarterly_review"
    },
    "metadata": {
        "type": "user_interaction",
        "tags": ["document", "report", "quarterly"],
        "confidence": 0.95
    },
    "context": {
        "user_id": "user_123",
        "org_id": "org_456",
        "session_id": "session_789"
    }
}

# Store in KSE
memory_id = await memory_service.store_memory(memory_data)

# Search semantically
results = await memory_service.search_memories(
    query="user document creation feedback",
    context={"user_id": "user_123", "org_id": "org_456"},
    limit=10
)
```

### Setting Up Local KSE Development

```bash
# Start vector database (Qdrant)
docker run -p 6333:6333 qdrant/qdrant:latest

# Configure memory service
export KSE_VECTOR_DB_URL=http://localhost:6333
export KSE_COLLECTION_NAME=liftos_memories
export KSE_VECTOR_SIZE=384  # Sentence transformer dimension

# Initialize KSE collections
python scripts/init_kse_collections.py

# Test KSE connectivity
python scripts/test_kse_connection.py
```

### Memory Development Patterns

```python
# Pattern 1: Contextual Memory Storage
async def store_user_action(user_id: str, org_id: str, action_data: Dict[str, Any]):
    """Store user action with proper context"""
    
    memory_key = f"user_action_{user_id}_{datetime.utcnow().isoformat()}"
    
    memory_data = {
        "key": memory_key,
        "value": {
            **action_data,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "user_interface"
        },
        "metadata": {
            "type": "user_action",
            "tags": extract_tags(action_data),
            "indexed_at": datetime.utcnow().isoformat()
        },
        "context": {
            "user_id": user_id,
            "org_id": org_id,
            "session_id": get_current_session_id()
        }
    }
    
    return await memory_service.store_memory(memory_data)

# Pattern 2: Intelligent Memory Retrieval
async def get_user_insights(user_id: str, org_id: str, insight_type: str):
    """Retrieve contextual insights for user"""
    
    # Build semantic query
    query = f"user behavior patterns {insight_type} insights"
    
    # Search with context
    results = await memory_service.search_memories(
        query=query,
        context={"user_id": user_id, "org_id": org_id},
        filters={
            "type": "user_action",
            "date_range": {
                "start": (datetime.utcnow() - timedelta(days=30)).isoformat(),
                "end": datetime.utcnow().isoformat()
            }
        },
        limit=50
    )
    
    # Process and aggregate insights
    insights = process_memory_results(results)
    
    return {
        "user_id": user_id,
        "insight_type": insight_type,
        "insights": insights,
        "confidence_score": calculate_confidence(results),
        "generated_at": datetime.utcnow().isoformat()
    }

# Pattern 3: Memory Context Management
async def initialize_user_memory_context(user_id: str, org_id: str):
    """Initialize memory context for new user"""
    
    context_data = {
        "user_id": user_id,
        "org_id": org_id,
        "created_at": datetime.utcnow().isoformat(),
        "memory_preferences": {
            "retention_days": 365,
            "privacy_level": "standard",
            "indexing_enabled": True
        }
    }
    
    await memory_service.create_memory_context(context_data)
```

## Development Workflow

### 1. Feature Development Process

```bash
# 1. Create feature branch
git checkout -b feature/user-memory-insights

# 2. Implement feature
# - Write code following standards
# - Add comprehensive tests
# - Update documentation

# 3. Run local tests
pytest tests/unit/
pytest tests/integration/
python scripts/test_production_features.py

# 4. Code quality checks
black .                    # Format code
isort .                   # Sort imports
flake8 .                  # Lint code
mypy .                    # Type checking

# 5. Commit changes
git add .
git commit -m "feat: add user memory insights functionality

- Implement semantic search for user behavior patterns
- Add insight aggregation and confidence scoring
- Include comprehensive test coverage
- Update API documentation

Closes #123"

# 6. Push and create PR
git push origin feature/user-memory-insights
# Create pull request via GitHub/GitLab
```

### 2. Testing Workflow

```bash
# Run specific test categories
pytest -m unit                    # Unit tests only
pytest -m integration            # Integration tests only
pytest -m e2e                    # End-to-end tests only
pytest -m "not slow"             # Exclude slow tests

# Run tests with coverage
pytest --cov=src --cov-report=html

# Run load tests
pytest -m load tests/test_load_performance.py

# Run specific service tests
pytest tests/unit/test_memory_service.py -v
pytest tests/integration/test_kse_integration.py -v
```

### 3. Local Development Commands

```bash
# Start development environment
make dev-start

# Stop development environment
make dev-stop

# Restart specific service
make restart-service SERVICE=memory

# View logs
make logs SERVICE=gateway
make logs-follow SERVICE=memory

# Run database migrations
make migrate

# Seed development data
make seed-data

# Generate API documentation
make docs-generate

# Run code quality checks
make lint
make format
make type-check

# Run all tests
make test

# Clean development environment
make clean
```

### 4. Debugging Workflow

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use debugger
import pdb; pdb.set_trace()  # Python debugger
import ipdb; ipdb.set_trace()  # Enhanced debugger

# Async debugging
import asyncio
import aiotools

async def debug_async_function():
    # Use aiotools for async debugging
    await aiotools.create_task_with_debug(your_async_function())

# Memory service debugging
from shared.logging.structured_logger import LiftOSLogger

logger = LiftOSLogger("debug_session", correlation_id="debug_123")

async def debug_memory_operation():
    logger.info("Starting memory operation debug")
    
    try:
        result = await memory_service.store_memory(test_data)
        logger.info("Memory stored successfully", memory_id=result)
    except Exception as e:
        logger.error("Memory operation failed", error=str(e), traceback=traceback.format_exc())
```

## Code Standards & Guidelines

### 1. Python Code Style

```python
# Follow PEP 8 with Black formatting
# Line length: 88 characters
# Use type hints for all functions

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import asyncio

class UserMemoryService:
    """Service for managing user memory operations.
    
    This service provides functionality for storing, retrieving,
    and analyzing user memory data using the KSE.
    """
    
    def __init__(self, kse_client: KSEClient, logger: LiftOSLogger):
        self.kse_client = kse_client
        self.logger = logger
    
    async def store_user_memory(
        self,
        user_id: str,
        memory_data: Dict[str, Any],
        context: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """Store user memory with proper validation and context.
        
        Args:
            user_id: Unique identifier for the user
            memory_data: Memory content to store
            context: Additional context information
            
        Returns:
            Memory ID if successful, None otherwise
            
        Raises:
            ValidationError: If memory data is invalid
            KSEError: If storage operation fails
        """
        # Validate input
        if not user_id or not memory_data:
            raise ValidationError("User ID and memory data are required")
        
        # Add correlation ID for tracing
        correlation_id = context.get("correlation_id") if context else None
        self.logger.info(
            "Storing user memory",
            user_id=user_id,
            memory_type=memory_data.get("type"),
            correlation_id=correlation_id
        )
        
        try:
            # Prepare memory for storage
            prepared_memory = self._prepare_memory_data(user_id, memory_data, context)
            
            # Store in KSE
            memory_id = await self.kse_client.store_memory(prepared_memory)
            
            self.logger.info(
                "User memory stored successfully",
                user_id=user_id,
                memory_id=memory_id,
                correlation_id=correlation_id
            )
            
            return memory_id
            
        except Exception as e:
            self.logger.error(
                "Failed to store user memory",
                user_id=user_id,
                error=str(e),
                correlation_id=correlation_id
            )
            raise KSEError(f"Memory storage failed: {e}") from e
    
    def _prepare_memory_data(
        self,
        user_id: str,
        memory_data: Dict[str, Any],
        context: Optional[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Prepare memory data for storage with proper structure."""
        return {
            "key": f"user_memory_{user_id}_{datetime.utcnow().isoformat()}",
            "value": memory_data,
            "metadata": {
                "type": "user_memory",
                "user_id": user_id,
                "created_at": datetime.utcnow().isoformat(),
                "tags": self._extract_tags(memory_data)
            },
            "context": context or {}
        }
    
    def _extract_tags(self, memory_data: Dict[str, Any]) -> List[str]:
        """Extract semantic tags from memory data."""
        tags = []
        
        # Extract entity types
        if "action" in memory_data:
            tags.append(f"action_{memory_data['action']}")
        
        if "category" in memory_data:
            tags.append(f"category_{memory_data['category']}")
        
        return tags
```

### 2. API Design Standards

```python
# FastAPI endpoint design
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from typing import List, Optional

router = APIRouter(prefix="/memory", tags=["Memory Operations"])

class MemoryStoreRequest(BaseModel):
    """Request model for storing memory."""
    
    key: str = Field(..., description="Unique memory identifier", min_length=1, max_length=255)
    value: Dict[str, Any] = Field(..., description="Memory content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "key": "user_preference_theme",
                "value": {"theme": "dark", "language": "en"},
                "metadata": {"type": "user_preference", "priority": "high"}
            }
        }

class MemoryStoreResponse(BaseModel):
    """Response model for memory storage."""
    
    memory_id: str = Field(..., description="Generated memory identifier")
    stored_at: datetime = Field(..., description="Storage timestamp")
    kse_indexed: bool = Field(..., description="Whether memory was indexed in KSE")

@router.post(
    "/store",
    response_model=MemoryStoreResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Store Memory",
    description="Store a memory item in the Knowledge Storage Engine with semantic indexing"
)
async def store_memory(
    request: MemoryStoreRequest,
    user_context: Dict[str, Any] = Depends(get_authenticated_user),
    memory_service: UserMemoryService = Depends(get_memory_service)
) -> MemoryStoreResponse:
    """Store memory with proper validation and error handling."""
    
    try:
        memory_id = await memory_service.store_user_memory(
            user_id=user_context["user_id"],
            memory_data=request.dict(),
            context={
                "org_id": user_context["org_id"],
                "correlation_id": get_correlation_id()
            }
        )
        
        if not memory_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to store memory"
            )
        
        return MemoryStoreResponse(
            memory_id=memory_id,
            stored_at=datetime.utcnow(),
            kse_indexed=True
        )
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid memory data: {e}"
        )
    except KSEError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Memory storage failed: {e}"
        )
```

### 3. Error Handling Standards

```python
# Custom exception hierarchy
class LiftOSError(Exception):
    """Base exception for Lift OS Core."""
    pass

class ValidationError(LiftOSError):
    """Raised when input validation fails."""
    pass

class AuthenticationError(LiftOSError):
    """Raised when authentication fails."""
    pass

class AuthorizationError(LiftOSError):
    """Raised when authorization fails."""
    pass

class KSEError(LiftOSError):
    """Raised when KSE operations fail."""
    pass

class ServiceUnavailableError(LiftOSError):
    """Raised when external service is unavailable."""
    pass

# Error handling middleware
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse

async def error_handling_middleware(request: Request, call_next):
    """Global error handling middleware."""
    
    try:
        response = await call_next(request)
        return response
        
    except ValidationError as e:
        return JSONResponse(
            status_code=400,
            content={
                "error": "validation_error",
                "message": str(e),
                "correlation_id": get_correlation_id()
            }
        )
    
    except AuthenticationError as e:
        return JSONResponse(
            status_code=401,
            content={
                "error": "authentication_error",
                "message": "Invalid or missing authentication",
                "correlation_id": get_correlation_id()
            }
        )
    
    except Exception as e:
        logger.error(
            "Unhandled exception",
            error=str(e),
            traceback=traceback.format_exc(),
            correlation_id=get_correlation_id()
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred",
                "correlation_id": get_correlation_id()
            }
        )
```

## Testing Guidelines

### 1. Test Structure

```python
# tests/unit/test_memory_service.py
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.services.memory_service import UserMemoryService
from src.exceptions import ValidationError, KSEError
from shared.logging.structured_logger import LiftOSLogger

class TestUserMemoryService:
    """Unit tests for UserMemoryService."""
    
    @pytest.fixture
    def mock_kse_client(self):
        """Mock KSE client for testing."""
        client = Mock()
        client.store_memory = AsyncMock(return_value="memory_123")
        client.search_memories = AsyncMock(return_value=[])
        return client
    
    @pytest.fixture
    def mock_logger(self):
        """Mock logger for testing."""
        return Mock(spec=LiftOSLogger)
    
    @pytest.fixture
    def memory_service(self, mock_kse_client, mock_logger):
        """Create memory service instance for testing."""
        return UserMemoryService(mock_kse_client, mock_logger)
    
    @pytest.mark.asyncio
    async def test_store_user_memory_success(self, memory_service, mock_kse_client):
        """Test successful memory storage."""
        
        # Arrange
        user_id = "user_123"
        memory_data = {"action": "login", "timestamp": "2025-01-07T12:00:00Z"}
        context = {"org_id": "org_456"}
        
        # Act
        result = await memory_service.store_user_memory(user_id, memory_data, context)
        
        # Assert
        assert result == "memory_123"
        mock_kse_client.store_memory.assert_called_once()
        
        # Verify prepared data structure
        call_args = mock_kse_client.store_memory.call_args[0][0]
        assert call_args["metadata"]["user_id"] == user_id
        assert call_args["metadata"]["type"] == "user_memory"
        assert call_args["value"] == memory_data
    
    @pytest.mark.asyncio
    async def test_store_user_memory_validation_error(self, memory_service):
        """Test memory storage with invalid input."""
        
        # Act & Assert
        with pytest.raises(ValidationError, match="User ID and memory data are required"):
            await memory_service.store_user_memory("", {})
    
    @pytest.mark.asyncio
    async def test_store_user_memory_kse_error(self, memory_service, mock_kse_client):
        """Test memory storage with KSE failure."""
        
        # Arrange
        mock_kse_client.store_memory.side_effect = Exception("KSE connection failed")
        
        # Act & Assert
        with pytest.raises(KSEError, match="Memory storage failed"):
            await memory_service.store_user_memory("user_123", {"test": "data"})
    
    def test_extract_tags(self, memory_service):
        """Test tag extraction from memory data."""
        
        # Arrange
        memory_data = {
            "action": "document_create",
            "category": "reports",
            "other_field": "value"
        }
        
        # Act
        tags = memory_service._extract_tags(memory_data)
        
        # Assert
        assert "action_document_create" in tags
        assert "category_reports" in tags
        assert len(tags) == 2
```

### 2. Integration Testing

```python
# tests/integration/test_memory_integration.py
import pytest
import aiohttp
from testcontainers import DockerCompose

@pytest.mark.integration
class TestMemoryIntegration:
    """Integration tests for memory service."""
    
    @pytest.fixture(scope="class")
    def docker_services(self):
        """Start required services for integration testing."""
        
        with DockerCompose(".", compose_file_name="docker-compose.test.yml") as compose:
            # Wait for services to be ready
            compose.wait_for("http://localhost:8003/health")  # Memory service
            compose.wait_for("http://localhost:6333/health")  # Vector DB
            yield compose
    
    @pytest.mark.asyncio
    async def test_memory_storage_and_retrieval(self, docker_services):
        """Test complete memory storage and retrieval workflow."""
        
        # Prepare test data
        memory_data = {
            "key": "integration_test_memory",
            "value": {
                "action": "test_action",
                "content": "This is integration test content",
                "metadata": {"test": True}
            },
            "context": {
                "user_id": "test_user",
                "org_id": "test_org"
            }
        }
        
        async with aiohttp.ClientSession() as session:
            # Store memory
            async with session.post(
                "http://localhost:8003/memory/store",
                json=memory_data,
                headers={"Authorization": "Bearer test_token"}
            ) as response:
                assert response.status == 201
                store_result = await response.json()
                memory_id = store_result["memory_id"]
            
            # Search for stored memory
            search_data = {
                "query": "integration test content",
                "context": {"user_id": "test_user", "org_id": "test_org"},
                "limit": 10
            }
            
            async with session.post(
                "http://localhost:8003/memory/search",
                json=search_data,
                headers={"Authorization": "Bearer test_token"}
            ) as response:
                assert response.status == 200
                search_result = await response.json()
                
                # Verify search results
                assert len(search_result["results"]) > 0
                assert any(
                    result["memory_id"]