# Lift OS Core - API Documentation

This document provides comprehensive API documentation for all Lift OS Core services.

## 🌐 Base URLs

### Development
- Gateway: `http://localhost:8000`
- Auth Service: `http://localhost:8001`
- Memory Service: `http://localhost:8002`
- Registry Service: `http://localhost:8003`
- Billing Service: `http://localhost:8004`
- Observability Service: `http://localhost:8005`

### Production
- Gateway: `https://api.your-domain.com`
- All services are accessed through the gateway in production

## 🔐 Authentication

All API requests (except public endpoints) require authentication via JWT tokens.

### Headers
```http
Authorization: Bearer <jwt_token>
Content-Type: application/json
```

### Getting a Token
```http
POST /api/auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "password123"
}
```

Response:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 86400,
  "user": {
    "id": "user_123",
    "email": "user@example.com",
    "name": "John Doe"
  }
}
```

## 📋 Gateway Service API

The Gateway Service acts as the main entry point and router for all API requests.

### Health Check
```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "services": {
    "auth": "healthy",
    "memory": "healthy",
    "registry": "healthy",
    "billing": "healthy",
    "observability": "healthy"
  }
}
```

### Service Status
```http
GET /api/status
Authorization: Bearer <token>
```

Response:
```json
{
  "gateway": {
    "status": "healthy",
    "version": "1.0.0",
    "uptime": 3600
  },
  "services": {
    "auth": {
      "status": "healthy",
      "response_time": 45
    },
    "memory": {
      "status": "healthy",
      "response_time": 32
    }
  }
}
```

## 🔑 Authentication Service API

### Register User
```http
POST /api/auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "password123",
  "name": "John Doe",
  "terms_accepted": true
}
```

Response:
```json
{
  "user": {
    "id": "user_123",
    "email": "user@example.com",
    "name": "John Doe",
    "created_at": "2024-01-15T10:30:00Z"
  },
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

### Login
```http
POST /api/auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "password123"
}
```

### OAuth Login
```http
GET /api/auth/oauth/{provider}?redirect_uri=https://your-app.com/callback
```

Supported providers: `google`, `github`, `microsoft`

### Refresh Token
```http
POST /api/auth/refresh
Authorization: Bearer <refresh_token>
```

### Get Current User
```http
GET /api/auth/me
Authorization: Bearer <token>
```

Response:
```json
{
  "id": "user_123",
  "email": "user@example.com",
  "name": "John Doe",
  "avatar_url": "https://example.com/avatar.jpg",
  "created_at": "2024-01-15T10:30:00Z",
  "last_login": "2024-01-15T10:30:00Z",
  "subscription": {
    "plan": "pro",
    "status": "active"
  }
}
```

### Update Profile
```http
PUT /api/auth/profile
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "John Smith",
  "avatar_url": "https://example.com/new-avatar.jpg"
}
```

### Change Password
```http
POST /api/auth/change-password
Authorization: Bearer <token>
Content-Type: application/json

{
  "current_password": "oldpassword",
  "new_password": "newpassword123"
}
```

### Logout
```http
POST /api/auth/logout
Authorization: Bearer <token>
```

## 🧠 Memory Service API

### Store Memory
```http
POST /api/memory/store
Authorization: Bearer <token>
Content-Type: application/json

{
  "content": "This is important information to remember",
  "context": "project_planning",
  "metadata": {
    "source": "meeting_notes",
    "priority": "high",
    "tags": ["planning", "strategy"]
  }
}
```

Response:
```json
{
  "memory_id": "mem_123",
  "content": "This is important information to remember",
  "context": "project_planning",
  "created_at": "2024-01-15T10:30:00Z",
  "embedding_id": "emb_456"
}
```

### Retrieve Memories
```http
GET /api/memory/retrieve?query=project planning&limit=10&context=project_planning
Authorization: Bearer <token>
```

Response:
```json
{
  "memories": [
    {
      "memory_id": "mem_123",
      "content": "This is important information to remember",
      "context": "project_planning",
      "relevance_score": 0.95,
      "created_at": "2024-01-15T10:30:00Z",
      "metadata": {
        "source": "meeting_notes",
        "priority": "high",
        "tags": ["planning", "strategy"]
      }
    }
  ],
  "total": 1,
  "query": "project planning"
}
```

### Search Memories
```http
POST /api/memory/search
Authorization: Bearer <token>
Content-Type: application/json

{
  "query": "project planning strategies",
  "filters": {
    "context": "project_planning",
    "tags": ["strategy"],
    "date_range": {
      "start": "2024-01-01T00:00:00Z",
      "end": "2024-01-31T23:59:59Z"
    }
  },
  "limit": 20,
  "offset": 0
}
```

### Get Memory by ID
```http
GET /api/memory/{memory_id}
Authorization: Bearer <token>
```

### Update Memory
```http
PUT /api/memory/{memory_id}
Authorization: Bearer <token>
Content-Type: application/json

{
  "content": "Updated content",
  "metadata": {
    "priority": "medium"
  }
}
```

### Delete Memory
```http
DELETE /api/memory/{memory_id}
Authorization: Bearer <token>
```

### List Contexts
```http
GET /api/memory/contexts
Authorization: Bearer <token>
```

Response:
```json
{
  "contexts": [
    {
      "name": "project_planning",
      "memory_count": 45,
      "last_updated": "2024-01-15T10:30:00Z"
    },
    {
      "name": "general",
      "memory_count": 123,
      "last_updated": "2024-01-15T09:15:00Z"
    }
  ]
}
```

## 📦 Registry Service API

### Register Module
```http
POST /api/registry/modules
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "my-custom-module",
  "version": "1.0.0",
  "description": "A custom module for data processing",
  "image": "my-registry/my-module:1.0.0",
  "config": {
    "port": 8080,
    "environment": {
      "API_KEY": "secret"
    }
  },
  "ui_components": [
    {
      "name": "Dashboard",
      "path": "/dashboard",
      "icon": "chart-bar"
    }
  ]
}
```

Response:
```json
{
  "module_id": "mod_123",
  "name": "my-custom-module",
  "version": "1.0.0",
  "status": "registered",
  "created_at": "2024-01-15T10:30:00Z",
  "registry_url": "https://registry.lift-os.com/modules/my-custom-module"
}
```

### List Modules
```http
GET /api/registry/modules?category=ai&status=active&limit=20
Authorization: Bearer <token>
```

Response:
```json
{
  "modules": [
    {
      "module_id": "mod_123",
      "name": "lift-causal",
      "version": "1.0.0",
      "description": "Causal modeling and analysis",
      "category": "ai",
      "status": "active",
      "downloads": 1250,
      "rating": 4.8,
      "created_at": "2024-01-15T10:30:00Z",
      "updated_at": "2024-01-15T10:30:00Z"
    }
  ],
  "total": 1,
  "page": 1,
  "limit": 20
}
```

### Get Module Details
```http
GET /api/registry/modules/{module_id}
Authorization: Bearer <token>
```

### Install Module
```http
POST /api/registry/modules/{module_id}/install
Authorization: Bearer <token>
Content-Type: application/json

{
  "environment": {
    "API_KEY": "user_specific_key"
  },
  "scaling": {
    "min_instances": 1,
    "max_instances": 3
  }
}
```

### Uninstall Module
```http
DELETE /api/registry/modules/{module_id}/install
Authorization: Bearer <token>
```

### List User Modules
```http
GET /api/registry/user/modules
Authorization: Bearer <token>
```

### Update Module
```http
PUT /api/registry/modules/{module_id}
Authorization: Bearer <token>
Content-Type: application/json

{
  "version": "1.1.0",
  "description": "Updated description",
  "config": {
    "new_feature": true
  }
}
```

## 💳 Billing Service API

### Get Subscription
```http
GET /api/billing/subscription
Authorization: Bearer <token>
```

Response:
```json
{
  "subscription": {
    "id": "sub_123",
    "plan": "pro",
    "status": "active",
    "current_period_start": "2024-01-01T00:00:00Z",
    "current_period_end": "2024-02-01T00:00:00Z",
    "cancel_at_period_end": false
  },
  "usage": {
    "api_calls": 15000,
    "storage_gb": 2.5,
    "modules_installed": 5
  },
  "limits": {
    "api_calls": 50000,
    "storage_gb": 10,
    "modules_installed": 20
  }
}
```

### Create Subscription
```http
POST /api/billing/subscription
Authorization: Bearer <token>
Content-Type: application/json

{
  "plan": "pro",
  "payment_method": "pm_1234567890"
}
```

### Update Subscription
```http
PUT /api/billing/subscription
Authorization: Bearer <token>
Content-Type: application/json

{
  "plan": "enterprise"
}
```

### Cancel Subscription
```http
DELETE /api/billing/subscription
Authorization: Bearer <token>
```

### Get Usage
```http
GET /api/billing/usage?start_date=2024-01-01&end_date=2024-01-31
Authorization: Bearer <token>
```

### Get Invoices
```http
GET /api/billing/invoices?limit=10
Authorization: Bearer <token>
```

### Create Payment Method
```http
POST /api/billing/payment-methods
Authorization: Bearer <token>
Content-Type: application/json

{
  "type": "card",
  "card": {
    "number": "4242424242424242",
    "exp_month": 12,
    "exp_year": 2025,
    "cvc": "123"
  }
}
```

## 📊 Observability Service API

### Get Metrics
```http
GET /api/observability/metrics?service=gateway&start_time=2024-01-15T00:00:00Z&end_time=2024-01-15T23:59:59Z
Authorization: Bearer <token>
```

Response:
```json
{
  "metrics": [
    {
      "name": "http_requests_total",
      "value": 15000,
      "timestamp": "2024-01-15T10:30:00Z",
      "labels": {
        "method": "GET",
        "status": "200"
      }
    }
  ],
  "timerange": {
    "start": "2024-01-15T00:00:00Z",
    "end": "2024-01-15T23:59:59Z"
  }
}
```

### Get Logs
```http
GET /api/observability/logs?service=auth&level=error&limit=100
Authorization: Bearer <token>
```

### Create Alert
```http
POST /api/observability/alerts
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "High Error Rate",
  "condition": "error_rate > 0.05",
  "service": "gateway",
  "notification": {
    "email": "admin@example.com",
    "webhook": "https://hooks.slack.com/..."
  }
}
```

### Get System Health
```http
GET /api/observability/health
Authorization: Bearer <token>
```

Response:
```json
{
  "overall_status": "healthy",
  "services": {
    "gateway": {
      "status": "healthy",
      "response_time": 45,
      "error_rate": 0.001
    },
    "auth": {
      "status": "healthy",
      "response_time": 32,
      "error_rate": 0.0
    }
  },
  "infrastructure": {
    "database": {
      "status": "healthy",
      "connections": 15,
      "query_time": 12
    },
    "redis": {
      "status": "healthy",
      "memory_usage": 0.45
    }
  }
}
```

## 🔧 Module Development API

### Module Template
```http
GET /api/registry/template
Authorization: Bearer <token>
```

### Validate Module
```http
POST /api/registry/validate
Authorization: Bearer <token>
Content-Type: application/json

{
  "module_config": {
    "name": "test-module",
    "version": "1.0.0",
    "image": "test/module:latest"
  }
}
```

### Deploy Module
```http
POST /api/registry/deploy
Authorization: Bearer <token>
Content-Type: application/json

{
  "module_id": "mod_123",
  "environment": "staging"
}
```

## 📝 Error Responses

All API endpoints return consistent error responses:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "field": "email",
      "issue": "Invalid email format"
    },
    "request_id": "req_123456789"
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `UNAUTHORIZED` | 401 | Invalid or missing authentication |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `VALIDATION_ERROR` | 400 | Invalid request data |
| `RATE_LIMITED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Server error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

## 📊 Rate Limiting

API requests are rate limited per user:

- **Free Plan**: 1,000 requests/hour
- **Pro Plan**: 10,000 requests/hour  
- **Enterprise Plan**: 100,000 requests/hour

Rate limit headers are included in responses:
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642248000
```

## 🔍 Pagination

List endpoints support pagination:

```http
GET /api/registry/modules?page=2&limit=20&sort=created_at&order=desc
```

Response includes pagination metadata:
```json
{
  "data": [...],
  "pagination": {
    "page": 2,
    "limit": 20,
    "total": 150,
    "pages": 8,
    "has_next": true,
    "has_prev": true
  }
}
```

## 🔗 Webhooks

Configure webhooks to receive real-time notifications:

```http
POST /api/webhooks
Authorization: Bearer <token>
Content-Type: application/json

{
  "url": "https://your-app.com/webhooks/lift-os",
  "events": ["module.installed", "subscription.updated"],
  "secret": "webhook_secret_key"
}
```

Webhook payload example:
```json
{
  "event": "module.installed",
  "data": {
    "module_id": "mod_123",
    "user_id": "user_456",
    "installed_at": "2024-01-15T10:30:00Z"
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "signature": "sha256=..."
}
```

## 📚 SDKs and Libraries

Official SDKs are available for:

- **Python**: `pip install lift-os-sdk`
- **JavaScript/Node.js**: `npm install @lift-os/sdk`
- **Go**: `go get github.com/lift-os/go-sdk`
- **Java**: Maven/Gradle dependency available

Example Python usage:
```python
from lift_os import LiftOSClient

client = LiftOSClient(api_key="your_api_key")

# Store memory
memory = client.memory.store(
    content="Important information",
    context="project_planning"
)

# Install module
client.registry.install_module("lift-causal")
```

## 🧪 Testing

Use the sandbox environment for testing:
- Base URL: `https://sandbox-api.lift-os.com`
- All data is reset daily
- No charges apply to sandbox usage

## 📞 Support

- **Documentation**: https://docs.lift-os.com
- **API Status**: https://status.lift-os.com
- **Support Email**: api-support@lift-os.com
- **Community**: https://community.lift-os.com