# LiftOS API Reference

## Overview

Complete API reference for LiftOS platform connections, data ingestion, and live data management.

**Base URL**: `https://api.liftos.com` (Production) | `http://localhost:8006` (Development)

**API Version**: v1

**Authentication**: Bearer JWT tokens

---

## Table of Contents

1. [Authentication](#authentication)
2. [Platform Connections](#platform-connections)
3. [Data Ingestion](#data-ingestion)
4. [Data Quality](#data-quality)
5. [Monitoring](#monitoring)
6. [Webhooks](#webhooks)
7. [Error Codes](#error-codes)

---

## Authentication

### JWT Token Structure

All API requests require a valid JWT token in the Authorization header:

```http
Authorization: Bearer <jwt_token>
X-Organization-ID: <org_id>
```

### Get Access Token

```http
POST /api/v1/auth/token
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "secure_password",
  "org_id": "org_123"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 86400,
  "refresh_token": "refresh_token_here",
  "user": {
    "id": "user_123",
    "email": "user@example.com",
    "roles": ["admin"],
    "org_id": "org_123"
  }
}
```

### Refresh Token

```http
POST /api/v1/auth/refresh
Content-Type: application/json

{
  "refresh_token": "refresh_token_here"
}
```

---

## Platform Connections

### Get Supported Platforms

```http
GET /api/v1/platform-connections/platforms
Authorization: Bearer <token>
```

**Response:**
```json
[
  {
    "id": "meta_business",
    "display_name": "Meta Business",
    "oauth_enabled": true,
    "auth_type": "oauth2",
    "required_scopes": ["ads_read", "ads_management"],
    "icon": "meta-icon.svg",
    "color": "#1877F2",
    "description": "Connect Facebook and Instagram advertising accounts",
    "documentation_url": "https://developers.facebook.com/docs/marketing-api"
  },
  {
    "id": "klaviyo",
    "display_name": "Klaviyo",
    "oauth_enabled": false,
    "auth_type": "api_key",
    "required_scopes": [],
    "icon": "klaviyo-icon.svg",
    "color": "#FF6900",
    "description": "Email marketing and automation platform"
  }
]
```

### Create Platform Connection

#### OAuth2 Flow Initiation

```http
POST /api/v1/platform-connections/oauth/initiate
Authorization: Bearer <token>
Content-Type: application/json

{
  "platform": "meta_business",
  "redirect_uri": "https://app.liftos.com/oauth/callback",
  "scopes": ["ads_read", "ads_management", "business_management"],
  "state": "optional_state_parameter"
}
```

**Response:**
```json
{
  "authorization_url": "https://www.facebook.com/v18.0/dialog/oauth?client_id=123&redirect_uri=...",
  "state": "secure_random_state_123",
  "expires_in": 600
}
```

#### Manual Credential Connection

```http
POST /api/v1/platform-connections/connections
Authorization: Bearer <token>
Content-Type: application/json

{
  "platform": "klaviyo",
  "connection_name": "Main Klaviyo Account",
  "credentials": {
    "api_key": "pk_live_abc123def456..."
  },
  "sync_frequency": "hourly",
  "auto_sync_enabled": true,
  "connection_config": {
    "data_sources": ["campaigns", "metrics", "lists"],
    "date_range_days": 90
  }
}
```

**Response:**
```json
{
  "id": "conn_123",
  "platform": "klaviyo",
  "connection_name": "Main Klaviyo Account",
  "status": "active",
  "created_at": "2024-01-01T10:00:00Z",
  "last_sync": null,
  "next_sync": "2024-01-01T11:00:00Z",
  "sync_frequency": "hourly",
  "auto_sync_enabled": true,
  "data_sources": [
    {
      "type": "campaigns",
      "enabled": true,
      "last_updated": null,
      "record_count": 0
    }
  ],
  "health_status": {
    "status": "pending",
    "last_check": null,
    "response_time_ms": null
  }
}
```

### List Connections

```http
GET /api/v1/platform-connections/connections
Authorization: Bearer <token>
```

**Query Parameters:**
- `platform` (optional): Filter by platform
- `status` (optional): Filter by status (active, error, disabled)
- `limit` (optional): Number of results (default: 50)
- `offset` (optional): Pagination offset

**Response:**
```json
{
  "connections": [
    {
      "id": "conn_123",
      "platform": "klaviyo",
      "connection_name": "Main Klaviyo Account",
      "status": "active",
      "last_sync": "2024-01-01T12:00:00Z",
      "next_sync": "2024-01-01T13:00:00Z",
      "sync_frequency": "hourly",
      "data_sources": [
        {
          "type": "campaigns",
          "last_updated": "2024-01-01T12:00:00Z",
          "record_count": 1250
        }
      ],
      "health_status": {
        "status": "healthy",
        "last_check": "2024-01-01T12:00:00Z",
        "response_time_ms": 245
      }
    }
  ],
  "total": 1,
  "limit": 50,
  "offset": 0
}
```

### Get Connection Details

```http
GET /api/v1/platform-connections/connections/{connection_id}
Authorization: Bearer <token>
```

**Response:**
```json
{
  "id": "conn_123",
  "platform": "klaviyo",
  "connection_name": "Main Klaviyo Account",
  "status": "active",
  "created_at": "2024-01-01T10:00:00Z",
  "updated_at": "2024-01-01T12:00:00Z",
  "last_sync": "2024-01-01T12:00:00Z",
  "next_sync": "2024-01-01T13:00:00Z",
  "sync_frequency": "hourly",
  "auto_sync_enabled": true,
  "connection_config": {
    "data_sources": ["campaigns", "metrics", "lists"],
    "date_range_days": 90
  },
  "data_sources": [
    {
      "type": "campaigns",
      "enabled": true,
      "last_updated": "2024-01-01T12:00:00Z",
      "record_count": 1250,
      "sync_status": "success"
    },
    {
      "type": "metrics",
      "enabled": true,
      "last_updated": "2024-01-01T12:00:00Z",
      "record_count": 5000,
      "sync_status": "success"
    }
  ],
  "health_status": {
    "status": "healthy",
    "last_check": "2024-01-01T12:00:00Z",
    "response_time_ms": 245,
    "error_count_24h": 0
  },
  "sync_history": [
    {
      "sync_id": "sync_456",
      "started_at": "2024-01-01T12:00:00Z",
      "completed_at": "2024-01-01T12:05:00Z",
      "status": "success",
      "records_processed": 6250,
      "errors": []
    }
  ]
}
```

### Update Connection

```http
PUT /api/v1/platform-connections/connections/{connection_id}
Authorization: Bearer <token>
Content-Type: application/json

{
  "connection_name": "Updated Klaviyo Account",
  "sync_frequency": "every_2_hours",
  "auto_sync_enabled": false,
  "connection_config": {
    "data_sources": ["campaigns", "metrics"],
    "date_range_days": 180
  }
}
```

### Delete Connection

```http
DELETE /api/v1/platform-connections/connections/{connection_id}
Authorization: Bearer <token>
```

**Response:**
```json
{
  "message": "Connection deleted successfully",
  "deleted_at": "2024-01-01T13:00:00Z"
}
```

### Test Connection

```http
POST /api/v1/platform-connections/connections/{connection_id}/test
Authorization: Bearer <token>
```

**Response:**
```json
{
  "status": "success",
  "response_time_ms": 234,
  "data_sources_available": [
    "campaigns",
    "metrics",
    "audiences",
    "lists"
  ],
  "api_rate_limit": {
    "remaining": 4950,
    "limit": 5000,
    "reset_time": "2024-01-01T13:00:00Z"
  },
  "platform_info": {
    "account_id": "abc123",
    "account_name": "My Company",
    "permissions": ["read", "write"]
  },
  "test_timestamp": "2024-01-01T12:30:00Z"
}
```

### Manual Sync

```http
POST /api/v1/platform-connections/connections/{connection_id}/sync
Authorization: Bearer <token>
Content-Type: application/json

{
  "sync_type": "manual",
  "data_sources": ["campaigns", "metrics"],
  "date_range": {
    "start_date": "2024-01-01",
    "end_date": "2024-01-31"
  },
  "priority": "high"
}
```

**Response:**
```json
{
  "sync_id": "sync_789",
  "status": "started",
  "estimated_duration_minutes": 5,
  "data_sources": ["campaigns", "metrics"],
  "started_at": "2024-01-01T13:00:00Z",
  "estimated_completion": "2024-01-01T13:05:00Z"
}
```

### Get Sync Status

```http
GET /api/v1/platform-connections/connections/{connection_id}/sync/{sync_id}
Authorization: Bearer <token>
```

**Response:**
```json
{
  "sync_id": "sync_789",
  "connection_id": "conn_123",
  "status": "running",
  "progress": 0.65,
  "started_at": "2024-01-01T13:00:00Z",
  "estimated_completion": "2024-01-01T13:05:00Z",
  "data_sources": [
    {
      "type": "campaigns",
      "status": "completed",
      "records_processed": 1250,
      "progress": 1.0
    },
    {
      "type": "metrics",
      "status": "running",
      "records_processed": 3200,
      "progress": 0.64
    }
  ],
  "errors": [],
  "warnings": [
    {
      "message": "Some historical data unavailable",
      "data_source": "metrics",
      "timestamp": "2024-01-01T13:02:00Z"
    }
  ]
}
```

### Bulk Sync

```http
POST /api/v1/platform-connections/bulk-sync
Authorization: Bearer <token>
Content-Type: application/json

{
  "connection_ids": ["conn_123", "conn_456", "conn_789"],
  "sync_type": "scheduled",
  "data_sources": ["campaigns", "metrics"],
  "priority": "normal"
}
```

**Response:**
```json
{
  "bulk_sync_id": "bulk_sync_123",
  "status": "started",
  "total_connections": 3,
  "sync_jobs": [
    {
      "connection_id": "conn_123",
      "sync_id": "sync_001",
      "status": "queued"
    },
    {
      "connection_id": "conn_456",
      "sync_id": "sync_002",
      "status": "queued"
    },
    {
      "connection_id": "conn_789",
      "sync_id": "sync_003",
      "status": "queued"
    }
  ],
  "started_at": "2024-01-01T13:00:00Z"
}
```

---

## Data Preferences

### Get Data Preferences

```http
GET /api/v1/platform-connections/preferences
Authorization: Bearer <token>
```

**Response:**
```json
{
  "prefer_live_data": true,
  "fallback_to_mock": true,
  "auto_sync_enabled": true,
  "data_retention_days": 90,
  "sync_frequency_default": "hourly",
  "quality_threshold": 0.95,
  "notification_preferences": {
    "sync_failures": true,
    "quality_alerts": true,
    "connection_issues": true
  },
  "data_export_settings": {
    "format": "json",
    "compression": true,
    "encryption": true
  }
}
```

### Update Data Preferences

```http
PUT /api/v1/platform-connections/preferences
Authorization: Bearer <token>
Content-Type: application/json

{
  "prefer_live_data": true,
  "fallback_to_mock": false,
  "auto_sync_enabled": true,
  "data_retention_days": 180,
  "quality_threshold": 0.90
}
```

---

## Data Ingestion

### Get Platform Data

```http
GET /api/v1/data-ingestion/platform-data
Authorization: Bearer <token>
```

**Query Parameters:**
- `platform` (required): Platform identifier
- `data_source` (required): Data source type
- `start_date` (optional): Start date (YYYY-MM-DD)
- `end_date` (optional): End date (YYYY-MM-DD)
- `limit` (optional): Number of records (default: 1000)
- `offset` (optional): Pagination offset

**Response:**
```json
{
  "platform": "klaviyo",
  "data_source": "campaigns",
  "date_range": {
    "start_date": "2024-01-01",
    "end_date": "2024-01-31"
  },
  "data": [
    {
      "campaign_id": "camp_123",
      "campaign_name": "Welcome Series",
      "status": "active",
      "created_at": "2024-01-01T10:00:00Z",
      "metrics": {
        "sent": 1250,
        "delivered": 1200,
        "opened": 480,
        "clicked": 96,
        "unsubscribed": 5
      }
    }
  ],
  "metadata": {
    "total_records": 1,
    "limit": 1000,
    "offset": 0,
    "data_freshness": "2024-01-01T12:00:00Z",
    "quality_score": 0.98
  }
}
```

### Ingest Custom Data

```http
POST /api/v1/data-ingestion/custom-data
Authorization: Bearer <token>
Content-Type: application/json

{
  "data_source": "custom_events",
  "data_type": "marketing_events",
  "data": [
    {
      "event_id": "evt_123",
      "event_type": "email_click",
      "timestamp": "2024-01-01T12:00:00Z",
      "user_id": "user_456",
      "properties": {
        "campaign_id": "camp_123",
        "email_subject": "Welcome to our platform",
        "link_url": "https://example.com/welcome"
      }
    }
  ],
  "metadata": {
    "source": "custom_tracking",
    "version": "1.0"
  }
}
```

---

## Data Quality

### Get Quality Metrics

```http
GET /api/v1/data-quality/metrics
Authorization: Bearer <token>
```

**Query Parameters:**
- `platform` (optional): Filter by platform
- `date_range` (optional): Date range for metrics
- `granularity` (optional): hour, day, week, month

**Response:**
```json
{
  "overall_quality_score": 0.96,
  "quality_dimensions": {
    "completeness": 0.98,
    "accuracy": 0.95,
    "timeliness": 0.97,
    "consistency": 0.94,
    "validity": 0.99
  },
  "platform_scores": [
    {
      "platform": "meta_business",
      "quality_score": 0.98,
      "completeness": 0.99,
      "accuracy": 0.97,
      "timeliness": 0.98,
      "last_assessment": "2024-01-01T12:00:00Z",
      "trend": "stable"
    },
    {
      "platform": "klaviyo",
      "quality_score": 0.94,
      "completeness": 0.96,
      "accuracy": 0.93,
      "timeliness": 0.95,
      "last_assessment": "2024-01-01T12:00:00Z",
      "trend": "improving"
    }
  ],
  "quality_trends": {
    "7_day_average": 0.95,
    "30_day_average": 0.94,
    "trend_direction": "improving"
  },
  "quality_issues": [
    {
      "platform": "klaviyo",
      "issue_type": "missing_data",
      "description": "Some campaign metrics missing for 2024-01-01",
      "severity": "medium",
      "detected_at": "2024-01-01T11:30:00Z"
    }
  ]
}
```

### Get Quality Issues

```http
GET /api/v1/data-quality/issues
Authorization: Bearer <token>
```

**Query Parameters:**
- `platform` (optional): Filter by platform
- `severity` (optional): low, medium, high, critical
- `status` (optional): open, resolved, ignored
- `limit` (optional): Number of results

**Response:**
```json
{
  "issues": [
    {
      "id": "issue_123",
      "platform": "klaviyo",
      "data_source": "campaigns",
      "issue_type": "missing_data",
      "severity": "medium",
      "status": "open",
      "description": "Campaign metrics missing for date range 2024-01-01 to 2024-01-02",
      "detected_at": "2024-01-01T11:30:00Z",
      "affected_records": 25,
      "resolution_suggestions": [
        "Re-sync affected date range",
        "Check platform API status",
        "Verify API permissions"
      ]
    }
  ],
  "summary": {
    "total_issues": 1,
    "by_severity": {
      "critical": 0,
      "high": 0,
      "medium": 1,
      "low": 0
    },
    "by_status": {
      "open": 1,
      "resolved": 0,
      "ignored": 0
    }
  }
}
```

### Run Quality Assessment

```http
POST /api/v1/data-quality/assess
Authorization: Bearer <token>
Content-Type: application/json

{
  "platform": "klaviyo",
  "data_sources": ["campaigns", "metrics"],
  "assessment_type": "comprehensive",
  "date_range": {
    "start_date": "2024-01-01",
    "end_date": "2024-01-31"
  }
}
```

**Response:**
```json
{
  "assessment_id": "assess_123",
  "status": "started",
  "estimated_duration_minutes": 10,
  "started_at": "2024-01-01T13:00:00Z"
}
```

---

## Monitoring

### System Health

```http
GET /api/v1/health
Authorization: Bearer <token>
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "1.4.0",
  "services": {
    "auth": {
      "status": "healthy",
      "response_time_ms": 45,
      "last_check": "2024-01-01T12:00:00Z"
    },
    "data_ingestion": {
      "status": "healthy",
      "response_time_ms": 120,
      "last_check": "2024-01-01T12:00:00Z"
    },
    "database": {
      "status": "healthy",
      "connection_pool": {
        "active": 5,
        "idle": 15,
        "max": 20
      }
    },
    "redis": {
      "status": "healthy",
      "memory_usage": "45MB",
      "connected_clients": 12
    }
  },
  "metrics": {
    "requests_per_minute": 1250,
    "average_response_time_ms": 85,
    "error_rate": 0.001,
    "active_connections": 1250
  }
}
```

### Connection Dashboard

```http
GET /api/v1/platform-connections/dashboard
Authorization: Bearer <token>
```

**Response:**
```json
{
  "summary": {
    "total_connections": 5,
    "active_connections": 4,
    "failed_connections": 1,
    "total_data_sources": 15,
    "last_sync_success_rate": 0.95
  },
  "platform_status": [
    {
      "platform": "meta_business",
      "connections": 2,
      "status": "healthy",
      "last_sync": "2024-01-01T12:00:00Z",
      "data_freshness_hours": 1,
      "error_rate_24h": 0.001
    },
    {
      "platform": "klaviyo",
      "connections": 1,
      "status": "warning",
      "last_sync": "2024-01-01T10:00:00Z",
      "data_freshness_hours": 3,
      "error_rate_24h": 0.05
    }
  ],
  "recent_syncs": [
    {
      "sync_id": "sync_123",
      "connection_id": "conn_456",
      "platform": "meta_business",
      "status": "success",
      "duration_minutes": 3,
      "records_processed": 5000,
      "completed_at": "2024-01-01T12:00:00Z"
    }
  ],
  "alerts": [
    {
      "id": "alert_123",
      "type": "sync_failure",
      "platform": "klaviyo",
      "message": "Sync failed due to rate limiting",
      "severity": "medium",
      "created_at": "2024-01-01T11:30:00Z"
    }
  ]
}
```

### Performance Metrics

```http
GET /api/v1/monitoring/metrics
Authorization: Bearer <token>
```

**Query Parameters:**
- `metric_type` (optional): response_time, throughput, error_rate
- `time_range` (optional): 1h, 24h, 7d, 30d
- `granularity` (optional): minute, hour, day

**Response:**
```json
{
  "time_range": "24h",
  "granularity": "hour",
  "metrics": {
    "response_time": {
      "average_ms": 245,
      "p95_ms": 450,
      "p99_ms": 800,
      "data_points": [
        {
          "timestamp": "2024-01-01T12:00:00Z",
          "value": 234
        }
      ]
    },
    "throughput": {
      "requests_per_second": 125,
      "data_points": [
        {
          "timestamp": "2024-01-01T12:00:00Z",
          "value": 130
        }
      ]
    },
    "error_rate": {
      "percentage": 0.1,
      "total_errors": 12,
      "total_requests": 12000,
      "data_points": [
        {
          "timestamp": "2024-01-01T12:00:00Z",
          "value": 0.08
        }
      ]
    }
  }
}
```

---

## Webhooks

### Register Webhook

```http
POST /api/v1/webhooks
Authorization: Bearer <token>
Content-Type: application/json

{
  "url": "https://your-app.com/webhooks/liftos",
  "events": [
    "connection.created",
    "connection.updated",
    "sync.completed",
    "sync.failed",
    "quality.alert"
  ],
  "secret": "your_webhook_secret",
  "active": true
}
```

**Response:**
```json
{
  "id": "webhook_123",
  "url": "https://your-app.com/webhooks/liftos",
  "events": [
    "connection.created",
    "connection.updated",
    "sync.completed",
    "sync.failed",
    "quality.alert"
  ],
  "secret": "your_webhook_secret",
  "active": true,
  "created_at": "2024-01-01T12:00:00Z"
}
```

### Webhook Event Types

| Event | Description | Payload |
|-------|-------------|---------|
| `connection.created` | New platform connection created | Connection object |
| `connection.updated` | Connection settings updated | Connection object |
| `connection.deleted` | Connection removed | Connection ID |
| `sync.started` | Data sync initiated | Sync object |
| `sync.completed` | Data sync finished successfully | Sync object |
| `sync.failed` | Data sync failed | Sync object with errors |
| `quality.alert` | Data quality issue detected | Quality issue object |

### Webhook Payload Example

```json
{
  "event": "sync.completed",
  "timestamp": "2024-01-01T12:05:00Z",
  "webhook_id": "webhook_123",
  "data": {
    "sync_id": "sync_456",
    "connection_id": "conn_123",
    "platform": "klaviyo",
    "status": "success",
    "duration_minutes": 5,
    "records_processed": 6250,
    "data_sources": ["campaigns", "metrics"],
    "started_at": "2024-01-01T12:00:00Z",
    "completed_at": "2024-01-01T12:05:00Z"
  }
}
```

---

## Error Codes

### HTTP Status Codes

| Code | Description |
|------|-------------|
| `200` | Success |
| `201` | Created |
| `400` | Bad Request |
| `401` | Unauthorized |
| `403` | Forbidden |
| `404` | Not Found |
| `409` | Conflict |
| `422` | Unprocessable Entity |
| `429` | Too Many Requests |
| `500` | Internal Server Error |
| `503` | Service Unavailable |

### Application Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `AUTH_REQUIRED` | Authentication required | 401 |
| `INVALID_TOKEN` | JWT token invalid or expired | 401 |
| `INSUFFICIENT_PERMISSIONS` | User lacks required permissions | 403 |
| `INVALID_CREDENTIALS` | Platform credentials invalid | 400 |
| `PLATFORM_NOT_SUPPORTED` | Platform not supported | 400 |
| `CONNECTION_NOT_FOUND` | Connection does not exist | 404 |
| `CONNECTION_ALREADY_EXISTS` | Connection already exists for platform | 409 |
| `SYNC_IN_PROGRESS` | Sync already running for connection | 409 |
| `RATE_LIMIT_EXCEEDED` | API rate limit exceeded | 429 |
| `PLATFORM_API_ERROR` | External platform API error | 503 |
| `DATA_QUALITY_FAILED` | Data quality validation failed | 422 |
| `WEBHOOK_DELIVERY_FAILED` | Webhook delivery failed | 500 |

### Error Response Format

```json
{
  "error": {
    "code": "INVALID_CREDENTIALS",
    "message": "The provided API credentials are invalid or expired",
    "details": {
      "platform": "klaviyo",
      "field": "api_key",
      "validation_errors": [
        "API key format is invalid"
      ]
    },
    "request_id": "req_123456",
    "timestamp": "2024-01-01T12:00:00Z",
    "documentation_url": "https://docs.liftos.com/errors/invalid-credentials"
  }
}
```

---

## Rate Limits

### Rate Limit Headers

All API responses include rate limit information:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 995
X-RateLimit-Reset: 1704067200
X-RateLimit-Window: 3600
```

### Rate Limit Tiers

| Endpoint Category | Rate Limit | Window |
|------------------|------------|---------|
| Authentication | 10 requests | 1 minute |
| Platform Connections | 100 requests | 1 hour |
| Data Ingestion | 500 requests | 1 hour |
| Data Quality | 200 requests | 1 hour |
| Monitoring | 1000 requests | 1 hour |
| Webhooks | 50 requests | 1 hour |

---

## SDKs and Libraries

### Official SDKs

- **Python**: `pip install liftos-sdk`
- **JavaScript/Node.js**: `npm install @liftos/sdk`
- **Go**: `go get github.com/liftos/go-sdk`

### Community SDKs

- **PHP**: Available on Packagist
- **Ruby**: Available as gem
- **Java**: Available on Maven Central

---

## Support

### API Support

- **Documentation**: [docs.liftos.com/api](https://docs.liftos.com/api)
- **Support Email**: api-support@liftos.com
- **Status Page**: [status.liftos.com](https://status.liftos.com)
- **Community Forum**: [community.liftos.com](https://community.liftos.com)

### Rate Limit Increases

Contact support@liftos.com for rate limit increases with:
- Current usage patterns
- Expected traffic volume
- Use case description
- Business justification