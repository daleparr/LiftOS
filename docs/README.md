# LiftOS Documentation

Welcome to the comprehensive documentation for LiftOS - Enterprise Marketing Intelligence Platform with Live Data Integration.

## 📚 Documentation Overview

This documentation covers all aspects of LiftOS, from basic setup to advanced enterprise security configurations.

### 🎯 Quick Navigation

| Document | Description | Audience |
|----------|-------------|----------|
| **[Live Data Mode Guide](LIVE_DATA_MODE_GUIDE.md)** | Complete guide to transitioning from demo to live data | Users, Administrators |
| **[Secure API Documentation](SECURE_API_DOCUMENTATION.md)** | Enterprise API security and authentication | Developers, Security Teams |
| **[API Reference](API_REFERENCE.md)** | Complete API endpoint documentation | Developers, Integrators |
| **[Security Guide](SECURITY_GUIDE.md)** | Security architecture and compliance | Security Teams, DevOps |

---

## 🚀 Getting Started

### For End Users

1. **[Live Data Mode Transition](LIVE_DATA_MODE_GUIDE.md#quick-start)**
   - 5-minute setup guide
   - Platform connection walkthrough
   - Data preferences configuration

2. **[Platform Connections](LIVE_DATA_MODE_GUIDE.md#platform-specific-setup)**
   - Meta Business (Facebook/Instagram)
   - Google Ads
   - Klaviyo
   - Shopify
   - HubSpot, Salesforce, and more

### For Developers

1. **[API Authentication](SECURE_API_DOCUMENTATION.md#authentication--authorization)**
   - JWT token management
   - Role-based access control
   - Permission system

2. **[API Endpoints](API_REFERENCE.md)**
   - Platform connections API
   - Data ingestion endpoints
   - Monitoring and health checks

### For Security Teams

1. **[Security Architecture](SECURITY_GUIDE.md#security-architecture)**
   - Defense in depth strategy
   - Encryption standards
   - Compliance frameworks

2. **[Incident Response](SECURITY_GUIDE.md#incident-response)**
   - Automated threat detection
   - Response procedures
   - Audit logging

---

## 🏗️ System Architecture

### High-Level Overview

```mermaid
graph TB
    subgraph "Frontend Layer"
        A[Streamlit Dashboard]
        B[Web Interface]
    end
    
    subgraph "API Gateway"
        C[Authentication]
        D[Rate Limiting]
        E[Load Balancer]
    end
    
    subgraph "Microservices"
        F[Data Ingestion Service]
        G[Auth Service]
        H[Channels Service]
        I[Memory Service]
    end
    
    subgraph "Security Layer"
        J[API Key Vault]
        K[Audit Logger]
        L[Encryption Engine]
    end
    
    subgraph "Data Layer"
        M[PostgreSQL]
        N[Redis Cache]
        O[File Storage]
    end
    
    subgraph "External APIs"
        P[Meta Business API]
        Q[Google Ads API]
        R[Klaviyo API]
        S[Shopify API]
    end
    
    A --> C
    B --> C
    C --> D
    D --> E
    E --> F
    E --> G
    E --> H
    E --> I
    
    F --> J
    F --> K
    G --> J
    G --> K
    
    F --> M
    F --> N
    G --> M
    H --> M
    I --> M
    
    F --> P
    F --> Q
    F --> R
    F --> S
    
    J --> L
    K --> L
```

### Core Components

| Component | Purpose | Technology |
|-----------|---------|------------|
| **Streamlit Frontend** | User interface and dashboards | Python, Streamlit |
| **Data Ingestion Service** | Platform API connections and data sync | FastAPI, Python |
| **Auth Service** | Authentication and authorization | FastAPI, JWT |
| **API Key Vault** | Secure credential storage | AES-256-GCM encryption |
| **Audit Logger** | Security event tracking | PostgreSQL, SOC 2 compliance |

---

## 🔐 Security Features

### Enterprise-Grade Security

LiftOS implements bank-level security standards:

- **🔒 AES-256-GCM Encryption**: All sensitive data encrypted at rest
- **🛡️ JWT Authentication**: Stateless token-based authentication
- **👥 RBAC Authorization**: Role-based access control with granular permissions
- **📊 SOC 2 Compliance**: Comprehensive audit logging and controls
- **🚨 Real-time Monitoring**: Automated threat detection and incident response

### Security Certifications

- ✅ **SOC 2 Type II** - Security, availability, and confidentiality
- ✅ **GDPR Compliant** - Data privacy and protection
- ✅ **ISO 27001** - Information security management
- ✅ **PCI DSS** - Payment card industry standards

---

## 🔌 Platform Integrations

### Supported Marketing Platforms

#### Tier 1 - Core Platforms
- **Meta Business** (Facebook/Instagram Ads)
- **Google Ads** (Search, Display, YouTube)
- **Klaviyo** (Email Marketing)

#### Tier 2 - E-commerce & CRM
- **Shopify** (E-commerce)
- **HubSpot** (CRM & Marketing)
- **Salesforce** (Enterprise CRM)
- **Stripe** (Payments)

#### Tier 3 - Analytics & Data
- **TikTok Ads** (Social Media)
- **Snowflake** (Data Warehouse)
- **Databricks** (Analytics)

#### Tier 4 - Extended Platforms
- **LinkedIn Ads** (B2B Marketing)
- **X Ads** (Twitter Advertising)
- **Zoho CRM** (Business Software)

### Integration Methods

| Method | Platforms | Security | Setup Time |
|--------|-----------|----------|------------|
| **OAuth 2.0** | Meta, Google, LinkedIn | Highest | 15-20 min |
| **API Key** | Klaviyo, Shopify, Stripe | High | 5-10 min |
| **Basic Auth** | Legacy systems | Medium | 10-15 min |

---

## 📊 Data Management

### Data Flow Architecture

```mermaid
graph LR
    A[Marketing Platforms] --> B[API Connectors]
    B --> C[Data Validation]
    C --> D[Quality Checks]
    D --> E[Encryption]
    E --> F[Database Storage]
    F --> G[Analytics Engine]
    G --> H[Dashboard]
    
    I[Mock Data] -.-> F
    J[Fallback Logic] --> I
    D --> J
```

### Data Quality Standards

| Metric | Target | Description |
|--------|--------|-------------|
| **Completeness** | > 95% | Percentage of expected data received |
| **Accuracy** | > 98% | Data validation success rate |
| **Timeliness** | < 2 hours | Data freshness requirement |
| **Consistency** | > 90% | Cross-platform data consistency |

### Data Retention Policies

```json
{
  "retention_policies": {
    "marketing_data": "2_years",
    "audit_logs": "7_years", 
    "api_keys": "until_revoked",
    "user_sessions": "24_hours",
    "temp_data": "24_hours"
  }
}
```

---

## 🚦 Operational Monitoring

### Health Monitoring

LiftOS provides comprehensive monitoring across all system components:

#### System Health Endpoints

```http
GET /api/v1/health                    # Overall system health
GET /api/v1/health/services           # Individual service status
GET /api/v1/health/connections        # Platform connection health
GET /api/v1/health/database           # Database connectivity
```

#### Key Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| **Response Time** | API response latency | > 1000ms |
| **Error Rate** | Failed requests percentage | > 1% |
| **Connection Health** | Platform API availability | < 95% |
| **Data Quality** | Quality score across platforms | < 90% |

### Alerting & Notifications

#### Alert Channels

- **📧 Email**: Critical system alerts
- **💬 Slack**: Real-time notifications
- **📱 SMS**: Emergency incidents
- **🔔 In-App**: User-facing notifications

#### Alert Types

```yaml
alerts:
  critical:
    - system_down
    - data_breach
    - security_incident
  
  warning:
    - high_error_rate
    - slow_response_time
    - connection_failure
  
  info:
    - sync_completed
    - user_login
    - configuration_change
```

---

## 🛠️ Development & Deployment

### Development Environment

#### Prerequisites

```bash
# System requirements
Python 3.11+
PostgreSQL 14+
Redis 6+
Node.js 18+ (for frontend builds)

# Environment setup
git clone https://github.com/liftos/liftos.git
cd liftos
pip install -r requirements.txt
```

#### Local Development

```bash
# Start services
cd services/data-ingestion && python app.py &
cd services/auth && python app.py &
cd services/channels && python app.py &

# Start frontend
cd liftos-streamlit && streamlit run app.py
```

### Production Deployment

#### Docker Deployment

```yaml
# docker-compose.yml
version: '3.8'
services:
  api:
    image: liftos/api:latest
    environment:
      - DATABASE_URL=postgresql://...
      - REDIS_URL=redis://...
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
    ports:
      - "8000:8000"
  
  frontend:
    image: liftos/frontend:latest
    environment:
      - API_URL=http://api:8000
    ports:
      - "8501:8501"
  
  database:
    image: postgres:14
    environment:
      - POSTGRES_DB=liftos
      - POSTGRES_USER=liftos
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
```

#### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: liftos-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: liftos-api
  template:
    metadata:
      labels:
        app: liftos-api
    spec:
      containers:
      - name: api
        image: liftos/api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: liftos-secrets
              key: database-url
```

---

## 📖 API Documentation

### Authentication

All API requests require authentication via JWT tokens:

```http
Authorization: Bearer <jwt_token>
X-Organization-ID: <org_id>
```

### Core Endpoints

#### Platform Connections

```http
GET    /api/v1/platform-connections/platforms     # List supported platforms
POST   /api/v1/platform-connections/connections   # Create connection
GET    /api/v1/platform-connections/connections   # List connections
PUT    /api/v1/platform-connections/connections/{id}  # Update connection
DELETE /api/v1/platform-connections/connections/{id}  # Delete connection
POST   /api/v1/platform-connections/connections/{id}/test  # Test connection
POST   /api/v1/platform-connections/connections/{id}/sync  # Manual sync
```

#### Data Quality

```http
GET  /api/v1/data-quality/metrics    # Quality metrics
GET  /api/v1/data-quality/issues     # Quality issues
POST /api/v1/data-quality/assess     # Run assessment
```

#### Monitoring

```http
GET /api/v1/health                   # System health
GET /api/v1/monitoring/metrics       # Performance metrics
GET /api/v1/platform-connections/dashboard  # Connection dashboard
```

### Rate Limits

| Endpoint Category | Rate Limit | Window |
|------------------|------------|---------|
| Authentication | 10 requests | 1 minute |
| Platform Connections | 100 requests | 1 hour |
| Data Sync | 50 requests | 1 hour |
| Analytics | 1000 requests | 1 hour |

---

## 🔧 Configuration

### Environment Variables

#### Core Configuration

```bash
# Application
APP_NAME=LiftOS
VERSION=1.4.0
DEBUG=false

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/liftos
REDIS_URL=redis://localhost:6379/0

# Security
JWT_SECRET_KEY=your_secure_jwt_secret_key
API_VAULT_MASTER_KEY=your_secure_master_key
ENCRYPTION_KEY=your_encryption_key

# Services
DATA_INGESTION_SERVICE_URL=http://localhost:8006
AUTH_SERVICE_URL=http://localhost:8001
CHANNELS_SERVICE_URL=http://localhost:8011
```

#### Feature Flags

```bash
# Demo vs Live Mode
DEMO_MODE=false
REQUIRE_AUTH=true
PREFER_LIVE_DATA=true
FALLBACK_TO_MOCK=true

# Features
ENABLE_CAUSAL=true
ENABLE_SURFACING=true
ENABLE_LLM=true
ENABLE_EXPERIMENTS=true
ENABLE_MEMORY=true
```

### Platform-Specific Configuration

#### Meta Business

```json
{
  "meta_business": {
    "app_id": "your_app_id",
    "app_secret": "your_app_secret",
    "api_version": "v18.0",
    "required_permissions": [
      "ads_read",
      "ads_management", 
      "business_management"
    ]
  }
}
```

#### Google Ads

```json
{
  "google_ads": {
    "client_id": "your_client_id",
    "client_secret": "your_client_secret",
    "developer_token": "your_developer_token",
    "api_version": "v14"
  }
}
```

---

## 🆘 Troubleshooting

### Common Issues

#### Connection Problems

**Issue**: "Service unavailable" errors
**Solution**: 
```bash
# Check service health
curl http://localhost:8006/health
curl http://localhost:8001/health

# Verify network connectivity
telnet api.facebook.com 443
telnet googleads.googleapis.com 443
```

#### Authentication Failures

**Issue**: "Invalid credentials" error
**Solution**:
```bash
# Verify credential format
# Meta: Should start with EAABwzLixnjYBO
# Klaviyo: Should start with pk_live_ or pk_test_
# Google: Check developer token format
```

#### Data Quality Issues

**Issue**: Low quality scores
**Solution**:
```python
# Check validation logs
GET /api/v1/data-quality/validation-logs?platform=meta_business

# Review sync errors  
GET /api/v1/platform-connections/connections/{id}/sync-history

# Adjust quality thresholds
PUT /api/v1/platform-connections/preferences
{
  "quality_threshold": 0.85
}
```

### Debug Mode

Enable debug logging:

```bash
export DEBUG=true
export LOG_LEVEL=DEBUG
export ENABLE_REQUEST_LOGGING=true
```

### Log Analysis

```bash
# Service logs
tail -f logs/data-ingestion.log
tail -f logs/auth-service.log

# Security audit logs
tail -f logs/security-audit.log

# Search for specific issues
grep "AUTH_FAILED" logs/auth-service.log
grep "RATE_LIMIT" logs/data-ingestion.log
grep "QUALITY_CHECK_FAILED" logs/data-ingestion.log
```

---

## 📞 Support & Resources

### Getting Help

#### Support Channels

- **📧 Technical Support**: support@liftos.com
- **🔒 Security Issues**: security@liftos.com  
- **📚 Documentation**: docs@liftos.com
- **💬 Community Forum**: [community.liftos.com](https://community.liftos.com)

#### Response Times

| Priority | Response Time | Resolution Time |
|----------|---------------|-----------------|
| **Critical** | 15 minutes | 4 hours |
| **High** | 1 hour | 24 hours |
| **Medium** | 4 hours | 72 hours |
| **Low** | 24 hours | 1 week |

### Additional Resources

#### Documentation Links

- **[API Reference](API_REFERENCE.md)** - Complete API documentation
- **[Security Guide](SECURITY_GUIDE.md)** - Security architecture and compliance
- **[Live Data Guide](LIVE_DATA_MODE_GUIDE.md)** - Platform integration walkthrough
- **[Secure API Docs](SECURE_API_DOCUMENTATION.md)** - Enterprise security features

#### External Resources

- **Platform Documentation**:
  - [Meta Business API](https://developers.facebook.com/docs/marketing-api)
  - [Google Ads API](https://developers.google.com/google-ads/api)
  - [Klaviyo API](https://developers.klaviyo.com/en)
  - [Shopify API](https://shopify.dev/api)

#### Training & Certification

- **[LiftOS Certification Program](https://training.liftos.com/certification)**
- **[Security Training](https://training.liftos.com/security)**
- **[API Integration Workshop](https://training.liftos.com/api-workshop)**

### Status & Updates

- **🔍 System Status**: [status.liftos.com](https://status.liftos.com)
- **📢 Release Notes**: [releases.liftos.com](https://releases.liftos.com)
- **🐛 Bug Reports**: [github.com/liftos/liftos/issues](https://github.com/liftos/liftos/issues)

---

## 📄 License & Legal

### Software License

LiftOS is proprietary software. See [LICENSE.md](../LICENSE.md) for terms and conditions.

### Privacy & Compliance

- **Privacy Policy**: [liftos.com/privacy](https://liftos.com/privacy)
- **Terms of Service**: [liftos.com/terms](https://liftos.com/terms)
- **SOC 2 Report**: Available upon request
- **GDPR Compliance**: [liftos.com/gdpr](https://liftos.com/gdpr)

### Third-Party Licenses

See [THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md) for open source dependencies.

---

**Last Updated**: January 2024  
**Documentation Version**: 1.4.0  
**LiftOS Version**: 1.4.0