# LiftOS Live Data Mode Guide

## Overview

This guide provides comprehensive instructions for transitioning from demo mode to live data mode in LiftOS, enabling real-time marketing intelligence from connected platforms.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Step-by-Step Transition](#step-by-step-transition)
4. [Platform-Specific Setup](#platform-specific-setup)
5. [Data Quality & Monitoring](#data-quality--monitoring)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

---

## Quick Start

### Current Status Check

Before transitioning to live data mode, verify your current configuration:

```bash
# Check current mode
echo $DEMO_MODE          # Should show 'true' in demo mode
echo $REQUIRE_AUTH       # Should show 'false' in demo mode
echo $PREFER_LIVE_DATA   # Should show 'false' in demo mode
```

### 5-Minute Setup

1. **Access Platform Connections**: Navigate to `🔗 Platform Connections` in LiftOS
2. **Connect First Platform**: Click "Connect New" → Select platform → Enter credentials
3. **Test Connection**: Use "Test Connection" to verify setup
4. **Enable Live Data**: Go to Settings tab → Check "Prefer Live Data"
5. **Monitor Dashboard**: View real-time data in the Dashboard tab

---

## Prerequisites

### System Requirements

- ✅ **LiftOS Services Running**: Data ingestion and auth services operational
- ✅ **Database Access**: PostgreSQL connection established
- ✅ **Network Access**: Outbound HTTPS connections to marketing platforms
- ✅ **SSL Certificates**: Valid certificates for secure API communication

### Marketing Platform Requirements

| Platform | Requirements | Setup Time |
|----------|-------------|------------|
| **Meta Business** | Business Manager access, App creation | 15 minutes |
| **Google Ads** | Google Ads account, API access enabled | 20 minutes |
| **Klaviyo** | Account admin access, API key generation | 5 minutes |
| **Shopify** | Store admin access, Private app creation | 10 minutes |
| **HubSpot** | Professional/Enterprise plan, API access | 15 minutes |

### Security Prerequisites

- **API Key Vault**: Configured with master encryption key
- **JWT Authentication**: Secret key configured
- **Audit Logging**: Database tables created
- **Rate Limiting**: Redis/memory cache available

---

## Step-by-Step Transition

### Phase 1: Environment Preparation

#### 1.1 Update Environment Variables

```bash
# Create .env file or update existing
cat > .env << EOF
# Live Data Mode Configuration
DEMO_MODE=false
REQUIRE_AUTH=true
PREFER_LIVE_DATA=true
FALLBACK_TO_MOCK=true
AUTO_SYNC_ENABLED=true

# Security Configuration
JWT_SECRET_KEY=your_secure_jwt_secret_key_here
API_VAULT_MASTER_KEY=your_secure_master_key_here

# Service URLs (adjust for your environment)
DATA_INGESTION_SERVICE_URL=http://localhost:8006
AUTH_SERVICE_URL=http://localhost:8001
CHANNELS_SERVICE_URL=http://localhost:8011

# Database Configuration
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/liftos
REDIS_URL=redis://localhost:6379/0
EOF
```

#### 1.2 Restart Services

```bash
# Restart all LiftOS services to apply new configuration
cd services/data-ingestion && python app.py &
cd services/auth && python app.py &
cd services/channels && python app.py &
cd liftos-streamlit && streamlit run app.py
```

### Phase 2: Platform Connection Setup

#### 2.1 Access Platform Connections Interface

1. Open LiftOS in your browser: `http://localhost:8501`
2. Navigate to **🔗 Platform Connections**
3. You should now see authentication required (no longer "Demo mode")

#### 2.2 Authentication Setup

If authentication is now required:

```python
# Create admin user (run once)
from shared.auth.user_manager import create_admin_user

admin_user = create_admin_user(
    email="admin@yourcompany.com",
    password="secure_password_123",
    org_id="your_org_id"
)
```

#### 2.3 Connect Your First Platform

**Option A: OAuth2 Platforms (Recommended)**

1. Click **"Connect New"** tab
2. Select platform (e.g., Meta Business)
3. Click **"Connect via OAuth"**
4. Complete OAuth flow in popup window
5. Verify connection appears in "My Connections"

**Option B: Manual Credentials**

1. Click **"Connect New"** tab
2. Select platform (e.g., Klaviyo)
3. Fill in credential form:
   ```
   Connection Name: Main Klaviyo Account
   API Key: pk_live_abc123def456...
   Sync Frequency: Hourly
   ```
4. Click **"Create Connection"**

### Phase 3: Data Preferences Configuration

#### 3.1 Configure Live Data Settings

Navigate to **Settings** tab in Platform Connections:

```json
{
  "prefer_live_data": true,        // ✅ Use live data when available
  "fallback_to_mock": true,        // ✅ Fallback to mock if live fails
  "auto_sync_enabled": true,       // ✅ Automatic data synchronization
  "data_retention_days": 90,       // Configure retention period
  "sync_frequency_default": "hourly", // Default sync frequency
  "quality_threshold": 0.95        // Minimum data quality score
}
```

#### 3.2 Test Connections

For each connected platform:

1. Go to **"My Connections"** tab
2. Click **"Test Connection"** button
3. Verify successful response:
   ```json
   {
     "status": "success",
     "response_time_ms": 234,
     "data_sources_available": ["campaigns", "metrics", "audiences"],
     "api_rate_limit": {
       "remaining": 4950,
       "reset_time": "2024-01-01T13:00:00Z"
     }
   }
   ```

### Phase 4: Data Synchronization

#### 4.1 Initial Data Sync

Perform initial data synchronization:

1. Click **"Sync Now"** for each connection
2. Monitor sync progress in Dashboard tab
3. Verify data appears in analytics dashboards

#### 4.2 Scheduled Sync Configuration

Configure automatic synchronization:

```json
{
  "sync_schedule": {
    "meta_business": "every_hour",
    "google_ads": "every_2_hours", 
    "klaviyo": "every_30_minutes",
    "shopify": "every_hour"
  },
  "sync_windows": {
    "start_time": "06:00",
    "end_time": "22:00",
    "timezone": "UTC"
  }
}
```

---

## Platform-Specific Setup

### Meta Business (Facebook/Instagram)

#### Prerequisites
- Facebook Business Manager account
- Admin access to advertising accounts
- App created in Facebook Developers

#### Setup Steps

1. **Create Facebook App**:
   ```
   - Go to developers.facebook.com
   - Create new app → Business type
   - Add Marketing API product
   - Configure OAuth redirect URI: https://your-domain.com/oauth/callback
   ```

2. **Get App Credentials**:
   ```
   App ID: 123456789012345
   App Secret: abc123def456ghi789
   ```

3. **Connect in LiftOS**:
   ```json
   {
     "platform": "meta_business",
     "auth_type": "oauth2",
     "credentials": {
       "app_id": "123456789012345",
       "app_secret": "abc123def456ghi789"
     },
     "scopes": ["ads_read", "ads_management", "business_management"]
   }
   ```

#### Available Data Sources
- **Campaigns**: Campaign performance metrics
- **Ad Sets**: Ad set level data and targeting
- **Ads**: Individual ad performance
- **Audiences**: Custom and lookalike audiences
- **Insights**: Detailed performance insights

### Google Ads

#### Prerequisites
- Google Ads account with API access
- Google Cloud project with Ads API enabled
- OAuth2 credentials configured

#### Setup Steps

1. **Enable Google Ads API**:
   ```
   - Go to console.cloud.google.com
   - Enable Google Ads API
   - Create OAuth2 credentials
   - Add authorized redirect URI
   ```

2. **Get Developer Token**:
   ```
   - Apply for developer token in Google Ads
   - Wait for approval (can take 24-48 hours)
   ```

3. **Connect in LiftOS**:
   ```json
   {
     "platform": "google_ads",
     "auth_type": "oauth2",
     "credentials": {
       "client_id": "your_client_id.googleusercontent.com",
       "client_secret": "your_client_secret",
       "developer_token": "your_developer_token"
     }
   }
   ```

### Klaviyo

#### Prerequisites
- Klaviyo account (any plan)
- Account admin access

#### Setup Steps

1. **Generate API Key**:
   ```
   - Go to Account → Settings → API Keys
   - Create Private API Key
   - Copy the key (starts with pk_live_ or pk_test_)
   ```

2. **Connect in LiftOS**:
   ```json
   {
     "platform": "klaviyo",
     "auth_type": "api_key",
     "credentials": {
       "api_key": "pk_live_abc123def456..."
     }
   }
   ```

#### Available Data Sources
- **Campaigns**: Email campaign performance
- **Flows**: Automated flow metrics
- **Lists**: Subscriber list data
- **Events**: Customer event tracking
- **Metrics**: Revenue and engagement metrics

### Shopify

#### Prerequisites
- Shopify store (any plan)
- Store admin access

#### Setup Steps

1. **Create Private App**:
   ```
   - Go to Apps → Manage private apps
   - Create private app
   - Enable Admin API access
   - Set required permissions
   ```

2. **Get Credentials**:
   ```
   API Key: your_api_key
   Password: your_password (acts as access token)
   Shop Domain: yourstore.myshopify.com
   ```

3. **Connect in LiftOS**:
   ```json
   {
     "platform": "shopify",
     "auth_type": "api_key",
     "credentials": {
       "shop_domain": "yourstore.myshopify.com",
       "access_token": "your_password"
     }
   }
   ```

---

## Data Quality & Monitoring

### Quality Metrics Dashboard

Access data quality monitoring at **📊 Data Quality Monitoring**:

#### Key Metrics

| Metric | Description | Target |
|--------|-------------|---------|
| **Completeness** | Percentage of expected data received | > 95% |
| **Accuracy** | Data validation success rate | > 98% |
| **Timeliness** | Data freshness (time since last update) | < 2 hours |
| **Consistency** | Cross-platform data consistency | > 90% |

#### Quality Alerts

Configure alerts for quality issues:

```json
{
  "quality_alerts": {
    "completeness_threshold": 0.90,
    "accuracy_threshold": 0.95,
    "timeliness_threshold_hours": 4,
    "notification_channels": ["email", "slack"]
  }
}
```

### Real-Time Monitoring

#### Connection Health Dashboard

Monitor platform connections at **📊 Dashboard** tab:

```json
{
  "connection_health": {
    "meta_business": {
      "status": "healthy",
      "last_sync": "2024-01-01T12:00:00Z",
      "response_time_ms": 245,
      "error_rate": 0.001
    },
    "google_ads": {
      "status": "warning",
      "last_sync": "2024-01-01T11:30:00Z",
      "response_time_ms": 1200,
      "error_rate": 0.05
    }
  }
}
```

#### Sync Status Monitoring

Track data synchronization progress:

```json
{
  "sync_status": {
    "active_syncs": 3,
    "completed_today": 24,
    "failed_syncs": 1,
    "average_duration_minutes": 5.2,
    "next_scheduled": "2024-01-01T13:00:00Z"
  }
}
```

### Performance Optimization

#### Sync Frequency Optimization

Optimize sync frequencies based on data patterns:

```python
# Recommended sync frequencies by platform
OPTIMAL_SYNC_FREQUENCIES = {
    "meta_business": "1_hour",      # High volume, frequent changes
    "google_ads": "2_hours",        # Moderate volume
    "klaviyo": "30_minutes",        # Real-time email metrics
    "shopify": "1_hour",            # E-commerce transactions
    "hubspot": "4_hours",           # CRM data, less frequent changes
    "salesforce": "6_hours"         # Enterprise CRM, stable data
}
```

---

## Troubleshooting

### Common Issues

#### 1. Authentication Failures

**Symptom**: "Invalid credentials" error
**Solutions**:
```bash
# Check credential format
curl -X POST "http://localhost:8006/api/v1/platform-connections/connections/test" \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"connection_id": "conn_123"}'

# Verify API key format
# Meta Business: Should start with EAABwzLixnjYBO
# Klaviyo: Should start with pk_live_ or pk_test_
# Google Ads: Developer token format varies
```

#### 2. Rate Limiting

**Symptom**: "Rate limit exceeded" errors
**Solutions**:
```json
{
  "rate_limit_handling": {
    "retry_strategy": "exponential_backoff",
    "max_retries": 3,
    "base_delay_seconds": 60,
    "respect_platform_limits": true
  }
}
```

#### 3. Data Quality Issues

**Symptom**: Low quality scores or missing data
**Solutions**:
```python
# Check data validation logs
GET /api/v1/data-quality/validation-logs?platform=meta_business

# Review sync errors
GET /api/v1/platform-connections/connections/{id}/sync-history

# Adjust quality thresholds
PUT /api/v1/platform-connections/preferences
{
  "quality_threshold": 0.85  # Lower threshold temporarily
}
```

#### 4. Service Connectivity

**Symptom**: "Service unavailable" errors
**Solutions**:
```bash
# Check service health
curl http://localhost:8006/health
curl http://localhost:8001/health

# Verify network connectivity
telnet api.facebook.com 443
telnet googleads.googleapis.com 443

# Check DNS resolution
nslookup api.facebook.com
nslookup googleads.googleapis.com
```

### Debug Mode

Enable debug logging for troubleshooting:

```bash
# Set debug environment variables
export DEBUG=true
export LOG_LEVEL=DEBUG
export ENABLE_REQUEST_LOGGING=true

# Restart services with debug logging
cd services/data-ingestion && python app.py
```

### Log Analysis

#### Key Log Locations

```bash
# Service logs
tail -f logs/data-ingestion.log
tail -f logs/auth-service.log
tail -f logs/platform-connections.log

# Security audit logs
tail -f logs/security-audit.log

# Database query logs
tail -f logs/database.log
```

#### Log Patterns to Monitor

```bash
# Authentication issues
grep "AUTH_FAILED" logs/auth-service.log

# API rate limiting
grep "RATE_LIMIT" logs/data-ingestion.log

# Data quality failures
grep "QUALITY_CHECK_FAILED" logs/data-ingestion.log

# Platform API errors
grep "PLATFORM_ERROR" logs/platform-connections.log
```

---

## Best Practices

### Security Best Practices

#### 1. Credential Management

```python
# ✅ DO: Use environment variables for sensitive data
API_KEY = os.getenv("KLAVIYO_API_KEY")

# ❌ DON'T: Hardcode credentials
API_KEY = "pk_live_abc123def456..."  # Never do this

# ✅ DO: Rotate credentials regularly
schedule_credential_rotation(
    platform="meta_business",
    rotation_interval_days=90
)

# ✅ DO: Use least privilege principle
required_scopes = ["ads_read"]  # Only request needed permissions
```

#### 2. Data Protection

```python
# ✅ DO: Encrypt sensitive data at rest
encrypted_data = vault.encrypt(sensitive_data)

# ✅ DO: Use HTTPS for all API calls
session = requests.Session()
session.verify = True  # Always verify SSL certificates

# ✅ DO: Implement data retention policies
configure_data_retention(
    marketing_data_days=730,    # 2 years
    audit_logs_days=2555,       # 7 years
    temp_data_hours=24          # 24 hours
)
```

### Performance Best Practices

#### 1. Sync Optimization

```python
# ✅ DO: Implement incremental syncs
sync_config = {
    "sync_type": "incremental",
    "last_sync_timestamp": "2024-01-01T12:00:00Z",
    "batch_size": 1000
}

# ✅ DO: Use appropriate sync frequencies
SYNC_FREQUENCIES = {
    "real_time_platforms": "15_minutes",    # Klaviyo, Shopify
    "hourly_platforms": "1_hour",           # Meta, Google Ads
    "daily_platforms": "24_hours"           # CRM systems
}

# ✅ DO: Implement circuit breakers
circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=300,
    expected_exception=PlatformAPIError
)
```

#### 2. Resource Management

```python
# ✅ DO: Use connection pooling
connection_pool = ConnectionPool(
    max_connections=20,
    max_keepalive_connections=5,
    keepalive_expiry=30
)

# ✅ DO: Implement caching
cache_config = {
    "platform_metadata": "1_hour",
    "user_preferences": "15_minutes",
    "api_responses": "5_minutes"
}

# ✅ DO: Monitor resource usage
monitor_metrics = [
    "cpu_usage",
    "memory_usage", 
    "database_connections",
    "api_response_times"
]
```

### Monitoring Best Practices

#### 1. Alerting Strategy

```yaml
# alerts.yaml
alerts:
  - name: "High Error Rate"
    condition: "error_rate > 0.05"
    duration: "5m"
    severity: "warning"
    
  - name: "Platform API Down"
    condition: "platform_availability < 0.95"
    duration: "2m"
    severity: "critical"
    
  - name: "Data Quality Degradation"
    condition: "data_quality_score < 0.90"
    duration: "10m"
    severity: "warning"
```

#### 2. Dashboard Configuration

```json
{
  "dashboards": {
    "operational": {
      "metrics": ["response_times", "error_rates", "throughput"],
      "refresh_interval": "30s"
    },
    "business": {
      "metrics": ["data_freshness", "platform_coverage", "sync_success_rate"],
      "refresh_interval": "5m"
    },
    "security": {
      "metrics": ["failed_logins", "api_key_usage", "suspicious_activity"],
      "refresh_interval": "1m"
    }
  }
}
```

### Data Governance

#### 1. Data Classification

```python
DATA_CLASSIFICATION = {
    "public": {
        "examples": ["aggregated_metrics", "public_campaign_data"],
        "retention": "indefinite",
        "encryption": "optional"
    },
    "internal": {
        "examples": ["detailed_analytics", "user_preferences"],
        "retention": "2_years",
        "encryption": "required"
    },
    "confidential": {
        "examples": ["api_keys", "personal_data"],
        "retention": "as_required",
        "encryption": "required_aes256"
    }
}
```

#### 2. Compliance Monitoring

```python
# GDPR compliance checks
def check_gdpr_compliance():
    return {
        "data_minimization": verify_minimal_data_collection(),
        "purpose_limitation": verify_purpose_compliance(),
        "data_accuracy": verify_data_accuracy(),
        "storage_limitation": verify_retention_policies(),
        "security": verify_security_measures()
    }

# SOC 2 compliance monitoring
def monitor_soc2_controls():
    return {
        "access_controls": audit_access_controls(),
        "system_monitoring": verify_monitoring_systems(),
        "change_management": audit_change_processes(),
        "incident_response": verify_incident_procedures()
    }
```

---

## Migration Checklist

### Pre-Migration

- [ ] **Environment Setup**: Configure environment variables
- [ ] **Service Health**: Verify all services are running
- [ ] **Database**: Confirm database connectivity and schema
- [ ] **Security**: Set up JWT secrets and encryption keys
- [ ] **Network**: Test connectivity to platform APIs

### Platform Connection

- [ ] **Meta Business**: OAuth app created and configured
- [ ] **Google Ads**: Developer token approved and API enabled
- [ ] **Klaviyo**: API key generated with required permissions
- [ ] **Shopify**: Private app created with necessary scopes
- [ ] **Additional Platforms**: Configure as needed

### Data Configuration

- [ ] **Preferences**: Set live data preferences
- [ ] **Sync Frequency**: Configure optimal sync schedules
- [ ] **Quality Thresholds**: Set appropriate quality standards
- [ ] **Retention Policies**: Configure data retention rules

### Testing & Validation

- [ ] **Connection Tests**: Verify all platform connections
- [ ] **Data Sync**: Perform initial data synchronization
- [ ] **Quality Checks**: Validate data quality metrics
- [ ] **Dashboard**: Confirm live data appears in dashboards
- [ ] **Alerts**: Test monitoring and alerting systems

### Post-Migration

- [ ] **Performance**: Monitor system performance
- [ ] **Security**: Review security audit logs
- [ ] **User Training**: Train users on live data features
- [ ] **Documentation**: Update internal documentation
- [ ] **Backup**: Implement data backup procedures

---

## Support & Resources

### Getting Help

- **Technical Support**: support@liftos.com
- **Documentation**: [docs.liftos.com](https://docs.liftos.com)
- **Community Forum**: [community.liftos.com](https://community.liftos.com)
- **Status Page**: [status.liftos.com](https://status.liftos.com)

### Additional Resources

- [Platform Integration Guides](https://docs.liftos.com/integrations)
- [Security Best Practices](https://docs.liftos.com/security)
- [API Reference](https://docs.liftos.com/api)
- [Troubleshooting Guide](https://docs.liftos.com/troubleshooting)

### Training Materials

- [Live Data Mode Video Tutorial](https://training.liftos.com/live-data)
- [Platform Connection Webinar](https://training.liftos.com/connections)
- [Security Configuration Workshop](https://training.liftos.com/security)