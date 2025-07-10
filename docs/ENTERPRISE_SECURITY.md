# LiftOS Enterprise Security Architecture

## Overview

LiftOS implements enterprise-grade security for API key management and authentication, designed to meet SOC 2 compliance requirements and handle confidential production data across 15+ marketing intelligence platforms.

## 🔐 Security Features

### Core Security Components

1. **API Key Vault** - AES-256-GCM encrypted storage with PBKDF2 key derivation
2. **Enhanced JWT Authentication** - RSA-256 signed tokens with refresh rotation
3. **Device Fingerprinting** - Multi-factor device identification and tracking
4. **Security Audit Logging** - SOC 2 compliant comprehensive audit trails
5. **Rate Limiting** - IP and endpoint-based request throttling
6. **Session Management** - Enhanced session tracking with security flags

### Supported Platforms

**Tier 1 (Core Marketing)**
- Meta Business API (Facebook/Instagram Ads)
- Google Ads API
- Klaviyo Email Marketing

**Tier 2 (E-commerce & CRM)**
- Shopify
- WooCommerce
- Amazon Seller Central
- HubSpot CRM
- Salesforce CRM

**Tier 3 (Payments & Social)**
- Stripe
- PayPal
- TikTok for Business
- LinkedIn Ads
- X (Twitter) Ads

**Tier 4 (Data & Analytics)**
- Snowflake
- Databricks
- Zoho CRM

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LiftOS Security Layer                    │
├─────────────────────────────────────────────────────────────┤
│  Enhanced Security Middleware                               │
│  ├── Rate Limiting                                          │
│  ├── Device Fingerprinting                                  │
│  ├── Risk Assessment                                        │
│  └── Security Headers                                       │
├─────────────────────────────────────────────────────────────┤
│  Enhanced JWT Manager                                       │
│  ├── RSA-256 Token Signing                                 │
│  ├── Refresh Token Rotation                                │
│  ├── Session Management                                     │
│  └── Token Revocation                                       │
├─────────────────────────────────────────────────────────────┤
│  API Key Vault                                             │
│  ├── AES-256-GCM Encryption                               │
│  ├── PBKDF2 Key Derivation                                │
│  ├── Automatic Key Rotation                               │
│  └── Usage Analytics                                       │
├─────────────────────────────────────────────────────────────┤
│  Security Audit Logger                                     │
│  ├── SOC 2 Compliance                                     │
│  ├── Event Classification                                  │
│  ├── Risk Scoring                                         │
│  └── Compliance Reporting                                 │
├─────────────────────────────────────────────────────────────┤
│  Database Security Models                                  │
│  ├── Encrypted API Keys                                   │
│  ├── Enhanced User Sessions                               │
│  ├── Security Audit Logs                                  │
│  ├── Revoked Tokens                                       │
│  └── Security Configuration                               │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Setup Security Infrastructure

```bash
# Run the security setup script
python scripts/setup_security.py [org-id]

# This will:
# - Create database tables
# - Generate RSA keys for JWT
# - Create master encryption key
# - Set up security configuration
# - Create environment template
```

### 2. Configure Environment

```bash
# Copy the environment template
cp .env.template .env

# Edit .env with your configuration
nano .env
```

### 3. Initialize Database

```bash
# Run database migrations
python shared/database/migration_runner.py

# Check migration status
python shared/database/migration_runner.py status
```

### 4. Test Security Setup

```bash
# Run security integration tests
python -m pytest tests/test_security_integration.py -v
```

## 📖 Usage Guide

### API Key Management

#### Store API Keys Securely

```python
from services.data_ingestion.enhanced_credential_manager import get_enhanced_credential_manager

manager = get_enhanced_credential_manager()

# Store Meta Business API credentials
credentials = {
    "access_token": "your-meta-access-token",
    "app_id": "your-app-id",
    "app_secret": "your-app-secret"
}

success = await manager.store_credentials_in_vault(
    org_id="your-org-id",
    provider="meta",
    credentials=credentials,
    created_by="user-id"
)
```

#### Retrieve API Keys

```python
# Retrieve credentials (automatically decrypted)
credentials = await manager.get_meta_business_credentials("your-org-id")

if credentials:
    access_token = credentials["access_token"]
    # Use the credentials for API calls
```

#### Rotate API Keys

```python
# Rotate credentials with new values
new_credentials = {
    "access_token": "new-meta-access-token",
    "app_id": "new-app-id", 
    "app_secret": "new-app-secret"
}

success = await manager.rotate_credentials(
    org_id="your-org-id",
    provider="meta",
    new_credentials=new_credentials,
    rotated_by="user-id"
)
```

### Authentication & Sessions

#### Create Authentication Session

```python
from shared.security.enhanced_jwt import get_enhanced_jwt_manager
from shared.security.enhanced_jwt import DeviceFingerprint

jwt_manager = get_enhanced_jwt_manager()

# Generate device fingerprint
device_fingerprint = DeviceFingerprint.generate_fingerprint(
    user_agent=request.headers.get('User-Agent'),
    ip_address=request.remote_addr
)

# Create token pair
access_token, refresh_token, session_info = await jwt_manager.create_token_pair(
    session=db_session,
    user_id="user-123",
    org_id="org-456",
    email="user@example.com",
    roles=["user", "admin"],
    permissions=["read", "write", "api_access"],
    device_fingerprint=device_fingerprint,
    ip_address=request.remote_addr,
    user_agent=request.headers.get('User-Agent')
)
```

#### Verify Access Token

```python
# Verify and decode access token
try:
    payload = await jwt_manager.verify_access_token(db_session, access_token)
    user_id = payload["sub"]
    org_id = payload["org_id"]
    roles = payload["roles"]
    permissions = payload["permissions"]
except ValueError as e:
    # Token is invalid or expired
    return {"error": "Invalid token"}, 401
```

#### Refresh Token

```python
# Refresh access token using refresh token
try:
    new_access_token, new_refresh_token = await jwt_manager.refresh_access_token(
        session=db_session,
        refresh_token=refresh_token,
        device_fingerprint=device_fingerprint,
        ip_address=request.remote_addr,
        user_agent=request.headers.get('User-Agent')
    )
except ValueError as e:
    # Refresh token is invalid or expired
    return {"error": "Invalid refresh token"}, 401
```

### Security Middleware Integration

#### Flask Application Setup

```python
from flask import Flask
from shared.security.enhanced_middleware import EnhancedSecurityMiddleware

app = Flask(__name__)

# Initialize security middleware
security_middleware = EnhancedSecurityMiddleware(app)

# Or initialize later
security_middleware = EnhancedSecurityMiddleware()
security_middleware.init_app(app)
```

#### Endpoint Protection

```python
from shared.security.enhanced_middleware import (
    require_auth, 
    require_roles, 
    require_permissions,
    require_low_risk
)

@app.route('/api/protected')
@require_auth
def protected_endpoint():
    user_id = g.security_context.user_id
    return {"message": f"Hello {user_id}"}

@app.route('/api/admin')
@require_auth
@require_roles('admin')
def admin_endpoint():
    return {"message": "Admin access granted"}

@app.route('/api/sensitive')
@require_auth
@require_permissions('sensitive_data_access')
@require_low_risk(max_risk=0.3)
def sensitive_endpoint():
    return {"message": "Sensitive data access granted"}
```

### Audit Logging

#### Log Security Events

```python
from shared.security.audit_logger import SecurityAuditLogger, SecurityEventType

audit_logger = SecurityAuditLogger()

# Log successful login
await audit_logger.log_security_event(
    session=db_session,
    event_type=SecurityEventType.LOGIN_SUCCESS,
    user_id="user-123",
    org_id="org-456",
    action="user_login",
    ip_address="192.168.1.100",
    user_agent="Mozilla/5.0...",
    success=True,
    details={"login_method": "password"}
)

# Log API key access
await audit_logger.log_api_key_access(
    session=db_session,
    org_id="org-456",
    provider="meta",
    key_name="default",
    action="retrieve",
    success=True,
    user_id="user-123"
)
```

## 🔧 Configuration

### Security Configuration

```python
# Default security configuration
{
    "max_concurrent_sessions": 3,
    "session_timeout_minutes": 480,  # 8 hours
    "require_mfa": False,
    "allowed_ip_ranges": ["0.0.0.0/0"],
    "api_rate_limits": {
        "auth": {"limit": 5, "window": 300},      # 5 attempts per 5 minutes
        "api": {"limit": 1000, "window": 3600},   # 1000 calls per hour
        "sensitive": {"limit": 10, "window": 60}   # 10 ops per minute
    },
    "password_policy": {
        "min_length": 12,
        "require_uppercase": True,
        "require_lowercase": True,
        "require_numbers": True,
        "require_symbols": True,
        "max_age_days": 90
    },
    "audit_retention_days": 365,
    "compliance_settings": {
        "soc2_compliance": True,
        "gdpr_compliance": True,
        "encryption_at_rest": True,
        "encryption_in_transit": True
    }
}
```

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/liftos

# JWT Configuration
JWT_ACCESS_SECRET=your-access-secret
JWT_REFRESH_SECRET=your-refresh-secret
JWT_PRIVATE_KEY_PATH=./security/keys/jwt_private_key.pem
JWT_PUBLIC_KEY_PATH=./security/keys/jwt_public_key.pem

# Encryption
MASTER_ENCRYPTION_KEY_PATH=./security/keys/master_encryption_key.txt

# Security Features
SECURITY_AUDIT_ENABLED=true
SECURITY_RATE_LIMITING_ENABLED=true
SECURITY_DEVICE_FINGERPRINTING_ENABLED=true
SECURITY_REQUIRE_HTTPS=true
```

## 🛡️ Security Best Practices

### Production Deployment

1. **Key Management**
   - Store RSA keys and master encryption key in secure key management service
   - Use environment variables for secrets, never hardcode
   - Implement key rotation schedule (90 days recommended)

2. **Network Security**
   - Use HTTPS/TLS 1.3 for all communications
   - Implement proper firewall rules
   - Use VPN or private networks for database access

3. **Access Control**
   - Implement principle of least privilege
   - Regular access reviews and permission audits
   - Multi-factor authentication for admin access

4. **Monitoring & Alerting**
   - Monitor security audit logs for anomalies
   - Set up alerts for failed authentication attempts
   - Track API key usage patterns

5. **Compliance**
   - Regular security assessments
   - Penetration testing
   - Compliance audits (SOC 2, GDPR)

### Key Rotation Schedule

- **API Keys**: 90 days or immediately if compromised
- **JWT Signing Keys**: 180 days
- **Master Encryption Key**: 365 days (with careful migration)
- **Database Credentials**: 90 days

## 📊 Monitoring & Analytics

### Security Metrics

```python
# Get credential usage analytics
analytics = await manager.get_credential_usage_analytics(
    org_id="your-org-id",
    provider="meta",  # Optional: specific provider
    days=30
)

# Analytics include:
# - Usage frequency
# - Success/error rates
# - Geographic distribution
# - Time-based patterns
```

### Audit Log Queries

```sql
-- Failed login attempts in last 24 hours
SELECT * FROM security_audit_logs 
WHERE event_type = 'LOGIN_FAILED' 
AND timestamp > NOW() - INTERVAL '24 hours';

-- High-risk security events
SELECT * FROM security_audit_logs 
WHERE risk_score > 0.7 
ORDER BY timestamp DESC;

-- API key access patterns
SELECT provider, COUNT(*) as access_count
FROM security_audit_logs 
WHERE event_type = 'API_KEY_ACCESS'
AND timestamp > NOW() - INTERVAL '7 days'
GROUP BY provider;
```

## 🔍 Troubleshooting

### Common Issues

1. **Database Connection Errors**
   ```bash
   # Check database connectivity
   python shared/database/migration_runner.py status
   ```

2. **JWT Token Verification Failures**
   ```bash
   # Verify RSA keys exist and are readable
   ls -la security/keys/
   ```

3. **API Key Decryption Errors**
   ```bash
   # Check master encryption key
   cat security/keys/master_encryption_key.txt
   ```

4. **Rate Limiting Issues**
   ```python
   # Check rate limit configuration
   from shared.security.enhanced_middleware import get_enhanced_security_middleware
   middleware = get_enhanced_security_middleware()
   print(middleware.default_rate_limits)
   ```

### Debug Mode

```python
# Enable debug logging
import logging
logging.getLogger('shared.security').setLevel(logging.DEBUG)
```

## 📚 API Reference

### API Key Vault

- `store_api_key()` - Store encrypted API key
- `get_api_key()` - Retrieve and decrypt API key
- `rotate_api_key()` - Rotate API key with new value
- `revoke_api_key()` - Revoke API key access
- `list_api_keys()` - List all API keys for organization
- `get_usage_analytics()` - Get usage statistics

### Enhanced JWT Manager

- `create_token_pair()` - Create access and refresh tokens
- `verify_access_token()` - Verify and decode access token
- `refresh_access_token()` - Refresh access token using refresh token
- `revoke_token()` - Revoke specific token

### Security Audit Logger

- `log_security_event()` - Log general security event
- `log_authentication_event()` - Log authentication-specific event
- `log_api_key_access()` - Log API key access event
- `log_suspicious_activity()` - Log suspicious activity

## 🤝 Contributing

### Adding New API Providers

1. **Update Provider Mapping**
   ```python
   # In enhanced_credential_manager.py
   self.provider_mapping = {
       "new_provider": "new_provider_key",
       # ... existing providers
   }
   ```

2. **Add Credential Methods**
   ```python
   async def get_new_provider_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
       # Implementation following existing pattern
   ```

3. **Update Tests**
   ```python
   # Add tests in test_security_integration.py
   ```

### Security Enhancements

1. Fork the repository
2. Create a feature branch
3. Implement security enhancement
4. Add comprehensive tests
5. Update documentation
6. Submit pull request

## 📄 License

This enterprise security implementation is part of LiftOS and follows the project's licensing terms.

## 🆘 Support

For security-related issues or questions:

1. **Security Issues**: Report privately to security team
2. **General Questions**: Create GitHub issue
3. **Documentation**: Check this guide and inline code documentation
4. **Enterprise Support**: Contact LiftOS enterprise support team

---

**⚠️ Security Notice**: This documentation contains security implementation details. Ensure proper access controls are in place when sharing or storing this information.