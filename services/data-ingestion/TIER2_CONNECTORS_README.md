# Tier 2 API Connectors - CRM and Payment Attribution
## LiftOS v1.3.0 - Advanced Marketing Attribution Systems

### Overview

The Tier 2 API connectors provide advanced CRM and payment attribution capabilities for LiftOS, enabling sophisticated causal marketing analysis across customer relationship management and payment processing platforms. These connectors implement dual processing architectures, advanced treatment assignment logic, and comprehensive KSE (Knowledge Space Embedding) integration.

### Supported Platforms

#### CRM Platforms
- **HubSpot CRM** - Lead scoring, deal attribution, and lifecycle stage tracking
- **Salesforce CRM** - Enterprise opportunity management and lead conversion analysis

#### Payment Platforms  
- **Stripe** - Payment intent analysis, subscription tracking, and customer behavior attribution
- **PayPal** - Transaction analysis, merchant attribution, and payment method insights

### Key Features

#### 🔄 Dual Processing Architecture
Each connector processes multiple conversion types:
- **HubSpot**: Deal conversions + MQL (Marketing Qualified Lead) conversions
- **Salesforce**: Opportunity conversions + Lead conversions  
- **Stripe**: Payment conversions + Subscription conversions
- **PayPal**: Payment conversions + Transaction conversions

#### 🎯 Advanced Treatment Assignment
Sophisticated attribution logic considering:
- **CRM Attribution**: Lead sources, campaign attribution, lifecycle stages
- **Payment Attribution**: UTM parameters, metadata analysis, customer journey tracking
- **Multi-touch Attribution**: First-touch, last-touch, and time-decay models
- **Cross-platform Integration**: Unified attribution across CRM and payment data

#### 🧠 KSE Universal Substrate Integration
- **Neural Content Generation**: AI-enhanced data enrichment
- **Conceptual Spaces**: Multi-dimensional knowledge representation
- **Knowledge Graphs**: Relationship mapping between entities
- **Semantic Enhancement**: Context-aware data processing

#### 📊 Data Quality & Confounder Detection
- **CRM Confounders**: Deal stage probability, lead scores, company size, sales velocity
- **Payment Confounders**: Payment amounts, customer history, geographic factors
- **Quality Scoring**: Comprehensive data quality assessment (0.0-1.0 scale)
- **Anomaly Detection**: Automated identification of data quality issues

### Installation & Setup

#### Prerequisites
```bash
# Python 3.9+ required
python --version

# Install Tier 2 dependencies
pip install -r requirements_tier2.txt
```

#### Environment Variables
```bash
# Core LiftOS services
export MEMORY_SERVICE_URL="http://localhost:8003"
export KSE_SERVICE_URL="http://localhost:8004"
export REDIS_URL="redis://localhost:6379"
export DATABASE_URL="postgresql://user:pass@localhost:5432/liftos"

# HubSpot Configuration
export HUBSPOT_API_KEY="your_hubspot_api_key"

# Salesforce Configuration  
export SALESFORCE_USERNAME="your_salesforce_username"
export SALESFORCE_PASSWORD="your_salesforce_password"
export SALESFORCE_SECURITY_TOKEN="your_security_token"
export SALESFORCE_CLIENT_ID="your_connected_app_client_id"
export SALESFORCE_CLIENT_SECRET="your_connected_app_client_secret"
export SALESFORCE_IS_SANDBOX="false"  # Set to "true" for sandbox

# Stripe Configuration
export STRIPE_API_KEY="sk_live_your_stripe_secret_key"
export STRIPE_WEBHOOK_SECRET="whsec_your_webhook_secret"

# PayPal Configuration
export PAYPAL_CLIENT_ID="your_paypal_client_id"
export PAYPAL_CLIENT_SECRET="your_paypal_client_secret"
export PAYPAL_IS_SANDBOX="false"  # Set to "true" for sandbox
export PAYPAL_WEBHOOK_ID="your_webhook_id"
```

### Platform-Specific Setup

#### HubSpot CRM Connector

**Authentication**: API Key (Bearer Token)
**Rate Limit**: 600 requests/minute
**Required Scopes**: 
- `crm.objects.deals.read`
- `crm.objects.contacts.read`
- `crm.objects.companies.read`
- `timeline`

**Setup Steps**:
1. Generate API key in HubSpot Developer Account
2. Configure OAuth scopes for data access
3. Set up webhook endpoints for real-time sync
4. Test connection with sample deal data

**Data Processing**:
- **Primary Events**: Deal conversions (closed-won deals)
- **Secondary Events**: MQL conversions (lifecycle stage changes)
- **Attribution Sources**: Original source, recent source, campaign data
- **Confounder Detection**: Deal probability, lead scores, company size

#### Salesforce CRM Connector

**Authentication**: OAuth 2.0 Username-Password Flow
**Rate Limit**: 4000 requests/hour
**Required Permissions**:
- `View All Data` or specific object permissions
- `API Enabled` user permission
- Connected App with OAuth settings

**Setup Steps**:
1. Create Connected App in Salesforce Setup
2. Configure OAuth settings and callback URLs
3. Generate security token for API access
4. Set up SOQL query permissions
5. Test with sample opportunity data

**Data Processing**:
- **Primary Events**: Opportunity conversions (closed-won opportunities)
- **Secondary Events**: Lead conversions (qualified leads)
- **Attribution Sources**: Lead source, campaign influence, opportunity source
- **Confounder Detection**: Opportunity probability, lead rating, account size

#### Stripe Payment Connector

**Authentication**: API Key (Secret Key)
**Rate Limit**: 80 requests/second
**Required Permissions**: Read access to payments, customers, subscriptions

**Setup Steps**:
1. Obtain secret API key from Stripe Dashboard
2. Configure webhook endpoints for real-time events
3. Set up metadata tracking for attribution
4. Test with sample payment data

**Data Processing**:
- **Primary Events**: Payment conversions (successful payment intents)
- **Secondary Events**: Subscription conversions (new subscriptions)
- **Attribution Sources**: UTM parameters in metadata, customer data
- **Confounder Detection**: Payment amounts, customer history, payment methods

#### PayPal Payment Connector

**Authentication**: OAuth 2.0 Client Credentials
**Rate Limit**: 300 requests/minute
**Required Scopes**: 
- `https://uri.paypal.com/services/payments/payment/authcapture`
- `https://uri.paypal.com/services/payments/refund`

**Setup Steps**:
1. Create PayPal Developer App
2. Configure OAuth 2.0 credentials
3. Set up webhook notifications
4. Configure custom field tracking
5. Test with sample transaction data

**Data Processing**:
- **Primary Events**: Payment conversions (approved payments)
- **Secondary Events**: Subscription conversions (recurring payments)
- **Attribution Sources**: Custom fields, invoice numbers, payer information
- **Confounder Detection**: Payment amounts, payer history, protection eligibility

### API Usage

#### Starting Data Sync
```python
import asyncio
from datetime import date
from connectors.hubspot_connector import HubSpotConnector

async def sync_hubspot_data():
    connector = HubSpotConnector(api_key="your_api_key")
    
    causal_data = await connector.extract_causal_marketing_data(
        org_id="your_org_id",
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 31),
        historical_data=[]
    )
    
    print(f"Extracted {len(causal_data)} causal marketing records")
    await connector.close()

# Run sync
asyncio.run(sync_hubspot_data())
```

#### REST API Endpoints
```bash
# Start sync job
POST /sync/start
{
  "platform": "hubspot",
  "date_range_start": "2024-01-01",
  "date_range_end": "2024-01-31",
  "sync_type": "full"
}

# Check sync status
GET /sync/jobs/{job_id}

# List all sync jobs
GET /sync/jobs
```

### Data Models

#### CausalMarketingData Structure
```python
{
  "record_id": "hubspot_deal_123456789",
  "data_source": "hubspot",
  "event_type": "conversion",
  "conversion_type": "deal_closed_won",
  "conversion_value": 5000.0,
  "conversion_date": "2024-01-15T10:30:00Z",
  "treatment_assignment": {
    "treatment_group": "paid_ads_google",
    "assignment_method": "source_based",
    "confidence_score": 0.95
  },
  "confounders": {
    "deal_stage_probability": 1.0,
    "lead_score": 85,
    "company_size": "mid_market"
  },
  "kse_enhancement": {
    "neural_content": "AI-generated insights...",
    "conceptual_space": {...},
    "knowledge_graph": {...}
  },
  "data_quality_score": 0.92
}
```

### Monitoring & Health Checks

#### Health Check Endpoint
```bash
GET /health
```

#### Metrics Available
- **Sync Performance**: Records processed, success rates, processing time
- **API Rate Limiting**: Request counts, rate limit status, throttling events
- **Data Quality**: Quality scores, validation errors, anomaly detection
- **KSE Integration**: Enhancement success rates, processing latency

#### Logging
```python
# Structured logging with correlation IDs
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "service": "tier2_connectors",
  "connector": "hubspot",
  "org_id": "org_123",
  "job_id": "job_456",
  "message": "Successfully processed 150 deals",
  "metrics": {
    "records_processed": 150,
    "processing_time_ms": 2500,
    "data_quality_avg": 0.89
  }
}
```

### Deployment

#### Automated Deployment
```bash
# Deploy Tier 2 connectors
python deploy_tier2_connectors.py

# Run tests
python -m pytest test_tier2_connectors.py -v

# Start service
uvicorn app:app --host 0.0.0.0 --port 8007
```

#### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements_tier2.txt .
RUN pip install -r requirements_tier2.txt

COPY . .
EXPOSE 8007

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8007"]
```

### Troubleshooting

#### Common Issues

**HubSpot Rate Limiting**
```bash
# Error: Rate limit exceeded (600/min)
# Solution: Implement exponential backoff
# Check: Rate limiter configuration in connector
```

**Salesforce Authentication**
```bash
# Error: Invalid username/password/security token
# Solution: Verify credentials and security token
# Check: Connected App OAuth settings
```

**Stripe Webhook Verification**
```bash
# Error: Webhook signature verification failed
# Solution: Verify webhook secret configuration
# Check: Endpoint URL and signature validation
```

**PayPal OAuth Token Expiry**
```bash
# Error: Access token expired
# Solution: Implement automatic token refresh
# Check: OAuth 2.0 refresh token flow
```

#### Debug Mode
```python
# Enable debug logging
import logging
logging.getLogger("tier2_connectors").setLevel(logging.DEBUG)

# Test connector individually
connector = HubSpotConnector(api_key="test", debug=True)
```

### Performance Optimization

#### Rate Limiting Best Practices
- **HubSpot**: Batch requests, use pagination efficiently
- **Salesforce**: Optimize SOQL queries, use bulk API for large datasets
- **Stripe**: Implement request queuing, use expand parameters
- **PayPal**: Cache authentication tokens, batch transaction requests

#### Data Processing Optimization
- **Parallel Processing**: Process multiple records concurrently
- **Incremental Sync**: Only sync changed data since last run
- **Caching**: Cache frequently accessed reference data
- **Compression**: Compress large data payloads

### Security Considerations

#### Credential Management
- Store API keys in secure credential manager
- Rotate credentials regularly
- Use environment-specific credentials
- Implement credential validation

#### Data Privacy
- Encrypt sensitive data in transit and at rest
- Implement data retention policies
- Support GDPR/CCPA compliance requirements
- Audit data access and processing

### Support & Documentation

#### Additional Resources
- **API Documentation**: Available at `/docs` endpoint
- **OpenAPI Spec**: Available at `/openapi.json`
- **Health Dashboard**: Available at `/health` endpoint
- **Metrics Dashboard**: Available at `/metrics` endpoint

#### Getting Help
- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Comprehensive guides and examples
- **Community**: Join LiftOS community discussions
- **Support**: Enterprise support available

---

**Last Updated**: January 2024  
**Version**: 1.3.0  
**Tier**: 2 (CRM and Payment Attribution)