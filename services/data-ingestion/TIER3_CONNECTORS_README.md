# Tier 3 Connectors: Social/Analytics/Data Platforms

## Overview

Tier 3 connectors extend LiftOS data ingestion capabilities to include social media advertising, data warehouses, and analytics platforms. These connectors provide advanced data processing, ML workflow analysis, and comprehensive data quality assessment.

## Supported Platforms

### 1. TikTok for Business
- **Platform**: Social media advertising
- **API**: TikTok Business API v1.3
- **Data Types**: Video campaigns, ad groups, creative performance
- **Key Features**:
  - Video advertising campaign analysis
  - Audience engagement metrics
  - Creative performance tracking
  - Causal attribution analysis

### 2. Snowflake
- **Platform**: Cloud data warehouse
- **API**: Snowflake Python Connector
- **Data Types**: Tables, queries, data quality metrics
- **Key Features**:
  - Data warehouse analysis
  - Query performance optimization
  - Data quality assessment
  - Storage efficiency analysis

### 3. Databricks
- **Platform**: Unified analytics platform
- **API**: Databricks REST API 2.0
- **Data Types**: Clusters, jobs, notebooks, ML experiments
- **Key Features**:
  - ML workflow analysis
  - Cluster utilization optimization
  - Job execution monitoring
  - MLflow experiment tracking

## Architecture

### Connector Structure
```
connectors/
├── tiktok_connector.py          # TikTok Business API integration
├── snowflake_connector.py       # Snowflake data warehouse
└── databricks_connector.py      # Databricks analytics platform
```

### Key Components

#### 1. TikTok Connector
```python
class TikTokConnector:
    - get_advertisers()          # List TikTok advertisers
    - get_campaigns()            # Campaign performance data
    - get_campaign_metrics()     # Detailed metrics
    - extract_causal_marketing_data()  # Causal insights
    - enhance_with_kse()         # KSE integration
```

#### 2. Snowflake Connector
```python
class SnowflakeConnector:
    - get_table_metadata()       # Table information
    - get_query_history()        # Query performance
    - assess_data_quality()      # Quality metrics
    - execute_query()            # Custom SQL execution
    - extract_causal_insights()  # Usage patterns
```

#### 3. Databricks Connector
```python
class DatabricksConnector:
    - get_clusters()             # Cluster information
    - get_jobs()                 # Job execution data
    - get_notebooks()            # Notebook metadata
    - get_ml_experiments()       # MLflow experiments
    - get_ml_runs()              # ML run details
```

## Configuration

### Environment Variables

#### TikTok for Business
```bash
TIKTOK_ACCESS_TOKEN=your_access_token
TIKTOK_APP_ID=your_app_id
TIKTOK_APP_SECRET=your_app_secret
```

#### Snowflake
```bash
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_USERNAME=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_SCHEMA=your_schema
```

#### Databricks
```bash
DATABRICKS_HOST=your_workspace_url
DATABRICKS_TOKEN=your_access_token
DATABRICKS_CLUSTER_ID=your_cluster_id
```

### Credential Files
Alternatively, store credentials in JSON files:
```
/app/credentials/{org_id}/
├── tiktok.json
├── snowflake.json
└── databricks.json
```

## Data Models

### TikTok Campaign Data
```python
@dataclass
class TikTokCampaign:
    campaign_id: str
    campaign_name: str
    objective: str
    status: str
    budget: float
    spend: float
    impressions: int
    clicks: int
    conversions: int
    ctr: float
    cpc: float
    cpm: float
    conversion_rate: float
    created_time: str
    updated_time: str
```

### Snowflake Table Metadata
```python
@dataclass
class SnowflakeTable:
    table_name: str
    schema_name: str
    database_name: str
    row_count: int
    column_count: int
    table_type: str
    created: str
    last_altered: str
    bytes: int
    retention_time: int
```

### Databricks Job Data
```python
@dataclass
class DatabricksJob:
    job_id: int
    run_id: int
    job_name: str
    state: str
    life_cycle_state: str
    result_state: Optional[str]
    start_time: int
    end_time: Optional[int]
    execution_duration: Optional[int]
    creator_user_name: str
    run_type: str
```

## KSE Integration

All Tier 3 connectors include Knowledge Space Embedding (KSE) integration:

### Causal Memory Entries
- **TikTok**: Video advertising effectiveness analysis
- **Snowflake**: Data quality and query performance relationships
- **Databricks**: ML workflow optimization patterns

### Causal Relationships
- **Creative Quality → Engagement Rate** (TikTok)
- **Data Quality → Query Performance** (Snowflake)
- **Cluster Configuration → Job Performance** (Databricks)

## API Endpoints

### Sync Data
```http
POST /sync/start
Content-Type: application/json
X-Org-Id: your-org-id
X-User-Id: your-user-id

{
  "platform": "tiktok|snowflake|databricks",
  "date_range_start": "2025-01-01",
  "date_range_end": "2025-01-07",
  "sync_type": "full"
}
```

### Response Format
```json
{
  "success": true,
  "message": "Sync job started for tiktok",
  "data": {
    "job_id": "uuid",
    "platform": "tiktok",
    "status": "pending"
  }
}
```

## Testing

### Run Tier 3 Tests
```bash
cd services/data-ingestion
python -m pytest test_tier3_connectors.py -v
```

### Manual Testing
```bash
python test_tier3_connectors.py
```

## Performance Considerations

### TikTok Connector
- **Rate Limits**: 1000 requests/hour per app
- **Batch Size**: 50 campaigns per request
- **Retry Logic**: Exponential backoff for rate limits

### Snowflake Connector
- **Connection Pooling**: Reuse connections for multiple queries
- **Query Optimization**: Use LIMIT for large result sets
- **Warehouse Management**: Auto-suspend after inactivity

### Databricks Connector
- **API Limits**: 1000 requests/hour per workspace
- **Pagination**: Handle large result sets
- **Cluster State**: Check cluster availability before operations

## Error Handling

### Common Error Scenarios
1. **Authentication Failures**: Invalid credentials
2. **Rate Limiting**: API quota exceeded
3. **Network Issues**: Connection timeouts
4. **Data Quality**: Missing or invalid data

### Retry Strategies
- **Exponential Backoff**: For rate limits and temporary failures
- **Circuit Breaker**: For persistent service failures
- **Graceful Degradation**: Continue with partial data

## Monitoring

### Key Metrics
- **Sync Success Rate**: Percentage of successful syncs
- **Data Quality Score**: Average quality across platforms
- **Processing Time**: Time to complete sync operations
- **Error Rate**: Frequency of sync failures

### Logging
- **Structured Logging**: JSON format for easy parsing
- **Correlation IDs**: Track requests across services
- **Performance Metrics**: Execution times and resource usage

## Deployment

### Dependencies
```bash
pip install snowflake-connector-python==3.6.0
pip install databricks-sql-connector==2.9.3
```

### Docker Configuration
```dockerfile
# Add to Dockerfile
RUN pip install snowflake-connector-python databricks-sql-connector
```

### Health Checks
```bash
curl http://localhost:8006/health
```

## Security

### Credential Management
- **Environment Variables**: For development
- **Vault Integration**: For production
- **Encryption**: All credentials encrypted at rest

### Network Security
- **TLS/SSL**: All API communications encrypted
- **IP Whitelisting**: Restrict access to known IPs
- **API Keys**: Rotate regularly

## Troubleshooting

### Common Issues

#### TikTok Connection Failures
```bash
# Check credentials
curl -H "Access-Token: $TIKTOK_ACCESS_TOKEN" \
     https://business-api.tiktok.com/open_api/v1.3/advertiser/info/
```

#### Snowflake Connection Issues
```python
# Test connection
import snowflake.connector
conn = snowflake.connector.connect(
    account='your_account',
    user='your_user',
    password='your_password'
)
```

#### Databricks Authentication
```bash
# Test API access
curl -H "Authorization: Bearer $DATABRICKS_TOKEN" \
     https://your-workspace.cloud.databricks.com/api/2.0/clusters/list
```

## Future Enhancements

### Planned Features
1. **Real-time Streaming**: Live data ingestion
2. **Advanced Analytics**: Predictive modeling
3. **Custom Dashboards**: Platform-specific visualizations
4. **Automated Optimization**: ML-driven recommendations

### Integration Roadmap
- **Additional Platforms**: LinkedIn Ads, Pinterest Business
- **Enhanced KSE**: Deeper causal analysis
- **Performance Optimization**: Parallel processing
- **Advanced Security**: Zero-trust architecture

## Support

For issues or questions:
1. Check logs in `/var/log/liftos/data-ingestion.log`
2. Review API documentation for each platform
3. Contact the LiftOS development team
4. Submit issues via the project repository

---

**Last Updated**: January 2025  
**Version**: 1.0.0  
**Maintainer**: LiftOS Data Engineering Team