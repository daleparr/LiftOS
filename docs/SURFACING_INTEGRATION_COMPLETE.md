# LiftOS Surfacing Module Integration - Complete Guide

## Overview

The LiftOS Surfacing Module has been successfully integrated into the LiftOS Core ecosystem. This integration bridges the existing Node.js/Fastify-based surfacing service with the Python/FastAPI-based LiftOS architecture.

## Integration Architecture

### Components Created

1. **Python Wrapper Module** (`modules/surfacing/`)
   - FastAPI-based wrapper that interfaces with the Node.js service
   - Full LiftOS authentication and memory integration
   - Compatible with LiftOS module system

2. **Docker Configuration**
   - Production and development Docker Compose configurations
   - Separate containers for Node.js service and Python wrapper
   - Health checks and proper networking

3. **Automation Scripts**
   - Windows batch file for automated setup
   - Python test suite for validation
   - Module registration automation

4. **Documentation**
   - Complete integration guide
   - API documentation
   - Troubleshooting instructions

## File Structure

```
LiftOS/
├── modules/surfacing/
│   ├── app.py                    # Python FastAPI wrapper
│   ├── module.json              # Module configuration
│   ├── requirements.txt         # Python dependencies
│   └── Dockerfile              # Python wrapper container
├── external/surfacing/          # Node.js service (cloned from GitHub)
├── Dockerfile.surfacing-service # Node.js service container
├── docker-compose.production.yml # Updated with surfacing services
├── docker-compose.dev.yml       # Development overrides
├── scripts/
│   ├── setup_surfacing.bat     # Windows setup automation
│   ├── setup_surfacing.sh      # Linux/Mac setup automation
│   ├── register_surfacing_module.py # Module registration
│   └── test_surfacing_integration.py # Integration tests
└── docs/
    ├── SURFACING_INTEGRATION_GUIDE.md
    └── SURFACING_INTEGRATION_COMPLETE.md
```

## Prerequisites

### System Requirements
- Docker Desktop installed and running
- Python 3.9+ with pip
- Git for repository cloning
- Windows PowerShell or Linux/Mac terminal

### LiftOS Core Services
The following LiftOS services must be running:
- API Gateway (port 8000)
- Authentication Service (port 8001)
- Registry Service (port 8002)
- Memory Service (port 8003)
- PostgreSQL Database (port 5432)
- Redis Cache (port 6379)

## Quick Start

### 1. Automated Setup (Recommended)

**Windows:**
```batch
scripts\setup_surfacing.bat
```

**Linux/Mac:**
```bash
chmod +x scripts/setup_surfacing.sh
./scripts/setup_surfacing.sh
```

### 2. Manual Setup

If you prefer manual setup or need to troubleshoot:

#### Step 1: Clone Surfacing Repository
```bash
mkdir -p external
cd external
git clone https://github.com/daleparr/Lift-os-surfacing.git surfacing
cd ..
```

#### Step 2: Build Docker Images
```bash
# Build Node.js surfacing service
docker build -f Dockerfile.surfacing-service -t liftos/surfacing-service:latest .

# Build Python wrapper module
docker build -f modules/surfacing/Dockerfile -t liftos/surfacing:latest modules/surfacing
```

#### Step 3: Start Services
```bash
# Development environment
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Production environment
docker-compose -f docker-compose.yml -f docker-compose.production.yml up -d
```

#### Step 4: Register Module
```bash
python scripts/register_surfacing_module.py
```

## Service Endpoints

### Direct Node.js Service
- **URL:** http://localhost:3001
- **Health:** GET /health
- **Analyze:** POST /api/analyze

### Python Wrapper Module
- **URL:** http://localhost:8007
- **Health:** GET /health
- **Analyze:** POST /api/v1/analyze
- **Batch Analyze:** POST /api/v1/batch-analyze
- **Optimize:** POST /api/v1/optimize

### Through LiftOS Gateway
- **Base URL:** http://localhost:8000
- **Module Endpoints:** /api/v1/modules/surfacing/*
- **Authentication:** Required (Bearer token)

## API Usage Examples

### 1. Direct Analysis (via Gateway)

```bash
curl -X POST http://localhost:8000/api/v1/modules/surfacing/analyze \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a document to analyze for key insights and patterns.",
    "user_id": "user-123",
    "options": {
      "extract_keywords": true,
      "analyze_sentiment": true,
      "store_in_memory": true,
      "memory_tags": ["analysis", "document"]
    }
  }'
```

### 2. Batch Analysis

```bash
curl -X POST http://localhost:8000/api/v1/modules/surfacing/batch-analyze \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {"id": "doc1", "text": "First document content"},
      {"id": "doc2", "text": "Second document content"}
    ],
    "user_id": "user-123",
    "options": {
      "extract_keywords": true,
      "parallel_processing": true
    }
  }'
```

### 3. Memory Integration

```bash
curl -X POST http://localhost:8000/api/v1/modules/surfacing/analyze \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Important business document requiring long-term storage.",
    "user_id": "user-123",
    "options": {
      "store_in_memory": true,
      "memory_tags": ["business", "important"],
      "memory_metadata": {
        "document_type": "business_plan",
        "priority": "high"
      }
    }
  }'
```

## Testing the Integration

### Automated Test Suite

Run the comprehensive test suite:

```bash
python scripts/test_surfacing_integration.py
```

The test suite validates:
- Service health and availability
- Authentication flow
- Direct Node.js service functionality
- Python wrapper endpoints
- Gateway integration
- Memory service integration
- Error handling
- Basic performance metrics

### Manual Testing

1. **Health Check All Services:**
   ```bash
   curl http://localhost:8000/health
   curl http://localhost:3001/health
   curl http://localhost:8007/health
   ```

2. **Test Direct Service:**
   ```bash
   curl -X POST http://localhost:3001/api/analyze \
     -H "Content-Type: application/json" \
     -d '{"text": "Test document", "options": {"extract_keywords": true}}'
   ```

3. **Test Through Gateway:**
   ```bash
   curl -X POST http://localhost:8000/api/v1/modules/surfacing/analyze \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"text": "Test document", "user_id": "test"}'
   ```

## Configuration

### Environment Variables

The integration uses the following environment variables:

```env
# Surfacing Service Configuration
SURFACING_SERVICE_URL=http://surfacing-service:3001
SURFACING_SERVICE_TIMEOUT=30

# Module Configuration
SURFACING_MODULE_PORT=8007
SURFACING_MODULE_HOST=0.0.0.0

# Memory Integration
MEMORY_SERVICE_URL=http://memory:8003
ENABLE_MEMORY_STORAGE=true

# Authentication
AUTH_SERVICE_URL=http://auth:8001
JWT_SECRET_KEY=your-secret-key
```

### Module Configuration

The module is configured via `modules/surfacing/module.json`:

```json
{
  "name": "surfacing",
  "version": "1.0.0",
  "description": "Advanced text analysis and surfacing capabilities",
  "capabilities": [
    "text_analysis",
    "keyword_extraction",
    "sentiment_analysis",
    "batch_processing",
    "memory_integration"
  ],
  "permissions": [
    "memory:read",
    "memory:write",
    "user:context"
  ],
  "endpoints": {
    "analyze": "/api/v1/analyze",
    "batch_analyze": "/api/v1/batch-analyze",
    "optimize": "/api/v1/optimize"
  }
}
```

## Monitoring and Observability

### Health Checks

All services include health check endpoints:
- **Node.js Service:** GET /health
- **Python Module:** GET /health
- **Through Gateway:** GET /api/v1/modules/surfacing/health

### Logging

Logs are available through Docker:

```bash
# View all surfacing logs
docker-compose logs surfacing-service surfacing

# Follow logs in real-time
docker-compose logs -f surfacing-service surfacing

# View specific service logs
docker-compose logs surfacing-service
docker-compose logs surfacing
```

### Metrics

The integration provides metrics through:
- Response time tracking
- Error rate monitoring
- Memory usage statistics
- Request volume metrics

## Troubleshooting

### Common Issues

1. **Docker Not Running**
   ```
   Error: Docker is not running
   Solution: Start Docker Desktop and ensure it's running
   ```

2. **Port Conflicts**
   ```
   Error: Port already in use
   Solution: Check for conflicting services on ports 3001, 8007
   ```

3. **Module Registration Failed**
   ```
   Error: Failed to register module
   Solution: Ensure registry service is running and accessible
   ```

4. **Authentication Errors**
   ```
   Error: 401 Unauthorized
   Solution: Verify JWT token is valid and properly formatted
   ```

### Debug Commands

```bash
# Check service status
docker-compose ps

# View service logs
docker-compose logs surfacing-service
docker-compose logs surfacing

# Test connectivity
curl http://localhost:3001/health
curl http://localhost:8007/health

# Check module registration
curl http://localhost:8002/api/v1/modules

# Restart services
docker-compose restart surfacing-service surfacing
```

### Performance Optimization

1. **Memory Usage:**
   - Monitor container memory usage
   - Adjust memory limits in docker-compose files
   - Implement caching for frequently accessed data

2. **Response Times:**
   - Enable connection pooling
   - Implement request caching
   - Optimize database queries

3. **Scalability:**
   - Use horizontal scaling with multiple container instances
   - Implement load balancing
   - Consider async processing for batch operations

## Security Considerations

### Authentication
- All requests require valid JWT tokens
- User context is properly propagated
- Service-to-service communication is secured

### Data Protection
- Sensitive data is encrypted in transit
- Memory storage follows LiftOS security policies
- Input validation prevents injection attacks

### Network Security
- Services communicate through internal Docker networks
- External access is controlled through the API gateway
- Health check endpoints are properly secured

## Development Guidelines

### Adding New Features

1. **Extend Node.js Service:**
   - Add new endpoints to the original surfacing service
   - Update API documentation

2. **Update Python Wrapper:**
   - Add corresponding endpoints in `modules/surfacing/app.py`
   - Implement proper error handling and validation
   - Update module configuration

3. **Test Integration:**
   - Add tests to `scripts/test_surfacing_integration.py`
   - Verify gateway routing works correctly
   - Test memory integration if applicable

### Code Quality

- Follow LiftOS coding standards
- Implement comprehensive error handling
- Add proper logging and monitoring
- Write unit and integration tests

## Production Deployment

### Prerequisites
- Production Docker environment
- Load balancer configuration
- Monitoring and alerting setup
- Backup and recovery procedures

### Deployment Steps

1. **Build Production Images:**
   ```bash
   docker build -f Dockerfile.surfacing-service -t liftos/surfacing-service:v1.0.0 .
   docker build -f modules/surfacing/Dockerfile -t liftos/surfacing:v1.0.0 modules/surfacing
   ```

2. **Deploy with Production Configuration:**
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.production.yml up -d
   ```

3. **Verify Deployment:**
   ```bash
   python scripts/test_surfacing_integration.py
   ```

4. **Monitor Services:**
   - Check health endpoints
   - Monitor logs for errors
   - Verify performance metrics

## Support and Maintenance

### Regular Maintenance
- Monitor service health and performance
- Update dependencies regularly
- Review and rotate security credentials
- Backup configuration and data

### Getting Help
- Check the troubleshooting section above
- Review service logs for error details
- Consult LiftOS Core documentation
- Contact the development team for complex issues

## Conclusion

The LiftOS Surfacing Module integration is now complete and ready for use. The integration provides:

✅ **Seamless Integration** - Node.js service works within LiftOS ecosystem
✅ **Full Authentication** - Proper JWT token validation and user context
✅ **Memory Integration** - Analysis results can be stored in LiftOS memory
✅ **API Gateway Support** - Unified access through LiftOS gateway
✅ **Docker Containerization** - Easy deployment and scaling
✅ **Comprehensive Testing** - Automated test suite for validation
✅ **Production Ready** - Full monitoring and error handling

The surfacing capabilities are now available to all LiftOS users and can be accessed through the standard LiftOS API patterns.