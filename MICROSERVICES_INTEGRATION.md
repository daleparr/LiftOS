# LiftOS Microservices Integration

This document provides comprehensive documentation for integrating external microservices with LiftOS Core using the established wrapper pattern.

## Overview

LiftOS Core supports integration of external Node.js and Python microservices through a standardized wrapper pattern that provides:

- **Seamless Integration**: External services remain unchanged while Python wrappers provide LiftOS compatibility
- **Authentication**: JWT-based authentication with user context propagation
- **Memory Integration**: Persistent storage of analysis results using KSE Memory SDK
- **API Gateway**: Unified routing and service discovery
- **Module Registry**: Dynamic module registration and capability discovery
- **Docker Orchestration**: Complete containerization with development and production configurations

## Integrated Microservices

### 1. Surfacing Microservice
**Repository**: https://github.com/daleparr/Lift-os-surfacing  
**Purpose**: Advanced text analysis, sentiment analysis, and content insights  
**Ports**: 3002 (Node.js service), 8007 (Python wrapper)

#### Capabilities
- Text sentiment analysis with confidence scores
- Named entity recognition and extraction
- Key phrase identification and ranking
- Content categorization and tagging
- Multi-language text processing
- Batch text analysis for large datasets

#### Key Endpoints
- `POST /analyze` - Comprehensive text analysis
- `POST /sentiment` - Sentiment analysis only
- `POST /entities` - Entity extraction only
- `POST /batch` - Batch processing for multiple texts
- `GET /health` - Service health check

### 2. Causal AI Microservice
**Repository**: https://github.com/daleparr/lift-causal-ai  
**Purpose**: Marketing attribution, causal inference, and budget optimization  
**Ports**: 3003 (Node.js/Python service), 8008 (Python wrapper)

#### Capabilities
- Multi-touch attribution modeling
- Media Mix Modeling (MMM) analysis
- Incrementality and lift measurement
- Budget optimization across channels
- Causal inference for marketing decisions
- Platform integration (Google Ads, Meta, TikTok, Klaviyo)

#### Key Endpoints
- `POST /attribution/analyze` - Attribution analysis
- `POST /mmm/analyze` - Media Mix Modeling
- `POST /lift/measure` - Lift measurement
- `POST /budget/optimize` - Budget optimization
- `POST /causal/inference` - Causal inference analysis
- `GET /health` - Service health check

### 3. LLM Microservice
**Repository**: https://github.com/daleparr/Lift-os-LLM
**Purpose**: Large Language Model evaluation, prompt engineering, and content generation
**Ports**: 3004 (Python service), 8009 (Python wrapper)

#### Capabilities
- Multi-provider LLM integration (OpenAI, Cohere, HuggingFace)
- Model evaluation and comparison with comprehensive metrics
- Prompt template management and optimization
- Content generation for marketing campaigns
- Model leaderboard and performance tracking
- Evaluation metrics: BLEU, ROUGE, BERTScore, RLHF scoring
- Batch processing for large-scale content generation
- A/B testing for prompt variations

#### Key Endpoints
- `POST /evaluate` - Comprehensive model evaluation
- `POST /generate` - Content generation with templates
- `POST /compare` - Compare multiple models
- `POST /metrics` - Calculate evaluation metrics
- `GET /leaderboard` - Model performance leaderboard
- `GET /templates` - Available prompt templates
- `POST /batch` - Batch content generation
- `GET /health` - Service health check

#### Evaluation Metrics
- **BLEU Score**: Measures n-gram overlap between generated and reference text
- **ROUGE Score**: Evaluates recall-oriented understanding for summarization
- **BERTScore**: Semantic similarity using contextual embeddings
- **RLHF Score**: Reinforcement Learning from Human Feedback evaluation

#### Prompt Templates
- **Marketing Copy**: Generate compelling marketing content
- **Product Descriptions**: Create detailed product descriptions
- **Email Campaigns**: Design effective email marketing content
- **Social Media**: Generate engaging social media posts

## Architecture Pattern

### Wrapper Pattern Implementation

```
External Service (Node.js/Python) ←→ Python Wrapper (FastAPI) ←→ LiftOS Core
```

Each integration follows this pattern:

1. **External Service**: Runs unchanged in its own container
2. **Python Wrapper**: FastAPI service that provides LiftOS integration
3. **LiftOS Integration**: Authentication, memory, registry, and API gateway

### Directory Structure

```
LiftOS/
├── modules/
│   ├── surfacing/
│   │   ├── app.py              # Python FastAPI wrapper
│   │   ├── module.json         # Module configuration
│   │   ├── requirements.txt    # Python dependencies
│   │   └── Dockerfile          # Container configuration
│   ├── causal/
│   │   ├── app.py              # Python FastAPI wrapper
│   │   ├── module.json         # Module configuration
│   │   ├── requirements.txt    # Python dependencies
│   │   └── Dockerfile          # Container configuration
│   └── llm/
│       ├── app.py              # Python FastAPI wrapper
│       ├── module.json         # Module configuration
│       ├── requirements.txt    # Python dependencies
│       └── Dockerfile          # Container configuration
├── external/
│   ├── surfacing/              # Cloned surfacing repository
│   ├── causal/                 # Cloned causal AI repository
│   └── llm/                    # Cloned LLM repository
├── scripts/
│   ├── setup_surfacing.bat     # Surfacing setup automation
│   ├── setup_causal.bat        # Causal AI setup automation
│   ├── setup_llm.bat           # LLM setup automation
│   ├── test_surfacing_integration.py
│   ├── test_causal_integration.py
│   ├── test_llm_integration.py
│   └── validate_complete_integration.py
├── Dockerfile.surfacing-service # External surfacing service
├── Dockerfile.causal-service   # External causal AI service
├── Dockerfile.llm-service      # External LLM service
├── docker-compose.production.yml
└── docker-compose.dev.yml
```

## Setup Instructions

### Prerequisites

1. **Docker Desktop** installed and running
2. **Git** for cloning repositories
3. **Python 3.9+** for running scripts
4. **Node.js 18+** for external services

### Quick Setup

#### 1. Clone External Repositories

```bash
# Clone surfacing microservice
git clone https://github.com/daleparr/Lift-os-surfacing external/surfacing

# Clone causal AI microservice
git clone https://github.com/daleparr/lift-causal-ai external/causal

# Clone LLM microservice
git clone https://github.com/daleparr/Lift-os-LLM external/llm
```

#### 2. Setup Surfacing Microservice

```bash
# Windows
scripts\setup_surfacing.bat

# Linux/Mac
chmod +x scripts/setup_surfacing.sh
./scripts/setup_surfacing.sh
```

#### 3. Setup Causal AI Microservice

```bash
# Windows
scripts\setup_causal.bat

# Linux/Mac
chmod +x scripts/setup_causal.sh
./scripts/setup_causal.sh
```

#### 4. Setup LLM Microservice

```bash
# Windows
scripts\setup_llm.bat

# Linux/Mac
chmod +x scripts/setup_llm.sh
./scripts/setup_llm.sh
```

#### 5. Validate Complete Integration

```bash
python scripts/validate_complete_integration.py
```

### Manual Setup

#### 1. Build Docker Images

```bash
# Build surfacing service
docker build -f Dockerfile.surfacing-service -t liftos-surfacing-service .

# Build surfacing module
docker build -f modules/surfacing/Dockerfile -t liftos-surfacing-module .

# Build causal service
docker build -f Dockerfile.causal-service -t liftos-causal-service .

# Build causal module
docker build -f modules/causal/Dockerfile -t liftos-causal-module .

# Build LLM service
docker build -f Dockerfile.llm-service -t liftos-llm-service .

# Build LLM module
docker build -f modules/llm/Dockerfile -t liftos-llm-module .
```

#### 2. Start Services

```bash
# Production
docker-compose -f docker-compose.production.yml up -d

# Development
docker-compose -f docker-compose.dev.yml up -d
```

#### 3. Register Modules

```bash
# Register surfacing module
python scripts/register_surfacing_module.py

# Register causal module
python scripts/register_causal_module.py

# Register LLM module
python scripts/register_llm_module.py
```

## Configuration

### Environment Variables

#### Production (.env.production)
```env
# Surfacing Service
SURFACING_SERVICE_URL=http://surfacing-service:3002
SURFACING_MODULE_URL=http://surfacing:8007

# Causal AI Service
CAUSAL_SERVICE_URL=http://causal-service:3003
CAUSAL_MODULE_URL=http://causal:8008

# LLM Service
LLM_SERVICE_URL=http://llm-service:3004
LLM_MODULE_URL=http://llm:8009

# Core Services
MEMORY_SERVICE_URL=http://memory:8003
AUTH_SERVICE_URL=http://auth:8001
REGISTRY_SERVICE_URL=http://registry:8005

# API Keys (for LLM providers)
OPENAI_API_KEY=your_openai_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
```

#### Development (.env.development)
```env
# Surfacing Service
SURFACING_SERVICE_URL=http://localhost:3002
SURFACING_MODULE_URL=http://localhost:8007

# Causal AI Service
CAUSAL_SERVICE_URL=http://localhost:3003
CAUSAL_MODULE_URL=http://localhost:8008

# LLM Service
LLM_SERVICE_URL=http://localhost:3004
LLM_MODULE_URL=http://localhost:8009

# Core Services
MEMORY_SERVICE_URL=http://localhost:8003
AUTH_SERVICE_URL=http://localhost:8001
REGISTRY_SERVICE_URL=http://localhost:8005

# API Keys (for LLM providers)
OPENAI_API_KEY=your_openai_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
```

### Port Allocation

| Service | Port | Purpose |
|---------|------|---------|
| API Gateway | 8000 | Main entry point |
| Auth Service | 8001 | Authentication |
| Memory Service | 8003 | Data persistence |
| Registry Service | 8005 | Module registry |
| Surfacing Service | 3002 | External Node.js service |
| Surfacing Module | 8007 | Python wrapper |
| Causal Service | 3003 | External Node.js/Python service |
| Causal Module | 8008 | Python wrapper |
| LLM Service | 3004 | External Python service |
| LLM Module | 8009 | Python wrapper |

## Usage Examples

### Text Analysis with Surfacing

```python
import requests

# Analyze text content
response = requests.post(
    "http://localhost:8007/analyze",
    json={
        "text": "Our marketing campaign exceeded expectations with 25% ROI improvement."
    },
    headers={"Authorization": "Bearer YOUR_JWT_TOKEN"}
)

result = response.json()
print(f"Sentiment: {result['sentiment']['label']}")
print(f"Entities: {result['entities']}")
```

### Attribution Analysis with Causal AI

```python
import requests

# Analyze campaign attribution
response = requests.post(
    "http://localhost:8008/attribution/analyze",
    json={
        "campaigns": [
            {
                "id": "camp_001",
                "platform": "google_ads",
                "spend": 10000,
                "conversions": 500,
                "revenue": 25000
            }
        ]
    },
    headers={"Authorization": "Bearer YOUR_JWT_TOKEN"}
)

result = response.json()
print(f"Attribution scores: {result['attribution_scores']}")
```

### LLM Content Generation and Evaluation

```python
import requests

# Generate marketing content using LLM
response = requests.post(
    "http://localhost:8009/generate",
    json={
        "prompt": "Create compelling marketing copy for a new eco-friendly product launch",
        "template": "marketing_copy",
        "context": {
            "product": "eco-friendly water bottle",
            "target_audience": "environmentally conscious consumers",
            "key_benefits": ["sustainable", "durable", "stylish"]
        }
    },
    headers={"Authorization": "Bearer YOUR_JWT_TOKEN"}
)

result = response.json()
print(f"Generated content: {result['content']}")

# Evaluate multiple models
evaluation_response = requests.post(
    "http://localhost:8009/evaluate",
    json={
        "prompt": "Write a product description for an eco-friendly water bottle",
        "models": ["gpt-4", "claude-3", "cohere-command"],
        "metrics": ["bleu", "rouge", "bertscore"]
    },
    headers={"Authorization": "Bearer YOUR_JWT_TOKEN"}
)

eval_result = evaluation_response.json()
print(f"Model performance: {eval_result['leaderboard']}")
```

### Integrated Workflow with LLM

```python
# 1. Analyze campaign text content
text_analysis = requests.post(
    "http://localhost:8007/analyze",
    json={"text": campaign_description}
).json()

# 2. Perform attribution analysis
attribution_analysis = requests.post(
    "http://localhost:8008/attribution/analyze",
    json={"campaigns": campaign_data}
).json()

# 3. Generate marketing recommendations using LLM
llm_recommendations = requests.post(
    "http://localhost:8009/generate",
    json={
        "prompt": "Generate marketing recommendations based on analysis results",
        "template": "marketing_recommendations",
        "context": {
            "sentiment": text_analysis["sentiment"],
            "attribution": attribution_analysis["attribution_scores"],
            "campaign_performance": campaign_data
        }
    }
).json()

# 4. Optimize budget based on all insights
budget_optimization = requests.post(
    "http://localhost:8008/budget/optimize",
    json={
        "total_budget": 50000,
        "sentiment_boost": text_analysis["sentiment"]["score"],
        "attribution_weights": attribution_analysis["attribution_scores"],
        "llm_recommendations": llm_recommendations["content"]
    }
).json()

print(f"Complete analysis with LLM insights: {budget_optimization}")
```

## Testing

### Individual Service Tests

```bash
# Test surfacing integration
python scripts/test_surfacing_integration.py

# Test causal AI integration
python scripts/test_causal_integration.py

# Test LLM integration
python scripts/test_llm_integration.py
```

### Complete Integration Test

```bash
# Test all three services working together
python scripts/validate_complete_integration.py
```

### Manual Testing

```bash
# Check service health
curl http://localhost:3002/health  # Surfacing service
curl http://localhost:8007/health  # Surfacing module
curl http://localhost:3003/health  # Causal service
curl http://localhost:8008/health  # Causal module
curl http://localhost:3004/health  # LLM service
curl http://localhost:8009/health  # LLM module

# Test through API gateway
curl http://localhost:8000/modules/surfacing/health
curl http://localhost:8000/modules/causal/health
curl http://localhost:8000/modules/llm/health
```

## Monitoring and Logs

### View Service Logs

```bash
# View all logs
docker-compose logs

# View specific service logs
docker-compose logs surfacing-service
docker-compose logs surfacing
docker-compose logs causal-service
docker-compose logs causal
docker-compose logs llm-service
docker-compose logs llm

# Follow logs in real-time
docker-compose logs -f surfacing
docker-compose logs -f causal
docker-compose logs -f llm
```

### Health Monitoring

All services provide health endpoints:
- Service health: `GET /health`
- Module health: `GET /health`
- Detailed status: `GET /status`

## Troubleshooting

### Common Issues

#### 1. Port Conflicts
```bash
# Check port usage
netstat -an | findstr :3002
netstat -an | findstr :8007

# Stop conflicting services
docker-compose down
```

#### 2. Service Not Starting
```bash
# Check Docker logs
docker-compose logs service-name

# Rebuild images
docker-compose build --no-cache service-name
```

#### 3. Module Registration Failed
```bash
# Check registry service
curl http://localhost:8005/health

# Re-register module
python scripts/register_surfacing_module.py
python scripts/register_causal_module.py
```

#### 4. Authentication Issues
```bash
# Check auth service
curl http://localhost:8001/health

# Verify JWT token format
```

### Debug Mode

Enable debug mode in development:

```bash
# Set debug environment variables
export DEBUG=true
export LOG_LEVEL=debug

# Start with debug logging
docker-compose -f docker-compose.dev.yml up
```

## Development

### Adding New Microservices

Follow the established pattern:

1. **Clone External Repository**
   ```bash
   git clone <repository-url> external/<service-name>
   ```

2. **Create Module Directory**
   ```bash
   mkdir modules/<service-name>
   ```

3. **Create Python Wrapper** (`modules/<service-name>/app.py`)
   - FastAPI application
   - LiftOS authentication integration
   - Memory service integration
   - External service communication

4. **Create Module Configuration** (`modules/<service-name>/module.json`)
   - Module metadata
   - Capabilities definition
   - Endpoint specifications

5. **Create Dockerfile** (`modules/<service-name>/Dockerfile`)
   - Python environment
   - Dependencies installation
   - Service configuration

6. **Create External Service Dockerfile** (`Dockerfile.<service-name>-service`)
   - External service environment
   - Service-specific configuration

7. **Update Docker Compose**
   - Add services to production and development configurations
   - Configure networking and dependencies

8. **Create Setup Scripts**
   - Automation script for deployment
   - Registration script for module registry
   - Integration test script

### Best Practices

1. **Wrapper Design**
   - Keep external services unchanged
   - Implement comprehensive error handling
   - Provide detailed logging
   - Follow LiftOS API patterns

2. **Configuration**
   - Use environment variables for configuration
   - Provide sensible defaults
   - Document all configuration options

3. **Testing**
   - Create comprehensive test suites
   - Test both individual and integrated functionality
   - Include performance and load testing

4. **Documentation**
   - Document all endpoints and capabilities
   - Provide usage examples
   - Include troubleshooting guides

## Security Considerations

### Authentication
- All module endpoints require JWT authentication
- User context is propagated through the system
- Service-to-service communication uses internal tokens

### Network Security
- Services communicate through Docker internal networks
- External access only through API gateway
- Environment-specific configurations

### Data Protection
- Sensitive data is encrypted in transit and at rest
- Memory service provides secure data persistence
- Audit logging for all operations

## Performance Optimization

### Caching
- Memory service provides intelligent caching
- Analysis results are cached for reuse
- Configurable cache TTL per analysis type

### Scaling
- Services can be horizontally scaled
- Load balancing through API gateway
- Database connection pooling

### Resource Management
- Container resource limits
- Memory usage optimization
- CPU allocation tuning

## Future Enhancements

### Planned Features
1. **Auto-scaling**: Automatic service scaling based on load
2. **Advanced Monitoring**: Comprehensive metrics and alerting
3. **Multi-tenancy**: Support for multiple organizations
4. **API Versioning**: Backward-compatible API evolution
5. **Real-time Processing**: Stream processing capabilities

### Integration Roadmap
1. **Additional Platforms**: Shopify, Amazon Ads, LinkedIn
2. **Advanced Analytics**: Predictive modeling, forecasting
3. **Automation**: Automated campaign optimization
4. **Reporting**: Advanced dashboard and reporting features

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review service logs
3. Run integration tests
4. Consult the LiftOS Core documentation

## License

This integration follows the same license as LiftOS Core. External microservices maintain their respective licenses.