# LiftOS Agentic Microservice

The Agentic microservice provides comprehensive AI agent testing and evaluation capabilities for marketing analytics within the LiftOS ecosystem. It enables creation, management, and systematic evaluation of marketing-focused AI agents with robust testing frameworks and performance metrics.

## Overview

This microservice is inspired by and adapted from the [AgentSIM](https://github.com/daleparr/AgentSIM) framework, providing a Python-based implementation tailored for LiftOS's marketing analytics use cases. It offers:

- **Agent Lifecycle Management**: Create, configure, and manage marketing AI agents
- **Comprehensive Testing Framework**: Execute systematic tests with marketing scenarios
- **Multi-Category Evaluation**: 5-category assessment framework (Functionality, Reliability, Performance, Security, Usability)
- **Marketing-Specific Test Cases**: Pre-built test scenarios for common marketing use cases
- **Integration with LiftOS Services**: Seamless integration with Memory, Auth, LLM, and Causal services

## Architecture

### Core Components

```
modules/agentic/
├── app.py                          # FastAPI application with REST endpoints
├── models/                         # Data models and schemas
│   ├── agent_models.py            # Agent configuration and capabilities
│   ├── evaluation_models.py       # Evaluation results and metrics
│   └── test_models.py             # Test cases and scenarios
├── core/                          # Business logic
│   ├── agent_manager.py           # Agent lifecycle management
│   ├── evaluation_engine.py       # Multi-category evaluation framework
│   └── test_orchestrator.py       # Test execution coordination
├── services/                      # External service integrations
│   ├── memory_service.py          # LiftOS Memory Service integration
│   └── auth_service.py            # Authentication and authorization
├── utils/                         # Utilities and configuration
│   ├── config.py                  # Configuration management
│   └── logging_config.py          # Structured logging
├── agents/                        # Pre-built agent library
│   └── marketing_agent_library.py # Marketing-specific agents
├── test_cases/                    # Test case library
│   └── marketing_test_library.py  # Marketing test scenarios
└── tests/                         # Unit tests
    ├── test_agent_manager.py      # Agent manager tests
    └── conftest.py                # Test configuration
```

### Key Features

#### 1. Agent Management
- **Agent Types**: Content optimization, campaign analysis, audience targeting, creative testing
- **Model Configuration**: Support for OpenAI, Anthropic, and other LLM providers
- **Marketing Context**: Target audience, brand voice, campaign objectives, budget constraints
- **Capability Framework**: Extensible capability system for agent specialization

#### 2. Testing Framework
- **Test Scenarios**: Campaign optimization, A/B testing, audience segmentation, creative performance
- **Success Criteria**: Configurable metrics with thresholds and weights
- **Test Orchestration**: Parallel execution, dependency management, timeout handling
- **Result Tracking**: Comprehensive test result storage and analysis

#### 3. Evaluation Engine
- **5-Category Assessment**:
  - **Functionality (30%)**: Core feature performance and accuracy
  - **Reliability (25%)**: Consistency and error handling
  - **Performance (20%)**: Speed, efficiency, and resource usage
  - **Security (15%)**: Data protection and access control
  - **Usability (10%)**: User experience and interface quality
- **Metric Scoring**: Weighted scoring system with detailed breakdowns
- **Recommendations**: Automated improvement suggestions

## API Endpoints

### Agent Management
- `POST /agents` - Create new agent
- `GET /agents/{agent_id}` - Get agent details
- `PUT /agents/{agent_id}` - Update agent configuration
- `DELETE /agents/{agent_id}` - Delete agent
- `GET /agents` - List agents with filtering
- `POST /agents/{agent_id}/sessions` - Start agent session
- `DELETE /sessions/{session_id}` - Stop agent session

### Testing
- `POST /tests/execute` - Execute test case
- `GET /tests/{test_id}/results` - Get test results
- `GET /tests/scenarios` - List available test scenarios
- `POST /tests/scenarios` - Create custom test scenario
- `GET /tests/results` - List test results with filtering

### Evaluation
- `POST /evaluations/evaluate` - Evaluate agent performance
- `GET /evaluations/{evaluation_id}` - Get evaluation results
- `GET /evaluations/agents/{agent_id}` - Get agent evaluation history
- `GET /evaluations/metrics` - Get evaluation metrics summary

### System
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics
- `GET /info` - Service information

## Configuration

### Environment Variables

```bash
# Service Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
PORT=8000

# LiftOS Service URLs
MEMORY_SERVICE_URL=http://memory:8000
AUTH_SERVICE_URL=http://auth:8000
OBSERVABILITY_SERVICE_URL=http://observability:8000
LLM_SERVICE_URL=http://llm:8000
CAUSAL_SERVICE_URL=http://causal:8000

# AI Provider Configuration
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Database Configuration
DATABASE_URL=postgresql://user:password@postgres:5432/agentic
REDIS_URL=redis://redis:6379

# Evaluation Configuration
DEFAULT_TIMEOUT_SECONDS=300
MAX_CONCURRENT_TESTS=10
EVALUATION_WEIGHTS_FUNCTIONALITY=0.30
EVALUATION_WEIGHTS_RELIABILITY=0.25
EVALUATION_WEIGHTS_PERFORMANCE=0.20
EVALUATION_WEIGHTS_SECURITY=0.15
EVALUATION_WEIGHTS_USABILITY=0.10
```

## Deployment

### Docker Deployment

1. **Build and run with Docker Compose:**
```bash
cd modules/agentic
docker-compose up -d
```

2. **Environment setup:**
```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

3. **Initialize database:**
```bash
# Database will be automatically initialized with init.sql
docker-compose logs postgres
```

### Manual Deployment

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set environment variables:**
```bash
export ENVIRONMENT=production
export DATABASE_URL=postgresql://user:password@localhost:5432/agentic
# ... other variables
```

3. **Run the service:**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Usage Examples

### Creating a Marketing Agent

```python
import httpx

# Create content optimization agent
agent_data = {
    "agent_id": "content_optimizer_001",
    "name": "Email Content Optimizer",
    "agent_type": "content_optimizer",
    "description": "Optimizes email marketing content for engagement",
    "capabilities": [
        {
            "name": "content_generation",
            "description": "Generate optimized email content",
            "parameters": {
                "max_length": 1000,
                "tone": "professional",
                "personalization": True
            }
        }
    ],
    "model_config": {
        "provider": "openai",
        "model_name": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 1000
    },
    "marketing_context": {
        "target_audience": "B2B professionals",
        "brand_voice": "professional",
        "campaign_objectives": ["engagement", "conversion"],
        "budget_constraints": {
            "max_cost_per_execution": 5.0
        }
    }
}

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8007/agents",
        json=agent_data,
        headers={"Authorization": "Bearer your_token"}
    )
    print(response.json())
```

### Executing a Test Case

```python
# Execute A/B testing scenario
test_data = {
    "test_case_id": "email_ab_test_001",
    "agent_id": "content_optimizer_001",
    "scenario_id": "ab_testing_scenario",
    "test_parameters": {
        "variants": ["subject_a", "subject_b"],
        "sample_size": 1000,
        "confidence_level": 0.95
    }
}

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8007/tests/execute",
        json=test_data,
        headers={"Authorization": "Bearer your_token"}
    )
    print(response.json())
```

### Getting Evaluation Results

```python
# Get agent evaluation results
async with httpx.AsyncClient() as client:
    response = await client.get(
        "http://localhost:8007/evaluations/agents/content_optimizer_001",
        headers={"Authorization": "Bearer your_token"}
    )
    evaluation = response.json()
    
    print(f"Overall Score: {evaluation['overall_score']}")
    print(f"Functionality: {evaluation['category_scores']['functionality']}")
    print(f"Reliability: {evaluation['category_scores']['reliability']}")
```

## Testing

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-mock

# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_agent_manager.py -v
```

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: Service integration testing
- **API Tests**: REST endpoint testing
- **Performance Tests**: Load and stress testing

## Monitoring and Observability

### Metrics

The service exposes Prometheus metrics at `/metrics`:

- `agentic_agents_total` - Total number of agents
- `agentic_tests_executed_total` - Total tests executed
- `agentic_evaluations_completed_total` - Total evaluations completed
- `agentic_test_duration_seconds` - Test execution duration
- `agentic_evaluation_scores` - Evaluation score distributions

### Logging

Structured logging with correlation IDs:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "service": "agentic",
  "correlation_id": "req_123456",
  "message": "Agent test completed",
  "agent_id": "content_optimizer_001",
  "test_id": "test_789",
  "duration_ms": 1500,
  "success": true
}
```

### Health Checks

- **Liveness**: `/health` - Service availability
- **Readiness**: `/health/ready` - Service readiness
- **Dependencies**: Checks Memory, Auth, and LLM services

## Integration with LiftOS

### Memory Service Integration
- Agent configuration storage
- Test result persistence
- Evaluation history tracking
- Session state management

### Auth Service Integration
- JWT token validation
- Role-based access control
- User permission checking
- Audit logging

### LLM Service Integration
- Model execution requests
- Cost tracking and budgeting
- Provider abstraction
- Response caching

### Causal Service Integration
- Campaign impact analysis
- Attribution modeling
- Performance correlation
- Recommendation generation

## Development

### Adding New Agent Types

1. **Define agent capabilities in `models/agent_models.py`**
2. **Implement agent logic in `agents/marketing_agent_library.py`**
3. **Create test scenarios in `test_cases/marketing_test_library.py`**
4. **Add evaluation criteria in `core/evaluation_engine.py`**

### Adding New Test Scenarios

1. **Define scenario in `models/test_models.py`**
2. **Implement test logic in `test_cases/marketing_test_library.py`**
3. **Configure success criteria and metrics**
4. **Add to test orchestrator execution flow**

### Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Ensure all tests pass
5. Submit a pull request

## License

This microservice is part of the LiftOS ecosystem and follows the same licensing terms.

## Support

For support and questions:
- Create an issue in the LiftOS repository
- Contact the development team
- Check the LiftOS documentation

---

**Note**: This microservice is designed to integrate seamlessly with the LiftOS ecosystem and requires proper configuration of dependent services (Memory, Auth, LLM, Causal) for full functionality.