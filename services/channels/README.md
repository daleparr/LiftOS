# Channels Service - Cross-Channel Budget Optimizer

The Channels service is a sophisticated microservice within the LiftOS ecosystem that provides advanced budget optimization capabilities for marketing channels. It transforms LiftOS from a diagnostic tool to a strategic copilot by delivering actionable budget reallocation recommendations based on causal inference, saturation curves, and Bayesian uncertainty quantification.

## 🎯 Overview

The Channels service enables CMOs and marketing teams to make data-driven budget allocation decisions across different marketing channels (Meta, Google, TikTok, etc.) by:

- **Multi-objective optimization** with advanced algorithms (NSGA-II, Bayesian optimization)
- **Monte Carlo simulation** for what-if scenario modeling
- **Saturation modeling** with Hill/Adstock functions for diminishing returns analysis
- **Bayesian inference** for uncertainty quantification and prior specification
- **Intelligent recommendations** with prioritized action plans

## 🏗️ Architecture

### Core Components

1. **Optimization Engine** (`engines/optimization_engine.py`)
   - Multi-objective budget optimization using NSGA-II and Bayesian methods
   - Constraint handling and convergence monitoring
   - Integration with causal effects and saturation models

2. **Simulation Engine** (`engines/simulation_engine.py`)
   - Monte Carlo simulation for scenario analysis
   - Risk assessment and uncertainty quantification
   - What-if modeling for strategic planning

3. **Saturation Engine** (`engines/saturation_engine.py`)
   - Hill, Adstock, Exponential, Logarithmic, and Power saturation functions
   - Automated model calibration and validation
   - Diminishing returns analysis

4. **Bayesian Engine** (`engines/bayesian_engine.py`)
   - Prior specification and posterior updating
   - Credible interval calculation
   - Model averaging and uncertainty estimation

5. **Recommendation Engine** (`engines/recommendation_engine.py`)
   - Intelligent recommendation generation
   - Priority scoring and ranking
   - Implementation planning and risk assessment

### Service Integration

The Channels service integrates with other LiftOS services through async HTTP clients:

- **Lift Causal Service**: Causal effect estimates and attribution
- **Data Ingestion Service**: Real-time and historical performance data
- **Memory Service**: Caching and persistent storage
- **Bayesian Analysis Service**: Advanced statistical modeling

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- PostgreSQL 13+
- Redis 6+
- Docker (optional)

### Installation

1. **Clone the repository**
   ```bash
   cd services/channels
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Run database migrations**
   ```bash
   alembic upgrade head
   ```

5. **Start the service**
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

### Docker Deployment

```bash
# Build the image
docker build -t liftos-channels .

# Run the container
docker run -p 8000:8000 \
  -e DATABASE_URL=postgresql://... \
  -e REDIS_URL=redis://... \
  liftos-channels
```

## 📊 API Endpoints

### Budget Optimization

**POST** `/optimize/budget`
```json
{
  "org_id": "org_123",
  "total_budget": 10000.0,
  "channels": ["google_ads", "meta_ads", "tiktok_ads"],
  "objectives": ["MAXIMIZE_REVENUE", "MAXIMIZE_ROAS"],
  "objective_weights": {
    "MAXIMIZE_REVENUE": 0.6,
    "MAXIMIZE_ROAS": 0.4
  },
  "constraints": [
    {
      "constraint_type": "MIN_SPEND",
      "channel_id": "google_ads",
      "min_value": 1000.0
    }
  ],
  "confidence_threshold": 0.8,
  "risk_tolerance": "moderate",
  "use_bayesian_optimization": true,
  "monte_carlo_samples": 10000
}
```

### Scenario Simulation

**POST** `/simulate/scenarios`
```json
{
  "org_id": "org_123",
  "channels": ["google_ads", "meta_ads"],
  "scenarios": [
    {
      "scenario_type": "BUDGET_CHANGE",
      "name": "20% Budget Increase",
      "parameters": {
        "budget_changes": {
          "google_ads": 1.2,
          "meta_ads": 1.2
        }
      }
    }
  ],
  "monte_carlo_samples": 10000,
  "confidence_level": 0.95
}
```

### Recommendations

**GET** `/recommendations/{org_id}`

Returns prioritized recommendations based on recent optimization results.

### Channel Management

**GET** `/channels/{org_id}`
**POST** `/channels/{org_id}`
**PUT** `/channels/{org_id}/{channel_id}`
**DELETE** `/channels/{org_id}/{channel_id}`

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CHANNELS_PORT` | Service port | `8000` |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://...` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379/0` |
| `CAUSAL_SERVICE_URL` | Causal service endpoint | `http://localhost:8003` |
| `DATA_INGESTION_SERVICE_URL` | Data ingestion endpoint | `http://localhost:8001` |
| `MEMORY_SERVICE_URL` | Memory service endpoint | `http://localhost:8002` |
| `BAYESIAN_ANALYSIS_SERVICE_URL` | Bayesian service endpoint | `http://localhost:8004` |
| `KSE_ENABLED` | Enable KSE integration | `true` |
| `KSE_BASE_URL` | KSE service endpoint | `http://localhost:9000` |

### Optimization Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `OPT_MAX_ITERATIONS` | Maximum optimization iterations | `1000` |
| `OPT_CONVERGENCE_TOL` | Convergence tolerance | `1e-6` |
| `OPT_POPULATION_SIZE` | Population size for genetic algorithms | `50` |
| `DEFAULT_MC_SAMPLES` | Default Monte Carlo samples | `10000` |
| `MAX_MC_SAMPLES` | Maximum Monte Carlo samples | `100000` |

## 🧪 Testing

### Run Unit Tests
```bash
pytest test_channels_service.py -v
```

### Run Integration Tests
```bash
pytest test_channels_service.py -v -m integration
```

### Run Performance Tests
```bash
pytest test_channels_service.py -v -m performance
```

## 📈 Usage Examples

### Basic Budget Optimization

```python
import httpx

async def optimize_budget():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/optimize/budget",
            json={
                "org_id": "my_org",
                "total_budget": 50000.0,
                "channels": ["google_ads", "meta_ads", "tiktok_ads"],
                "objectives": ["MAXIMIZE_REVENUE"],
                "confidence_threshold": 0.8
            }
        )
        result = response.json()
        print(f"Recommended allocation: {result['recommended_allocation']}")
        print(f"Expected improvement: {result['performance_improvement']}")
```

### Scenario Analysis

```python
async def run_scenario_analysis():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/simulate/scenarios",
            json={
                "org_id": "my_org",
                "channels": ["google_ads", "meta_ads"],
                "scenarios": [
                    {
                        "scenario_type": "BUDGET_CHANGE",
                        "name": "Budget Cut Scenario",
                        "parameters": {
                            "budget_changes": {
                                "google_ads": 0.8,
                                "meta_ads": 0.8
                            }
                        }
                    }
                ],
                "monte_carlo_samples": 5000
            }
        )
        result = response.json()
        for scenario in result['scenario_results']:
            print(f"Scenario: {scenario['scenario_name']}")
            print(f"Expected revenue: ${scenario['expected_performance']['revenue']['mean']:,.0f}")
            print(f"Risk level: {scenario['risk_metrics']['probability_of_loss']:.1%}")
```

## 🔍 Monitoring and Observability

### Health Checks

- **Service Health**: `GET /health`
- **Dependency Health**: `GET /health/dependencies`
- **Detailed Status**: `GET /status`

### Metrics

The service exposes Prometheus metrics on port 9090:

- `channels_optimization_requests_total`
- `channels_optimization_duration_seconds`
- `channels_simulation_requests_total`
- `channels_recommendation_generation_duration_seconds`

### Logging

Structured JSON logging with configurable levels:

```json
{
  "timestamp": "2025-01-09T15:30:45Z",
  "level": "INFO",
  "service": "channels",
  "org_id": "org_123",
  "optimization_id": "opt_456",
  "message": "Budget optimization completed",
  "duration_ms": 2500,
  "channels_optimized": 3
}
```

## 🔒 Security

### Authentication

The service integrates with LiftOS authentication:

- JWT token validation
- Organization-based access control
- Rate limiting per organization

### Data Protection

- Sensitive data encryption at rest
- Secure inter-service communication
- Audit logging for all optimization requests

## 🚀 Performance

### Optimization Performance

- **Small problems** (≤5 channels): < 5 seconds
- **Medium problems** (6-15 channels): < 30 seconds  
- **Large problems** (16+ channels): < 2 minutes

### Simulation Performance

- **10K samples**: < 10 seconds
- **100K samples**: < 60 seconds
- **Parallel execution** for multiple scenarios

### Scalability

- Horizontal scaling with load balancer
- Async processing for long-running optimizations
- Redis caching for frequently accessed data

## 🛠️ Development

### Project Structure

```
services/channels/
├── app.py                          # FastAPI application
├── config.py                       # Configuration management
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container configuration
├── test_channels_service.py        # Test suite
├── models/
│   └── channels.py                 # Pydantic data models
├── engines/
│   ├── optimization_engine.py      # Budget optimization
│   ├── simulation_engine.py        # Monte Carlo simulation
│   ├── saturation_engine.py        # Saturation modeling
│   ├── bayesian_engine.py          # Bayesian inference
│   └── recommendation_engine.py    # Recommendation generation
└── integrations/
    └── service_clients.py          # Service integration clients
```

### Adding New Optimization Objectives

1. Add objective to `OptimizationObjective` enum in `models/channels.py`
2. Implement objective function in `optimization_engine.py`
3. Add objective weights to default configuration
4. Update API documentation

### Adding New Saturation Functions

1. Add function to `SaturationFunction` enum
2. Implement function in `saturation_engine.py`
3. Add fitting method for parameter estimation
4. Update model validation logic

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Run the test suite
5. Submit a pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all functions
- Add docstrings for public methods
- Maintain test coverage > 80%

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:

- **Documentation**: [LiftOS Docs](https://docs.liftos.ai)
- **Issues**: [GitHub Issues](https://github.com/liftos/liftos/issues)
- **Slack**: #channels-service

## 🗺️ Roadmap

### Q1 2025
- [ ] Advanced constraint handling
- [ ] Multi-period optimization
- [ ] Enhanced visualization APIs

### Q2 2025
- [ ] Real-time optimization
- [ ] A/B testing integration
- [ ] Advanced attribution modeling

### Q3 2025
- [ ] Machine learning model integration
- [ ] Automated model retraining
- [ ] Advanced risk modeling