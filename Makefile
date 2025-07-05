# Lift OS Core - Development Makefile

.PHONY: help dev-up dev-down logs test clean build init-memory create-module register-module surfacing-up surfacing-down surfacing-register surfacing-status surfacing-test

# Default target
help:
	@echo "Lift OS Core - Available Commands:"
	@echo ""
	@echo "  dev-up          Start all services in development mode"
	@echo "  dev-down        Stop all services"
	@echo "  logs            View logs from all services"
	@echo "  test            Run all tests"
	@echo "  clean           Clean up containers and volumes"
	@echo "  build           Build all Docker images"
	@echo "  init-memory     Initialize sample memory contexts"
	@echo "  create-module   Create new module (usage: make create-module name=my-module)"
	@echo "  register-module Register module with core (usage: make register-module name=my-module)"
	@echo ""
	@echo "Surfacing Module Commands:"
	@echo "  surfacing-up    Start surfacing services only"
	@echo "  surfacing-down  Stop surfacing services"
	@echo "  surfacing-register Register surfacing module with registry"
	@echo "  surfacing-status   Check surfacing module status"
	@echo "  surfacing-test     Test surfacing module functionality"
	@echo ""

# Development environment
dev-up:
	@echo "🚀 Starting Lift OS Core development environment..."
	@cp .env.example .env 2>/dev/null || true
	@docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
	@echo "✅ Services started!"
	@echo ""
	@echo "Access points:"
	@echo "  - UI Shell: http://localhost:3000"
	@echo "  - API Gateway: http://localhost:8000"
	@echo "  - API Docs: http://localhost:8000/docs"
	@echo "  - Auth Service: http://localhost:8001/docs"
	@echo "  - Billing Service: http://localhost:8002/docs"
	@echo "  - Memory Service: http://localhost:8003/docs"
	@echo "  - Observability Service: http://localhost:8004/docs"
	@echo "  - Registry Service: http://localhost:8005/docs"
	@echo "  - Lift Causal Module: http://localhost:9001/docs"
	@echo "  - Surfacing Module: http://localhost:9005/docs"
	@echo "  - Surfacing Service: http://localhost:3000/docs"
	@echo ""

dev-down:
	@echo "🛑 Stopping Lift OS Core services..."
	@docker-compose -f docker-compose.yml -f docker-compose.dev.yml down
	@echo "✅ Services stopped!"

# Logging
logs:
	@docker-compose -f docker-compose.yml -f docker-compose.dev.yml logs -f

logs-service:
	@docker-compose -f docker-compose.yml -f docker-compose.dev.yml logs -f $(service)

# Testing
test:
	@echo "🧪 Running tests..."
	@docker-compose -f docker-compose.yml -f docker-compose.dev.yml exec gateway python -m pytest tests/ -v
	@docker-compose -f docker-compose.yml -f docker-compose.dev.yml exec auth python -m pytest tests/ -v
	@docker-compose -f docker-compose.yml -f docker-compose.dev.yml exec memory python -m pytest tests/ -v
	@docker-compose -f docker-compose.yml -f docker-compose.dev.yml exec billing python -m pytest tests/ -v
	@docker-compose -f docker-compose.yml -f docker-compose.dev.yml exec observability python -m pytest tests/ -v
	@docker-compose -f docker-compose.yml -f docker-compose.dev.yml exec registry python -m pytest tests/ -v
	@docker-compose -f docker-compose.yml -f docker-compose.dev.yml exec lift-causal python -m pytest tests/ -v

test-service:
	@docker-compose -f docker-compose.yml -f docker-compose.dev.yml exec $(service) python -m pytest tests/ -v

# Cleanup
clean:
	@echo "🧹 Cleaning up..."
	@docker-compose -f docker-compose.yml -f docker-compose.dev.yml down -v
	@docker system prune -f
	@echo "✅ Cleanup complete!"

# Build
build:
	@echo "🔨 Building all Docker images..."
	@docker-compose -f docker-compose.yml -f docker-compose.dev.yml build
	@echo "✅ Build complete!"

# Memory initialization
init-memory:
	@echo "🧠 Initializing sample memory contexts..."
	@docker-compose -f docker-compose.yml -f docker-compose.dev.yml exec memory python scripts/init_memory.py
	@echo "✅ Memory contexts initialized!"

# Module management
create-module:
	@if [ -z "$(name)" ]; then \
		echo "❌ Error: Please provide a module name. Usage: make create-module name=my-module"; \
		exit 1; \
	fi
	@echo "📦 Creating module: $(name)..."
	@mkdir -p modules/$(name)
	@cp -r modules/_template/* modules/$(name)/
	@sed -i 's/template_module/$(name)/g' modules/$(name)/*
	@echo "✅ Module $(name) created in modules/$(name)/"

register-module:
	@if [ -z "$(name)" ]; then \
		echo "❌ Error: Please provide a module name. Usage: make register-module name=my-module"; \
		exit 1; \
	fi
	@echo "📝 Registering module: $(name)..."
	@curl -X POST http://localhost:8004/modules \
		-H "Content-Type: application/json" \
		-d @modules/$(name)/module.json
	@echo "✅ Module $(name) registered!"

# Database operations
db-migrate:
	@echo "🗃️ Running database migrations..."
	@docker-compose -f docker-compose.yml -f docker-compose.dev.yml exec gateway alembic upgrade head
	@echo "✅ Migrations complete!"

db-reset:
	@echo "🗃️ Resetting database..."
	@docker-compose -f docker-compose.yml -f docker-compose.dev.yml down -v
	@docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d postgres redis
	@sleep 5
	@docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
	@echo "✅ Database reset complete!"

# Development utilities
shell-gateway:
	@docker-compose -f docker-compose.yml -f docker-compose.dev.yml exec gateway bash

shell-auth:
	@docker-compose -f docker-compose.yml -f docker-compose.dev.yml exec auth bash

shell-memory:
	@docker-compose -f docker-compose.yml -f docker-compose.dev.yml exec memory bash

shell-billing:
	@docker-compose -f docker-compose.yml -f docker-compose.dev.yml exec billing bash

shell-observability:
	@docker-compose -f docker-compose.yml -f docker-compose.dev.yml exec observability bash

shell-registry:
	@docker-compose -f docker-compose.yml -f docker-compose.dev.yml exec registry bash

shell-ui:
	@docker-compose -f docker-compose.yml -f docker-compose.dev.yml exec ui-shell bash

shell-lift-causal:
	@docker-compose -f docker-compose.yml -f docker-compose.dev.yml exec lift-causal bash

shell-lift-eval:
	@docker-compose -f docker-compose.yml -f docker-compose.dev.yml exec lift-eval bash

# Surfacing Module Commands
surfacing-up:
	@echo "🌊 Starting Surfacing services..."
	@cp .env.example .env 2>/dev/null || true
	@docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d surfacing-service surfacing
	@echo "✅ Surfacing services started!"
	@echo ""
	@echo "Access points:"
	@echo "  - Surfacing Module: http://localhost:9005/docs"
	@echo "  - Surfacing Service: http://localhost:3000/docs"
	@echo "  - Gateway Access: http://localhost:8000/modules/surfacing/"
	@echo ""

surfacing-down:
	@echo "🛑 Stopping Surfacing services..."
	@docker-compose -f docker-compose.yml -f docker-compose.dev.yml stop surfacing surfacing-service
	@echo "✅ Surfacing services stopped!"

surfacing-register:
	@echo "📝 Registering Surfacing module..."
	@python scripts/register_surfacing_module.py register
	@echo "✅ Surfacing module registered!"

surfacing-status:
	@echo "🔍 Checking Surfacing module status..."
	@python scripts/register_surfacing_module.py status

surfacing-test:
	@echo "🧪 Testing Surfacing module..."
	@echo "Testing surfacing service health..."
	@curl -f http://localhost:3000/health || echo "❌ Surfacing service not responding"
	@echo ""
	@echo "Testing surfacing module health..."
	@curl -f http://localhost:9005/health || echo "❌ Surfacing module not responding"
	@echo ""
	@echo "Testing gateway access..."
	@curl -f http://localhost:8000/modules/surfacing/ || echo "❌ Gateway access failed"
	@echo ""
	@echo "✅ Surfacing tests completed!"

shell-surfacing:
	@docker-compose -f docker-compose.yml -f docker-compose.dev.yml exec surfacing bash

shell-surfacing-service:
	@docker-compose -f docker-compose.yml -f docker-compose.dev.yml exec surfacing-service sh

logs-surfacing:
	@docker-compose -f docker-compose.yml -f docker-compose.dev.yml logs -f surfacing surfacing-service

# Testing commands
.PHONY: test test-unit test-integration test-e2e test-coverage

test: test-unit test-integration
	@echo "All tests completed"

test-unit:
	@echo "Running unit tests..."
	@docker-compose exec gateway pytest tests/ -m "unit" -v

test-integration:
	@echo "Running integration tests..."
	@docker-compose exec gateway pytest tests/ -m "integration" -v

test-e2e:
	@echo "Running end-to-end tests..."
	@docker-compose exec gateway pytest tests/ -m "e2e" -v

test-coverage:
	@echo "Running tests with coverage..."
	@docker-compose exec gateway pytest tests/ --cov=. --cov-report=html --cov-report=term

test-performance:
	@echo "Running performance tests..."
	@docker-compose exec gateway pytest tests/ -m "slow" -v

# Production commands
prod-up:
	@echo "🚀 Starting Lift OS Core in production mode..."
	@docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
	@echo "✅ Production services started!"

prod-down:
	@docker-compose -f docker-compose.yml -f docker-compose.prod.yml down

# Health checks
health:
	@echo "🏥 Checking service health..."
	@curl -s http://localhost:8000/health || echo "❌ Gateway unhealthy"
	@curl -s http://localhost:8001/health || echo "❌ Auth unhealthy"
	@curl -s http://localhost:8002/health || echo "❌ Memory unhealthy"
	@curl -s http://localhost:8003/health || echo "❌ Billing unhealthy"
	@curl -s http://localhost:8004/health || echo "❌ Registry unhealthy"
	@curl -s http://localhost:8005/health || echo "❌ Observability unhealthy"
	@echo "✅ Health check complete!"