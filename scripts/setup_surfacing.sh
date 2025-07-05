#!/bin/bash

# LiftOS Surfacing Integration Setup Script
set -e

echo "🌊 LiftOS Surfacing Integration Setup"
echo "===================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    print_error "Please run this script from the LiftOS root directory"
    exit 1
fi

print_status "Starting LiftOS Surfacing integration setup..."

# Step 1: Clone the surfacing repository
print_status "Step 1: Cloning surfacing repository..."
if [ ! -d "surfacing-service" ]; then
    print_status "Cloning Lift-os-surfacing repository..."
    git clone https://github.com/daleparr/Lift-os-surfacing.git surfacing-service
    print_success "Surfacing repository cloned successfully"
else
    print_warning "Surfacing service directory already exists, skipping clone"
fi

# Step 2: Check if .env file exists
print_status "Step 2: Setting up environment configuration..."
if [ ! -f ".env" ]; then
    print_status "Creating .env file from template..."
    cp .env.example .env
    print_success "Environment file created"
else
    print_warning ".env file already exists"
fi

# Step 3: Build the services
print_status "Step 3: Building Docker images..."
print_status "Building surfacing services..."
docker-compose -f docker-compose.yml -f docker-compose.dev.yml build surfacing-service surfacing
print_success "Docker images built successfully"

# Step 4: Start core services first
print_status "Step 4: Starting core LiftOS services..."
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d postgres redis
print_status "Waiting for database to be ready..."
sleep 10

docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d auth registry memory
print_status "Waiting for core services to be ready..."
sleep 15

# Step 5: Start surfacing services
print_status "Step 5: Starting surfacing services..."
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d surfacing-service surfacing
print_status "Waiting for surfacing services to be ready..."
sleep 10

# Step 6: Start gateway
print_status "Step 6: Starting API gateway..."
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d gateway
sleep 5

# Step 7: Health checks
print_status "Step 7: Performing health checks..."

# Check core services
services=("postgres:5432" "redis:6379" "auth:8001" "registry:8005" "memory:8003" "gateway:8000")
for service in "${services[@]}"; do
    service_name=$(echo $service | cut -d':' -f1)
    port=$(echo $service | cut -d':' -f2)
    
    print_status "Checking $service_name service..."
    if docker-compose -f docker-compose.yml -f docker-compose.dev.yml exec -T $service_name curl -f http://localhost:$port/health >/dev/null 2>&1; then
        print_success "$service_name is healthy"
    else
        print_warning "$service_name health check failed, but continuing..."
    fi
done

# Check surfacing services
print_status "Checking surfacing service..."
if curl -f http://localhost:3000/health >/dev/null 2>&1; then
    print_success "Surfacing service is healthy"
else
    print_warning "Surfacing service health check failed"
fi

print_status "Checking surfacing module..."
if curl -f http://localhost:9005/health >/dev/null 2>&1; then
    print_success "Surfacing module is healthy"
else
    print_warning "Surfacing module health check failed"
fi

# Step 8: Register the surfacing module
print_status "Step 8: Registering surfacing module..."
if python scripts/register_surfacing_module.py register; then
    print_success "Surfacing module registered successfully"
else
    print_warning "Module registration failed, you can try again later with: make surfacing-register"
fi

# Step 9: Final verification
print_status "Step 9: Final verification..."
print_status "Testing gateway access to surfacing module..."
if curl -f http://localhost:8000/modules/surfacing/ >/dev/null 2>&1; then
    print_success "Gateway access to surfacing module working"
else
    print_warning "Gateway access test failed"
fi

# Display final status
echo ""
echo "🎉 LiftOS Surfacing Integration Setup Complete!"
echo "=============================================="
echo ""
echo "Access Points:"
echo "  🌐 UI Shell:           http://localhost:3000"
echo "  🚪 API Gateway:        http://localhost:8000"
echo "  📚 API Documentation:  http://localhost:8000/docs"
echo "  🌊 Surfacing Module:   http://localhost:9005/docs"
echo "  🔧 Surfacing Service:  http://localhost:3000/docs"
echo "  🎯 Gateway Access:     http://localhost:8000/modules/surfacing/"
echo ""
echo "Quick Commands:"
echo "  📊 Check status:       make surfacing-status"
echo "  🧪 Run tests:          make surfacing-test"
echo "  📝 View logs:          make logs-surfacing"
echo "  🐚 Shell access:       make shell-surfacing"
echo ""
echo "Example API Call:"
echo "  curl -X POST http://localhost:8000/modules/surfacing/api/v1/analyze \\"
echo "    -H \"Authorization: Bearer <token>\" \\"
echo "    -H \"Content-Type: application/json\" \\"
echo "    -d '{\"url\": \"https://example.com/product\", \"analysis_type\": \"comprehensive\"}'"
echo ""

# Check if everything is running
print_status "Current service status:"
docker-compose -f docker-compose.yml -f docker-compose.dev.yml ps

print_success "Setup completed! The surfacing module is now integrated with LiftOS Core."