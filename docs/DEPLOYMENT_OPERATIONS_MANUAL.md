# Deployment and Operations Manual

## Overview

This manual provides comprehensive guidance for deploying, operating, and maintaining the Lift OS Core platform in production environments.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Infrastructure Setup](#infrastructure-setup)
3. [Deployment Strategies](#deployment-strategies)
4. [Configuration Management](#configuration-management)
5. [Monitoring & Observability](#monitoring--observability)
6. [Security Operations](#security-operations)
7. [Backup & Recovery](#backup--recovery)
8. [Scaling & Performance](#scaling--performance)
9. [Troubleshooting](#troubleshooting)
10. [Maintenance Procedures](#maintenance-procedures)

## System Requirements

### Minimum Requirements

#### Development Environment
- **CPU**: 4 cores, 2.5GHz
- **RAM**: 8GB
- **Storage**: 50GB SSD
- **Network**: 100 Mbps
- **OS**: Linux (Ubuntu 20.04+), macOS 11+, Windows 10+

#### Production Environment
- **CPU**: 8 cores, 3.0GHz per node
- **RAM**: 16GB per node (32GB recommended)
- **Storage**: 200GB SSD per node
- **Network**: 1 Gbps
- **OS**: Linux (Ubuntu 20.04 LTS or CentOS 8+)

### Recommended Production Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer (HAProxy/NGINX)           │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┼───────────────────────────────────────┐
│                Gateway Cluster (3 nodes)                   │
├─────────────────────┼───────────────────────────────────────┤
│  ┌─────────────┐   │   ┌─────────────┐   ┌─────────────┐   │
│  │ Gateway-1   │   │   │ Gateway-2   │   │ Gateway-3   │   │
│  │ Port: 8000  │   │   │ Port: 8000  │   │ Port: 8000  │   │
│  └─────────────┘   │   └─────────────┘   └─────────────┘   │
└─────────────────────┼───────────────────────────────────────┘
                      │
┌─────────────────────┼───────────────────────────────────────┐
│                Core Services Cluster                       │
├─────────────────────┼───────────────────────────────────────┤
│  ┌─────────────┐   │   ┌─────────────┐   ┌─────────────┐   │
│  │ Auth        │   │   │ Memory      │   │ Registry    │   │
│  │ Port: 8001  │   │   │ Port: 8003  │   │ Port: 8005  │   │
│  └─────────────┘   │   └─────────────┘   └─────────────┘   │
└─────────────────────┼───────────────────────────────────────┘
                      │
┌─────────────────────┼───────────────────────────────────────┐
│                Data Layer                                   │
├─────────────────────┼───────────────────────────────────────┤
│  ┌─────────────┐   │   ┌─────────────┐   ┌─────────────┐   │
│  │ PostgreSQL  │   │   │ Redis       │   │ Vector DB   │   │
│  │ (Primary)   │   │   │ (Cache)     │   │ (KSE)       │   │
│  └─────────────┘   │   └─────────────┘   └─────────────┘   │
└─────────────────────┼───────────────────────────────────────┘
```

## Infrastructure Setup

### 1. Container Orchestration (Docker Compose)

#### Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  # Load Balancer
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - gateway-1
      - gateway-2
      - gateway-3
    networks:
      - liftos-network
    restart: unless-stopped

  # Gateway Cluster
  gateway-1:
    image: liftos/gateway:${VERSION}
    environment:
      - NODE_ID=gateway-1
      - CLUSTER_MODE=true
      - AUTH_SERVICE_URL=http://auth:8001
      - MEMORY_SERVICE_URL=http://memory:8003
      - REGISTRY_SERVICE_URL=http://registry:8005
      - REDIS_URL=redis://redis:6379
    depends_on:
      - auth
      - memory
      - registry
      - redis
    networks:
      - liftos-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  gateway-2:
    image: liftos/gateway:${VERSION}
    environment:
      - NODE_ID=gateway-2
      - CLUSTER_MODE=true
      - AUTH_SERVICE_URL=http://auth:8001
      - MEMORY_SERVICE_URL=http://memory:8003
      - REGISTRY_SERVICE_URL=http://registry:8005
      - REDIS_URL=redis://redis:6379
    depends_on:
      - auth
      - memory
      - registry
      - redis
    networks:
      - liftos-network
    restart: unless-stopped

  gateway-3:
    image: liftos/gateway:${VERSION}
    environment:
      - NODE_ID=gateway-3
      - CLUSTER_MODE=true
      - AUTH_SERVICE_URL=http://auth:8001
      - MEMORY_SERVICE_URL=http://memory:8003
      - REGISTRY_SERVICE_URL=http://registry:8005
      - REDIS_URL=redis://redis:6379
    depends_on:
      - auth
      - memory
      - registry
      - redis
    networks:
      - liftos-network
    restart: unless-stopped

  # Core Services
  auth:
    image: liftos/auth:${VERSION}
    environment:
      - DATABASE_URL=postgresql://liftos:${DB_PASSWORD}@postgres:5432/liftos_auth
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
    networks:
      - liftos-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  memory:
    image: liftos/memory:${VERSION}
    environment:
      - KSE_VECTOR_DB_URL=http://vectordb:8080
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://liftos:${DB_PASSWORD}@postgres:5432/liftos_memory
    depends_on:
      - postgres
      - redis
      - vectordb
    networks:
      - liftos-network
    restart: unless-stopped
    volumes:
      - memory_data:/app/data

  registry:
    image: liftos/registry:${VERSION}
    environment:
      - DATABASE_URL=postgresql://liftos:${DB_PASSWORD}@postgres:5432/liftos_registry
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
    networks:
      - liftos-network
    restart: unless-stopped

  # Data Layer
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=liftos
      - POSTGRES_USER=liftos
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./postgres/init:/docker-entrypoint-initdb.d
    networks:
      - liftos-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U liftos"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    networks:
      - liftos-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  vectordb:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - vectordb_data:/qdrant/storage
    networks:
      - liftos-network
    restart: unless-stopped

  # Monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    networks:
      - liftos-network
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
    networks:
      - liftos-network
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  vectordb_data:
  memory_data:
  prometheus_data:
  grafana_data:

networks:
  liftos-network:
    driver: bridge
```

### 2. Kubernetes Deployment

#### Namespace Configuration

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: liftos-core
  labels:
    name: liftos-core
    environment: production
```

#### ConfigMap for Environment Variables

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: liftos-config
  namespace: liftos-core
data:
  AUTH_SERVICE_URL: "http://auth-service:8001"
  MEMORY_SERVICE_URL: "http://memory-service:8003"
  REGISTRY_SERVICE_URL: "http://registry-service:8005"
  REDIS_URL: "redis://redis-service:6379"
  KSE_VECTOR_DB_URL: "http://vectordb-service:8080"
  LOG_LEVEL: "INFO"
  ENVIRONMENT: "production"
```

#### Secret Management

```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: liftos-secrets
  namespace: liftos-core
type: Opaque
data:
  DB_PASSWORD: <base64-encoded-password>
  JWT_SECRET_KEY: <base64-encoded-secret>
  REDIS_PASSWORD: <base64-encoded-password>
  GRAFANA_PASSWORD: <base64-encoded-password>
```

#### Gateway Deployment

```yaml
# k8s/gateway-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gateway
  namespace: liftos-core
  labels:
    app: gateway
    component: core
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: gateway
  template:
    metadata:
      labels:
        app: gateway
        component: core
    spec:
      containers:
      - name: gateway
        image: liftos/gateway:1.0.0
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: liftos-config
        - secretRef:
            name: liftos-secrets
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        securityContext:
          runAsNonRoot: true
          runAsUser: 1000
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
---
apiVersion: v1
kind: Service
metadata:
  name: gateway-service
  namespace: liftos-core
spec:
  selector:
    app: gateway
  ports:
  - protocol: TCP
    port: 8000
    targetPort: 8000
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: gateway-ingress
  namespace: liftos-core
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
  - hosts:
    - api.liftos.com
    secretName: liftos-tls
  rules:
  - host: api.liftos.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: gateway-service
            port:
              number: 8000
```

## Deployment Strategies

### 1. Blue-Green Deployment

```bash
#!/bin/bash
# scripts/blue-green-deploy.sh

set -e

ENVIRONMENT=${1:-production}
VERSION=${2:-latest}
CURRENT_COLOR=$(kubectl get service gateway-service -o jsonpath='{.spec.selector.color}' 2>/dev/null || echo "blue")
NEW_COLOR=$([ "$CURRENT_COLOR" = "blue" ] && echo "green" || echo "blue")

echo "Current deployment: $CURRENT_COLOR"
echo "Deploying to: $NEW_COLOR"

# Deploy new version
kubectl set image deployment/gateway-$NEW_COLOR gateway=liftos/gateway:$VERSION
kubectl set image deployment/auth-$NEW_COLOR auth=liftos/auth:$VERSION
kubectl set image deployment/memory-$NEW_COLOR memory=liftos/memory:$VERSION
kubectl set image deployment/registry-$NEW_COLOR registry=liftos/registry:$VERSION

# Wait for rollout
kubectl rollout status deployment/gateway-$NEW_COLOR --timeout=300s
kubectl rollout status deployment/auth-$NEW_COLOR --timeout=300s
kubectl rollout status deployment/memory-$NEW_COLOR --timeout=300s
kubectl rollout status deployment/registry-$NEW_COLOR --timeout=300s

# Health check
echo "Performing health checks..."
for i in {1..30}; do
    if kubectl exec deployment/gateway-$NEW_COLOR -- curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "Health check passed"
        break
    fi
    echo "Health check attempt $i/30 failed, retrying..."
    sleep 10
done

# Switch traffic
kubectl patch service gateway-service -p '{"spec":{"selector":{"color":"'$NEW_COLOR'"}}}'
kubectl patch service auth-service -p '{"spec":{"selector":{"color":"'$NEW_COLOR'"}}}'
kubectl patch service memory-service -p '{"spec":{"selector":{"color":"'$NEW_COLOR'"}}}'
kubectl patch service registry-service -p '{"spec":{"selector":{"color":"'$NEW_COLOR'"}}}'

echo "Deployment completed successfully"
echo "Traffic switched to: $NEW_COLOR"
```

### 2. Rolling Deployment

```bash
#!/bin/bash
# scripts/rolling-deploy.sh

set -e

VERSION=${1:-latest}

echo "Starting rolling deployment of version: $VERSION"

# Update deployments
kubectl set image deployment/gateway gateway=liftos/gateway:$VERSION
kubectl set image deployment/auth auth=liftos/auth:$VERSION
kubectl set image deployment/memory memory=liftos/memory:$VERSION
kubectl set image deployment/registry registry=liftos/registry:$VERSION

# Monitor rollout
kubectl rollout status deployment/gateway --timeout=600s
kubectl rollout status deployment/auth --timeout=600s
kubectl rollout status deployment/memory --timeout=600s
kubectl rollout status deployment/registry --timeout=600s

echo "Rolling deployment completed successfully"
```

### 3. Canary Deployment

```yaml
# k8s/canary-deployment.yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: gateway-rollout
  namespace: liftos-core
spec:
  replicas: 5
  strategy:
    canary:
      steps:
      - setWeight: 20
      - pause: {duration: 10m}
      - setWeight: 40
      - pause: {duration: 10m}
      - setWeight: 60
      - pause: {duration: 10m}
      - setWeight: 80
      - pause: {duration: 10m}
      canaryService: gateway-canary-service
      stableService: gateway-stable-service
      trafficRouting:
        nginx:
          stableIngress: gateway-stable-ingress
          annotationPrefix: nginx.ingress.kubernetes.io
          additionalIngressAnnotations:
            canary-by-header: X-Canary
  selector:
    matchLabels:
      app: gateway
  template:
    metadata:
      labels:
        app: gateway
    spec:
      containers:
      - name: gateway
        image: liftos/gateway:1.0.0
        ports:
        - containerPort: 8000
```

## Configuration Management

### 1. Environment Configuration

```bash
# environments/production.env
# Core Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false

# Service URLs
AUTH_SERVICE_URL=http://auth-service:8001
MEMORY_SERVICE_URL=http://memory-service:8003
REGISTRY_SERVICE_URL=http://registry-service:8005

# Database Configuration
DATABASE_URL=postgresql://liftos:${DB_PASSWORD}@postgres-primary:5432/liftos
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Redis Configuration
REDIS_URL=redis://redis-cluster:6379
REDIS_POOL_SIZE=10
REDIS_TIMEOUT=5

# KSE Configuration
KSE_VECTOR_DB_URL=http://vectordb-cluster:8080
KSE_BATCH_SIZE=1000
KSE_INDEX_REPLICAS=2

# Security Configuration
JWT_SECRET_KEY=${JWT_SECRET_KEY}
JWT_EXPIRATION=3600
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=60

# Monitoring Configuration
PROMETHEUS_ENABLED=true
METRICS_PORT=9090
HEALTH_CHECK_INTERVAL=30

# Performance Configuration
WORKER_PROCESSES=4
MAX_CONNECTIONS=1000
KEEPALIVE_TIMEOUT=65
```

### 2. Secrets Management with HashiCorp Vault

```python
# scripts/vault_setup.py
import hvac
import os

def setup_vault_secrets():
    """Setup secrets in HashiCorp Vault"""
    
    client = hvac.Client(url=os.getenv('VAULT_URL'))
    client.token = os.getenv('VAULT_TOKEN')
    
    # Create KV secrets engine
    client.sys.enable_secrets_engine(
        backend_type='kv',
        path='liftos',
        options={'version': '2'}
    )
    
    # Store secrets
    secrets = {
        'database': {
            'password': os.getenv('DB_PASSWORD'),
            'url': os.getenv('DATABASE_URL')
        },
        'jwt': {
            'secret_key': os.getenv('JWT_SECRET_KEY'),
            'algorithm': 'HS256'
        },
        'redis': {
            'password': os.getenv('REDIS_PASSWORD'),
            'url': os.getenv('REDIS_URL')
        }
    }
    
    for secret_path, secret_data in secrets.items():
        client.secrets.kv.v2.create_or_update_secret(
            path=f'liftos/{secret_path}',
            secret=secret_data
        )
    
    print("Vault secrets configured successfully")

if __name__ == "__main__":
    setup_vault_secrets()
```

### 3. Configuration Validation

```python
# scripts/validate_config.py
import os
import sys
from urllib.parse import urlparse

def validate_configuration():
    """Validate production configuration"""
    
    errors = []
    
    # Required environment variables
    required_vars = [
        'DATABASE_URL',
        'JWT_SECRET_KEY',
        'REDIS_URL',
        'AUTH_SERVICE_URL',
        'MEMORY_SERVICE_URL',
        'REGISTRY_SERVICE_URL'
    ]
    
    for var in required_vars:
        if not os.getenv(var):
            errors.append(f"Missing required environment variable: {var}")
    
    # Validate URLs
    url_vars = ['DATABASE_URL', 'REDIS_URL', 'AUTH_SERVICE_URL', 'MEMORY_SERVICE_URL']
    for var in url_vars:
        url = os.getenv(var)
        if url:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                errors.append(f"Invalid URL format for {var}: {url}")
    
    # Validate JWT secret strength
    jwt_secret = os.getenv('JWT_SECRET_KEY')
    if jwt_secret and len(jwt_secret) < 32:
        errors.append("JWT_SECRET_KEY should be at least 32 characters long")
    
    # Validate numeric configurations
    numeric_vars = {
        'DATABASE_POOL_SIZE': (1, 100),
        'REDIS_POOL_SIZE': (1, 50),
        'WORKER_PROCESSES': (1, 16)
    }
    
    for var, (min_val, max_val) in numeric_vars.items():
        value = os.getenv(var)
        if value:
            try:
                num_value = int(value)
                if not min_val <= num_value <= max_val:
                    errors.append(f"{var} should be between {min_val} and {max_val}")
            except ValueError:
                errors.append(f"{var} should be a valid integer")
    
    if errors:
        print("Configuration validation failed:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    else:
        print("Configuration validation passed")

if __name__ == "__main__":
    validate_configuration()
```

## Monitoring & Observability

### 1. Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'liftos-gateway'
    static_configs:
      - targets: ['gateway-1:8000', 'gateway-2:8000', 'gateway-3:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'liftos-auth'
    static_configs:
      - targets: ['auth:8001']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'liftos-memory'
    static_configs:
      - targets: ['memory:8003']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'liftos-registry'
    static_configs:
      - targets: ['registry:8005']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis-exporter'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
```

### 2. Alert Rules

```yaml
# monitoring/alert_rules.yml
groups:
- name: liftos_alerts
  rules:
  - alert: ServiceDown
    expr: up == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Service {{ $labels.instance }} is down"
      description: "{{ $labels.instance }} has been down for more than 1 minute"

  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High error rate on {{ $labels.instance }}"
      description: "Error rate is {{ $value }} errors per second"

  - alert: HighResponseTime
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High response time on {{ $labels.instance }}"
      description: "95th percentile response time is {{ $value }} seconds"

  - alert: DatabaseConnectionsHigh
    expr: pg_stat_activity_count > 80
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High database connections"
      description: "Database has {{ $value }} active connections"

  - alert: MemoryUsageHigh
    expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High memory usage on {{ $labels.instance }}"
      description: "Memory usage is {{ $value | humanizePercentage }}"

  - alert: DiskSpaceHigh
    expr: (node_filesystem_size_bytes - node_filesystem_free_bytes) / node_filesystem_size_bytes > 0.9
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High disk usage on {{ $labels.instance }}"
      description: "Disk usage is {{ $value | humanizePercentage }}"
```

### 3. Grafana Dashboards

```json
{
  "dashboard": {
    "id": null,
    "title": "Lift OS Core - System Overview",
    "tags": ["liftos", "overview"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total[5m])) by (service)",
            "legendFormat": "{{ service }}"
          }
        ],
        "yAxes": [
          {
            "label": "Requests/sec"
          }
        ]
      },
      {
        "id": 2,
        "title": "Response Time (95th percentile)",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, service))",
            "legendFormat": "{{ service }}"
          }
        ],
        "yAxes": [
          {
            "label": "Seconds"
          }
        ]
      },
      {
        "id": 3,
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{status=~\"5..\"}[5m])) by (service) / sum(rate(http_requests_total[5m])) by (service)",
            "legendFormat": "{{ service }}"
          }
        ],
        "yAxes": [
          {
            "label": "Error Rate",
            "max": 1,
            "min": 0
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
```

## Security Operations

### 1. SSL/TLS Configuration

```nginx
# nginx/nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream gateway_backend {
        least_conn;
        server gateway-1:8000 max_fails=3 fail_timeout=30s;
        server gateway-2:8000 max_fails=3 fail_timeout=30s;
        server gateway-3:8000 max_fails=3 fail_timeout=30s;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=login:10m rate=1r/s;

    server {
        listen 80;
        server_name api.liftos.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name api.liftos.com;

        # SSL Configuration
        ssl_certificate /etc/nginx/ssl/liftos.crt;
        ssl_certificate_key /etc/nginx/ssl/liftos.key;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;
        ssl_session_cache shared:SSL:10m;
        ssl_session_timeout 10m;

        # Security Headers
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
        add_header X-Content-Type-Options nosniff always;
        add_header X-Frame-Options DENY always;
        add_header X-XSS-Protection "1; mode=block" always;
        add_header Referrer-Policy "strict-origin-when-cross-origin" always;

        # Rate limiting
        limit_req zone=api burst=20 nodelay;

        location /auth/login {
            limit_req zone=login burst=5 nodelay;
            proxy_pass http://gateway_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr