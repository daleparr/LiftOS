# Lift OS Core - Deployment Guide

This guide covers deploying Lift OS Core in various environments, from development to production.

## 🚀 Quick Start (Development)

### Prerequisites
- Docker & Docker Compose
- Node.js 18+ (for UI development)
- Python 3.11+ (for service development)
- Make (optional, for convenience commands)

### Development Deployment

```bash
# Clone the repository
git clone <repository-url>
cd lift-os-core

# Copy environment template
cp .env.example .env

# Edit environment variables
nano .env

# Start all services
make dev-up

# Wait for services to be ready (2-3 minutes)
make health-check

# Access the UI
open http://localhost:3000
```

## 🏗️ Production Deployment

### Docker Compose (Recommended for small-medium deployments)

1. **Prepare Production Environment**
```bash
# Create production directory
mkdir -p /opt/lift-os-core
cd /opt/lift-os-core

# Copy deployment files
cp docker-compose.yml .
cp docker-compose.prod.yml .
cp .env.example .env.prod
```

2. **Configure Production Environment**
```bash
# Edit production environment
nano .env.prod
```

Required production environment variables:
```env
# Database
DATABASE_URL=postgresql://user:password@postgres:5432/lift_os_prod
REDIS_URL=redis://redis:6379/0

# Security
JWT_SECRET=your-super-secure-jwt-secret-here
ENCRYPTION_KEY=your-32-character-encryption-key

# External Services
KSE_API_KEY=your-kse-api-key
STRIPE_SECRET_KEY=sk_live_your_stripe_key
STRIPE_WEBHOOK_SECRET=whsec_your_webhook_secret

# OAuth (Optional)
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
GITHUB_CLIENT_ID=your-github-client-id
GITHUB_CLIENT_SECRET=your-github-client-secret

# Monitoring
PROMETHEUS_PORT=9090
LOG_LEVEL=INFO
METRICS_ENABLED=true

# Domain
DOMAIN=your-domain.com
SSL_EMAIL=admin@your-domain.com
```

3. **Deploy with SSL**
```bash
# Start production stack
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Check service health
docker-compose ps
docker-compose logs -f
```

### Kubernetes Deployment

1. **Create Namespace**
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: lift-os
```

2. **Deploy Core Services**
```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/postgres.yaml
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/services/
kubectl apply -f k8s/ingress.yaml
```

3. **Monitor Deployment**
```bash
# Check pod status
kubectl get pods -n lift-os

# Check service status
kubectl get services -n lift-os

# View logs
kubectl logs -f deployment/gateway -n lift-os
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DATABASE_URL` | PostgreSQL connection string | - | Yes |
| `REDIS_URL` | Redis connection string | - | Yes |
| `JWT_SECRET` | JWT signing secret | - | Yes |
| `KSE_API_KEY` | KSE Memory API key | - | Yes |
| `STRIPE_SECRET_KEY` | Stripe API key | - | No |
| `LOG_LEVEL` | Logging level | INFO | No |
| `DOMAIN` | Application domain | localhost | No |

### Service Configuration

Each service can be configured via environment variables or configuration files:

#### Gateway Service
```env
GATEWAY_PORT=8000
CORS_ORIGINS=http://localhost:3000,https://your-domain.com
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
```

#### Auth Service
```env
AUTH_PORT=8001
JWT_EXPIRATION_HOURS=24
PASSWORD_MIN_LENGTH=8
ENABLE_OAUTH=true
```

#### Memory Service
```env
MEMORY_PORT=8002
KSE_ENVIRONMENT=production
KSE_DEFAULT_DOMAIN=general
KSE_MAX_CONTEXTS=1000
```

## 🔒 Security Configuration

### SSL/TLS Setup

1. **Using Let's Encrypt (Recommended)**
```yaml
# docker-compose.prod.yml
services:
  traefik:
    image: traefik:v2.10
    command:
      - --certificatesresolvers.letsencrypt.acme.email=${SSL_EMAIL}
      - --certificatesresolvers.letsencrypt.acme.storage=/letsencrypt/acme.json
      - --certificatesresolvers.letsencrypt.acme.httpchallenge.entrypoint=web
    volumes:
      - ./letsencrypt:/letsencrypt
```

2. **Using Custom Certificates**
```bash
# Place certificates in ssl/ directory
mkdir ssl/
cp your-domain.crt ssl/
cp your-domain.key ssl/
```

### Security Headers
```yaml
# Configure in gateway service
security:
  headers:
    X-Frame-Options: DENY
    X-Content-Type-Options: nosniff
    X-XSS-Protection: "1; mode=block"
    Strict-Transport-Security: "max-age=31536000; includeSubDomains"
```

## 📊 Monitoring & Observability

### Prometheus Metrics
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'lift-os-gateway'
    static_configs:
      - targets: ['gateway:8000']
  
  - job_name: 'lift-os-auth'
    static_configs:
      - targets: ['auth:8001']
```

### Grafana Dashboards
```bash
# Import pre-built dashboards
curl -X POST \
  http://admin:admin@localhost:3001/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @grafana/lift-os-dashboard.json
```

### Log Aggregation
```yaml
# docker-compose.prod.yml
services:
  loki:
    image: grafana/loki:latest
    ports:
      - "3100:3100"
    
  promtail:
    image: grafana/promtail:latest
    volumes:
      - /var/log:/var/log:ro
      - ./promtail-config.yml:/etc/promtail/config.yml
```

## 🔄 Backup & Recovery

### Database Backup
```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups/$(date +%Y-%m-%d)"
mkdir -p $BACKUP_DIR

# PostgreSQL backup
docker-compose exec postgres pg_dump -U postgres lift_os > $BACKUP_DIR/postgres.sql

# Redis backup
docker-compose exec redis redis-cli BGSAVE
docker cp $(docker-compose ps -q redis):/data/dump.rdb $BACKUP_DIR/redis.rdb
```

### Restore Procedure
```bash
# Restore PostgreSQL
docker-compose exec postgres psql -U postgres -d lift_os < /backups/postgres.sql

# Restore Redis
docker cp /backups/redis.rdb $(docker-compose ps -q redis):/data/dump.rdb
docker-compose restart redis
```

## 📈 Scaling

### Horizontal Scaling
```yaml
# docker-compose.scale.yml
services:
  gateway:
    deploy:
      replicas: 3
  
  auth:
    deploy:
      replicas: 2
  
  memory:
    deploy:
      replicas: 2
```

### Load Balancing
```yaml
# nginx.conf
upstream gateway {
    server gateway_1:8000;
    server gateway_2:8000;
    server gateway_3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://gateway;
    }
}
```

## 🚨 Troubleshooting

### Common Issues

1. **Services Not Starting**
```bash
# Check logs
docker-compose logs service-name

# Check resource usage
docker stats

# Restart specific service
docker-compose restart service-name
```

2. **Database Connection Issues**
```bash
# Test database connectivity
docker-compose exec gateway python -c "
import psycopg2
conn = psycopg2.connect('$DATABASE_URL')
print('Database connection successful')
"
```

3. **Memory Service Issues**
```bash
# Test KSE API connectivity
curl -H "Authorization: Bearer $KSE_API_KEY" \
  https://api.kse.com/v1/health
```

### Health Checks
```bash
# Check all service health
make health-check

# Individual service health
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health
```

## 🔄 Updates & Maintenance

### Rolling Updates
```bash
# Pull latest images
docker-compose pull

# Rolling restart
docker-compose up -d --no-deps gateway
docker-compose up -d --no-deps auth
docker-compose up -d --no-deps memory
```

### Database Migrations
```bash
# Run migrations
docker-compose exec gateway python -m alembic upgrade head
docker-compose exec auth python -m alembic upgrade head
docker-compose exec memory python -m alembic upgrade head
```

## 📋 Deployment Checklist

### Pre-Deployment
- [ ] Environment variables configured
- [ ] SSL certificates ready
- [ ] Database backup completed
- [ ] External service credentials verified
- [ ] Resource requirements met

### Post-Deployment
- [ ] All services healthy
- [ ] Database migrations applied
- [ ] SSL certificates valid
- [ ] Monitoring dashboards accessible
- [ ] Backup procedures tested
- [ ] Load testing completed

### Production Readiness
- [ ] Security headers configured
- [ ] Rate limiting enabled
- [ ] Log aggregation setup
- [ ] Alerting configured
- [ ] Documentation updated
- [ ] Team trained on operations

## 📞 Support

For deployment issues:
1. Check the troubleshooting section
2. Review service logs
3. Consult the monitoring dashboards
4. Contact the development team

## 🔗 Additional Resources

- [Architecture Documentation](./ARCHITECTURE.md)
- [API Documentation](./API.md)
- [Security Guide](./SECURITY.md)
- [Performance Tuning](./PERFORMANCE.md)