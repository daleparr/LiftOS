"""
Configuration utilities for Lift OS Core
"""
import os
from typing import Optional, Dict, Any
from functools import lru_cache


class Config:
    """Base configuration class"""
    
    # Core settings
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key")
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://lift_user:lift_password@localhost:5432/lift_os")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # JWT
    JWT_SECRET: str = os.getenv("JWT_SECRET", "jwt-secret-key")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_EXPIRATION_HOURS: int = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
    
    # KSE Memory
    KSE_API_KEY: Optional[str] = os.getenv("KSE_API_KEY")
    KSE_ENVIRONMENT: str = os.getenv("KSE_ENVIRONMENT", "development")
    KSE_DEFAULT_DOMAIN: str = os.getenv("KSE_DEFAULT_DOMAIN", "general")
    KSE_MAX_CONTEXTS: int = int(os.getenv("KSE_MAX_CONTEXTS", "100"))
    
    # OAuth
    GOOGLE_CLIENT_ID: Optional[str] = os.getenv("GOOGLE_CLIENT_ID")
    GOOGLE_CLIENT_SECRET: Optional[str] = os.getenv("GOOGLE_CLIENT_SECRET")
    GITHUB_CLIENT_ID: Optional[str] = os.getenv("GITHUB_CLIENT_ID")
    GITHUB_CLIENT_SECRET: Optional[str] = os.getenv("GITHUB_CLIENT_SECRET")
    
    # Stripe
    STRIPE_SECRET_KEY: Optional[str] = os.getenv("STRIPE_SECRET_KEY")
    STRIPE_PUBLISHABLE_KEY: Optional[str] = os.getenv("STRIPE_PUBLISHABLE_KEY")
    STRIPE_WEBHOOK_SECRET: Optional[str] = os.getenv("STRIPE_WEBHOOK_SECRET")
    
    # Service URLs
    GATEWAY_URL: str = os.getenv("GATEWAY_URL", "http://localhost:8000")
    AUTH_SERVICE_URL: str = os.getenv("AUTH_SERVICE_URL", "http://localhost:8001")
    BILLING_SERVICE_URL: str = os.getenv("BILLING_SERVICE_URL", "http://localhost:8002")
    MEMORY_SERVICE_URL: str = os.getenv("MEMORY_SERVICE_URL", "http://localhost:8003")
    OBSERVABILITY_SERVICE_URL: str = os.getenv("OBSERVABILITY_SERVICE_URL", "http://localhost:8004")
    REGISTRY_SERVICE_URL: str = os.getenv("REGISTRY_SERVICE_URL", "http://localhost:8005")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Monitoring
    METRICS_ENABLED: bool = os.getenv("METRICS_ENABLED", "true").lower() == "true"
    PROMETHEUS_PORT: int = int(os.getenv("PROMETHEUS_PORT", "9090"))
    
    @classmethod
    def is_development(cls) -> bool:
        """Check if running in development mode"""
        return cls.ENVIRONMENT == "development"
    
    @classmethod
    def is_production(cls) -> bool:
        """Check if running in production mode"""
        return cls.ENVIRONMENT == "production"
    
    @classmethod
    def get_database_config(cls) -> Dict[str, Any]:
        """Get database configuration"""
        return {
            "url": cls.DATABASE_URL,
            "echo": cls.DEBUG,
            "pool_pre_ping": True,
            "pool_recycle": 300
        }
    
    @classmethod
    def get_redis_config(cls) -> Dict[str, Any]:
        """Get Redis configuration"""
        return {
            "url": cls.REDIS_URL,
            "decode_responses": True,
            "health_check_interval": 30
        }
    
    @classmethod
    def get_kse_config(cls) -> Dict[str, Any]:
        """Get KSE Memory configuration"""
        return {
            "api_key": cls.KSE_API_KEY,
            "environment": cls.KSE_ENVIRONMENT,
            "default_domain": cls.KSE_DEFAULT_DOMAIN,
            "max_contexts": cls.KSE_MAX_CONTEXTS
        }
    
    @classmethod
    def get_jwt_config(cls) -> Dict[str, Any]:
        """Get JWT configuration"""
        return {
            "secret": cls.JWT_SECRET,
            "algorithm": cls.JWT_ALGORITHM,
            "expiration_hours": cls.JWT_EXPIRATION_HOURS
        }
    
    @classmethod
    def get_oauth_config(cls) -> Dict[str, Any]:
        """Get OAuth configuration"""
        return {
            "google": {
                "client_id": cls.GOOGLE_CLIENT_ID,
                "client_secret": cls.GOOGLE_CLIENT_SECRET
            },
            "github": {
                "client_id": cls.GITHUB_CLIENT_ID,
                "client_secret": cls.GITHUB_CLIENT_SECRET
            }
        }
    
    @classmethod
    def get_stripe_config(cls) -> Dict[str, Any]:
        """Get Stripe configuration"""
        return {
            "secret_key": cls.STRIPE_SECRET_KEY,
            "publishable_key": cls.STRIPE_PUBLISHABLE_KEY,
            "webhook_secret": cls.STRIPE_WEBHOOK_SECRET
        }


class ServiceConfig(Config):
    """Service-specific configuration"""
    
    def __init__(self, service_name: str, port: int):
        self.SERVICE_NAME = service_name
        self.PORT = port
        self.HOST = "0.0.0.0"
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information"""
        return {
            "name": self.SERVICE_NAME,
            "host": self.HOST,
            "port": self.PORT,
            "environment": self.ENVIRONMENT,
            "debug": self.DEBUG
        }


@lru_cache()
def get_config() -> Config:
    """Get cached configuration instance"""
    return Config()


@lru_cache()
def get_service_config(service_name: str, port: int) -> ServiceConfig:
    """Get cached service configuration"""
    return ServiceConfig(service_name, port)


# Environment-specific configurations
class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = "DEBUG"


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = "INFO"


class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    DATABASE_URL = "postgresql://test_user:test_password@localhost:5432/test_lift_os"
    REDIS_URL = "redis://localhost:6379/1"


def get_config_by_environment(environment: str = None) -> Config:
    """Get configuration based on environment"""
    env = environment or os.getenv("ENVIRONMENT", "development")
    
    if env == "development":
        return DevelopmentConfig()
    elif env == "production":
        return ProductionConfig()
    elif env == "testing":
        return TestingConfig()
    else:
        return Config()