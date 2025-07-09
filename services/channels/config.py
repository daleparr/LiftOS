"""
Configuration for Channels Service
"""

import os
from typing import Optional, List
from pydantic import BaseSettings, Field


class ChannelsConfig(BaseSettings):
    """Configuration settings for the Channels service"""
    
    # Service configuration
    service_name: str = "channels"
    service_version: str = "1.1.0"
    service_port: int = Field(default=8000, env="CHANNELS_PORT")
    service_host: str = Field(default="0.0.0.0", env="CHANNELS_HOST")
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    
    # Database configuration
    database_url: str = Field(
        default="postgresql+asyncpg://liftos:liftos@localhost:5432/liftos_channels",
        env="DATABASE_URL"
    )
    
    # Redis configuration
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    # KSE SDK configuration
    kse_enabled: bool = Field(default=True, env="KSE_ENABLED")
    kse_api_key: Optional[str] = Field(default=None, env="KSE_API_KEY")
    kse_base_url: str = Field(default="http://localhost:9000", env="KSE_BASE_URL")
    
    # Service URLs
    causal_service_url: str = Field(
        default="http://localhost:8003", 
        env="CAUSAL_SERVICE_URL"
    )
    data_ingestion_service_url: str = Field(
        default="http://localhost:8001", 
        env="DATA_INGESTION_SERVICE_URL"
    )
    memory_service_url: str = Field(
        default="http://localhost:8002", 
        env="MEMORY_SERVICE_URL"
    )
    bayesian_analysis_service_url: str = Field(
        default="http://localhost:8004", 
        env="BAYESIAN_ANALYSIS_SERVICE_URL"
    )
    
    # Optimization engine configuration
    optimization_max_iterations: int = Field(default=1000, env="OPT_MAX_ITERATIONS")
    optimization_convergence_tolerance: float = Field(default=1e-6, env="OPT_CONVERGENCE_TOL")
    optimization_population_size: int = Field(default=50, env="OPT_POPULATION_SIZE")
    
    # Simulation configuration
    default_monte_carlo_samples: int = Field(default=10000, env="DEFAULT_MC_SAMPLES")
    max_monte_carlo_samples: int = Field(default=100000, env="MAX_MC_SAMPLES")
    simulation_timeout_seconds: int = Field(default=300, env="SIMULATION_TIMEOUT")
    
    # Saturation modeling configuration
    saturation_min_data_points: int = Field(default=10, env="SAT_MIN_DATA_POINTS")
    saturation_calibration_window_days: int = Field(default=90, env="SAT_CALIBRATION_DAYS")
    saturation_confidence_threshold: float = Field(default=0.7, env="SAT_CONFIDENCE_THRESHOLD")
    
    # Bayesian configuration
    bayesian_credible_interval_level: float = Field(default=0.95, env="BAYESIAN_CREDIBLE_LEVEL")
    bayesian_mcmc_samples: int = Field(default=5000, env="BAYESIAN_MCMC_SAMPLES")
    bayesian_burn_in_samples: int = Field(default=1000, env="BAYESIAN_BURN_IN")
    
    # Recommendation engine configuration
    recommendation_min_confidence: float = Field(default=0.6, env="REC_MIN_CONFIDENCE")
    recommendation_high_impact_threshold: float = Field(default=0.15, env="REC_HIGH_IMPACT")
    
    # Security configuration
    secret_key: str = Field(
        default="your-secret-key-change-in-production", 
        env="SECRET_KEY"
    )
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # CORS configuration
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8501"],
        env="CORS_ORIGINS"
    )
    
    # Logging configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")  # json or text
    
    # Monitoring configuration
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    
    # Rate limiting
    rate_limit_requests_per_minute: int = Field(default=100, env="RATE_LIMIT_RPM")
    rate_limit_burst: int = Field(default=20, env="RATE_LIMIT_BURST")
    
    # Background tasks
    enable_background_tasks: bool = Field(default=True, env="ENABLE_BACKGROUND_TASKS")
    task_queue_url: str = Field(default="redis://localhost:6379/1", env="TASK_QUEUE_URL")
    
    # Cache configuration
    cache_ttl_seconds: int = Field(default=3600, env="CACHE_TTL_SECONDS")  # 1 hour
    cache_max_size: int = Field(default=1000, env="CACHE_MAX_SIZE")
    
    # Feature flags
    enable_advanced_optimization: bool = Field(default=True, env="ENABLE_ADVANCED_OPT")
    enable_bayesian_optimization: bool = Field(default=True, env="ENABLE_BAYESIAN_OPT")
    enable_simulation_engine: bool = Field(default=True, env="ENABLE_SIMULATION")
    enable_recommendation_engine: bool = Field(default=True, env="ENABLE_RECOMMENDATIONS")
    
    # Performance tuning
    max_concurrent_optimizations: int = Field(default=5, env="MAX_CONCURRENT_OPT")
    optimization_timeout_seconds: int = Field(default=600, env="OPT_TIMEOUT_SECONDS")
    
    # Data retention
    optimization_result_retention_days: int = Field(default=90, env="OPT_RESULT_RETENTION_DAYS")
    simulation_result_retention_days: int = Field(default=30, env="SIM_RESULT_RETENTION_DAYS")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


class DevelopmentConfig(ChannelsConfig):
    """Development environment configuration"""
    
    environment: str = "development"
    debug: bool = True
    log_level: str = "DEBUG"
    
    # Use local services
    database_url: str = "postgresql+asyncpg://liftos:liftos@localhost:5432/liftos_channels_dev"
    redis_url: str = "redis://localhost:6379/0"
    
    # Relaxed limits for development
    rate_limit_requests_per_minute: int = 1000
    optimization_timeout_seconds: int = 120
    simulation_timeout_seconds: int = 60


class ProductionConfig(ChannelsConfig):
    """Production environment configuration"""
    
    environment: str = "production"
    debug: bool = False
    log_level: str = "INFO"
    
    # Production database (should be set via environment variables)
    database_url: str = Field(env="DATABASE_URL")
    redis_url: str = Field(env="REDIS_URL")
    
    # Stricter limits for production
    rate_limit_requests_per_minute: int = 60
    max_concurrent_optimizations: int = 3
    
    # Security
    secret_key: str = Field(env="SECRET_KEY")
    
    # CORS - should be more restrictive in production
    cors_origins: List[str] = Field(env="CORS_ORIGINS")


class TestConfig(ChannelsConfig):
    """Test environment configuration"""
    
    environment: str = "test"
    debug: bool = True
    log_level: str = "WARNING"
    
    # Use test databases
    database_url: str = "postgresql+asyncpg://liftos:liftos@localhost:5432/liftos_channels_test"
    redis_url: str = "redis://localhost:6379/15"  # Use different Redis DB for tests
    
    # Faster execution for tests
    optimization_max_iterations: int = 100
    default_monte_carlo_samples: int = 1000
    saturation_min_data_points: int = 5
    
    # Disable external services for tests
    kse_enabled: bool = False
    enable_background_tasks: bool = False
    enable_metrics: bool = False


def get_config() -> ChannelsConfig:
    """Get configuration based on environment"""
    
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionConfig()
    elif env == "test":
        return TestConfig()
    else:
        return DevelopmentConfig()


# Global config instance
config = get_config()