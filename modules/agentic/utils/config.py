"""
Configuration for Agentic Module

Handles configuration loading and validation for the Agentic microservice.
"""

import os
import logging
from typing import Optional, Dict, Any
from pydantic import BaseSettings, Field, validator

logger = logging.getLogger(__name__)


class AgenticConfig(BaseSettings):
    """
    Configuration settings for the Agentic microservice.
    
    This class handles all configuration parameters including
    service URLs, API keys, and operational settings.
    """
    
    # Service URLs
    memory_service_url: str = Field(
        default="http://localhost:8001",
        description="URL for the LiftOS Memory Service"
    )
    
    auth_service_url: str = Field(
        default="http://localhost:8002", 
        description="URL for the LiftOS Auth Service"
    )
    
    observability_service_url: str = Field(
        default="http://localhost:8003",
        description="URL for the LiftOS Observability Service"
    )
    
    # AI Service Configuration
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key for agent operations"
    )
    
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key for agent operations"
    )
    
    default_ai_provider: str = Field(
        default="openai",
        description="Default AI provider (openai, anthropic)"
    )
    
    default_model: str = Field(
        default="gpt-4",
        description="Default AI model to use"
    )
    
    # Operational Settings
    max_concurrent_tests: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of concurrent test executions"
    )
    
    max_concurrent_evaluations: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of concurrent evaluations"
    )
    
    default_test_timeout: int = Field(
        default=300,
        ge=30,
        le=3600,
        description="Default test timeout in seconds"
    )
    
    default_evaluation_timeout: int = Field(
        default=600,
        ge=60,
        le=3600,
        description="Default evaluation timeout in seconds"
    )
    
    # Cache Settings
    agent_cache_ttl: int = Field(
        default=3600,
        ge=300,
        description="Agent cache TTL in seconds"
    )
    
    evaluation_cache_ttl: int = Field(
        default=1800,
        ge=300,
        description="Evaluation cache TTL in seconds"
    )
    
    # Logging Configuration
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    
    # Performance Settings
    enable_performance_monitoring: bool = Field(
        default=True,
        description="Enable performance monitoring"
    )
    
    enable_cost_tracking: bool = Field(
        default=True,
        description="Enable AI cost tracking"
    )
    
    # Security Settings
    enable_auth: bool = Field(
        default=True,
        description="Enable authentication and authorization"
    )
    
    api_key_header: str = Field(
        default="X-API-Key",
        description="Header name for API key authentication"
    )
    
    # Data Retention
    evaluation_retention_days: int = Field(
        default=90,
        ge=7,
        description="Number of days to retain evaluation results"
    )
    
    test_result_retention_days: int = Field(
        default=30,
        ge=7,
        description="Number of days to retain test results"
    )
    
    # Feature Flags
    enable_experimental_features: bool = Field(
        default=False,
        description="Enable experimental features"
    )
    
    enable_advanced_analytics: bool = Field(
        default=True,
        description="Enable advanced analytics features"
    )
    
    # Custom Settings
    custom_settings: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom configuration settings"
    )
    
    class Config:
        """Pydantic configuration."""
        env_prefix = "AGENTIC_"
        case_sensitive = False
        
    @validator('default_ai_provider')
    def validate_ai_provider(cls, v):
        """Validate AI provider."""
        valid_providers = ['openai', 'anthropic', 'azure', 'local']
        if v not in valid_providers:
            raise ValueError(f"AI provider must be one of {valid_providers}")
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
    
    def get_ai_config(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """Get AI configuration for a specific provider."""
        provider = provider or self.default_ai_provider
        
        config = {
            "provider": provider,
            "model": self.default_model
        }
        
        if provider == "openai":
            config["api_key"] = self.openai_api_key
        elif provider == "anthropic":
            config["api_key"] = self.anthropic_api_key
        
        return config
    
    def get_service_urls(self) -> Dict[str, str]:
        """Get all service URLs."""
        return {
            "memory": self.memory_service_url,
            "auth": self.auth_service_url,
            "observability": self.observability_service_url
        }
    
    def get_timeout_config(self) -> Dict[str, int]:
        """Get timeout configuration."""
        return {
            "test_timeout": self.default_test_timeout,
            "evaluation_timeout": self.default_evaluation_timeout
        }
    
    def get_cache_config(self) -> Dict[str, int]:
        """Get cache configuration."""
        return {
            "agent_ttl": self.agent_cache_ttl,
            "evaluation_ttl": self.evaluation_cache_ttl
        }
    
    def get_retention_config(self) -> Dict[str, int]:
        """Get data retention configuration."""
        return {
            "evaluations": self.evaluation_retention_days,
            "test_results": self.test_result_retention_days
        }
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled."""
        feature_map = {
            "experimental": self.enable_experimental_features,
            "analytics": self.enable_advanced_analytics,
            "auth": self.enable_auth,
            "performance_monitoring": self.enable_performance_monitoring,
            "cost_tracking": self.enable_cost_tracking
        }
        
        return feature_map.get(feature, False)
    
    def validate_configuration(self) -> bool:
        """Validate the complete configuration."""
        try:
            # Check required AI credentials
            if self.default_ai_provider == "openai" and not self.openai_api_key:
                logger.warning("OpenAI API key not configured")
                return False
            
            if self.default_ai_provider == "anthropic" and not self.anthropic_api_key:
                logger.warning("Anthropic API key not configured")
                return False
            
            # Validate service URLs
            for service, url in self.get_service_urls().items():
                if not url.startswith(('http://', 'https://')):
                    logger.warning(f"Invalid URL for {service}: {url}")
                    return False
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    @classmethod
    def load_from_env(cls) -> 'AgenticConfig':
        """Load configuration from environment variables."""
        return cls()
    
    @classmethod
    def load_from_file(cls, config_file: str) -> 'AgenticConfig':
        """Load configuration from a file."""
        # This would implement file-based configuration loading
        # For now, just load from environment
        return cls.load_from_env()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = self.dict()
        
        # Mask sensitive information
        if config_dict.get('openai_api_key'):
            config_dict['openai_api_key'] = '***masked***'
        if config_dict.get('anthropic_api_key'):
            config_dict['anthropic_api_key'] = '***masked***'
        
        return config_dict