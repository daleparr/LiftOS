import os
import streamlit as st
from typing import Dict, Any

def load_config() -> Dict[str, Any]:
    """Load application configuration"""
    return {
        'app_name': 'LiftOS Hub',
        'version': '1.4.0',
        'debug': os.getenv('DEBUG', 'false').lower() == 'true',
        'services': get_service_urls(),
        'auth': get_auth_config(),
        'features': get_feature_flags()
    }

def get_service_urls() -> Dict[str, str]:
    """Get microservice URLs"""
    return {
        'gateway': os.getenv('GATEWAY_SERVICE_URL', 'http://localhost:8000'),
        'auth': os.getenv('AUTH_SERVICE_URL', 'http://localhost:8001'),
        'billing': os.getenv('BILLING_SERVICE_URL', 'http://localhost:8002'),
        'memory': os.getenv('MEMORY_SERVICE_URL', 'http://localhost:8003'),
        'observability': os.getenv('OBSERVABILITY_SERVICE_URL', 'http://localhost:8004'),
        'registry': os.getenv('REGISTRY_SERVICE_URL', 'http://localhost:8005'),
        'data_ingestion': os.getenv('DATA_INGESTION_SERVICE_URL', 'http://localhost:8006'),
        'surfacing': os.getenv('SURFACING_SERVICE_URL', 'http://localhost:8007'),
        'causal': os.getenv('CAUSAL_SERVICE_URL', 'http://localhost:8008'),
        'llm': os.getenv('LLM_SERVICE_URL', 'http://localhost:8009'),
        'channels': os.getenv('CHANNELS_SERVICE_URL', 'http://localhost:8011')
    }

def get_auth_config() -> Dict[str, Any]:
    """Get authentication configuration"""
    return {
        'auth_service_url': os.getenv('AUTH_SERVICE_URL', 'http://localhost:8005'),
        'session_timeout': int(os.getenv('SESSION_TIMEOUT', '3600')),
        'require_auth': os.getenv('REQUIRE_AUTH', 'false').lower() == 'true',
        'demo_mode': os.getenv('DEMO_MODE', 'true').lower() == 'true'
    }

def get_feature_flags() -> Dict[str, bool]:
    """Get feature flags"""
    return {
        'enable_causal': os.getenv('ENABLE_CAUSAL', 'true').lower() == 'true',
        'enable_surfacing': os.getenv('ENABLE_SURFACING', 'true').lower() == 'true',
        'enable_llm': os.getenv('ENABLE_LLM', 'true').lower() == 'true',
        'enable_experiments': os.getenv('ENABLE_EXPERIMENTS', 'true').lower() == 'true',
        'enable_memory': os.getenv('ENABLE_MEMORY', 'true').lower() == 'true',
        'enable_observability': os.getenv('ENABLE_OBSERVABILITY', 'true').lower() == 'true',
        'enable_data_transformations': os.getenv('ENABLE_DATA_TRANSFORMATIONS', 'true').lower() == 'true',
        'enable_system_health': os.getenv('ENABLE_SYSTEM_HEALTH', 'true').lower() == 'true',
        'enable_channels': os.getenv('ENABLE_CHANNELS', 'true').lower() == 'true',
        'enable_real_time_monitoring': os.getenv('ENABLE_REAL_TIME_MONITORING', 'true').lower() == 'true',
        'enable_bayesian': os.getenv('ENABLE_BAYESIAN', 'true').lower() == 'true'
    }

def get_api_timeout() -> int:
    """Get API timeout in seconds"""
    return int(os.getenv('API_TIMEOUT', '30'))

def get_cache_ttl() -> int:
    """Get cache TTL in seconds"""
    return int(os.getenv('CACHE_TTL', '300'))

def is_development() -> bool:
    """Check if running in development mode"""
    return os.getenv('ENVIRONMENT', 'development').lower() == 'development'

def get_log_level() -> str:
    """Get logging level"""
    return os.getenv('LOG_LEVEL', 'INFO').upper()