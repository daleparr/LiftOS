"""
Secrets management for Lift OS Core services
Supports multiple backends: environment variables, AWS Secrets Manager, HashiCorp Vault
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SecretConfig:
    """Configuration for secret retrieval"""
    name: str
    key: Optional[str] = None  # For nested secrets
    default: Optional[str] = None
    required: bool = True

class SecretsBackend(ABC):
    """Abstract base class for secrets backends"""
    
    @abstractmethod
    async def get_secret(self, secret_name: str) -> Union[str, Dict[str, Any]]:
        """Retrieve a secret by name"""
        pass
    
    @abstractmethod
    async def get_secret_value(self, secret_name: str, key: str) -> str:
        """Retrieve a specific key from a secret"""
        pass

class EnvironmentSecretsBackend(SecretsBackend):
    """Environment variables secrets backend"""
    
    async def get_secret(self, secret_name: str) -> Union[str, Dict[str, Any]]:
        """Get secret from environment variable"""
        value = os.getenv(secret_name)
        if value is None:
            raise ValueError(f"Environment variable {secret_name} not found")
        
        # Try to parse as JSON for complex secrets
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    
    async def get_secret_value(self, secret_name: str, key: str) -> str:
        """Get specific key from environment variable"""
        # Try compound key first (SECRET_NAME_KEY)
        compound_key = f"{secret_name}_{key}".upper()
        value = os.getenv(compound_key)
        
        if value is not None:
            return value
        
        # Try getting the secret as JSON and extracting key
        secret = await self.get_secret(secret_name)
        if isinstance(secret, dict):
            return secret.get(key)
        
        raise ValueError(f"Key {key} not found in secret {secret_name}")

class AWSSecretsManagerBackend(SecretsBackend):
    """AWS Secrets Manager backend"""
    
    def __init__(self, region_name: str = "us-east-1"):
        self.region_name = region_name
        self._client = None
    
    @property
    def client(self):
        """Lazy initialization of AWS client"""
        if self._client is None:
            try:
                import boto3
                from botocore.exceptions import ClientError
                self._client = boto3.client('secretsmanager', region_name=self.region_name)
                self.ClientError = ClientError
            except ImportError:
                raise ImportError("boto3 is required for AWS Secrets Manager backend")
        return self._client
    
    async def get_secret(self, secret_name: str) -> Union[str, Dict[str, Any]]:
        """Retrieve secret from AWS Secrets Manager"""
        try:
            response = self.client.get_secret_value(SecretId=secret_name)
            secret_string = response['SecretString']
            
            # Try to parse as JSON
            try:
                return json.loads(secret_string)
            except json.JSONDecodeError:
                return secret_string
                
        except self.ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ResourceNotFoundException':
                raise ValueError(f"Secret {secret_name} not found in AWS Secrets Manager")
            elif error_code == 'InvalidRequestException':
                raise ValueError(f"Invalid request for secret {secret_name}")
            elif error_code == 'InvalidParameterException':
                raise ValueError(f"Invalid parameter for secret {secret_name}")
            else:
                logger.error(f"AWS Secrets Manager error: {e}")
                raise
    
    async def get_secret_value(self, secret_name: str, key: str) -> str:
        """Get specific key from AWS secret"""
        secret = await self.get_secret(secret_name)
        
        if isinstance(secret, dict):
            if key not in secret:
                raise ValueError(f"Key {key} not found in secret {secret_name}")
            return secret[key]
        else:
            raise ValueError(f"Secret {secret_name} is not a JSON object")

class HashiCorpVaultBackend(SecretsBackend):
    """HashiCorp Vault backend"""
    
    def __init__(self, url: str, token: str, mount_point: str = "secret"):
        self.url = url
        self.token = token
        self.mount_point = mount_point
        self._client = None
    
    @property
    def client(self):
        """Lazy initialization of Vault client"""
        if self._client is None:
            try:
                import hvac
                self._client = hvac.Client(url=self.url, token=self.token)
                if not self._client.is_authenticated():
                    raise ValueError("Failed to authenticate with HashiCorp Vault")
            except ImportError:
                raise ImportError("hvac is required for HashiCorp Vault backend")
        return self._client
    
    async def get_secret(self, secret_name: str) -> Union[str, Dict[str, Any]]:
        """Retrieve secret from HashiCorp Vault"""
        try:
            response = self.client.secrets.kv.v2.read_secret_version(
                path=secret_name,
                mount_point=self.mount_point
            )
            return response['data']['data']
        except Exception as e:
            logger.error(f"Vault error retrieving {secret_name}: {e}")
            raise ValueError(f"Failed to retrieve secret {secret_name} from Vault")
    
    async def get_secret_value(self, secret_name: str, key: str) -> str:
        """Get specific key from Vault secret"""
        secret = await self.get_secret(secret_name)
        
        if key not in secret:
            raise ValueError(f"Key {key} not found in secret {secret_name}")
        
        return secret[key]

class SecretsManager:
    """Main secrets manager with multiple backend support"""
    
    def __init__(self, backend: SecretsBackend = None):
        self.backend = backend or self._get_default_backend()
        self._cache: Dict[str, Any] = {}
        self.cache_enabled = os.getenv("SECRETS_CACHE_ENABLED", "true").lower() == "true"
    
    def _get_default_backend(self) -> SecretsBackend:
        """Determine the default backend based on environment"""
        backend_type = os.getenv("SECRETS_BACKEND", "environment").lower()
        
        if backend_type == "aws":
            region = os.getenv("AWS_REGION", "us-east-1")
            return AWSSecretsManagerBackend(region_name=region)
        elif backend_type == "vault":
            vault_url = os.getenv("VAULT_URL")
            vault_token = os.getenv("VAULT_TOKEN")
            if not vault_url or not vault_token:
                raise ValueError("VAULT_URL and VAULT_TOKEN must be set for Vault backend")
            return HashiCorpVaultBackend(vault_url, vault_token)
        else:
            return EnvironmentSecretsBackend()
    
    async def get_secret(self, secret_name: str, use_cache: bool = True) -> Union[str, Dict[str, Any]]:
        """Get a secret with optional caching"""
        cache_key = f"secret:{secret_name}"
        
        # Check cache first
        if use_cache and self.cache_enabled and cache_key in self._cache:
            logger.debug(f"Retrieved secret {secret_name} from cache")
            return self._cache[cache_key]
        
        # Retrieve from backend
        try:
            secret = await self.backend.get_secret(secret_name)
            
            # Cache the result
            if self.cache_enabled:
                self._cache[cache_key] = secret
                logger.debug(f"Cached secret {secret_name}")
            
            return secret
            
        except Exception as e:
            logger.error(f"Failed to retrieve secret {secret_name}: {e}")
            raise
    
    async def get_secret_value(self, secret_name: str, key: str, use_cache: bool = True) -> str:
        """Get a specific value from a secret"""
        cache_key = f"secret:{secret_name}:{key}"
        
        # Check cache first
        if use_cache and self.cache_enabled and cache_key in self._cache:
            logger.debug(f"Retrieved secret value {secret_name}.{key} from cache")
            return self._cache[cache_key]
        
        # Retrieve from backend
        try:
            value = await self.backend.get_secret_value(secret_name, key)
            
            # Cache the result
            if self.cache_enabled:
                self._cache[cache_key] = value
                logger.debug(f"Cached secret value {secret_name}.{key}")
            
            return value
            
        except Exception as e:
            logger.error(f"Failed to retrieve secret value {secret_name}.{key}: {e}")
            raise
    
    async def get_config(self, configs: Dict[str, SecretConfig]) -> Dict[str, str]:
        """Get multiple secrets based on configuration"""
        result = {}
        
        for config_key, secret_config in configs.items():
            try:
                if secret_config.key:
                    # Get specific key from secret
                    value = await self.get_secret_value(secret_config.name, secret_config.key)
                else:
                    # Get entire secret
                    secret = await self.get_secret(secret_config.name)
                    value = secret if isinstance(secret, str) else json.dumps(secret)
                
                result[config_key] = value
                
            except Exception as e:
                if secret_config.required:
                    logger.error(f"Required secret {secret_config.name} not found: {e}")
                    raise
                else:
                    logger.warning(f"Optional secret {secret_config.name} not found, using default")
                    result[config_key] = secret_config.default
        
        return result
    
    def clear_cache(self):
        """Clear the secrets cache"""
        self._cache.clear()
        logger.info("Secrets cache cleared")

# Global secrets manager instance
_secrets_manager: Optional[SecretsManager] = None

def get_secrets_manager() -> SecretsManager:
    """Get the global secrets manager instance"""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager

# Convenience functions for common secrets
async def get_database_config() -> Dict[str, str]:
    """Get database configuration from secrets"""
    secrets = get_secrets_manager()
    
    configs = {
        "database_url": SecretConfig("DATABASE_URL", required=True),
        "database_user": SecretConfig("DATABASE_USER", default="lift_user"),
        "database_password": SecretConfig("DATABASE_PASSWORD", required=True),
        "database_name": SecretConfig("DATABASE_NAME", default="lift_os"),
    }
    
    return await secrets.get_config(configs)

async def get_jwt_config() -> Dict[str, str]:
    """Get JWT configuration from secrets"""
    secrets = get_secrets_manager()
    
    configs = {
        "jwt_secret": SecretConfig("JWT_SECRET", required=True),
        "jwt_algorithm": SecretConfig("JWT_ALGORITHM", default="HS256"),
        "jwt_expiry_hours": SecretConfig("JWT_EXPIRY_HOURS", default="24"),
    }
    
    return await secrets.get_config(configs)

async def get_service_config(service_name: str) -> Dict[str, str]:
    """Get service-specific configuration"""
    secrets = get_secrets_manager()
    
    configs = {
        "api_key": SecretConfig(f"{service_name.upper()}_API_KEY", required=False),
        "service_url": SecretConfig(f"{service_name.upper()}_URL", required=False),
        "timeout": SecretConfig(f"{service_name.upper()}_TIMEOUT", default="30"),
    }
    
    return await secrets.get_config(configs)