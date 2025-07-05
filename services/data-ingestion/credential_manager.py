"""
Credential Management for Data Ingestion Service
Handles secure storage and retrieval of API credentials
"""
import os
import json
from typing import Dict, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class CredentialProvider(Enum):
    """Supported credential providers"""
    ENVIRONMENT = "environment"
    FILE = "file"
    VAULT = "vault"  # Future implementation

class CredentialManager:
    """Manages API credentials for external platforms"""
    
    def __init__(self, provider: CredentialProvider = CredentialProvider.ENVIRONMENT):
        self.provider = provider
        self._credentials_cache = {}
    
    async def get_meta_business_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get Meta Business API credentials"""
        cache_key = f"meta_business_{org_id}"
        
        if cache_key in self._credentials_cache:
            return self._credentials_cache[cache_key]
        
        credentials = None
        
        if self.provider == CredentialProvider.ENVIRONMENT:
            credentials = self._get_meta_env_credentials()
        elif self.provider == CredentialProvider.FILE:
            credentials = self._get_meta_file_credentials(org_id)
        
        if credentials:
            self._credentials_cache[cache_key] = credentials
            logger.info(f"Retrieved Meta Business credentials for org {org_id}")
        else:
            logger.warning(f"No Meta Business credentials found for org {org_id}")
        
        return credentials
    
    async def get_google_ads_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get Google Ads API credentials"""
        cache_key = f"google_ads_{org_id}"
        
        if cache_key in self._credentials_cache:
            return self._credentials_cache[cache_key]
        
        credentials = None
        
        if self.provider == CredentialProvider.ENVIRONMENT:
            credentials = self._get_google_env_credentials()
        elif self.provider == CredentialProvider.FILE:
            credentials = self._get_google_file_credentials(org_id)
        
        if credentials:
            self._credentials_cache[cache_key] = credentials
            logger.info(f"Retrieved Google Ads credentials for org {org_id}")
        else:
            logger.warning(f"No Google Ads credentials found for org {org_id}")
        
        return credentials
    
    async def get_klaviyo_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get Klaviyo API credentials"""
        cache_key = f"klaviyo_{org_id}"
        
        if cache_key in self._credentials_cache:
            return self._credentials_cache[cache_key]
        
        credentials = None
        
        if self.provider == CredentialProvider.ENVIRONMENT:
            credentials = self._get_klaviyo_env_credentials()
        elif self.provider == CredentialProvider.FILE:
            credentials = self._get_klaviyo_file_credentials(org_id)
        
        if credentials:
            self._credentials_cache[cache_key] = credentials
            logger.info(f"Retrieved Klaviyo credentials for org {org_id}")
        else:
            logger.warning(f"No Klaviyo credentials found for org {org_id}")
        
        return credentials
    
    def _get_meta_env_credentials(self) -> Optional[Dict[str, str]]:
        """Get Meta Business credentials from environment variables"""
        access_token = os.getenv("META_ACCESS_TOKEN")
        app_id = os.getenv("META_APP_ID")
        app_secret = os.getenv("META_APP_SECRET")
        
        if access_token and app_id and app_secret:
            return {
                "access_token": access_token,
                "app_id": app_id,
                "app_secret": app_secret
            }
        return None
    
    def _get_google_env_credentials(self) -> Optional[Dict[str, str]]:
        """Get Google Ads credentials from environment variables"""
        developer_token = os.getenv("GOOGLE_ADS_DEVELOPER_TOKEN")
        client_id = os.getenv("GOOGLE_ADS_CLIENT_ID")
        client_secret = os.getenv("GOOGLE_ADS_CLIENT_SECRET")
        refresh_token = os.getenv("GOOGLE_ADS_REFRESH_TOKEN")
        
        if developer_token and client_id and client_secret and refresh_token:
            return {
                "developer_token": developer_token,
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token
            }
        return None
    
    def _get_klaviyo_env_credentials(self) -> Optional[Dict[str, str]]:
        """Get Klaviyo credentials from environment variables"""
        api_key = os.getenv("KLAVIYO_API_KEY")
        
        if api_key:
            return {
                "api_key": api_key
            }
        return None
    
    def _get_meta_file_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get Meta Business credentials from file (future implementation)"""
        credentials_file = f"/app/credentials/{org_id}/meta_business.json"
        return self._load_credentials_from_file(credentials_file)
    
    def _get_google_file_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get Google Ads credentials from file (future implementation)"""
        credentials_file = f"/app/credentials/{org_id}/google_ads.json"
        return self._load_credentials_from_file(credentials_file)
    
    def _get_klaviyo_file_credentials(self, org_id: str) -> Optional[Dict[str, str]]:
        """Get Klaviyo credentials from file (future implementation)"""
        credentials_file = f"/app/credentials/{org_id}/klaviyo.json"
        return self._load_credentials_from_file(credentials_file)
    
    def _load_credentials_from_file(self, file_path: str) -> Optional[Dict[str, str]]:
        """Load credentials from JSON file"""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading credentials from {file_path}: {str(e)}")
        return None
    
    def clear_cache(self):
        """Clear credentials cache"""
        self._credentials_cache.clear()
        logger.info("Credentials cache cleared")
    
    def validate_credentials(self, platform: str, credentials: Dict[str, str]) -> bool:
        """Validate that credentials contain required fields"""
        required_fields = {
            "meta_business": ["access_token", "app_id", "app_secret"],
            "google_ads": ["developer_token", "client_id", "client_secret", "refresh_token"],
            "klaviyo": ["api_key"]
        }
        
        if platform not in required_fields:
            return False
        
        return all(field in credentials for field in required_fields[platform])

# Global credential manager instance
credential_manager = CredentialManager()