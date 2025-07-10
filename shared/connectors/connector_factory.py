"""
Connector Factory
Factory pattern for creating platform-specific connectors with unified interface
"""

from typing import Dict, Any, Optional, Type, Protocol
from abc import ABC, abstractmethod
from datetime import datetime, date
import logging

from shared.models.marketing import DataSource
from shared.utils.logging import setup_logging

logger = setup_logging("connector_factory")

class PlatformConnector(Protocol):
    """Protocol defining the interface for all platform connectors"""
    
    async def authenticate(self, credentials: Dict[str, str]) -> bool:
        """Authenticate with the platform"""
        ...
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test the connection and return status"""
        ...
    
    async def extract_data(self, start_date: date, end_date: date, **kwargs) -> list:
        """Extract data from the platform"""
        ...
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        ...
    
    async def close(self) -> None:
        """Close the connection"""
        ...

class BaseConnector(ABC):
    """Base class for all platform connectors"""
    
    def __init__(self, credentials: Dict[str, str]):
        self.credentials = credentials
        self.authenticated = False
        self.logger = setup_logging(f"connector_{self.__class__.__name__.lower()}")
    
    @abstractmethod
    async def authenticate(self, credentials: Dict[str, str]) -> bool:
        """Authenticate with the platform"""
        pass
    
    @abstractmethod
    async def test_connection(self) -> Dict[str, Any]:
        """Test the connection and return status"""
        pass
    
    @abstractmethod
    async def extract_data(self, start_date: date, end_date: date, **kwargs) -> list:
        """Extract data from the platform"""
        pass
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information - default implementation"""
        return {
            "platform": self.__class__.__name__.replace("Connector", "").lower(),
            "authenticated": self.authenticated,
            "credentials_provided": bool(self.credentials)
        }
    
    async def close(self) -> None:
        """Close the connection - default implementation"""
        self.authenticated = False

class MockConnector(BaseConnector):
    """Mock connector for testing and fallback scenarios"""
    
    def __init__(self, platform: str, credentials: Dict[str, str] = None):
        super().__init__(credentials or {})
        self.platform = platform
    
    async def authenticate(self, credentials: Dict[str, str]) -> bool:
        """Mock authentication - always succeeds"""
        self.authenticated = True
        return True
    
    async def test_connection(self) -> Dict[str, Any]:
        """Mock connection test"""
        return {
            "success": True,
            "platform": self.platform,
            "message": "Mock connection successful",
            "data_preview": {
                "campaigns": 5,
                "last_updated": datetime.utcnow().isoformat(),
                "mock_data": True
            }
        }
    
    async def extract_data(self, start_date: date, end_date: date, **kwargs) -> list:
        """Generate mock data"""
        from shared.utils.mock_data_generator import generate_mock_marketing_data
        
        return generate_mock_marketing_data(
            platform=self.platform,
            start_date=start_date,
            end_date=end_date,
            **kwargs
        )

class ConnectorFactory:
    """Factory for creating platform connectors"""
    
    def __init__(self):
        self._connectors: Dict[str, Type[BaseConnector]] = {}
        self._register_default_connectors()
    
    def _register_default_connectors(self):
        """Register default platform connectors"""
        try:
            # Import and register Tier 0 connectors (Core Marketing)
            from services.data_ingestion.connectors.meta_business_connector import MetaBusinessConnector
            self.register_connector(DataSource.META_BUSINESS, MetaBusinessConnector)
        except ImportError:
            logger.warning("MetaBusinessConnector not available")
        
        try:
            from services.data_ingestion.connectors.google_ads_connector import GoogleAdsConnector
            self.register_connector(DataSource.GOOGLE_ADS, GoogleAdsConnector)
        except ImportError:
            logger.warning("GoogleAdsConnector not available")
        
        try:
            from services.data_ingestion.connectors.klaviyo_connector import KlaviyoConnector
            self.register_connector(DataSource.KLAVIYO, KlaviyoConnector)
        except ImportError:
            logger.warning("KlaviyoConnector not available")
        
        # Import and register Tier 1 connectors (E-commerce)
        try:
            from services.data_ingestion.connectors.shopify_connector import ShopifyConnector
            self.register_connector(DataSource.SHOPIFY, ShopifyConnector)
        except ImportError:
            logger.warning("ShopifyConnector not available")
        
        try:
            from services.data_ingestion.connectors.woocommerce_connector import WooCommerceConnector
            self.register_connector(DataSource.WOOCOMMERCE, WooCommerceConnector)
        except ImportError:
            logger.warning("WooCommerceConnector not available")
        
        try:
            from services.data_ingestion.connectors.amazon_connector import AmazonConnector
            self.register_connector(DataSource.AMAZON_SELLER_CENTRAL, AmazonConnector)
        except ImportError:
            logger.warning("AmazonConnector not available")
        
        # Import and register Tier 2 connectors (CRM/Payment)
        try:
            from services.data_ingestion.connectors.hubspot_connector import HubSpotConnector
            self.register_connector(DataSource.HUBSPOT, HubSpotConnector)
        except ImportError:
            logger.warning("HubSpotConnector not available")
        
        try:
            from services.data_ingestion.connectors.salesforce_connector import SalesforceConnector
            self.register_connector(DataSource.SALESFORCE, SalesforceConnector)
        except ImportError:
            logger.warning("SalesforceConnector not available")
        
        try:
            from services.data_ingestion.connectors.stripe_connector import StripeConnector
            self.register_connector(DataSource.STRIPE, StripeConnector)
        except ImportError:
            logger.warning("StripeConnector not available")
        
        try:
            from services.data_ingestion.connectors.paypal_connector import PayPalConnector
            self.register_connector(DataSource.PAYPAL, PayPalConnector)
        except ImportError:
            logger.warning("PayPalConnector not available")
        
        # Import and register Tier 3 connectors (Social/Analytics/Data)
        try:
            from services.data_ingestion.connectors.tiktok_connector import TikTokConnector
            self.register_connector(DataSource.TIKTOK, TikTokConnector)
        except ImportError:
            logger.warning("TikTokConnector not available")
        
        try:
            from services.data_ingestion.connectors.snowflake_connector import SnowflakeConnector
            self.register_connector(DataSource.SNOWFLAKE, SnowflakeConnector)
        except ImportError:
            logger.warning("SnowflakeConnector not available")
        
        try:
            from services.data_ingestion.connectors.databricks_connector import DatabricksConnector
            self.register_connector(DataSource.DATABRICKS, DatabricksConnector)
        except ImportError:
            logger.warning("DatabricksConnector not available")
        
        # Import and register Tier 4 connectors (Extended Social/CRM)
        try:
            from services.data_ingestion.connectors.zoho_crm_connector import ZohoCRMConnector
            self.register_connector(DataSource.ZOHO_CRM, ZohoCRMConnector)
        except ImportError:
            logger.warning("ZohoCRMConnector not available")
        
        try:
            from services.data_ingestion.connectors.linkedin_ads_connector import LinkedInAdsConnector
            self.register_connector(DataSource.LINKEDIN_ADS, LinkedInAdsConnector)
        except ImportError:
            logger.warning("LinkedInAdsConnector not available")
        
        try:
            from services.data_ingestion.connectors.x_ads_connector import XAdsConnector
            self.register_connector(DataSource.X_ADS, XAdsConnector)
        except ImportError:
            logger.warning("XAdsConnector not available")
    
    def register_connector(self, platform: DataSource, connector_class: Type[BaseConnector]):
        """Register a connector for a platform"""
        self._connectors[platform.value] = connector_class
        logger.info(f"Registered connector for {platform.value}")
    
    def create_connector(self, platform: DataSource, credentials: Dict[str, str], 
                        use_mock: bool = False) -> BaseConnector:
        """Create a connector instance for the specified platform"""
        platform_key = platform.value
        
        if use_mock or platform_key not in self._connectors:
            if platform_key not in self._connectors:
                logger.warning(f"No connector available for {platform_key}, using mock")
            return MockConnector(platform_key, credentials)
        
        connector_class = self._connectors[platform_key]
        return connector_class(credentials)
    
    def get_supported_platforms(self) -> list:
        """Get list of supported platforms"""
        return list(self._connectors.keys())
    
    def is_platform_supported(self, platform: DataSource) -> bool:
        """Check if a platform is supported"""
        return platform.value in self._connectors

class ConnectorManager:
    """Manager for handling multiple connector instances"""
    
    def __init__(self):
        self.factory = ConnectorFactory()
        self._active_connectors: Dict[str, BaseConnector] = {}
    
    async def get_connector(self, connection_id: str, platform: DataSource, 
                          credentials: Dict[str, str], use_mock: bool = False) -> BaseConnector:
        """Get or create a connector instance"""
        if connection_id in self._active_connectors:
            return self._active_connectors[connection_id]
        
        connector = self.factory.create_connector(platform, credentials, use_mock)
        
        # Authenticate the connector
        try:
            authenticated = await connector.authenticate(credentials)
            if not authenticated:
                logger.error(f"Failed to authenticate connector for {platform.value}")
                # Fall back to mock connector
                connector = MockConnector(platform.value, credentials)
                await connector.authenticate(credentials)
        except Exception as e:
            logger.error(f"Error authenticating connector for {platform.value}: {str(e)}")
            # Fall back to mock connector
            connector = MockConnector(platform.value, credentials)
            await connector.authenticate(credentials)
        
        self._active_connectors[connection_id] = connector
        return connector
    
    async def test_connector(self, connection_id: str, platform: DataSource, 
                           credentials: Dict[str, str]) -> Dict[str, Any]:
        """Test a connector without caching it"""
        connector = self.factory.create_connector(platform, credentials)
        
        try:
            authenticated = await connector.authenticate(credentials)
            if authenticated:
                test_result = await connector.test_connection()
                await connector.close()
                return test_result
            else:
                return {
                    "success": False,
                    "platform": platform.value,
                    "message": "Authentication failed",
                    "error_code": "AUTH_FAILED"
                }
        except Exception as e:
            await connector.close()
            return {
                "success": False,
                "platform": platform.value,
                "message": str(e),
                "error_code": "CONNECTION_ERROR"
            }
    
    async def close_connector(self, connection_id: str):
        """Close and remove a connector"""
        if connection_id in self._active_connectors:
            connector = self._active_connectors[connection_id]
            await connector.close()
            del self._active_connectors[connection_id]
    
    async def close_all_connectors(self):
        """Close all active connectors"""
        for connection_id in list(self._active_connectors.keys()):
            await self.close_connector(connection_id)
    
    def get_supported_platforms(self) -> list:
        """Get list of supported platforms"""
        return self.factory.get_supported_platforms()

# Global connector manager instance
connector_manager = ConnectorManager()

def get_connector_manager() -> ConnectorManager:
    """Get the global connector manager instance"""
    return connector_manager