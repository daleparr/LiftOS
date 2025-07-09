"""
Test suite for Tier 0 (Legacy) API connectors
Tests Meta Business, Google Ads, and Klaviyo connectors with KSE integration
"""
import pytest
import asyncio
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import json

# Import connectors
from connectors.meta_business_connector import MetaBusinessConnector, create_meta_business_connector
from connectors.google_ads_connector import GoogleAdsConnector, create_google_ads_connector
from connectors.klaviyo_connector import KlaviyoConnector, create_klaviyo_connector

# Import shared models
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.models.marketing import MetaBusinessData, GoogleAdsData, KlaviyoData


class TestMetaBusinessConnector:
    """Test suite for Meta Business connector"""
    
    @pytest.fixture
    def mock_credentials(self):
        return {
            "access_token": "test_access_token",
            "app_id": "test_app_id",
            "app_secret": "test_app_secret"
        }
    
    @pytest.fixture
    def mock_ad_accounts_response(self):
        return {
            "data": [
                {
                    "id": "act_123456789",
                    "name": "Test Ad Account",
                    "account_status": 1,
                    "currency": "USD",
                    "timezone_name": "America/Los_Angeles"
                }
            ]
        }
    
    @pytest.fixture
    def mock_campaigns_response(self):
        return {
            "data": [
                {
                    "id": "23843185618250384",
                    "name": "Test Campaign",
                    "status": "ACTIVE",
                    "objective": "CONVERSIONS",
                    "daily_budget": "5000",
                    "created_time": "2024-01-01T00:00:00+0000",
                    "updated_time": "2024-01-02T00:00:00+0000"
                }
            ]
        }
    
    @pytest.fixture
    def mock_insights_response(self):
        return {
            "data": [
                {
                    "spend": "100.50",
                    "impressions": "10000",
                    "clicks": "500",
                    "reach": "8000",
                    "frequency": "1.25",
                    "cpm": "10.05",
                    "cpc": "0.201",
                    "ctr": "5.0",
                    "video_views": "200",
                    "actions": [
                        {"action_type": "purchase", "value": "50"},
                        {"action_type": "add_to_cart", "value": "150"}
                    ]
                }
            ]
        }
    
    @pytest.mark.asyncio
    async def test_meta_business_connector_initialization(self, mock_credentials):
        """Test Meta Business connector initialization"""
        connector = MetaBusinessConnector(
            access_token=mock_credentials["access_token"],
            app_id=mock_credentials["app_id"],
            app_secret=mock_credentials["app_secret"]
        )
        
        assert connector.access_token == mock_credentials["access_token"]
        assert connector.app_id == mock_credentials["app_id"]
        assert connector.app_secret == mock_credentials["app_secret"]
        assert connector.base_url == "https://graph.facebook.com/v18.0"
        assert connector.kse_integration is not None
        assert connector.causal_transformer is not None
        
        await connector.close()
    
    @pytest.mark.asyncio
    async def test_get_ad_accounts(self, mock_credentials, mock_ad_accounts_response):
        """Test getting ad accounts"""
        connector = MetaBusinessConnector(**mock_credentials)
        
        with patch.object(connector.client, 'get') as mock_get:
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_ad_accounts_response
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            accounts = await connector.get_ad_accounts()
            
            assert len(accounts) == 1
            assert accounts[0]["id"] == "act_123456789"
            assert accounts[0]["name"] == "Test Ad Account"
            
        await connector.close()
    
    @pytest.mark.asyncio
    async def test_extract_data_with_kse(self, mock_credentials):
        """Test data extraction with KSE integration"""
        connector = MetaBusinessConnector(**mock_credentials)
        
        # Mock KSE integration
        with patch.object(connector, 'get_ad_accounts') as mock_accounts, \
             patch.object(connector, 'get_campaigns') as mock_campaigns, \
             patch.object(connector.kse_integration, 'enhance_marketing_data') as mock_kse:
            
            mock_accounts.return_value = [{"id": "act_123456789"}]
            mock_campaigns.return_value = [
                {
                    "id": "campaign_123",
                    "name": "Test Campaign",
                    "status": "ACTIVE",
                    "objective": "CONVERSIONS",
                    "insights": {
                        "spend": "100.50",
                        "impressions": "10000",
                        "clicks": "500"
                    }
                }
            ]
            
            # Mock enhanced data
            enhanced_data = [
                MetaBusinessData(
                    campaign_id="campaign_123",
                    campaign_name="Test Campaign",
                    account_id="act_123456789",
                    date_start=date.today() - timedelta(days=7),
                    date_end=date.today(),
                    spend=100.50,
                    impressions=10000,
                    clicks=500,
                    reach=8000,
                    frequency=1.25,
                    cpm=10.05,
                    cpc=0.201,
                    ctr=5.0,
                    video_views=200,
                    actions=[],
                    objective="CONVERSIONS",
                    status="ACTIVE",
                    created_time="2024-01-01T00:00:00+0000",
                    updated_time="2024-01-02T00:00:00+0000"
                )
            ]
            mock_kse.return_value = enhanced_data
            
            result = await connector.extract_data(
                date.today() - timedelta(days=7),
                date.today()
            )
            
            assert len(result) == 1
            assert result[0].campaign_id == "campaign_123"
            assert result[0].spend == 100.50
            mock_kse.assert_called_once()
            
        await connector.close()
    
    @pytest.mark.asyncio
    async def test_validate_connection(self, mock_credentials):
        """Test connection validation"""
        connector = MetaBusinessConnector(**mock_credentials)
        
        with patch.object(connector.client, 'get') as mock_get:
            mock_response = AsyncMock()
            mock_response.json.return_value = {"id": "123", "name": "Test User"}
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            is_valid = await connector.validate_connection()
            assert is_valid is True
            
        await connector.close()
    
    @pytest.mark.asyncio
    async def test_create_meta_business_connector(self, mock_credentials):
        """Test connector factory function"""
        with patch('connectors.meta_business_connector.MetaBusinessConnector.validate_connection') as mock_validate:
            mock_validate.return_value = True
            
            connector = await create_meta_business_connector(mock_credentials)
            assert isinstance(connector, MetaBusinessConnector)
            mock_validate.assert_called_once()
            
            await connector.close()


class TestGoogleAdsConnector:
    """Test suite for Google Ads connector"""
    
    @pytest.fixture
    def mock_credentials(self):
        return {
            "developer_token": "test_developer_token",
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "refresh_token": "test_refresh_token"
        }
    
    @pytest.fixture
    def mock_token_response(self):
        return {
            "access_token": "test_access_token",
            "expires_in": 3600,
            "token_type": "Bearer"
        }
    
    @pytest.fixture
    def mock_customers_response(self):
        return {
            "resourceNames": [
                "customers/1234567890",
                "customers/0987654321"
            ]
        }
    
    @pytest.mark.asyncio
    async def test_google_ads_connector_initialization(self, mock_credentials):
        """Test Google Ads connector initialization"""
        connector = GoogleAdsConnector(**mock_credentials)
        
        assert connector.developer_token == mock_credentials["developer_token"]
        assert connector.client_id == mock_credentials["client_id"]
        assert connector.client_secret == mock_credentials["client_secret"]
        assert connector.refresh_token == mock_credentials["refresh_token"]
        assert connector.base_url == "https://googleads.googleapis.com/v14"
        assert connector.kse_integration is not None
        assert connector.causal_transformer is not None
        
        await connector.close()
    
    @pytest.mark.asyncio
    async def test_authenticate(self, mock_credentials, mock_token_response):
        """Test authentication"""
        connector = GoogleAdsConnector(**mock_credentials)
        
        with patch.object(connector.client, 'post') as mock_post:
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_token_response
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response
            
            await connector.authenticate()
            
            assert connector.access_token == "test_access_token"
            
        await connector.close()
    
    @pytest.mark.asyncio
    async def test_get_customers(self, mock_credentials, mock_customers_response):
        """Test getting customers"""
        connector = GoogleAdsConnector(**mock_credentials)
        connector.access_token = "test_access_token"
        
        with patch.object(connector.client, 'get') as mock_get:
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_customers_response
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            customers = await connector.get_customers()
            
            assert len(customers) == 2
            assert "customers/1234567890" in customers
            
        await connector.close()
    
    @pytest.mark.asyncio
    async def test_extract_data_with_kse(self, mock_credentials):
        """Test data extraction with KSE integration"""
        connector = GoogleAdsConnector(**mock_credentials)
        
        with patch.object(connector, 'get_customers') as mock_customers, \
             patch.object(connector, 'get_campaigns') as mock_campaigns, \
             patch.object(connector.kse_integration, 'enhance_marketing_data') as mock_kse:
            
            mock_customers.return_value = ["customers/1234567890"]
            mock_campaigns.return_value = [
                {
                    "campaign": {"id": "123", "name": "Test Campaign"},
                    "metrics": {"costMicros": "100000000", "impressions": "10000"},
                    "segments": {"date": "2024-01-01"}
                }
            ]
            
            enhanced_data = [
                GoogleAdsData(
                    campaign_id="123",
                    campaign_name="Test Campaign",
                    customer_id="1234567890",
                    date_start=date.today() - timedelta(days=7),
                    date_end=date.today(),
                    cost_micros=100000000,
                    impressions=10000,
                    clicks=500,
                    conversions=10.0,
                    conversions_value=1000.0,
                    advertising_channel_type="SEARCH",
                    status="ENABLED",
                    date="2024-01-01"
                )
            ]
            mock_kse.return_value = enhanced_data
            
            result = await connector.extract_data(
                date.today() - timedelta(days=7),
                date.today()
            )
            
            assert len(result) == 1
            assert result[0].campaign_id == "123"
            mock_kse.assert_called_once()
            
        await connector.close()


class TestKlaviyoConnector:
    """Test suite for Klaviyo connector"""
    
    @pytest.fixture
    def mock_credentials(self):
        return {
            "api_key": "test_api_key"
        }
    
    @pytest.fixture
    def mock_campaigns_response(self):
        return {
            "data": [
                {
                    "id": "campaign_123",
                    "attributes": {
                        "name": "Test Email Campaign",
                        "status": "sent",
                        "send_time": "2024-01-01T12:00:00Z",
                        "subject_line": "Test Subject",
                        "campaign_type": "regular"
                    }
                }
            ]
        }
    
    @pytest.mark.asyncio
    async def test_klaviyo_connector_initialization(self, mock_credentials):
        """Test Klaviyo connector initialization"""
        connector = KlaviyoConnector(api_key=mock_credentials["api_key"])
        
        assert connector.api_key == mock_credentials["api_key"]
        assert connector.base_url == "https://a.klaviyo.com/api"
        assert connector.kse_integration is not None
        assert connector.causal_transformer is not None
        
        await connector.close()
    
    @pytest.mark.asyncio
    async def test_get_campaigns(self, mock_credentials, mock_campaigns_response):
        """Test getting campaigns"""
        connector = KlaviyoConnector(api_key=mock_credentials["api_key"])
        
        with patch.object(connector, 'get_campaign_metrics') as mock_metrics, \
             patch.object(connector.client, 'get') as mock_get:
            
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_campaigns_response
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            mock_metrics.return_value = {
                "sent_count": 1000,
                "delivered_count": 950,
                "open_count": 300,
                "click_count": 50
            }
            
            campaigns = await connector.get_campaigns(
                date.today() - timedelta(days=7),
                date.today()
            )
            
            assert len(campaigns) == 1
            assert campaigns[0]["id"] == "campaign_123"
            assert campaigns[0]["metrics"]["sent_count"] == 1000
            
        await connector.close()
    
    @pytest.mark.asyncio
    async def test_extract_data_with_kse(self, mock_credentials):
        """Test data extraction with KSE integration"""
        connector = KlaviyoConnector(api_key=mock_credentials["api_key"])
        
        with patch.object(connector, 'get_campaigns') as mock_campaigns, \
             patch.object(connector.kse_integration, 'enhance_marketing_data') as mock_kse:
            
            mock_campaigns.return_value = [
                {
                    "id": "campaign_123",
                    "attributes": {
                        "name": "Test Campaign",
                        "status": "sent",
                        "send_time": "2024-01-01T12:00:00Z"
                    },
                    "metrics": {
                        "sent_count": 1000,
                        "delivered_count": 950,
                        "open_count": 300,
                        "click_count": 50
                    }
                }
            ]
            
            enhanced_data = [
                KlaviyoData(
                    campaign_id="campaign_123",
                    campaign_name="Test Campaign",
                    date_start=date.today() - timedelta(days=7),
                    date_end=date.today(),
                    sent_count=1000,
                    delivered_count=950,
                    open_count=300,
                    click_count=50,
                    unsubscribe_count=5,
                    bounce_count=50,
                    open_rate=0.316,
                    click_rate=0.053,
                    unsubscribe_rate=0.005,
                    bounce_rate=0.05,
                    subject_line="Test Subject",
                    send_time="2024-01-01T12:00:00Z",
                    campaign_type="regular",
                    status="sent"
                )
            ]
            mock_kse.return_value = enhanced_data
            
            result = await connector.extract_data(
                date.today() - timedelta(days=7),
                date.today()
            )
            
            assert len(result) == 1
            assert result[0].campaign_id == "campaign_123"
            assert result[0].sent_count == 1000
            mock_kse.assert_called_once()
            
        await connector.close()
    
    @pytest.mark.asyncio
    async def test_validate_connection(self, mock_credentials):
        """Test connection validation"""
        connector = KlaviyoConnector(api_key=mock_credentials["api_key"])
        
        with patch.object(connector.client, 'get') as mock_get:
            mock_response = AsyncMock()
            mock_response.json.return_value = {
                "data": [
                    {
                        "attributes": {
                            "contact_information": {
                                "organization_name": "Test Organization"
                            }
                        }
                    }
                ]
            }
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            is_valid = await connector.validate_connection()
            assert is_valid is True
            
        await connector.close()


# Integration tests
class TestTier0Integration:
    """Integration tests for all Tier 0 connectors"""
    
    @pytest.mark.asyncio
    async def test_all_connectors_have_kse_integration(self):
        """Test that all connectors have KSE integration"""
        meta_connector = MetaBusinessConnector("token", "app_id", "secret")
        google_connector = GoogleAdsConnector("dev_token", "client_id", "secret", "refresh")
        klaviyo_connector = KlaviyoConnector("api_key")
        
        assert hasattr(meta_connector, 'kse_integration')
        assert hasattr(google_connector, 'kse_integration')
        assert hasattr(klaviyo_connector, 'kse_integration')
        
        assert hasattr(meta_connector, 'causal_transformer')
        assert hasattr(google_connector, 'causal_transformer')
        assert hasattr(klaviyo_connector, 'causal_transformer')
        
        await meta_connector.close()
        await google_connector.close()
        await klaviyo_connector.close()
    
    @pytest.mark.asyncio
    async def test_all_connectors_have_extract_causal_data(self):
        """Test that all connectors have causal data extraction"""
        meta_connector = MetaBusinessConnector("token", "app_id", "secret")
        google_connector = GoogleAdsConnector("dev_token", "client_id", "secret", "refresh")
        klaviyo_connector = KlaviyoConnector("api_key")
        
        assert hasattr(meta_connector, 'extract_causal_data')
        assert hasattr(google_connector, 'extract_causal_data')
        assert hasattr(klaviyo_connector, 'extract_causal_data')
        
        await meta_connector.close()
        await google_connector.close()
        await klaviyo_connector.close()


if __name__ == "__main__":
    pytest.main([__file__])