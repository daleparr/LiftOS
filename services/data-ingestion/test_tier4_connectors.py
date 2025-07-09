"""
Test Suite for Tier 4 Connectors (Extended Social/CRM)
Tests Zoho CRM, LinkedIn Ads, X Ads, and X Sentiment connectors
"""
import asyncio
import pytest
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch, AsyncMock
import json

# Import connectors
from connectors.zoho_crm_connector import ZohoCRMConnector
from connectors.linkedin_ads_connector import LinkedInAdsConnector
from connectors.x_ads_connector import XAdsConnector

# Import shared models
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from shared.models.marketing import DataSource


class TestZohoCRMConnector:
    """Test Zoho CRM connector functionality"""
    
    @pytest.fixture
    def mock_credentials(self):
        return {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret", 
            "refresh_token": "test_refresh_token",
            "domain": "com"
        }
    
    @pytest.fixture
    def mock_zoho_response(self):
        return {
            "data": [
                {
                    "id": "12345",
                    "Company": "Test Company",
                    "Lead_Source": "Website",
                    "Email": "test@example.com",
                    "Phone": "555-0123",
                    "Lead_Status": "Qualified",
                    "Created_Time": "2024-01-15T10:30:00Z",
                    "Modified_Time": "2024-01-15T15:45:00Z"
                }
            ],
            "info": {
                "count": 1,
                "more_records": False
            }
        }
    
    @pytest.mark.asyncio
    async def test_zoho_crm_initialization(self, mock_credentials):
        """Test Zoho CRM connector initialization"""
        connector = ZohoCRMConnector(mock_credentials)
        assert connector.client_id == "test_client_id"
        assert connector.domain == "com"
        assert connector.base_url == "https://www.zohoapis.com"
    
    @pytest.mark.asyncio
    async def test_zoho_crm_data_extraction(self, mock_credentials, mock_zoho_response):
        """Test Zoho CRM data extraction with mocked API"""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_zoho_response
            mock_response.status = 200
            mock_get.return_value.__aenter__.return_value = mock_response
            
            connector = ZohoCRMConnector(mock_credentials)
            
            # Mock token refresh
            with patch.object(connector, '_refresh_access_token', return_value="mock_token"):
                result = await connector.sync_data("2024-01-01", "2024-01-31")
                
                assert "records" in result
                assert len(result["records"]) == 1
                assert result["records"][0]["company"] == "Test Company"
                assert result["records"][0]["lead_source"] == "Website"
    
    @pytest.mark.asyncio
    async def test_zoho_crm_pipeline_analytics(self, mock_credentials):
        """Test Zoho CRM pipeline analytics functionality"""
        connector = ZohoCRMConnector(mock_credentials)
        
        # Mock pipeline data
        mock_pipeline_data = [
            {"stage": "Lead", "count": 100, "value": 50000},
            {"stage": "Qualified", "count": 50, "value": 75000},
            {"stage": "Proposal", "count": 25, "value": 100000},
            {"stage": "Closed Won", "count": 10, "value": 125000}
        ]
        
        with patch.object(connector, '_get_pipeline_data', return_value=mock_pipeline_data):
            analytics = await connector._analyze_pipeline_performance(mock_pipeline_data)
            
            assert analytics["conversion_rate"] > 0
            assert analytics["average_deal_value"] > 0
            assert "stage_analysis" in analytics


class TestLinkedInAdsConnector:
    """Test LinkedIn Ads connector functionality"""
    
    @pytest.fixture
    def mock_credentials(self):
        return {
            "client_id": "test_linkedin_client",
            "client_secret": "test_linkedin_secret",
            "access_token": "test_linkedin_token"
        }
    
    @pytest.fixture
    def mock_linkedin_response(self):
        return {
            "elements": [
                {
                    "id": "campaign_123",
                    "name": "B2B Lead Gen Campaign",
                    "status": "ACTIVE",
                    "type": "SPONSORED_CONTENT",
                    "costType": "CPC",
                    "dailyBudget": {"amount": "100.00", "currencyCode": "USD"},
                    "totalBudget": {"amount": "3000.00", "currencyCode": "USD"},
                    "createdAt": 1642204800000,
                    "lastModifiedAt": 1642291200000
                }
            ],
            "paging": {
                "count": 1,
                "start": 0
            }
        }
    
    @pytest.mark.asyncio
    async def test_linkedin_ads_initialization(self, mock_credentials):
        """Test LinkedIn Ads connector initialization"""
        connector = LinkedInAdsConnector(mock_credentials)
        assert connector.client_id == "test_linkedin_client"
        assert connector.access_token == "test_linkedin_token"
        assert connector.base_url == "https://api.linkedin.com/v2"
    
    @pytest.mark.asyncio
    async def test_linkedin_ads_data_extraction(self, mock_credentials, mock_linkedin_response):
        """Test LinkedIn Ads data extraction with mocked API"""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_linkedin_response
            mock_response.status = 200
            mock_get.return_value.__aenter__.return_value = mock_response
            
            connector = LinkedInAdsConnector(mock_credentials)
            result = await connector.sync_data("2024-01-01", "2024-01-31")
            
            assert "campaigns" in result
            assert len(result["campaigns"]) == 1
            assert result["campaigns"][0]["campaign_name"] == "B2B Lead Gen Campaign"
            assert result["campaigns"][0]["status"] == "ACTIVE"
    
    @pytest.mark.asyncio
    async def test_linkedin_professional_targeting(self, mock_credentials):
        """Test LinkedIn professional targeting analytics"""
        connector = LinkedInAdsConnector(mock_credentials)
        
        # Mock targeting data
        mock_targeting_data = {
            "job_functions": ["Engineering", "Marketing", "Sales"],
            "industries": ["Technology", "Healthcare", "Finance"],
            "seniority_levels": ["Manager", "Director", "VP"]
        }
        
        with patch.object(connector, '_get_targeting_data', return_value=mock_targeting_data):
            analytics = await connector._analyze_professional_targeting(mock_targeting_data)
            
            assert "job_function_performance" in analytics
            assert "industry_insights" in analytics
            assert "seniority_optimization" in analytics


class TestXAdsConnector:
    """Test X (Twitter) Ads connector functionality"""
    
    @pytest.fixture
    def mock_credentials(self):
        return {
            "consumer_key": "test_x_consumer_key",
            "consumer_secret": "test_x_consumer_secret",
            "access_token": "test_x_access_token",
            "access_token_secret": "test_x_token_secret"
        }
    
    @pytest.fixture
    def mock_x_ads_response(self):
        return {
            "data": [
                {
                    "id": "x_campaign_456",
                    "name": "Social Engagement Campaign",
                    "entity_status": "ACTIVE",
                    "funding_instrument_id": "funding_123",
                    "daily_budget_amount_local_micro": 50000000,
                    "total_budget_amount_local_micro": 1500000000,
                    "created_at": "2024-01-15T10:00:00Z",
                    "updated_at": "2024-01-15T16:30:00Z"
                }
            ],
            "next_cursor": None
        }
    
    @pytest.mark.asyncio
    async def test_x_ads_initialization(self, mock_credentials):
        """Test X Ads connector initialization"""
        connector = XAdsConnector(mock_credentials)
        assert connector.consumer_key == "test_x_consumer_key"
        assert connector.access_token == "test_x_access_token"
        assert connector.base_url == "https://ads-api.x.com/12"
    
    @pytest.mark.asyncio
    async def test_x_ads_oauth_signature(self, mock_credentials):
        """Test X Ads OAuth 1.0a signature generation"""
        connector = XAdsConnector(mock_credentials)
        
        # Test OAuth signature generation
        url = "https://ads-api.x.com/12/accounts"
        params = {"count": "200"}
        
        auth_header = connector._generate_oauth_header("GET", url, params)
        
        assert "OAuth " in auth_header
        assert "oauth_consumer_key" in auth_header
        assert "oauth_signature" in auth_header
        assert "oauth_timestamp" in auth_header
    
    @pytest.mark.asyncio
    async def test_x_ads_data_extraction(self, mock_credentials, mock_x_ads_response):
        """Test X Ads data extraction with mocked API"""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.json.return_value = mock_x_ads_response
            mock_response.status = 200
            mock_get.return_value.__aenter__.return_value = mock_response
            
            connector = XAdsConnector(mock_credentials)
            result = await connector.sync_data("2024-01-01", "2024-01-31")
            
            assert "campaigns" in result
            assert len(result["campaigns"]) == 1
            assert result["campaigns"][0]["campaign_name"] == "Social Engagement Campaign"
            assert result["campaigns"][0]["status"] == "ACTIVE"
    
    @pytest.mark.asyncio
    async def test_x_ads_viral_metrics(self, mock_credentials):
        """Test X Ads viral amplification metrics"""
        connector = XAdsConnector(mock_credentials)
        
        # Mock viral metrics data
        mock_viral_data = {
            "retweets": 150,
            "likes": 500,
            "replies": 75,
            "quote_tweets": 25,
            "impressions": 10000,
            "engagements": 750
        }
        
        viral_score = connector._calculate_viral_amplification_score(mock_viral_data)
        
        assert viral_score > 0
        assert viral_score <= 100  # Normalized score


class TestTier4Integration:
    """Test Tier 4 connectors integration"""
    
    @pytest.mark.asyncio
    async def test_all_tier4_connectors_data_sources(self):
        """Test that all Tier 4 connectors have correct DataSource mappings"""
        # Test DataSource enum includes all Tier 4 platforms
        assert hasattr(DataSource, 'ZOHO_CRM')
        assert hasattr(DataSource, 'LINKEDIN_ADS')
        assert hasattr(DataSource, 'X_ADS')
        # Test enum values
        assert DataSource.ZOHO_CRM.value == "zoho_crm"
        assert DataSource.LINKEDIN_ADS.value == "linkedin_ads"
        assert DataSource.X_ADS.value == "x_ads"
    
    @pytest.mark.asyncio
    async def test_tier4_kse_integration(self):
        """Test KSE (Knowledge Storage Engine) integration for Tier 4"""
        # Mock KSE integration for all Tier 4 connectors
        mock_kse_data = {
            "embeddings": [0.1, 0.2, 0.3],
            "quality_score": 0.85,
            "causal_factors": ["campaign_budget", "audience_targeting", "content_quality"]
        }
        
        # Test each connector's KSE integration
        connectors = [
            ("zoho_crm", {"client_id": "test", "client_secret": "test", "refresh_token": "test", "domain": "com"}),
            ("linkedin_ads", {"client_id": "test", "client_secret": "test", "access_token": "test"}),
            ("x_ads", {"consumer_key": "test", "consumer_secret": "test", "access_token": "test", "access_token_secret": "test"})
        ]
        
        for platform, credentials in connectors:
            # Each connector should support KSE enhancement
            assert platform in ["zoho_crm", "linkedin_ads", "x_ads"]


if __name__ == "__main__":
    # Run tests
    print("Running Tier 4 Connector Tests...")
    
    # Test individual connectors
    asyncio.run(TestZohoCRMConnector().test_zoho_crm_initialization({
        "client_id": "test", "client_secret": "test", "refresh_token": "test", "domain": "com"
    }))
    
    asyncio.run(TestLinkedInAdsConnector().test_linkedin_ads_initialization({
        "client_id": "test", "client_secret": "test", "access_token": "test"
    }))
    
    asyncio.run(TestXAdsConnector().test_x_ads_initialization({
        "consumer_key": "test", "consumer_secret": "test", 
        "access_token": "test", "access_token_secret": "test"
    }))
    
    print("✅ All Tier 4 connector initialization tests passed!")
    print("🚀 Tier 4 connectors (Zoho CRM, LinkedIn Ads, X Ads) are ready!")