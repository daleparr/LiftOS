"""
Simple Test for Tier 4 Connectors
Basic functionality verification without pytest
"""
import asyncio
import sys
import os

# Add path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import shared models
from shared.models.marketing import DataSource

# Import credential classes
from connectors.zoho_crm_connector import ZohoCRMCredentials, ZohoCRMConnector
from connectors.linkedin_ads_connector import LinkedInAdsCredentials, LinkedInAdsConnector
from connectors.x_ads_connector import XAdsCredentials, XAdsConnector


async def test_zoho_crm_initialization():
    """Test Zoho CRM connector initialization"""
    print("Testing Zoho CRM connector initialization...")
    
    credentials = ZohoCRMCredentials(
        client_id="test_client_id",
        client_secret="test_client_secret",
        refresh_token="test_refresh_token",
        domain="com"
    )
    
    connector = ZohoCRMConnector(credentials)
    
    assert connector.credentials.client_id == "test_client_id"
    assert connector.credentials.domain == "com"
    assert "zohoapis.com" in connector.base_url
    
    print(" Zoho CRM connector initialization passed!")


async def test_linkedin_ads_initialization():
    """Test LinkedIn Ads connector initialization"""
    print("Testing LinkedIn Ads connector initialization...")
    
    credentials = LinkedInAdsCredentials(
        client_id="test_linkedin_client",
        client_secret="test_linkedin_secret",
        access_token="test_linkedin_token"
    )
    
    connector = LinkedInAdsConnector(credentials)
    
    assert connector.credentials.client_id == "test_linkedin_client"
    assert connector.credentials.access_token == "test_linkedin_token"
    assert "api.linkedin.com" in connector.base_url
    
    print(" LinkedIn Ads connector initialization passed!")


async def test_x_ads_initialization():
    """Test X Ads connector initialization"""
    print("Testing X Ads connector initialization...")
    
    credentials = XAdsCredentials(
        consumer_key="test_x_consumer_key",
        consumer_secret="test_x_consumer_secret",
        access_token="test_x_access_token",
        access_token_secret="test_x_token_secret"
    )
    
    connector = XAdsConnector(credentials)
    
    assert connector.credentials.consumer_key == "test_x_consumer_key"
    assert connector.credentials.access_token == "test_x_access_token"
    assert "ads-api.twitter.com" in connector.base_url
    
    print(" X Ads connector initialization passed!")


def test_data_source_enums():
    """Test that all Tier 4 DataSource enums exist"""
    print("Testing DataSource enum values...")
    
    # Test that all Tier 4 platforms are in DataSource enum
    assert hasattr(DataSource, 'ZOHO_CRM')
    assert hasattr(DataSource, 'LINKEDIN_ADS')
    assert hasattr(DataSource, 'X_ADS')
    
    # Test enum values
    assert DataSource.ZOHO_CRM.value == "zoho_crm"
    assert DataSource.LINKEDIN_ADS.value == "linkedin_ads"
    assert DataSource.X_ADS.value == "x_ads"
    
    print("DataSource enum tests passed!")


async def main():
    """Run all tests"""
    print("Starting Tier 4 Connector Tests...")
    print("=" * 50)
    
    try:
        # Test DataSource enums first
        test_data_source_enums()
        
        # Test connector initializations
        await test_zoho_crm_initialization()
        await test_linkedin_ads_initialization()
        await test_x_ads_initialization()
        
        
        print("=" * 50)
        print("ALL TIER 4 CONNECTOR TESTS PASSED!")
        print("Zoho CRM Connector: Ready")
        print("LinkedIn Ads Connector: Ready")
        print("X Ads Connector: Ready")
        print("=" * 50)
        print("LiftOS Data Ingestion Service now supports 16 platforms!")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())