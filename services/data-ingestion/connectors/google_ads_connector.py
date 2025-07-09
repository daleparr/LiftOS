"""
Google Ads API Connector for LiftOS Data Ingestion Service
Handles data extraction from Google Ads advertising platform
"""
import asyncio
import time
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any, Union
import httpx
import json
import logging

# Import shared modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from shared.models.marketing import GoogleAdsData
from shared.models.causal_marketing import CausalMarketingData, DataQualityAssessment
from shared.utils.causal_transforms import CausalDataTransformer
from shared.mmm_spine_integration import EnhancedKSEIntegration

logger = logging.getLogger(__name__)

class GoogleAdsConnector:
    """Google Ads API connector with KSE integration and causal data extraction"""
    
    def __init__(self, developer_token: str, client_id: str, client_secret: str, refresh_token: str):
        self.developer_token = developer_token
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self.access_token = None
        self.base_url = "https://googleads.googleapis.com/v14"
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Initialize causal transformer and KSE integration
        self.causal_transformer = CausalDataTransformer()
        self.kse_integration = EnhancedKSEIntegration()
    
    async def authenticate(self):
        """Get access token using refresh token"""
        try:
            url = "https://oauth2.googleapis.com/token"
            data = {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "refresh_token": self.refresh_token,
                "grant_type": "refresh_token"
            }
            
            response = await self.client.post(url, data=data)
            response.raise_for_status()
            
            token_data = await response.json()
            self.access_token = token_data["access_token"]
            
        except Exception as e:
            logger.error(f"Failed to authenticate with Google Ads: {str(e)}")
            raise
    
    async def get_customers(self) -> List[Dict[str, Any]]:
        """Get available customer accounts"""
        try:
            if not self.access_token:
                await self.authenticate()
            
            url = f"{self.base_url}/customers:listAccessibleCustomers"
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "developer-token": self.developer_token
            }
            
            response = await self.client.get(url, headers=headers)
            response.raise_for_status()
            
            data = await response.json()
            return data.get("resourceNames", [])
            
        except Exception as e:
            logger.error(f"Failed to get Google Ads customers: {str(e)}")
            raise
    
    async def get_campaigns(self, customer_id: str, date_start: date, date_end: date) -> List[Dict[str, Any]]:
        """Get campaigns data from Google Ads API"""
        try:
            if not self.access_token:
                await self.authenticate()
            
            # Google Ads Reporting API query
            query = f"""
                SELECT 
                    campaign.id,
                    campaign.name,
                    campaign.status,
                    campaign.advertising_channel_type,
                    metrics.cost_micros,
                    metrics.impressions,
                    metrics.clicks,
                    metrics.conversions,
                    metrics.conversions_value,
                    segments.date
                FROM campaign 
                WHERE segments.date BETWEEN '{date_start.strftime('%Y-%m-%d')}' AND '{date_end.strftime('%Y-%m-%d')}'
            """
            
            url = f"{self.base_url}/customers/{customer_id}/googleAds:searchStream"
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "developer-token": self.developer_token,
                "Content-Type": "application/json"
            }
            
            payload = {"query": query}
            
            response = await self.client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            # Process streaming response
            campaigns_data = []
            for line in response.text.strip().split('\n'):
                if line:
                    result = json.loads(line)
                    if "results" in result:
                        campaigns_data.extend(result["results"])
            
            return campaigns_data
            
        except Exception as e:
            logger.error(f"Failed to get Google Ads campaigns: {str(e)}")
            raise
    
    async def extract_data(self, date_start: date, date_end: date, 
                          customer_ids: Optional[List[str]] = None) -> List[GoogleAdsData]:
        """Extract comprehensive data from Google Ads API with KSE integration"""
        try:
            logger.info(f"Starting Google Ads data extraction for {date_start} to {date_end}")
            
            # Get customers if not specified
            if not customer_ids:
                customers = await self.get_customers()
                customer_ids = [customer.split('/')[-1] for customer in customers]
            
            all_data = []
            
            for customer_id in customer_ids:
                # Extract campaigns data
                campaigns = await self.get_campaigns(customer_id, date_start, date_end)
                
                for campaign_result in campaigns:
                    campaign = campaign_result.get("campaign", {})
                    metrics = campaign_result.get("metrics", {})
                    segments = campaign_result.get("segments", {})
                    
                    # Transform to GoogleAdsData model
                    google_ads_data = GoogleAdsData(
                        campaign_id=str(campaign.get("id", "")),
                        campaign_name=campaign.get("name", ""),
                        customer_id=customer_id,
                        date_start=date_start,
                        date_end=date_end,
                        cost_micros=int(metrics.get("costMicros", 0)),
                        impressions=int(metrics.get("impressions", 0)),
                        clicks=int(metrics.get("clicks", 0)),
                        conversions=float(metrics.get("conversions", 0)),
                        conversions_value=float(metrics.get("conversionsValue", 0)),
                        advertising_channel_type=campaign.get("advertisingChannelType", ""),
                        status=campaign.get("status", ""),
                        date=segments.get("date", date_start.strftime('%Y-%m-%d'))
                    )
                    
                    all_data.append(google_ads_data)
            
            # Apply KSE integration
            enhanced_data = await self.kse_integration.enhance_marketing_data(all_data)
            
            logger.info(f"Successfully extracted {len(enhanced_data)} Google Ads records")
            return enhanced_data
            
        except Exception as e:
            logger.error(f"Failed to extract Google Ads data: {str(e)}")
            raise
    
    async def extract_causal_data(self, date_start: date, date_end: date,
                                 customer_ids: Optional[List[str]] = None) -> CausalMarketingData:
        """Extract causal marketing data with treatment assignment and confounder detection"""
        try:
            logger.info(f"Starting Google Ads causal data extraction for {date_start} to {date_end}")
            
            # Extract base data
            base_data = await self.extract_data(date_start, date_end, customer_ids)
            
            # Transform to causal data format
            causal_data = await self.causal_transformer.transform_google_ads_data(
                base_data, date_start, date_end
            )
            
            # Assess data quality
            quality_assessment = await self.causal_transformer.assess_data_quality(causal_data)
            
            logger.info(f"Successfully extracted causal data with quality score: {quality_assessment.overall_score}")
            return causal_data
            
        except Exception as e:
            logger.error(f"Failed to extract Google Ads causal data: {str(e)}")
            raise
    
    async def get_attribution_data(self, date_start: date, date_end: date,
                                  attribution_window: int = 7) -> List[Dict[str, Any]]:
        """Get attribution data for causal analysis"""
        try:
            logger.info(f"Extracting Google Ads attribution data with {attribution_window}-day window")
            
            # Get customers
            customers = await self.get_customers()
            attribution_data = []
            
            for customer_resource in customers:
                customer_id = customer_resource.split('/')[-1]
                
                # Enhanced query for attribution data
                query = f"""
                    SELECT 
                        campaign.id,
                        campaign.name,
                        metrics.cost_micros,
                        metrics.impressions,
                        metrics.clicks,
                        metrics.conversions,
                        metrics.conversions_value,
                        metrics.view_through_conversions,
                        segments.date,
                        segments.conversion_action_name,
                        segments.conversion_attribution_event_type
                    FROM campaign 
                    WHERE segments.date BETWEEN '{date_start.strftime('%Y-%m-%d')}' AND '{date_end.strftime('%Y-%m-%d')}'
                """
                
                url = f"{self.base_url}/customers/{customer_id}/googleAds:searchStream"
                headers = {
                    "Authorization": f"Bearer {self.access_token}",
                    "developer-token": self.developer_token,
                    "Content-Type": "application/json"
                }
                
                payload = {"query": query}
                
                response = await self.client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                
                # Process streaming response
                for line in response.text.strip().split('\n'):
                    if line:
                        result = json.loads(line)
                        if "results" in result:
                            for campaign_result in result["results"]:
                                campaign = campaign_result.get("campaign", {})
                                metrics = campaign_result.get("metrics", {})
                                segments = campaign_result.get("segments", {})
                                
                                attribution_record = {
                                    "customer_id": customer_id,
                                    "campaign_id": str(campaign.get("id", "")),
                                    "campaign_name": campaign.get("name", ""),
                                    "date": segments.get("date"),
                                    "attribution_window": attribution_window,
                                    "cost_micros": int(metrics.get("costMicros", 0)),
                                    "impressions": int(metrics.get("impressions", 0)),
                                    "clicks": int(metrics.get("clicks", 0)),
                                    "conversions": float(metrics.get("conversions", 0)),
                                    "conversions_value": float(metrics.get("conversionsValue", 0)),
                                    "view_through_conversions": float(metrics.get("viewThroughConversions", 0)),
                                    "conversion_action_name": segments.get("conversionActionName", ""),
                                    "attribution_event_type": segments.get("conversionAttributionEventType", "")
                                }
                                attribution_data.append(attribution_record)
            
            logger.info(f"Successfully extracted {len(attribution_data)} attribution records")
            return attribution_data
            
        except Exception as e:
            logger.error(f"Failed to extract Google Ads attribution data: {str(e)}")
            return []
    
    async def validate_connection(self) -> bool:
        """Validate Google Ads API connection"""
        try:
            await self.authenticate()
            
            # Test with a simple customer list request
            customers = await self.get_customers()
            
            logger.info(f"Google Ads connection validated with {len(customers)} accessible customers")
            return True
            
        except Exception as e:
            logger.error(f"Google Ads connection validation failed: {str(e)}")
            return False
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

# Factory function for creating connector instances
async def create_google_ads_connector(credentials: Dict[str, str]) -> GoogleAdsConnector:
    """Create and validate Google Ads connector instance"""
    connector = GoogleAdsConnector(
        developer_token=credentials["developer_token"],
        client_id=credentials["client_id"],
        client_secret=credentials["client_secret"],
        refresh_token=credentials["refresh_token"]
    )
    
    # Validate connection
    is_valid = await connector.validate_connection()
    if not is_valid:
        await connector.close()
        raise ValueError("Invalid Google Ads credentials")
    
    return connector