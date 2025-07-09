"""
Meta Business API Connector for LiftOS Data Ingestion Service
Handles data extraction from Meta Business (Facebook) advertising platform
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

from shared.models.marketing import MetaBusinessData
from shared.models.causal_marketing import CausalMarketingData, DataQualityAssessment
from shared.utils.causal_transforms import CausalDataTransformer
from shared.mmm_spine_integration import EnhancedKSEIntegration

logger = logging.getLogger(__name__)

class MetaBusinessConnector:
    """Meta Business API connector with KSE integration and causal data extraction"""
    
    def __init__(self, access_token: str, app_id: str, app_secret: str):
        self.access_token = access_token
        self.app_id = app_id
        self.app_secret = app_secret
        self.base_url = "https://graph.facebook.com/v18.0"
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Initialize causal transformer and KSE integration
        self.causal_transformer = CausalDataTransformer()
        self.kse_integration = EnhancedKSEIntegration()
    
    async def get_ad_accounts(self) -> List[Dict[str, Any]]:
        """Get available ad accounts"""
        try:
            url = f"{self.base_url}/me/adaccounts"
            params = {
                "access_token": self.access_token,
                "fields": "id,name,account_status,currency,timezone_name"
            }
            
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            data = await response.json()
            return data.get("data", [])
            
        except Exception as e:
            logger.error(f"Failed to get Meta ad accounts: {str(e)}")
            raise
    
    async def get_campaigns(self, account_id: str, date_start: date, date_end: date) -> List[Dict[str, Any]]:
        """Get campaigns data from Meta Business API"""
        try:
            url = f"{self.base_url}/{account_id}/campaigns"
            params = {
                "access_token": self.access_token,
                "fields": "id,name,status,objective,daily_budget,lifetime_budget,created_time,updated_time",
                "limit": 100
            }
            
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            data = await response.json()
            campaigns = data.get("data", [])
            
            # Get insights for each campaign
            campaign_insights = []
            for campaign in campaigns:
                insights = await self.get_campaign_insights(
                    campaign["id"], date_start, date_end
                )
                campaign_data = {**campaign, "insights": insights}
                campaign_insights.append(campaign_data)
            
            return campaign_insights
            
        except Exception as e:
            logger.error(f"Failed to get Meta campaigns: {str(e)}")
            raise
    
    async def get_campaign_insights(self, campaign_id: str, date_start: date, date_end: date) -> Dict[str, Any]:
        """Get campaign insights from Meta Business API"""
        try:
            url = f"{self.base_url}/{campaign_id}/insights"
            params = {
                "access_token": self.access_token,
                "fields": "spend,impressions,clicks,actions,cpm,cpc,ctr,frequency,reach,video_views",
                "time_range": json.dumps({
                    "since": date_start.strftime("%Y-%m-%d"),
                    "until": date_end.strftime("%Y-%m-%d")
                }),
                "level": "campaign"
            }
            
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            data = await response.json()
            insights = data.get("data", [])
            
            return insights[0] if insights else {}
            
        except Exception as e:
            logger.error(f"Failed to get Meta campaign insights: {str(e)}")
            return {}
    
    async def extract_data(self, date_start: date, date_end: date, 
                          account_ids: Optional[List[str]] = None) -> List[MetaBusinessData]:
        """Extract comprehensive data from Meta Business API with KSE integration"""
        try:
            logger.info(f"Starting Meta Business data extraction for {date_start} to {date_end}")
            
            # Get ad accounts if not specified
            if not account_ids:
                accounts = await self.get_ad_accounts()
                account_ids = [acc["id"] for acc in accounts]
            
            all_data = []
            
            for account_id in account_ids:
                # Extract campaigns data
                campaigns = await self.get_campaigns(account_id, date_start, date_end)
                
                for campaign in campaigns:
                    # Transform to MetaBusinessData model
                    meta_data = MetaBusinessData(
                        campaign_id=campaign["id"],
                        campaign_name=campaign["name"],
                        account_id=account_id,
                        date_start=date_start,
                        date_end=date_end,
                        spend=float(campaign.get("insights", {}).get("spend", 0)),
                        impressions=int(campaign.get("insights", {}).get("impressions", 0)),
                        clicks=int(campaign.get("insights", {}).get("clicks", 0)),
                        reach=int(campaign.get("insights", {}).get("reach", 0)),
                        frequency=float(campaign.get("insights", {}).get("frequency", 0)),
                        cpm=float(campaign.get("insights", {}).get("cpm", 0)),
                        cpc=float(campaign.get("insights", {}).get("cpc", 0)),
                        ctr=float(campaign.get("insights", {}).get("ctr", 0)),
                        video_views=int(campaign.get("insights", {}).get("video_views", 0)),
                        actions=campaign.get("insights", {}).get("actions", []),
                        objective=campaign.get("objective", ""),
                        status=campaign.get("status", ""),
                        created_time=campaign.get("created_time", ""),
                        updated_time=campaign.get("updated_time", "")
                    )
                    
                    all_data.append(meta_data)
            
            # Apply KSE integration
            enhanced_data = await self.kse_integration.enhance_marketing_data(all_data)
            
            logger.info(f"Successfully extracted {len(enhanced_data)} Meta Business records")
            return enhanced_data
            
        except Exception as e:
            logger.error(f"Failed to extract Meta Business data: {str(e)}")
            raise
    
    async def extract_causal_data(self, date_start: date, date_end: date,
                                 account_ids: Optional[List[str]] = None) -> CausalMarketingData:
        """Extract causal marketing data with treatment assignment and confounder detection"""
        try:
            logger.info(f"Starting Meta Business causal data extraction for {date_start} to {date_end}")
            
            # Extract base data
            base_data = await self.extract_data(date_start, date_end, account_ids)
            
            # Transform to causal data format
            causal_data = await self.causal_transformer.transform_meta_business_data(
                base_data, date_start, date_end
            )
            
            # Assess data quality
            quality_assessment = await self.causal_transformer.assess_data_quality(causal_data)
            
            logger.info(f"Successfully extracted causal data with quality score: {quality_assessment.overall_score}")
            return causal_data
            
        except Exception as e:
            logger.error(f"Failed to extract Meta Business causal data: {str(e)}")
            raise
    
    async def get_attribution_data(self, date_start: date, date_end: date,
                                  attribution_window: int = 7) -> List[Dict[str, Any]]:
        """Get attribution data for causal analysis"""
        try:
            logger.info(f"Extracting Meta Business attribution data with {attribution_window}-day window")
            
            # Get ad accounts
            accounts = await self.get_ad_accounts()
            attribution_data = []
            
            for account in accounts:
                account_id = account["id"]
                
                # Get attribution insights
                url = f"{self.base_url}/{account_id}/insights"
                params = {
                    "access_token": self.access_token,
                    "fields": "spend,impressions,clicks,actions,attribution_setting",
                    "time_range": json.dumps({
                        "since": date_start.strftime("%Y-%m-%d"),
                        "until": date_end.strftime("%Y-%m-%d")
                    }),
                    "level": "account",
                    "action_attribution_windows": f"['{attribution_window}d_click','{attribution_window}d_view']"
                }
                
                response = await self.client.get(url, params=params)
                response.raise_for_status()
                
                data = await response.json()
                insights = data.get("data", [])
                
                for insight in insights:
                    attribution_record = {
                        "account_id": account_id,
                        "date": insight.get("date_start"),
                        "attribution_window": attribution_window,
                        "spend": float(insight.get("spend", 0)),
                        "impressions": int(insight.get("impressions", 0)),
                        "clicks": int(insight.get("clicks", 0)),
                        "actions": insight.get("actions", []),
                        "attribution_setting": insight.get("attribution_setting", {})
                    }
                    attribution_data.append(attribution_record)
            
            logger.info(f"Successfully extracted {len(attribution_data)} attribution records")
            return attribution_data
            
        except Exception as e:
            logger.error(f"Failed to extract Meta Business attribution data: {str(e)}")
            return []
    
    async def validate_connection(self) -> bool:
        """Validate Meta Business API connection"""
        try:
            url = f"{self.base_url}/me"
            params = {
                "access_token": self.access_token,
                "fields": "id,name"
            }
            
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            data = await response.json()
            logger.info(f"Meta Business connection validated for user: {data.get('name', 'Unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Meta Business connection validation failed: {str(e)}")
            return False
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

# Factory function for creating connector instances
async def create_meta_business_connector(credentials: Dict[str, str]) -> MetaBusinessConnector:
    """Create and validate Meta Business connector instance"""
    connector = MetaBusinessConnector(
        access_token=credentials["access_token"],
        app_id=credentials["app_id"],
        app_secret=credentials["app_secret"]
    )
    
    # Validate connection
    is_valid = await connector.validate_connection()
    if not is_valid:
        await connector.close()
        raise ValueError("Invalid Meta Business credentials")
    
    return connector