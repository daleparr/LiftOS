"""
Klaviyo API Connector for LiftOS Data Ingestion Service
Handles data extraction from Klaviyo email marketing platform
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

from shared.models.marketing import KlaviyoData
from shared.models.causal_marketing import CausalMarketingData, DataQualityAssessment
from shared.utils.causal_transforms import CausalDataTransformer
from shared.mmm_spine_integration import EnhancedKSEIntegration

logger = logging.getLogger(__name__)

class KlaviyoConnector:
    """Klaviyo API connector with KSE integration and causal data extraction"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://a.klaviyo.com/api"
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Initialize causal transformer and KSE integration
        self.causal_transformer = CausalDataTransformer()
        self.kse_integration = EnhancedKSEIntegration()
    
    async def get_campaigns(self, date_start: date, date_end: date) -> List[Dict[str, Any]]:
        """Get campaigns data from Klaviyo API"""
        try:
            url = f"{self.base_url}/campaigns"
            headers = {
                "Authorization": f"Klaviyo-API-Key {self.api_key}",
                "revision": "2024-02-15"
            }
            
            params = {
                "filter": f"greater-than(send_time,{date_start.isoformat()}),less-than(send_time,{date_end.isoformat()})",
                "include": "campaign-messages",
                "page[size]": 100
            }
            
            response = await self.client.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = await response.json()
            campaigns = data.get("data", [])
            
            # Get metrics for each campaign
            campaign_metrics = []
            for campaign in campaigns:
                metrics = await self.get_campaign_metrics(campaign["id"])
                campaign_data = {**campaign, "metrics": metrics}
                campaign_metrics.append(campaign_data)
            
            return campaign_metrics
            
        except Exception as e:
            logger.error(f"Failed to get Klaviyo campaigns: {str(e)}")
            raise
    
    async def get_campaign_metrics(self, campaign_id: str) -> Dict[str, Any]:
        """Get comprehensive campaign metrics from Klaviyo API"""
        try:
            # Get campaign performance metrics
            url = f"{self.base_url}/campaign-recipient-estimations"
            headers = {
                "Authorization": f"Klaviyo-API-Key {self.api_key}",
                "revision": "2024-02-15"
            }
            
            params = {
                "filter": f"equals(campaign_id,{campaign_id})"
            }
            
            response = await self.client.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = await response.json()
            metrics = data.get("data", {})
            
            # Get additional metrics like opens, clicks, etc.
            additional_metrics = await self.get_campaign_performance(campaign_id)
            if additional_metrics:
                metrics.update(additional_metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get Klaviyo campaign metrics: {str(e)}")
            return {}
    
    async def get_campaign_performance(self, campaign_id: str) -> Dict[str, Any]:
        """Get detailed campaign performance metrics"""
        try:
            # This would use Klaviyo's reporting endpoints for detailed metrics
            url = f"{self.base_url}/campaigns/{campaign_id}/campaign-messages"
            headers = {
                "Authorization": f"Klaviyo-API-Key {self.api_key}",
                "revision": "2024-02-15"
            }
            
            response = await self.client.get(url, headers=headers)
            response.raise_for_status()
            
            data = await response.json()
            messages = data.get("data", [])
            
            # Aggregate performance metrics
            total_sent = 0
            total_delivered = 0
            total_opens = 0
            total_clicks = 0
            total_unsubscribes = 0
            total_bounces = 0
            
            for message in messages:
                attrs = message.get("attributes", {})
                total_sent += attrs.get("sent_at_count", 0)
                total_delivered += attrs.get("delivered_count", 0)
                total_opens += attrs.get("open_count", 0)
                total_clicks += attrs.get("click_count", 0)
                total_unsubscribes += attrs.get("unsubscribe_count", 0)
                total_bounces += attrs.get("bounce_count", 0)
            
            return {
                "sent_count": total_sent,
                "delivered_count": total_delivered,
                "open_count": total_opens,
                "click_count": total_clicks,
                "unsubscribe_count": total_unsubscribes,
                "bounce_count": total_bounces,
                "open_rate": (total_opens / total_delivered) if total_delivered > 0 else 0,
                "click_rate": (total_clicks / total_delivered) if total_delivered > 0 else 0,
                "unsubscribe_rate": (total_unsubscribes / total_delivered) if total_delivered > 0 else 0,
                "bounce_rate": (total_bounces / total_sent) if total_sent > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get Klaviyo campaign performance: {str(e)}")
            return {}
    
    async def extract_data(self, date_start: date, date_end: date) -> List[KlaviyoData]:
        """Extract comprehensive data from Klaviyo API with KSE integration"""
        try:
            logger.info(f"Starting Klaviyo data extraction for {date_start} to {date_end}")
            
            # Extract campaigns data
            campaigns = await self.get_campaigns(date_start, date_end)
            
            all_data = []
            
            for campaign in campaigns:
                attrs = campaign.get("attributes", {})
                metrics = campaign.get("metrics", {})
                
                # Transform to KlaviyoData model
                klaviyo_data = KlaviyoData(
                    campaign_id=campaign.get("id", ""),
                    campaign_name=attrs.get("name", ""),
                    date_start=date_start,
                    date_end=date_end,
                    sent_count=metrics.get("sent_count", 0),
                    delivered_count=metrics.get("delivered_count", 0),
                    open_count=metrics.get("open_count", 0),
                    click_count=metrics.get("click_count", 0),
                    unsubscribe_count=metrics.get("unsubscribe_count", 0),
                    bounce_count=metrics.get("bounce_count", 0),
                    open_rate=metrics.get("open_rate", 0.0),
                    click_rate=metrics.get("click_rate", 0.0),
                    unsubscribe_rate=metrics.get("unsubscribe_rate", 0.0),
                    bounce_rate=metrics.get("bounce_rate", 0.0),
                    subject_line=attrs.get("subject_line", ""),
                    send_time=attrs.get("send_time", ""),
                    campaign_type=attrs.get("campaign_type", ""),
                    status=attrs.get("status", "")
                )
                
                all_data.append(klaviyo_data)
            
            # Apply KSE integration
            enhanced_data = await self.kse_integration.enhance_marketing_data(all_data)
            
            logger.info(f"Successfully extracted {len(enhanced_data)} Klaviyo records")
            return enhanced_data
            
        except Exception as e:
            logger.error(f"Failed to extract Klaviyo data: {str(e)}")
            raise
    
    async def extract_causal_data(self, date_start: date, date_end: date) -> CausalMarketingData:
        """Extract causal marketing data with treatment assignment and confounder detection"""
        try:
            logger.info(f"Starting Klaviyo causal data extraction for {date_start} to {date_end}")
            
            # Extract base data
            base_data = await self.extract_data(date_start, date_end)
            
            # Transform to causal data format
            causal_data = await self.causal_transformer.transform_klaviyo_data(
                base_data, date_start, date_end
            )
            
            # Assess data quality
            quality_assessment = await self.causal_transformer.assess_data_quality(causal_data)
            
            logger.info(f"Successfully extracted causal data with quality score: {quality_assessment.overall_score}")
            return causal_data
            
        except Exception as e:
            logger.error(f"Failed to extract Klaviyo causal data: {str(e)}")
            raise
    
    async def get_attribution_data(self, date_start: date, date_end: date,
                                  attribution_window: int = 7) -> List[Dict[str, Any]]:
        """Get attribution data for causal analysis"""
        try:
            logger.info(f"Extracting Klaviyo attribution data with {attribution_window}-day window")
            
            # Get campaigns
            campaigns = await self.get_campaigns(date_start, date_end)
            attribution_data = []
            
            for campaign in campaigns:
                attrs = campaign.get("attributes", {})
                metrics = campaign.get("metrics", {})
                
                # Get flow data for attribution analysis
                flow_data = await self.get_flow_attribution(campaign.get("id", ""), attribution_window)
                
                attribution_record = {
                    "campaign_id": campaign.get("id", ""),
                    "campaign_name": attrs.get("name", ""),
                    "send_time": attrs.get("send_time", ""),
                    "attribution_window": attribution_window,
                    "sent_count": metrics.get("sent_count", 0),
                    "delivered_count": metrics.get("delivered_count", 0),
                    "open_count": metrics.get("open_count", 0),
                    "click_count": metrics.get("click_count", 0),
                    "conversion_count": flow_data.get("conversion_count", 0),
                    "revenue_attributed": flow_data.get("revenue_attributed", 0.0),
                    "flow_triggers": flow_data.get("flow_triggers", []),
                    "segment_data": flow_data.get("segment_data", {})
                }
                attribution_data.append(attribution_record)
            
            logger.info(f"Successfully extracted {len(attribution_data)} attribution records")
            return attribution_data
            
        except Exception as e:
            logger.error(f"Failed to extract Klaviyo attribution data: {str(e)}")
            return []
    
    async def get_flow_attribution(self, campaign_id: str, attribution_window: int) -> Dict[str, Any]:
        """Get flow and conversion attribution data"""
        try:
            # This would integrate with Klaviyo's flow and conversion tracking
            url = f"{self.base_url}/flows"
            headers = {
                "Authorization": f"Klaviyo-API-Key {self.api_key}",
                "revision": "2024-02-15"
            }
            
            params = {
                "filter": f"any(trigger_filters.campaign_id,['{campaign_id}'])",
                "include": "flow-actions"
            }
            
            response = await self.client.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = await response.json()
            flows = data.get("data", [])
            
            # Aggregate flow attribution data
            total_conversions = 0
            total_revenue = 0.0
            flow_triggers = []
            
            for flow in flows:
                attrs = flow.get("attributes", {})
                total_conversions += attrs.get("conversion_count", 0)
                total_revenue += attrs.get("revenue_attributed", 0.0)
                flow_triggers.append({
                    "flow_id": flow.get("id", ""),
                    "flow_name": attrs.get("name", ""),
                    "trigger_type": attrs.get("trigger_type", "")
                })
            
            return {
                "conversion_count": total_conversions,
                "revenue_attributed": total_revenue,
                "flow_triggers": flow_triggers,
                "segment_data": {}  # Would be populated with segment analysis
            }
            
        except Exception as e:
            logger.error(f"Failed to get Klaviyo flow attribution: {str(e)}")
            return {}
    
    async def validate_connection(self) -> bool:
        """Validate Klaviyo API connection"""
        try:
            url = f"{self.base_url}/accounts"
            headers = {
                "Authorization": f"Klaviyo-API-Key {self.api_key}",
                "revision": "2024-02-15"
            }
            
            response = await self.client.get(url, headers=headers)
            response.raise_for_status()
            
            data = await response.json()
            accounts = data.get("data", [])
            
            if accounts:
                account_name = accounts[0].get("attributes", {}).get("contact_information", {}).get("organization_name", "Unknown")
                logger.info(f"Klaviyo connection validated for account: {account_name}")
                return True
            else:
                logger.warning("Klaviyo connection validated but no accounts found")
                return True
            
        except Exception as e:
            logger.error(f"Klaviyo connection validation failed: {str(e)}")
            return False
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

# Factory function for creating connector instances
async def create_klaviyo_connector(credentials: Dict[str, str]) -> KlaviyoConnector:
    """Create and validate Klaviyo connector instance"""
    connector = KlaviyoConnector(
        api_key=credentials["api_key"]
    )
    
    # Validate connection
    is_valid = await connector.validate_connection()
    if not is_valid:
        await connector.close()
        raise ValueError("Invalid Klaviyo credentials")
    
    return connector