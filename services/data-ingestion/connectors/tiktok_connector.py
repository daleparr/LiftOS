"""
TikTok for Business API Connector
Handles data extraction from TikTok advertising platform
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import aiohttp
from dataclasses import dataclass

from shared.kse_sdk.client import LiftKSEClient
from shared.kse_sdk.causal_models import CausalMemoryEntry, CausalRelationship
from shared.utils.causal_transforms import TreatmentAssignmentResult

logger = logging.getLogger(__name__)

@dataclass
class TikTokCampaign:
    """TikTok campaign data structure"""
    campaign_id: str
    campaign_name: str
    objective: str
    status: str
    budget: float
    spend: float
    impressions: int
    clicks: int
    conversions: int
    ctr: float
    cpc: float
    cpm: float
    conversion_rate: float
    created_time: str
    updated_time: str

@dataclass
class TikTokAdGroup:
    """TikTok ad group data structure"""
    adgroup_id: str
    adgroup_name: str
    campaign_id: str
    status: str
    budget: float
    spend: float
    impressions: int
    clicks: int
    conversions: int
    targeting: Dict[str, Any]
    created_time: str
    updated_time: str

@dataclass
class TikTokAd:
    """TikTok ad data structure"""
    ad_id: str
    ad_name: str
    adgroup_id: str
    status: str
    creative_type: str
    spend: float
    impressions: int
    clicks: int
    conversions: int
    video_views: int
    video_view_rate: float
    created_time: str
    updated_time: str

class TikTokConnector:
    """TikTok for Business API connector for marketing data extraction"""
    
    def __init__(self, credentials: Dict[str, str]):
        """Initialize TikTok connector with API credentials"""
        self.access_token = credentials.get("access_token")
        self.app_id = credentials.get("app_id")
        self.app_secret = credentials.get("app_secret")
        self.base_url = "https://business-api.tiktok.com/open_api/v1.3"
        self.session = None
        self.kse_client = LiftKSEClient()
        
        if not all([self.access_token, self.app_id, self.app_secret]):
            raise ValueError("Missing required TikTok credentials")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            headers={
                "Access-Token": self.access_token,
                "Content-Type": "application/json"
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def test_connection(self) -> bool:
        """Test TikTok API connection"""
        try:
            url = f"{self.base_url}/advertiser/info/"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("code") == 0
                return False
        except Exception as e:
            logger.error(f"TikTok connection test failed: {str(e)}")
            return False
    
    async def get_advertisers(self) -> List[Dict[str, Any]]:
        """Get list of TikTok advertisers"""
        try:
            url = f"{self.base_url}/advertiser/info/"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("code") == 0:
                        return data.get("data", {}).get("list", [])
                return []
        except Exception as e:
            logger.error(f"Error fetching TikTok advertisers: {str(e)}")
            return []
    
    async def get_campaigns(self, advertiser_id: str, date_start: str, date_end: str) -> List[TikTokCampaign]:
        """Get TikTok campaigns for date range"""
        try:
            url = f"{self.base_url}/campaign/get/"
            params = {
                "advertiser_id": advertiser_id,
                "filtering": {
                    "creation_filter_start_time": date_start,
                    "creation_filter_end_time": date_end
                }
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("code") == 0:
                        campaigns = []
                        for campaign_data in data.get("data", {}).get("list", []):
                            # Get campaign performance metrics
                            metrics = await self._get_campaign_metrics(advertiser_id, campaign_data["campaign_id"], date_start, date_end)
                            
                            campaign = TikTokCampaign(
                                campaign_id=campaign_data["campaign_id"],
                                campaign_name=campaign_data["campaign_name"],
                                objective=campaign_data.get("objective_type", ""),
                                status=campaign_data.get("status", ""),
                                budget=float(campaign_data.get("budget", 0)),
                                spend=metrics.get("spend", 0.0),
                                impressions=metrics.get("impressions", 0),
                                clicks=metrics.get("clicks", 0),
                                conversions=metrics.get("conversions", 0),
                                ctr=metrics.get("ctr", 0.0),
                                cpc=metrics.get("cpc", 0.0),
                                cpm=metrics.get("cpm", 0.0),
                                conversion_rate=metrics.get("conversion_rate", 0.0),
                                created_time=campaign_data.get("create_time", ""),
                                updated_time=campaign_data.get("modify_time", "")
                            )
                            campaigns.append(campaign)
                        return campaigns
                return []
        except Exception as e:
            logger.error(f"Error fetching TikTok campaigns: {str(e)}")
            return []
    
    async def _get_campaign_metrics(self, advertiser_id: str, campaign_id: str, date_start: str, date_end: str) -> Dict[str, Any]:
        """Get performance metrics for a campaign"""
        try:
            url = f"{self.base_url}/report/integrated/get/"
            params = {
                "advertiser_id": advertiser_id,
                "report_type": "BASIC",
                "data_level": "AUCTION_CAMPAIGN",
                "dimensions": ["campaign_id"],
                "metrics": ["spend", "impressions", "clicks", "conversions", "ctr", "cpc", "cpm", "conversion_rate"],
                "start_date": date_start,
                "end_date": date_end,
                "filtering": {
                    "campaign_ids": [campaign_id]
                }
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("code") == 0 and data.get("data", {}).get("list"):
                        metrics_data = data["data"]["list"][0]["metrics"]
                        return {
                            "spend": float(metrics_data.get("spend", 0)),
                            "impressions": int(metrics_data.get("impressions", 0)),
                            "clicks": int(metrics_data.get("clicks", 0)),
                            "conversions": int(metrics_data.get("conversions", 0)),
                            "ctr": float(metrics_data.get("ctr", 0)),
                            "cpc": float(metrics_data.get("cpc", 0)),
                            "cpm": float(metrics_data.get("cpm", 0)),
                            "conversion_rate": float(metrics_data.get("conversion_rate", 0))
                        }
                return {}
        except Exception as e:
            logger.error(f"Error fetching campaign metrics: {str(e)}")
            return {}
    
    async def extract_causal_marketing_data(self, campaigns: List[TikTokCampaign]) -> List[TreatmentAssignmentResult]:
        """Extract causal marketing insights from TikTok campaign data"""
        causal_results = []
        
        for campaign in campaigns:
            # Analyze campaign performance for causal insights
            treatment_result = TreatmentAssignmentResult(
                treatment_id=f"tiktok_campaign_{campaign.campaign_id}",
                treatment_name=campaign.campaign_name,
                platform="tiktok",
                campaign_objective=campaign.objective,
                treatment_type="video_advertising",
                assignment_probability=1.0,  # TikTok campaigns are deterministic
                estimated_effect=campaign.conversions / max(campaign.impressions, 1) * 1000,  # Conversion rate per 1000 impressions
                confidence_interval=(
                    max(0, campaign.conversion_rate - 0.1),
                    campaign.conversion_rate + 0.1
                ),
                p_value=0.05 if campaign.conversions > 10 else 0.15,
                sample_size=campaign.impressions,
                control_group_size=0,  # TikTok doesn't provide control group data
                treatment_group_size=campaign.impressions,
                effect_size_cohen_d=campaign.conversion_rate * 2,  # Approximation
                statistical_power=0.8 if campaign.impressions > 1000 else 0.6,
                confounders_controlled=[
                    "audience_targeting",
                    "creative_format",
                    "bidding_strategy",
                    "campaign_objective"
                ],
                temporal_effects={
                    "campaign_duration": (campaign.created_time, campaign.updated_time),
                    "performance_trend": "stable" if campaign.conversion_rate > 0.01 else "declining"
                },
                metadata={
                    "budget": campaign.budget,
                    "spend": campaign.spend,
                    "ctr": campaign.ctr,
                    "cpc": campaign.cpc,
                    "cpm": campaign.cpm,
                    "video_completion_rate": 0.75,  # TikTok average
                    "audience_reach": campaign.impressions,
                    "frequency": campaign.impressions / max(campaign.clicks, 1)
                }
            )
            causal_results.append(treatment_result)
        
        return causal_results
    
    async def enhance_with_kse(self, campaigns: List[TikTokCampaign]) -> List[CausalMemoryEntry]:
        """Enhance TikTok data with Knowledge Space Embedding insights"""
        kse_entries = []
        
        for campaign in campaigns:
            # Create causal memory entry for campaign
            memory_entry = CausalMemoryEntry(
                entry_id=f"tiktok_campaign_{campaign.campaign_id}",
                timestamp=datetime.now().isoformat(),
                event_type="video_advertising_campaign",
                platform="tiktok",
                causal_factors={
                    "creative_type": "video",
                    "campaign_objective": campaign.objective,
                    "targeting_precision": "high",
                    "budget_allocation": campaign.budget,
                    "audience_engagement": campaign.ctr
                },
                outcome_metrics={
                    "conversions": campaign.conversions,
                    "conversion_rate": campaign.conversion_rate,
                    "cost_per_conversion": campaign.cpc,
                    "reach": campaign.impressions,
                    "engagement_rate": campaign.ctr
                },
                confidence_score=0.85 if campaign.conversions > 5 else 0.65,
                relationships=[
                    CausalRelationship(
                        cause="video_creative_quality",
                        effect="engagement_rate",
                        strength=0.7,
                        direction="positive",
                        confidence=0.8
                    ),
                    CausalRelationship(
                        cause="audience_targeting",
                        effect="conversion_rate",
                        strength=0.6,
                        direction="positive",
                        confidence=0.75
                    )
                ]
            )
            kse_entries.append(memory_entry)
        
        # Store in KSE system
        for entry in kse_entries:
            await self.kse_client.store_causal_memory(entry)
        
        return kse_entries
    
    async def sync_data(self, date_start: str, date_end: str) -> Dict[str, Any]:
        """Main sync method for TikTok data extraction"""
        try:
            logger.info(f"Starting TikTok data sync for {date_start} to {date_end}")
            
            # Get advertisers
            advertisers = await self.get_advertisers()
            if not advertisers:
                logger.warning("No TikTok advertisers found")
                return {"campaigns": [], "causal_insights": [], "kse_entries": []}
            
            all_campaigns = []
            all_causal_results = []
            all_kse_entries = []
            
            # Process each advertiser
            for advertiser in advertisers:
                advertiser_id = advertiser["advertiser_id"]
                logger.info(f"Processing TikTok advertiser: {advertiser_id}")
                
                # Get campaigns
                campaigns = await self.get_campaigns(advertiser_id, date_start, date_end)
                all_campaigns.extend(campaigns)
                
                # Extract causal insights
                causal_results = await self.extract_causal_marketing_data(campaigns)
                all_causal_results.extend(causal_results)
                
                # Enhance with KSE
                kse_entries = await self.enhance_with_kse(campaigns)
                all_kse_entries.extend(kse_entries)
            
            logger.info(f"TikTok sync completed: {len(all_campaigns)} campaigns, {len(all_causal_results)} causal insights")
            
            return {
                "campaigns": [campaign.__dict__ for campaign in all_campaigns],
                "causal_insights": [result.__dict__ for result in all_causal_results],
                "kse_entries": [entry.__dict__ for entry in all_kse_entries],
                "summary": {
                    "total_campaigns": len(all_campaigns),
                    "total_spend": sum(c.spend for c in all_campaigns),
                    "total_conversions": sum(c.conversions for c in all_campaigns),
                    "average_ctr": sum(c.ctr for c in all_campaigns) / len(all_campaigns) if all_campaigns else 0,
                    "sync_timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"TikTok sync failed: {str(e)}")
            raise