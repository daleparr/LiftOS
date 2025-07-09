"""
LinkedIn Ads Connector for LiftOS Data Ingestion Service
Integrates with LinkedIn Marketing API v2 for professional advertising data
"""
import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, AsyncGenerator
from dataclasses import dataclass
import logging

# Import shared modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from shared.models.causal_marketing import CausalMarketingData, DataQualityAssessment
from shared.kse_sdk.client import LiftKSEClient
from shared.utils.causal_transforms import CausalDataTransformer

logger = logging.getLogger(__name__)


@dataclass
class LinkedInAdsCredentials:
    """LinkedIn Ads API credentials"""
    client_id: str
    client_secret: str
    access_token: str
    refresh_token: Optional[str] = None
    
    
@dataclass
class LinkedInAdsData:
    """LinkedIn Ads data structure"""
    # Campaign data
    campaigns: List[Dict[str, Any]]
    campaign_groups: List[Dict[str, Any]]
    creatives: List[Dict[str, Any]]
    
    # Performance data
    campaign_insights: List[Dict[str, Any]]
    audience_insights: List[Dict[str, Any]]
    demographic_insights: List[Dict[str, Any]]
    
    # Targeting data
    targeting_criteria: List[Dict[str, Any]]
    audience_segments: List[Dict[str, Any]]
    
    # Professional insights
    industry_performance: Dict[str, Any]
    job_function_performance: Dict[str, Any]
    seniority_performance: Dict[str, Any]
    
    # Metadata
    sync_timestamp: datetime
    total_spend: float
    total_impressions: int
    total_clicks: int
    data_quality_score: float


class LinkedInAdsConnector:
    """LinkedIn Marketing API connector with professional advertising analysis"""
    
    def __init__(self, credentials: LinkedInAdsCredentials, kse_client: Optional[LiftKSEClient] = None):
        self.credentials = credentials
        self.kse_client = kse_client
        self.causal_transformer = CausalDataTransformer()
        self.base_url = "https://api.linkedin.com/v2"
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make authenticated request to LinkedIn Marketing API"""
        headers = {
            'Authorization': f'Bearer {self.credentials.access_token}',
            'Content-Type': 'application/json',
            'LinkedIn-Version': '202312'
        }
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 401:
                    raise Exception("LinkedIn Ads API authentication failed")
                else:
                    error_text = await response.text()
                    raise Exception(f"LinkedIn Ads API request failed: {error_text}")
                    
        except Exception as e:
            logger.error(f"Error making LinkedIn Ads API request to {endpoint}: {str(e)}")
            raise
            
    async def get_campaigns(self, date_from: datetime, date_to: datetime) -> List[Dict[str, Any]]:
        """Fetch campaigns data from LinkedIn Ads"""
        try:
            params = {
                'q': 'search',
                'search.status.values[0]': 'ACTIVE',
                'search.status.values[1]': 'PAUSED',
                'search.status.values[2]': 'COMPLETED',
                'count': 100
            }
            
            campaigns = []
            start = 0
            
            while True:
                params['start'] = start
                response = await self._make_request('adCampaignsV2', params)
                
                if 'elements' in response:
                    batch_campaigns = response['elements']
                    
                    # Filter by date range if creation date is available
                    filtered_campaigns = []
                    for campaign in batch_campaigns:
                        # LinkedIn doesn't always provide creation date in list, so we include all
                        filtered_campaigns.append(campaign)
                        
                    campaigns.extend(filtered_campaigns)
                    
                    # Check pagination
                    paging = response.get('paging', {})
                    if len(batch_campaigns) < 100 or not paging.get('links', {}).get('next'):
                        break
                        
                    start += 100
                else:
                    break
                    
            logger.info(f"Fetched {len(campaigns)} campaigns from LinkedIn Ads")
            return campaigns
            
        except Exception as e:
            logger.error(f"Error fetching LinkedIn Ads campaigns: {str(e)}")
            return []
            
    async def get_campaign_groups(self) -> List[Dict[str, Any]]:
        """Fetch campaign groups from LinkedIn Ads"""
        try:
            params = {
                'q': 'search',
                'search.status.values[0]': 'ACTIVE',
                'search.status.values[1]': 'PAUSED',
                'count': 100
            }
            
            campaign_groups = []
            start = 0
            
            while True:
                params['start'] = start
                response = await self._make_request('adCampaignGroupsV2', params)
                
                if 'elements' in response:
                    batch_groups = response['elements']
                    campaign_groups.extend(batch_groups)
                    
                    if len(batch_groups) < 100:
                        break
                        
                    start += 100
                else:
                    break
                    
            logger.info(f"Fetched {len(campaign_groups)} campaign groups from LinkedIn Ads")
            return campaign_groups
            
        except Exception as e:
            logger.error(f"Error fetching LinkedIn Ads campaign groups: {str(e)}")
            return []
            
    async def get_creatives(self, campaigns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fetch creatives for campaigns"""
        try:
            if not campaigns:
                return []
                
            campaign_ids = [str(campaign.get('id', '')) for campaign in campaigns[:50]]  # Limit for API
            
            params = {
                'q': 'search',
                'search.campaignIds[0]': campaign_ids[0] if campaign_ids else '',
                'count': 100
            }
            
            creatives = []
            start = 0
            
            while True:
                params['start'] = start
                response = await self._make_request('adCreativesV2', params)
                
                if 'elements' in response:
                    batch_creatives = response['elements']
                    creatives.extend(batch_creatives)
                    
                    if len(batch_creatives) < 100:
                        break
                        
                    start += 100
                else:
                    break
                    
            logger.info(f"Fetched {len(creatives)} creatives from LinkedIn Ads")
            return creatives
            
        except Exception as e:
            logger.error(f"Error fetching LinkedIn Ads creatives: {str(e)}")
            return []
            
    async def get_campaign_insights(self, campaigns: List[Dict[str, Any]], date_from: datetime, date_to: datetime) -> List[Dict[str, Any]]:
        """Fetch campaign performance insights"""
        try:
            if not campaigns:
                return []
                
            # Format dates for LinkedIn API
            date_range = {
                'start.day': date_from.day,
                'start.month': date_from.month,
                'start.year': date_from.year,
                'end.day': date_to.day,
                'end.month': date_to.month,
                'end.year': date_to.year
            }
            
            insights = []
            
            # Get insights for each campaign (batch processing for efficiency)
            for i in range(0, len(campaigns), 20):  # Process in batches of 20
                batch_campaigns = campaigns[i:i+20]
                campaign_ids = [str(campaign.get('id', '')) for campaign in batch_campaigns]
                
                params = {
                    'q': 'analytics',
                    'pivot': 'CAMPAIGN',
                    'timeGranularity': 'DAILY',
                    'campaigns[0]': campaign_ids[0] if campaign_ids else '',
                    'fields': 'impressions,clicks,costInUsd,externalWebsiteConversions,dateRange',
                    **date_range
                }
                
                try:
                    response = await self._make_request('adAnalyticsV2', params)
                    
                    if 'elements' in response:
                        insights.extend(response['elements'])
                        
                except Exception as e:
                    logger.warning(f"Error fetching insights for campaign batch: {str(e)}")
                    continue
                    
            logger.info(f"Fetched {len(insights)} campaign insights from LinkedIn Ads")
            return insights
            
        except Exception as e:
            logger.error(f"Error fetching LinkedIn Ads campaign insights: {str(e)}")
            return []
            
    async def get_audience_insights(self, campaigns: List[Dict[str, Any]], date_from: datetime, date_to: datetime) -> List[Dict[str, Any]]:
        """Fetch audience demographic insights"""
        try:
            if not campaigns:
                return []
                
            date_range = {
                'start.day': date_from.day,
                'start.month': date_from.month,
                'start.year': date_from.year,
                'end.day': date_to.day,
                'end.month': date_to.month,
                'end.year': date_to.year
            }
            
            audience_insights = []
            
            # Get demographic breakdowns
            demographic_pivots = ['COMPANY_SIZE', 'INDUSTRY', 'JOB_FUNCTION', 'SENIORITY']
            
            for pivot in demographic_pivots:
                try:
                    params = {
                        'q': 'analytics',
                        'pivot': pivot,
                        'timeGranularity': 'ALL_DAYS',
                        'fields': 'impressions,clicks,costInUsd,externalWebsiteConversions',
                        **date_range
                    }
                    
                    response = await self._make_request('adAnalyticsV2', params)
                    
                    if 'elements' in response:
                        for element in response['elements']:
                            element['pivot_type'] = pivot
                            audience_insights.append(element)
                            
                except Exception as e:
                    logger.warning(f"Error fetching {pivot} insights: {str(e)}")
                    continue
                    
            logger.info(f"Fetched {len(audience_insights)} audience insights from LinkedIn Ads")
            return audience_insights
            
        except Exception as e:
            logger.error(f"Error fetching LinkedIn Ads audience insights: {str(e)}")
            return []
            
    async def get_professional_insights(self, audience_insights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze professional targeting performance"""
        try:
            professional_insights = {
                'industry_performance': {},
                'job_function_performance': {},
                'seniority_performance': {},
                'company_size_performance': {}
            }
            
            for insight in audience_insights:
                pivot_type = insight.get('pivot_type', '')
                pivot_value = insight.get('pivot', '')
                
                metrics = {
                    'impressions': insight.get('impressions', 0),
                    'clicks': insight.get('clicks', 0),
                    'cost': insight.get('costInUsd', 0),
                    'conversions': insight.get('externalWebsiteConversions', 0)
                }
                
                # Calculate derived metrics
                if metrics['impressions'] > 0:
                    metrics['ctr'] = metrics['clicks'] / metrics['impressions']
                if metrics['clicks'] > 0:
                    metrics['cpc'] = metrics['cost'] / metrics['clicks']
                    metrics['conversion_rate'] = metrics['conversions'] / metrics['clicks']
                    
                if pivot_type == 'INDUSTRY':
                    professional_insights['industry_performance'][pivot_value] = metrics
                elif pivot_type == 'JOB_FUNCTION':
                    professional_insights['job_function_performance'][pivot_value] = metrics
                elif pivot_type == 'SENIORITY':
                    professional_insights['seniority_performance'][pivot_value] = metrics
                elif pivot_type == 'COMPANY_SIZE':
                    professional_insights['company_size_performance'][pivot_value] = metrics
                    
            logger.info("Analyzed professional targeting insights")
            return professional_insights
            
        except Exception as e:
            logger.error(f"Error analyzing professional insights: {str(e)}")
            return {}
            
    async def sync_data(self, date_from: datetime, date_to: datetime) -> LinkedInAdsData:
        """Sync comprehensive LinkedIn Ads data"""
        try:
            logger.info(f"Starting LinkedIn Ads data sync from {date_from} to {date_to}")
            
            # Fetch campaigns first
            campaigns = await self.get_campaigns(date_from, date_to)
            
            # Fetch other data concurrently
            campaign_groups_task = self.get_campaign_groups()
            creatives_task = self.get_creatives(campaigns)
            campaign_insights_task = self.get_campaign_insights(campaigns, date_from, date_to)
            audience_insights_task = self.get_audience_insights(campaigns, date_from, date_to)
            
            campaign_groups, creatives, campaign_insights, audience_insights = await asyncio.gather(
                campaign_groups_task, creatives_task, campaign_insights_task, audience_insights_task
            )
            
            # Analyze professional insights
            professional_insights = await self.get_professional_insights(audience_insights)
            
            # Calculate totals and quality score
            total_spend = sum(insight.get('costInUsd', 0) for insight in campaign_insights)
            total_impressions = sum(insight.get('impressions', 0) for insight in campaign_insights)
            total_clicks = sum(insight.get('clicks', 0) for insight in campaign_insights)
            
            data_quality_score = self._calculate_data_quality(campaigns, campaign_insights, creatives)
            
            linkedin_data = LinkedInAdsData(
                campaigns=campaigns,
                campaign_groups=campaign_groups,
                creatives=creatives,
                campaign_insights=campaign_insights,
                audience_insights=audience_insights,
                demographic_insights=[],  # Can be extended
                targeting_criteria=[],  # Can be extended
                audience_segments=[],  # Can be extended
                industry_performance=professional_insights.get('industry_performance', {}),
                job_function_performance=professional_insights.get('job_function_performance', {}),
                seniority_performance=professional_insights.get('seniority_performance', {}),
                sync_timestamp=datetime.utcnow(),
                total_spend=total_spend,
                total_impressions=total_impressions,
                total_clicks=total_clicks,
                data_quality_score=data_quality_score
            )
            
            logger.info(f"LinkedIn Ads sync completed: ${total_spend:.2f} spend, {total_impressions} impressions, quality score: {data_quality_score:.2f}")
            return linkedin_data
            
        except Exception as e:
            logger.error(f"Error syncing LinkedIn Ads data: {str(e)}")
            raise
            
    def _calculate_data_quality(self, campaigns: List[Dict], insights: List[Dict], creatives: List[Dict]) -> float:
        """Calculate data quality score for LinkedIn Ads data"""
        try:
            if not campaigns:
                return 0.0
                
            quality_scores = []
            
            # Check campaign data quality
            for campaign in campaigns:
                score = 0.0
                if campaign.get('name'):
                    score += 0.2
                if campaign.get('status'):
                    score += 0.2
                if campaign.get('type'):
                    score += 0.2
                if campaign.get('targetingCriteria'):
                    score += 0.2
                if campaign.get('costType'):
                    score += 0.2
                quality_scores.append(score)
                
            # Check insights data quality
            for insight in insights:
                score = 0.0
                if insight.get('impressions') is not None:
                    score += 0.3
                if insight.get('clicks') is not None:
                    score += 0.3
                if insight.get('costInUsd') is not None:
                    score += 0.4
                quality_scores.append(score)
                
            return sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating LinkedIn Ads data quality: {str(e)}")
            return 0.0
            
    async def enhance_with_kse(self, linkedin_data: LinkedInAdsData, context: str) -> CausalMarketingData:
        """Enhance LinkedIn Ads data with KSE causal analysis"""
        try:
            if not self.kse_client:
                logger.warning("KSE client not available for LinkedIn Ads enhancement")
                return self._create_basic_causal_data(linkedin_data)
                
            # Prepare professional advertising insights for causal analysis
            professional_insights = {
                'b2b_targeting_effectiveness': self._analyze_b2b_targeting(linkedin_data),
                'professional_audience_optimization': self._analyze_professional_audiences(linkedin_data),
                'industry_performance_patterns': self._analyze_industry_patterns(linkedin_data),
                'linkedin_attribution_analysis': self._analyze_linkedin_attribution(linkedin_data)
            }
            
            # Store in KSE for causal memory
            kse_entry = {
                'source': 'linkedin_ads',
                'timestamp': linkedin_data.sync_timestamp.isoformat(),
                'data_type': 'professional_advertising',
                'insights': professional_insights,
                'context': context,
                'total_spend': linkedin_data.total_spend,
                'total_impressions': linkedin_data.total_impressions,
                'data_quality_score': linkedin_data.data_quality_score
            }
            
            await self.kse_client.store_causal_memory(
                memory_id=f"linkedin_ads_{context}_{int(linkedin_data.sync_timestamp.timestamp())}",
                content=kse_entry,
                context=context
            )
            
            # Transform to causal marketing data
            causal_data = await self.causal_transformer.transform_advertising_data(
                linkedin_data, professional_insights
            )
            
            logger.info("LinkedIn Ads data enhanced with KSE causal analysis")
            return causal_data
            
        except Exception as e:
            logger.error(f"Error enhancing LinkedIn Ads data with KSE: {str(e)}")
            return self._create_basic_causal_data(linkedin_data)
            
    def _analyze_b2b_targeting(self, data: LinkedInAdsData) -> Dict[str, Any]:
        """Analyze B2B targeting effectiveness"""
        try:
            return {
                'industry_targeting_performance': data.industry_performance,
                'job_function_targeting_performance': data.job_function_performance,
                'seniority_targeting_performance': data.seniority_performance,
                'b2b_conversion_patterns': {
                    'total_campaigns': len(data.campaigns),
                    'professional_reach': data.total_impressions,
                    'engagement_quality': data.total_clicks / data.total_impressions if data.total_impressions > 0 else 0
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing B2B targeting: {str(e)}")
            return {}
            
    def _analyze_professional_audiences(self, data: LinkedInAdsData) -> Dict[str, Any]:
        """Analyze professional audience optimization"""
        try:
            # Find best performing segments
            best_industries = {}
            best_job_functions = {}
            
            for industry, metrics in data.industry_performance.items():
                if metrics.get('conversion_rate', 0) > 0:
                    best_industries[industry] = metrics['conversion_rate']
                    
            for job_function, metrics in data.job_function_performance.items():
                if metrics.get('conversion_rate', 0) > 0:
                    best_job_functions[job_function] = metrics['conversion_rate']
                    
            return {
                'top_performing_industries': dict(sorted(best_industries.items(), key=lambda x: x[1], reverse=True)[:5]),
                'top_performing_job_functions': dict(sorted(best_job_functions.items(), key=lambda x: x[1], reverse=True)[:5]),
                'audience_optimization_score': data.data_quality_score
            }
        except Exception as e:
            logger.error(f"Error analyzing professional audiences: {str(e)}")
            return {}
            
    def _analyze_industry_patterns(self, data: LinkedInAdsData) -> Dict[str, Any]:
        """Analyze industry-specific performance patterns"""
        try:
            industry_analysis = {}
            
            for industry, metrics in data.industry_performance.items():
                industry_analysis[industry] = {
                    'performance_score': metrics.get('conversion_rate', 0) * 100,
                    'cost_efficiency': metrics.get('cpc', 0),
                    'reach_quality': metrics.get('ctr', 0) * 100
                }
                
            return {
                'industry_performance_analysis': industry_analysis,
                'cross_industry_insights': {
                    'most_cost_effective': min(industry_analysis.items(), key=lambda x: x[1]['cost_efficiency'])[0] if industry_analysis else None,
                    'highest_engagement': max(industry_analysis.items(), key=lambda x: x[1]['reach_quality'])[0] if industry_analysis else None
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing industry patterns: {str(e)}")
            return {}
            
    def _analyze_linkedin_attribution(self, data: LinkedInAdsData) -> Dict[str, Any]:
        """Analyze LinkedIn-specific attribution patterns"""
        try:
            total_conversions = sum(insight.get('externalWebsiteConversions', 0) for insight in data.campaign_insights)
            
            return {
                'linkedin_attribution_score': total_conversions / data.total_clicks if data.total_clicks > 0 else 0,
                'professional_network_impact': {
                    'total_professional_reach': data.total_impressions,
                    'professional_engagement_rate': data.total_clicks / data.total_impressions if data.total_impressions > 0 else 0,
                    'b2b_conversion_efficiency': total_conversions / data.total_spend if data.total_spend > 0 else 0
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing LinkedIn attribution: {str(e)}")
            return {}
            
    def _create_basic_causal_data(self, linkedin_data: LinkedInAdsData) -> CausalMarketingData:
        """Create basic causal marketing data without KSE enhancement"""
        return CausalMarketingData(
            source="linkedin_ads",
            timestamp=linkedin_data.sync_timestamp,
            raw_data=linkedin_data.__dict__,
            causal_factors={
                'campaign_count': len(linkedin_data.campaigns),
                'total_spend': linkedin_data.total_spend,
                'professional_reach': linkedin_data.total_impressions
            },
            treatment_assignment={
                'b2b_targeting': 'active',
                'professional_advertising': 'active',
                'linkedin_optimization': 'active'
            },
            outcome_metrics={
                'total_impressions': linkedin_data.total_impressions,
                'total_clicks': linkedin_data.total_clicks,
                'total_spend': linkedin_data.total_spend,
                'ctr': linkedin_data.total_clicks / linkedin_data.total_impressions if linkedin_data.total_impressions > 0 else 0
            },
            data_quality=DataQualityAssessment(
                completeness=linkedin_data.data_quality_score,
                consistency=0.85,
                validity=0.90,
                timeliness=0.95,
                overall_score=linkedin_data.data_quality_score
            )
        )


async def create_linkedin_ads_connector(credentials: Dict[str, str], kse_client: Optional[LiftKSEClient] = None) -> LinkedInAdsConnector:
    """Factory function to create LinkedIn Ads connector"""
    linkedin_credentials = LinkedInAdsCredentials(
        client_id=credentials['client_id'],
        client_secret=credentials['client_secret'],
        access_token=credentials['access_token'],
        refresh_token=credentials.get('refresh_token')
    )
    
    return LinkedInAdsConnector(linkedin_credentials, kse_client)