"""
X (Twitter) Ads Connector for LiftOS Data Ingestion Service
Integrates with X Ads API v12 for social media advertising data
"""
import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, AsyncGenerator
from dataclasses import dataclass
import logging
import hashlib
import hmac
import base64
import urllib.parse
import time

# Import shared modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from shared.models.causal_marketing import CausalMarketingData, DataQualityAssessment
from shared.kse_sdk.client import LiftKSEClient
from shared.utils.causal_transforms import CausalDataTransformer

logger = logging.getLogger(__name__)


@dataclass
class XAdsCredentials:
    """X (Twitter) Ads API credentials"""
    consumer_key: str
    consumer_secret: str
    access_token: str
    access_token_secret: str
    
    
@dataclass
class XAdsData:
    """X (Twitter) Ads data structure"""
    # Campaign data
    campaigns: List[Dict[str, Any]]
    line_items: List[Dict[str, Any]]
    promoted_tweets: List[Dict[str, Any]]
    
    # Performance data
    campaign_stats: List[Dict[str, Any]]
    line_item_stats: List[Dict[str, Any]]
    promoted_tweet_stats: List[Dict[str, Any]]
    
    # Audience data
    tailored_audiences: List[Dict[str, Any]]
    audience_insights: List[Dict[str, Any]]
    
    # Creative data
    media_creatives: List[Dict[str, Any]]
    tweet_performance: List[Dict[str, Any]]
    
    # Social insights
    engagement_metrics: Dict[str, Any]
    viral_metrics: Dict[str, Any]
    conversation_metrics: Dict[str, Any]
    
    # Metadata
    sync_timestamp: datetime
    total_spend: float
    total_impressions: int
    total_engagements: int
    data_quality_score: float


class XAdsConnector:
    """X (Twitter) Ads API connector with social media advertising analysis"""
    
    def __init__(self, credentials: XAdsCredentials, kse_client: Optional[LiftKSEClient] = None):
        self.credentials = credentials
        self.kse_client = kse_client
        self.causal_transformer = CausalDataTransformer()
        self.base_url = "https://ads-api.twitter.com/12"
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            
    def _generate_oauth_header(self, method: str, url: str, params: Optional[Dict] = None) -> str:
        """Generate OAuth 1.0a authorization header for X API"""
        oauth_params = {
            'oauth_consumer_key': self.credentials.consumer_key,
            'oauth_token': self.credentials.access_token,
            'oauth_signature_method': 'HMAC-SHA1',
            'oauth_timestamp': str(int(time.time())),
            'oauth_nonce': hashlib.md5(str(time.time()).encode()).hexdigest(),
            'oauth_version': '1.0'
        }
        
        # Combine OAuth params with request params
        all_params = oauth_params.copy()
        if params:
            all_params.update(params)
            
        # Create parameter string
        param_string = '&'.join([f"{k}={urllib.parse.quote(str(v), safe='')}" 
                                for k, v in sorted(all_params.items())])
        
        # Create signature base string
        signature_base = f"{method.upper()}&{urllib.parse.quote(url, safe='')}&{urllib.parse.quote(param_string, safe='')}"
        
        # Create signing key
        signing_key = f"{urllib.parse.quote(self.credentials.consumer_secret, safe='')}&{urllib.parse.quote(self.credentials.access_token_secret, safe='')}"
        
        # Generate signature
        signature = base64.b64encode(
            hmac.new(signing_key.encode(), signature_base.encode(), hashlib.sha1).digest()
        ).decode()
        
        oauth_params['oauth_signature'] = signature
        
        # Create authorization header
        auth_header = 'OAuth ' + ', '.join([f'{k}="{urllib.parse.quote(str(v), safe="")}"' 
                                           for k, v in sorted(oauth_params.items())])
        
        return auth_header
        
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make authenticated request to X Ads API"""
        url = f"{self.base_url}/{endpoint}"
        
        # Generate OAuth header
        auth_header = self._generate_oauth_header('GET', url, params)
        
        headers = {
            'Authorization': auth_header,
            'Content-Type': 'application/json'
        }
        
        try:
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 401:
                    raise Exception("X Ads API authentication failed")
                else:
                    error_text = await response.text()
                    raise Exception(f"X Ads API request failed: {error_text}")
                    
        except Exception as e:
            logger.error(f"Error making X Ads API request to {endpoint}: {str(e)}")
            raise
            
    async def get_accounts(self) -> List[Dict[str, Any]]:
        """Get advertising accounts"""
        try:
            response = await self._make_request('accounts')
            accounts = response.get('data', [])
            logger.info(f"Fetched {len(accounts)} X Ads accounts")
            return accounts
        except Exception as e:
            logger.error(f"Error fetching X Ads accounts: {str(e)}")
            return []
            
    async def get_campaigns(self, account_id: str, date_from: datetime, date_to: datetime) -> List[Dict[str, Any]]:
        """Fetch campaigns data from X Ads"""
        try:
            params = {
                'account_id': account_id,
                'count': 200,
                'with_deleted': 'false'
            }
            
            campaigns = []
            cursor = None
            
            while True:
                if cursor:
                    params['cursor'] = cursor
                    
                response = await self._make_request(f'accounts/{account_id}/campaigns', params)
                
                if 'data' in response:
                    batch_campaigns = response['data']
                    
                    # Filter by date range if creation date is available
                    filtered_campaigns = []
                    for campaign in batch_campaigns:
                        # X API provides created_at timestamp
                        if 'created_at' in campaign:
                            created_at = datetime.fromisoformat(campaign['created_at'].replace('Z', '+00:00'))
                            if date_from <= created_at <= date_to:
                                filtered_campaigns.append(campaign)
                        else:
                            filtered_campaigns.append(campaign)  # Include if no date available
                            
                    campaigns.extend(filtered_campaigns)
                    
                    # Check pagination
                    next_cursor = response.get('next_cursor')
                    if not next_cursor:
                        break
                    cursor = next_cursor
                else:
                    break
                    
            logger.info(f"Fetched {len(campaigns)} campaigns from X Ads")
            return campaigns
            
        except Exception as e:
            logger.error(f"Error fetching X Ads campaigns: {str(e)}")
            return []
            
    async def get_line_items(self, account_id: str, campaign_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch line items (ad groups) for campaigns"""
        try:
            if not campaign_ids:
                return []
                
            params = {
                'account_id': account_id,
                'campaign_ids': ','.join(campaign_ids[:50]),  # Limit for API
                'count': 200,
                'with_deleted': 'false'
            }
            
            line_items = []
            cursor = None
            
            while True:
                if cursor:
                    params['cursor'] = cursor
                    
                response = await self._make_request(f'accounts/{account_id}/line_items', params)
                
                if 'data' in response:
                    batch_line_items = response['data']
                    line_items.extend(batch_line_items)
                    
                    next_cursor = response.get('next_cursor')
                    if not next_cursor:
                        break
                    cursor = next_cursor
                else:
                    break
                    
            logger.info(f"Fetched {len(line_items)} line items from X Ads")
            return line_items
            
        except Exception as e:
            logger.error(f"Error fetching X Ads line items: {str(e)}")
            return []
            
    async def get_promoted_tweets(self, account_id: str, line_item_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch promoted tweets for line items"""
        try:
            if not line_item_ids:
                return []
                
            params = {
                'account_id': account_id,
                'line_item_ids': ','.join(line_item_ids[:50]),
                'count': 200,
                'with_deleted': 'false'
            }
            
            promoted_tweets = []
            cursor = None
            
            while True:
                if cursor:
                    params['cursor'] = cursor
                    
                response = await self._make_request(f'accounts/{account_id}/promoted_tweets', params)
                
                if 'data' in response:
                    batch_tweets = response['data']
                    promoted_tweets.extend(batch_tweets)
                    
                    next_cursor = response.get('next_cursor')
                    if not next_cursor:
                        break
                    cursor = next_cursor
                else:
                    break
                    
            logger.info(f"Fetched {len(promoted_tweets)} promoted tweets from X Ads")
            return promoted_tweets
            
        except Exception as e:
            logger.error(f"Error fetching X Ads promoted tweets: {str(e)}")
            return []
            
    async def get_campaign_stats(self, account_id: str, campaign_ids: List[str], date_from: datetime, date_to: datetime) -> List[Dict[str, Any]]:
        """Fetch campaign performance statistics"""
        try:
            if not campaign_ids:
                return []
                
            params = {
                'account_id': account_id,
                'entity_ids': ','.join(campaign_ids[:50]),
                'entity': 'CAMPAIGN',
                'start_time': date_from.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'end_time': date_to.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'granularity': 'DAY',
                'metric_groups': 'ENGAGEMENT,BILLING,VIDEO,WEB_CONVERSION'
            }
            
            stats = []
            cursor = None
            
            while True:
                if cursor:
                    params['cursor'] = cursor
                    
                response = await self._make_request(f'stats/accounts/{account_id}', params)
                
                if 'data' in response:
                    batch_stats = response['data']
                    stats.extend(batch_stats)
                    
                    next_cursor = response.get('next_cursor')
                    if not next_cursor:
                        break
                    cursor = next_cursor
                else:
                    break
                    
            logger.info(f"Fetched {len(stats)} campaign stats from X Ads")
            return stats
            
        except Exception as e:
            logger.error(f"Error fetching X Ads campaign stats: {str(e)}")
            return []
            
    async def get_tailored_audiences(self, account_id: str) -> List[Dict[str, Any]]:
        """Fetch tailored audiences for targeting analysis"""
        try:
            params = {
                'account_id': account_id,
                'count': 200
            }
            
            audiences = []
            cursor = None
            
            while True:
                if cursor:
                    params['cursor'] = cursor
                    
                response = await self._make_request(f'accounts/{account_id}/tailored_audiences', params)
                
                if 'data' in response:
                    batch_audiences = response['data']
                    audiences.extend(batch_audiences)
                    
                    next_cursor = response.get('next_cursor')
                    if not next_cursor:
                        break
                    cursor = next_cursor
                else:
                    break
                    
            logger.info(f"Fetched {len(audiences)} tailored audiences from X Ads")
            return audiences
            
        except Exception as e:
            logger.error(f"Error fetching X Ads tailored audiences: {str(e)}")
            return []
            
    async def get_social_insights(self, stats: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze social media specific insights"""
        try:
            engagement_metrics = {
                'total_engagements': 0,
                'total_retweets': 0,
                'total_likes': 0,
                'total_replies': 0,
                'total_follows': 0
            }
            
            viral_metrics = {
                'viral_impressions': 0,
                'viral_engagements': 0,
                'viral_rate': 0.0
            }
            
            conversation_metrics = {
                'conversation_rate': 0.0,
                'reply_rate': 0.0,
                'engagement_rate': 0.0
            }
            
            total_impressions = 0
            
            for stat in stats:
                metrics = stat.get('id_data', [{}])[0].get('metrics', {})
                
                # Engagement metrics
                engagement_metrics['total_engagements'] += metrics.get('engagements', 0)
                engagement_metrics['total_retweets'] += metrics.get('retweets', 0)
                engagement_metrics['total_likes'] += metrics.get('likes', 0)
                engagement_metrics['total_replies'] += metrics.get('replies', 0)
                engagement_metrics['total_follows'] += metrics.get('follows', 0)
                
                # Viral metrics
                viral_metrics['viral_impressions'] += metrics.get('viral_impressions', 0)
                viral_metrics['viral_engagements'] += metrics.get('viral_engagements', 0)
                
                total_impressions += metrics.get('impressions', 0)
                
            # Calculate rates
            if total_impressions > 0:
                conversation_metrics['engagement_rate'] = engagement_metrics['total_engagements'] / total_impressions
                conversation_metrics['reply_rate'] = engagement_metrics['total_replies'] / total_impressions
                
            if engagement_metrics['total_engagements'] > 0:
                viral_metrics['viral_rate'] = viral_metrics['viral_engagements'] / engagement_metrics['total_engagements']
                
            return {
                'engagement_metrics': engagement_metrics,
                'viral_metrics': viral_metrics,
                'conversation_metrics': conversation_metrics
            }
            
        except Exception as e:
            logger.error(f"Error analyzing social insights: {str(e)}")
            return {}
            
    async def sync_data(self, date_from: datetime, date_to: datetime) -> XAdsData:
        """Sync comprehensive X Ads data"""
        try:
            logger.info(f"Starting X Ads data sync from {date_from} to {date_to}")
            
            # Get accounts first
            accounts = await self.get_accounts()
            if not accounts:
                raise Exception("No X Ads accounts found")
                
            account_id = accounts[0]['id']  # Use first account
            
            # Fetch campaigns
            campaigns = await self.get_campaigns(account_id, date_from, date_to)
            campaign_ids = [campaign['id'] for campaign in campaigns]
            
            # Fetch other data concurrently
            line_items_task = self.get_line_items(account_id, campaign_ids)
            campaign_stats_task = self.get_campaign_stats(account_id, campaign_ids, date_from, date_to)
            tailored_audiences_task = self.get_tailored_audiences(account_id)
            
            line_items, campaign_stats, tailored_audiences = await asyncio.gather(
                line_items_task, campaign_stats_task, tailored_audiences_task
            )
            
            # Get line item IDs for promoted tweets
            line_item_ids = [item['id'] for item in line_items]
            promoted_tweets = await self.get_promoted_tweets(account_id, line_item_ids)
            
            # Analyze social insights
            social_insights = await self.get_social_insights(campaign_stats)
            
            # Calculate totals and quality score
            total_spend = sum(
                stat.get('id_data', [{}])[0].get('metrics', {}).get('billed_charge_local_micro', 0) / 1000000
                for stat in campaign_stats
            )
            total_impressions = sum(
                stat.get('id_data', [{}])[0].get('metrics', {}).get('impressions', 0)
                for stat in campaign_stats
            )
            total_engagements = social_insights.get('engagement_metrics', {}).get('total_engagements', 0)
            
            data_quality_score = self._calculate_data_quality(campaigns, campaign_stats, promoted_tweets)
            
            x_ads_data = XAdsData(
                campaigns=campaigns,
                line_items=line_items,
                promoted_tweets=promoted_tweets,
                campaign_stats=campaign_stats,
                line_item_stats=[],  # Can be extended
                promoted_tweet_stats=[],  # Can be extended
                tailored_audiences=tailored_audiences,
                audience_insights=[],  # Can be extended
                media_creatives=[],  # Can be extended
                tweet_performance=[],  # Can be extended
                engagement_metrics=social_insights.get('engagement_metrics', {}),
                viral_metrics=social_insights.get('viral_metrics', {}),
                conversation_metrics=social_insights.get('conversation_metrics', {}),
                sync_timestamp=datetime.utcnow(),
                total_spend=total_spend,
                total_impressions=total_impressions,
                total_engagements=total_engagements,
                data_quality_score=data_quality_score
            )
            
            logger.info(f"X Ads sync completed: ${total_spend:.2f} spend, {total_impressions} impressions, {total_engagements} engagements, quality score: {data_quality_score:.2f}")
            return x_ads_data
            
        except Exception as e:
            logger.error(f"Error syncing X Ads data: {str(e)}")
            raise
            
    def _calculate_data_quality(self, campaigns: List[Dict], stats: List[Dict], tweets: List[Dict]) -> float:
        """Calculate data quality score for X Ads data"""
        try:
            if not campaigns:
                return 0.0
                
            quality_scores = []
            
            # Check campaign data quality
            for campaign in campaigns:
                score = 0.0
                if campaign.get('name'):
                    score += 0.25
                if campaign.get('funding_instrument_id'):
                    score += 0.25
                if campaign.get('entity_status'):
                    score += 0.25
                if campaign.get('objective'):
                    score += 0.25
                quality_scores.append(score)
                
            # Check stats data quality
            for stat in stats:
                score = 0.0
                metrics = stat.get('id_data', [{}])[0].get('metrics', {})
                if metrics.get('impressions') is not None:
                    score += 0.4
                if metrics.get('engagements') is not None:
                    score += 0.3
                if metrics.get('billed_charge_local_micro') is not None:
                    score += 0.3
                quality_scores.append(score)
                
            return sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating X Ads data quality: {str(e)}")
            return 0.0
            
    async def enhance_with_kse(self, x_ads_data: XAdsData, context: str) -> CausalMarketingData:
        """Enhance X Ads data with KSE causal analysis"""
        try:
            if not self.kse_client:
                logger.warning("KSE client not available for X Ads enhancement")
                return self._create_basic_causal_data(x_ads_data)
                
            # Prepare social media advertising insights for causal analysis
            social_insights = {
                'viral_amplification_analysis': self._analyze_viral_amplification(x_ads_data),
                'conversation_driving_effectiveness': self._analyze_conversation_effectiveness(x_ads_data),
                'social_engagement_optimization': self._analyze_engagement_optimization(x_ads_data),
                'x_attribution_analysis': self._analyze_x_attribution(x_ads_data)
            }
            
            # Store in KSE for causal memory
            kse_entry = {
                'source': 'x_ads',
                'timestamp': x_ads_data.sync_timestamp.isoformat(),
                'data_type': 'social_media_advertising',
                'insights': social_insights,
                'context': context,
                'total_spend': x_ads_data.total_spend,
                'total_impressions': x_ads_data.total_impressions,
                'total_engagements': x_ads_data.total_engagements,
                'data_quality_score': x_ads_data.data_quality_score
            }
            
            await self.kse_client.store_causal_memory(
                memory_id=f"x_ads_{context}_{int(x_ads_data.sync_timestamp.timestamp())}",
                content=kse_entry,
                context=context
            )
            
            # Transform to causal marketing data
            causal_data = await self.causal_transformer.transform_advertising_data(
                x_ads_data, social_insights
            )
            
            logger.info("X Ads data enhanced with KSE causal analysis")
            return causal_data
            
        except Exception as e:
            logger.error(f"Error enhancing X Ads data with KSE: {str(e)}")
            return self._create_basic_causal_data(x_ads_data)
            
    def _analyze_viral_amplification(self, data: XAdsData) -> Dict[str, Any]:
        """Analyze viral amplification patterns"""
        try:
            return {
                'viral_rate': data.viral_metrics.get('viral_rate', 0),
                'viral_impressions': data.viral_metrics.get('viral_impressions', 0),
                'amplification_score': data.viral_metrics.get('viral_engagements', 0) / data.total_engagements if data.total_engagements > 0 else 0,
                'organic_reach_multiplier': data.viral_metrics.get('viral_impressions', 0) / data.total_impressions if data.total_impressions > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error analyzing viral amplification: {str(e)}")
            return {}
            
    def _analyze_conversation_effectiveness(self, data: XAdsData) -> Dict[str, Any]:
        """Analyze conversation driving effectiveness"""
        try:
            return {
                'conversation_rate': data.conversation_metrics.get('conversation_rate', 0),
                'reply_rate': data.conversation_metrics.get('reply_rate', 0),
                'engagement_quality': {
                    'retweets': data.engagement_metrics.get('total_retweets', 0),
                    'likes': data.engagement_metrics.get('total_likes', 0),
                    'replies': data.engagement_metrics.get('total_replies', 0),
                    'follows': data.engagement_metrics.get('total_follows', 0)
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing conversation effectiveness: {str(e)}")
            return {}
            
    def _analyze_engagement_optimization(self, data: XAdsData) -> Dict[str, Any]:
        """Analyze social engagement optimization"""
        try:
            total_engagements = data.engagement_metrics.get('total_engagements', 0)
            
            return {
                'engagement_rate': data.conversation_metrics.get('engagement_rate', 0),
                'engagement_distribution': {
                    'retweet_share': data.engagement_metrics.get('total_retweets', 0) / total_engagements if total_engagements > 0 else 0,
                    'like_share': data.engagement_metrics.get('total_likes', 0) / total_engagements if total_engagements > 0 else 0,
                    'reply_share': data.engagement_metrics.get('total_replies', 0) / total_engagements if total_engagements > 0 else 0
                },
                'social_optimization_score': data.data_quality_score
            }
        except Exception as e:
            logger.error(f"Error analyzing engagement optimization: {str(e)}")
            return {}
            
    def _analyze_x_attribution(self, data: XAdsData) -> Dict[str, Any]:
        """Analyze X-specific attribution patterns"""
        try:
            return {
                'social_attribution_score': data.total_engagements / data.total_impressions if data.total_impressions > 0 else 0,
                'x_platform_impact': {
                    'total_social_reach': data.total_impressions,
                    'social_engagement_rate': data.total_engagements / data.total_impressions if data.total_impressions > 0 else 0,
                    'cost_per_engagement': data.total_spend / data.total_engagements if data.total_engagements > 0 else 0,
                    'viral_amplification_factor': data.viral_metrics.get('viral_rate', 0)
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing X attribution: {str(e)}")
            return {}
            
    def _create_basic_causal_data(self, x_ads_data: XAdsData) -> CausalMarketingData:
        """Create basic causal marketing data without KSE enhancement"""
        return CausalMarketingData(
            source="x_ads",
            timestamp=x_ads_data.sync_timestamp,
            raw_data=x_ads_data.__dict__,
            causal_factors={
                'campaign_count': len(x_ads_data.campaigns),
                'total_spend': x_ads_data.total_spend,
                'social_reach': x_ads_data.total_impressions,
                'engagement_volume': x_ads_data.total_engagements
            },
            treatment_assignment={
                'social_advertising': 'active',
                'viral_optimization': 'active',
                'x_engagement': 'active'
            },
            outcome_metrics={
                'total_impressions': x_ads_data.total_impressions,
                'total_engagements': x_ads_data.total_engagements,
                'total_spend': x_ads_data.total_spend,
                'engagement_rate': x_ads_data.total_engagements / x_ads_data.total_impressions if x_ads_data.total_impressions > 0 else 0
            },
            data_quality=DataQualityAssessment(
                completeness=x_ads_data.data_quality_score,
                consistency=0.85,
                validity=0.90,
                timeliness=0.95,
                overall_score=x_ads_data.data_quality_score
            )
        )


async def create_x_ads_connector(credentials: Dict[str, str], kse_client: Optional[LiftKSEClient] = None) -> XAdsConnector:
    """Factory function to create X Ads connector"""
    x_credentials = XAdsCredentials(
        consumer_key=credentials['consumer_key'],
        consumer_secret=credentials['consumer_secret'],
        access_token=credentials['access_token'],
        access_token_secret=credentials['access_token_secret']
    )
    
    return XAdsConnector(x_credentials, kse_client)