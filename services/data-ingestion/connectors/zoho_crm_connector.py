"""
Zoho CRM Connector for LiftOS Data Ingestion Service
Integrates with Zoho CRM API v2 for comprehensive customer relationship management data
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
class ZohoCRMCredentials:
    """Zoho CRM API credentials"""
    client_id: str
    client_secret: str
    refresh_token: str
    access_token: Optional[str] = None
    domain: str = "com"  # com, eu, in, com.au, jp
    
    
@dataclass
class ZohoCRMData:
    """Zoho CRM data structure"""
    # Lead data
    leads: List[Dict[str, Any]]
    contacts: List[Dict[str, Any]]
    accounts: List[Dict[str, Any]]
    deals: List[Dict[str, Any]]
    
    # Activity data
    calls: List[Dict[str, Any]]
    meetings: List[Dict[str, Any]]
    tasks: List[Dict[str, Any]]
    emails: List[Dict[str, Any]]
    
    # Sales data
    quotes: List[Dict[str, Any]]
    sales_orders: List[Dict[str, Any]]
    invoices: List[Dict[str, Any]]
    
    # Analytics
    pipeline_analytics: Dict[str, Any]
    conversion_metrics: Dict[str, Any]
    
    # Metadata
    sync_timestamp: datetime
    total_records: int
    data_quality_score: float


class ZohoCRMConnector:
    """Zoho CRM API connector with causal analysis capabilities"""
    
    def __init__(self, credentials: ZohoCRMCredentials, kse_client: Optional[LiftKSEClient] = None):
        self.credentials = credentials
        self.kse_client = kse_client
        self.causal_transformer = CausalDataTransformer()
        self.base_url = f"https://www.zohoapis.{credentials.domain}/crm/v2"
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        await self._refresh_access_token()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            
    async def _refresh_access_token(self):
        """Refresh Zoho access token using refresh token"""
        try:
            refresh_url = f"https://accounts.zoho.{self.credentials.domain}/oauth/v2/token"
            
            data = {
                'refresh_token': self.credentials.refresh_token,
                'client_id': self.credentials.client_id,
                'client_secret': self.credentials.client_secret,
                'grant_type': 'refresh_token'
            }
            
            async with self.session.post(refresh_url, data=data) as response:
                if response.status == 200:
                    token_data = await response.json()
                    self.credentials.access_token = token_data['access_token']
                    logger.info("Zoho CRM access token refreshed successfully")
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to refresh token: {error_text}")
                    
        except Exception as e:
            logger.error(f"Error refreshing Zoho CRM token: {str(e)}")
            raise
            
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make authenticated request to Zoho CRM API"""
        if not self.credentials.access_token:
            await self._refresh_access_token()
            
        headers = {
            'Authorization': f'Zoho-oauthtoken {self.credentials.access_token}',
            'Content-Type': 'application/json'
        }
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 401:
                    # Token expired, refresh and retry
                    await self._refresh_access_token()
                    headers['Authorization'] = f'Zoho-oauthtoken {self.credentials.access_token}'
                    
                    async with self.session.get(url, headers=headers, params=params) as retry_response:
                        if retry_response.status == 200:
                            return await retry_response.json()
                        else:
                            error_text = await retry_response.text()
                            raise Exception(f"API request failed: {error_text}")
                            
                elif response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"API request failed: {error_text}")
                    
        except Exception as e:
            logger.error(f"Error making Zoho CRM API request to {endpoint}: {str(e)}")
            raise
            
    async def get_leads(self, date_from: datetime, date_to: datetime) -> List[Dict[str, Any]]:
        """Fetch leads data from Zoho CRM"""
        try:
            params = {
                'fields': 'all',
                'per_page': 200,
                'sort_by': 'Modified_Time',
                'sort_order': 'desc'
            }
            
            leads = []
            page = 1
            
            while True:
                params['page'] = page
                response = await self._make_request('Leads', params)
                
                if 'data' in response:
                    batch_leads = response['data']
                    
                    # Filter by date range
                    filtered_leads = []
                    for lead in batch_leads:
                        modified_time = datetime.fromisoformat(lead.get('Modified_Time', '').replace('Z', '+00:00'))
                        if date_from <= modified_time <= date_to:
                            filtered_leads.append(lead)
                            
                    leads.extend(filtered_leads)
                    
                    # Check if we have more pages
                    if len(batch_leads) < 200 or not response.get('info', {}).get('more_records'):
                        break
                        
                    page += 1
                else:
                    break
                    
            logger.info(f"Fetched {len(leads)} leads from Zoho CRM")
            return leads
            
        except Exception as e:
            logger.error(f"Error fetching Zoho CRM leads: {str(e)}")
            return []
            
    async def get_contacts(self, date_from: datetime, date_to: datetime) -> List[Dict[str, Any]]:
        """Fetch contacts data from Zoho CRM"""
        try:
            params = {
                'fields': 'all',
                'per_page': 200,
                'sort_by': 'Modified_Time',
                'sort_order': 'desc'
            }
            
            contacts = []
            page = 1
            
            while True:
                params['page'] = page
                response = await self._make_request('Contacts', params)
                
                if 'data' in response:
                    batch_contacts = response['data']
                    
                    # Filter by date range
                    filtered_contacts = []
                    for contact in batch_contacts:
                        modified_time = datetime.fromisoformat(contact.get('Modified_Time', '').replace('Z', '+00:00'))
                        if date_from <= modified_time <= date_to:
                            filtered_contacts.append(contact)
                            
                    contacts.extend(filtered_contacts)
                    
                    if len(batch_contacts) < 200 or not response.get('info', {}).get('more_records'):
                        break
                        
                    page += 1
                else:
                    break
                    
            logger.info(f"Fetched {len(contacts)} contacts from Zoho CRM")
            return contacts
            
        except Exception as e:
            logger.error(f"Error fetching Zoho CRM contacts: {str(e)}")
            return []
            
    async def get_deals(self, date_from: datetime, date_to: datetime) -> List[Dict[str, Any]]:
        """Fetch deals data from Zoho CRM"""
        try:
            params = {
                'fields': 'all',
                'per_page': 200,
                'sort_by': 'Modified_Time',
                'sort_order': 'desc'
            }
            
            deals = []
            page = 1
            
            while True:
                params['page'] = page
                response = await self._make_request('Deals', params)
                
                if 'data' in response:
                    batch_deals = response['data']
                    
                    # Filter by date range
                    filtered_deals = []
                    for deal in batch_deals:
                        modified_time = datetime.fromisoformat(deal.get('Modified_Time', '').replace('Z', '+00:00'))
                        if date_from <= modified_time <= date_to:
                            filtered_deals.append(deal)
                            
                    deals.extend(filtered_deals)
                    
                    if len(batch_deals) < 200 or not response.get('info', {}).get('more_records'):
                        break
                        
                    page += 1
                else:
                    break
                    
            logger.info(f"Fetched {len(deals)} deals from Zoho CRM")
            return deals
            
        except Exception as e:
            logger.error(f"Error fetching Zoho CRM deals: {str(e)}")
            return []
            
    async def get_activities(self, date_from: datetime, date_to: datetime) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch activities data from Zoho CRM"""
        activities = {
            'calls': [],
            'meetings': [],
            'tasks': [],
            'emails': []
        }
        
        try:
            for activity_type in activities.keys():
                params = {
                    'fields': 'all',
                    'per_page': 200,
                    'sort_by': 'Modified_Time',
                    'sort_order': 'desc'
                }
                
                page = 1
                while True:
                    params['page'] = page
                    endpoint = activity_type.capitalize()
                    response = await self._make_request(endpoint, params)
                    
                    if 'data' in response:
                        batch_activities = response['data']
                        
                        # Filter by date range
                        filtered_activities = []
                        for activity in batch_activities:
                            modified_time = datetime.fromisoformat(activity.get('Modified_Time', '').replace('Z', '+00:00'))
                            if date_from <= modified_time <= date_to:
                                filtered_activities.append(activity)
                                
                        activities[activity_type].extend(filtered_activities)
                        
                        if len(batch_activities) < 200 or not response.get('info', {}).get('more_records'):
                            break
                            
                        page += 1
                    else:
                        break
                        
            logger.info(f"Fetched activities from Zoho CRM: {sum(len(v) for v in activities.values())} total")
            return activities
            
        except Exception as e:
            logger.error(f"Error fetching Zoho CRM activities: {str(e)}")
            return activities
            
    async def get_analytics_data(self) -> Dict[str, Any]:
        """Fetch analytics and pipeline data from Zoho CRM"""
        try:
            # Get pipeline analytics
            pipeline_response = await self._make_request('settings/pipeline')
            
            # Get conversion metrics (custom analytics)
            analytics_data = {
                'pipeline_stages': pipeline_response.get('pipeline', []),
                'conversion_metrics': {
                    'lead_to_contact_rate': 0.0,
                    'contact_to_deal_rate': 0.0,
                    'deal_close_rate': 0.0,
                    'average_deal_size': 0.0,
                    'sales_cycle_length': 0.0
                }
            }
            
            logger.info("Fetched Zoho CRM analytics data")
            return analytics_data
            
        except Exception as e:
            logger.error(f"Error fetching Zoho CRM analytics: {str(e)}")
            return {}
            
    async def sync_data(self, date_from: datetime, date_to: datetime) -> ZohoCRMData:
        """Sync comprehensive CRM data from Zoho"""
        try:
            logger.info(f"Starting Zoho CRM data sync from {date_from} to {date_to}")
            
            # Fetch all data concurrently
            leads_task = self.get_leads(date_from, date_to)
            contacts_task = self.get_contacts(date_from, date_to)
            deals_task = self.get_deals(date_from, date_to)
            activities_task = self.get_activities(date_from, date_to)
            analytics_task = self.get_analytics_data()
            
            # Wait for all tasks to complete
            leads, contacts, deals, activities, analytics = await asyncio.gather(
                leads_task, contacts_task, deals_task, activities_task, analytics_task
            )
            
            # Calculate total records and data quality
            total_records = len(leads) + len(contacts) + len(deals) + sum(len(v) for v in activities.values())
            data_quality_score = self._calculate_data_quality(leads, contacts, deals)
            
            zoho_data = ZohoCRMData(
                leads=leads,
                contacts=contacts,
                accounts=[],  # Can be extended
                deals=deals,
                calls=activities.get('calls', []),
                meetings=activities.get('meetings', []),
                tasks=activities.get('tasks', []),
                emails=activities.get('emails', []),
                quotes=[],  # Can be extended
                sales_orders=[],  # Can be extended
                invoices=[],  # Can be extended
                pipeline_analytics=analytics.get('pipeline_stages', []),
                conversion_metrics=analytics.get('conversion_metrics', {}),
                sync_timestamp=datetime.utcnow(),
                total_records=total_records,
                data_quality_score=data_quality_score
            )
            
            logger.info(f"Zoho CRM sync completed: {total_records} records, quality score: {data_quality_score:.2f}")
            return zoho_data
            
        except Exception as e:
            logger.error(f"Error syncing Zoho CRM data: {str(e)}")
            raise
            
    def _calculate_data_quality(self, leads: List[Dict], contacts: List[Dict], deals: List[Dict]) -> float:
        """Calculate data quality score for CRM data"""
        try:
            total_records = len(leads) + len(contacts) + len(deals)
            if total_records == 0:
                return 0.0
                
            quality_scores = []
            
            # Check leads quality
            for lead in leads:
                score = 0.0
                if lead.get('Email'):
                    score += 0.3
                if lead.get('Phone'):
                    score += 0.2
                if lead.get('Company'):
                    score += 0.2
                if lead.get('Lead_Status'):
                    score += 0.3
                quality_scores.append(score)
                
            # Check contacts quality
            for contact in contacts:
                score = 0.0
                if contact.get('Email'):
                    score += 0.3
                if contact.get('Phone'):
                    score += 0.2
                if contact.get('Account_Name'):
                    score += 0.2
                if contact.get('Title'):
                    score += 0.3
                quality_scores.append(score)
                
            # Check deals quality
            for deal in deals:
                score = 0.0
                if deal.get('Amount'):
                    score += 0.4
                if deal.get('Stage'):
                    score += 0.3
                if deal.get('Closing_Date'):
                    score += 0.3
                quality_scores.append(score)
                
            return sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating Zoho CRM data quality: {str(e)}")
            return 0.0
            
    async def enhance_with_kse(self, zoho_data: ZohoCRMData, context: str) -> CausalMarketingData:
        """Enhance Zoho CRM data with KSE causal analysis"""
        try:
            if not self.kse_client:
                logger.warning("KSE client not available for Zoho CRM enhancement")
                return self._create_basic_causal_data(zoho_data)
                
            # Prepare CRM data for causal analysis
            crm_insights = {
                'lead_conversion_patterns': self._analyze_lead_conversion(zoho_data),
                'deal_progression_analysis': self._analyze_deal_progression(zoho_data),
                'activity_effectiveness': self._analyze_activity_effectiveness(zoho_data),
                'pipeline_optimization': self._analyze_pipeline_optimization(zoho_data)
            }
            
            # Store in KSE for causal memory
            kse_entry = {
                'source': 'zoho_crm',
                'timestamp': zoho_data.sync_timestamp.isoformat(),
                'data_type': 'crm_analytics',
                'insights': crm_insights,
                'context': context,
                'total_records': zoho_data.total_records,
                'data_quality_score': zoho_data.data_quality_score
            }
            
            await self.kse_client.store_causal_memory(
                memory_id=f"zoho_crm_{context}_{int(zoho_data.sync_timestamp.timestamp())}",
                content=kse_entry,
                context=context
            )
            
            # Transform to causal marketing data
            causal_data = await self.causal_transformer.transform_crm_data(
                zoho_data, crm_insights
            )
            
            logger.info("Zoho CRM data enhanced with KSE causal analysis")
            return causal_data
            
        except Exception as e:
            logger.error(f"Error enhancing Zoho CRM data with KSE: {str(e)}")
            return self._create_basic_causal_data(zoho_data)
            
    def _analyze_lead_conversion(self, data: ZohoCRMData) -> Dict[str, Any]:
        """Analyze lead conversion patterns"""
        try:
            lead_stages = {}
            for lead in data.leads:
                stage = lead.get('Lead_Status', 'Unknown')
                lead_stages[stage] = lead_stages.get(stage, 0) + 1
                
            return {
                'stage_distribution': lead_stages,
                'total_leads': len(data.leads),
                'conversion_opportunities': len([l for l in data.leads if l.get('Lead_Status') in ['Qualified', 'Hot']])
            }
        except Exception as e:
            logger.error(f"Error analyzing lead conversion: {str(e)}")
            return {}
            
    def _analyze_deal_progression(self, data: ZohoCRMData) -> Dict[str, Any]:
        """Analyze deal progression patterns"""
        try:
            deal_stages = {}
            total_value = 0
            
            for deal in data.deals:
                stage = deal.get('Stage', 'Unknown')
                deal_stages[stage] = deal_stages.get(stage, 0) + 1
                
                amount = deal.get('Amount', 0)
                if isinstance(amount, (int, float)):
                    total_value += amount
                    
            return {
                'stage_distribution': deal_stages,
                'total_deals': len(data.deals),
                'total_pipeline_value': total_value,
                'average_deal_size': total_value / len(data.deals) if data.deals else 0
            }
        except Exception as e:
            logger.error(f"Error analyzing deal progression: {str(e)}")
            return {}
            
    def _analyze_activity_effectiveness(self, data: ZohoCRMData) -> Dict[str, Any]:
        """Analyze activity effectiveness"""
        try:
            activity_counts = {
                'calls': len(data.calls),
                'meetings': len(data.meetings),
                'tasks': len(data.tasks),
                'emails': len(data.emails)
            }
            
            return {
                'activity_distribution': activity_counts,
                'total_activities': sum(activity_counts.values()),
                'activity_per_deal': sum(activity_counts.values()) / len(data.deals) if data.deals else 0
            }
        except Exception as e:
            logger.error(f"Error analyzing activity effectiveness: {str(e)}")
            return {}
            
    def _analyze_pipeline_optimization(self, data: ZohoCRMData) -> Dict[str, Any]:
        """Analyze pipeline optimization opportunities"""
        try:
            return {
                'pipeline_health_score': data.data_quality_score,
                'conversion_metrics': data.conversion_metrics,
                'optimization_recommendations': [
                    'Improve lead qualification process',
                    'Increase activity frequency for stalled deals',
                    'Implement automated follow-up sequences'
                ]
            }
        except Exception as e:
            logger.error(f"Error analyzing pipeline optimization: {str(e)}")
            return {}
            
    def _create_basic_causal_data(self, zoho_data: ZohoCRMData) -> CausalMarketingData:
        """Create basic causal marketing data without KSE enhancement"""
        return CausalMarketingData(
            source="zoho_crm",
            timestamp=zoho_data.sync_timestamp,
            raw_data=zoho_data.__dict__,
            causal_factors={
                'lead_volume': len(zoho_data.leads),
                'deal_volume': len(zoho_data.deals),
                'activity_volume': len(zoho_data.calls) + len(zoho_data.meetings)
            },
            treatment_assignment={
                'crm_optimization': 'active',
                'lead_nurturing': 'active',
                'pipeline_management': 'active'
            },
            outcome_metrics={
                'total_records': zoho_data.total_records,
                'data_quality_score': zoho_data.data_quality_score,
                'pipeline_value': sum(deal.get('Amount', 0) for deal in zoho_data.deals if isinstance(deal.get('Amount'), (int, float)))
            },
            data_quality=DataQualityAssessment(
                completeness=zoho_data.data_quality_score,
                consistency=0.85,
                validity=0.90,
                timeliness=0.95,
                overall_score=zoho_data.data_quality_score
            )
        )


async def create_zoho_crm_connector(credentials: Dict[str, str], kse_client: Optional[LiftKSEClient] = None) -> ZohoCRMConnector:
    """Factory function to create Zoho CRM connector"""
    zoho_credentials = ZohoCRMCredentials(
        client_id=credentials['client_id'],
        client_secret=credentials['client_secret'],
        refresh_token=credentials['refresh_token'],
        domain=credentials.get('domain', 'com')
    )
    
    return ZohoCRMConnector(zoho_credentials, kse_client)