"""
HubSpot CRM Connector with Advanced Attribution Analysis
Tier 2 API Connector for LiftOS v1.3.0

Features:
- Complete HubSpot CRM API integration (Contacts, Deals, Companies, Activities)
- Lead scoring and lifecycle stage attribution
- Marketing qualified lead (MQL) to customer attribution
- Sales pipeline velocity analysis
- Contact engagement scoring and attribution
- Advanced CRM treatment assignment logic
- Full KSE Universal Substrate integration
"""
import asyncio
import httpx
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import json
import logging

# Import shared modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from shared.models.causal_marketing import (
    CausalMarketingData, TreatmentType, RandomizationUnit,
    ConfounderVariable
)
from shared.kse_sdk.causal_models import CausalMemoryEntry, CausalRelationship
from shared.utils.causal_transforms import CausalDataTransformer, TreatmentAssignmentResult
from shared.kse_sdk.client import LiftKSEClient

logger = logging.getLogger(__name__)


class HubSpotLifecycleStage(str, Enum):
    """HubSpot contact lifecycle stages"""
    SUBSCRIBER = "subscriber"
    LEAD = "lead"
    MARKETING_QUALIFIED_LEAD = "marketingqualifiedlead"
    SALES_QUALIFIED_LEAD = "salesqualifiedlead"
    OPPORTUNITY = "opportunity"
    CUSTOMER = "customer"
    EVANGELIST = "evangelist"
    OTHER = "other"


class HubSpotDealStage(str, Enum):
    """HubSpot deal pipeline stages"""
    APPOINTMENT_SCHEDULED = "appointmentscheduled"
    QUALIFIED_TO_BUY = "qualifiedtobuy"
    PRESENTATION_SCHEDULED = "presentationscheduled"
    DECISION_MAKER_BOUGHT_IN = "decisionmakerboughtin"
    CONTRACT_SENT = "contractsent"
    CLOSED_WON = "closedwon"
    CLOSED_LOST = "closedlost"


class HubSpotContactData(BaseModel):
    """HubSpot contact data model"""
    id: str
    email: Optional[str] = None
    firstname: Optional[str] = None
    lastname: Optional[str] = None
    company: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    lifecyclestage: Optional[HubSpotLifecycleStage] = None
    lead_status: Optional[str] = None
    hs_lead_status: Optional[str] = None
    hubspotscore: Optional[float] = None
    createdate: Optional[datetime] = None
    lastmodifieddate: Optional[datetime] = None
    hs_analytics_source: Optional[str] = None
    hs_analytics_source_data_1: Optional[str] = None
    hs_analytics_source_data_2: Optional[str] = None
    hs_latest_source: Optional[str] = None
    hs_latest_source_data_1: Optional[str] = None
    hs_latest_source_data_2: Optional[str] = None
    first_conversion_event_name: Optional[str] = None
    first_conversion_date: Optional[datetime] = None
    recent_conversion_event_name: Optional[str] = None
    recent_conversion_date: Optional[datetime] = None
    num_conversion_events: Optional[int] = 0
    hs_email_open_count: Optional[int] = 0
    hs_email_click_count: Optional[int] = 0
    hs_email_bounce_count: Optional[int] = 0
    hs_social_num_broadcast_clicks: Optional[int] = 0
    hs_time_to_move_from_lead_to_customer: Optional[float] = None
    hs_time_to_move_from_mql_to_customer: Optional[float] = None
    hs_time_to_move_from_sql_to_customer: Optional[float] = None


class HubSpotDealData(BaseModel):
    """HubSpot deal data model"""
    id: str
    dealname: Optional[str] = None
    amount: Optional[float] = None
    dealstage: Optional[HubSpotDealStage] = None
    pipeline: Optional[str] = None
    closedate: Optional[datetime] = None
    createdate: Optional[datetime] = None
    hs_lastmodifieddate: Optional[datetime] = None
    hubspot_owner_id: Optional[str] = None
    dealtype: Optional[str] = None
    hs_deal_stage_probability: Optional[float] = None
    hs_analytics_source: Optional[str] = None
    hs_analytics_source_data_1: Optional[str] = None
    hs_analytics_source_data_2: Optional[str] = None
    hs_campaign: Optional[str] = None
    hs_deal_amount_calculation_preference: Optional[str] = None
    hs_time_in_dealstage: Optional[float] = None
    days_to_close: Optional[int] = None
    num_contacted_notes: Optional[int] = 0
    num_notes: Optional[int] = 0
    hs_num_times_contacted: Optional[int] = 0
    associated_contact_ids: List[str] = Field(default_factory=list)
    associated_company_ids: List[str] = Field(default_factory=list)


class HubSpotCompanyData(BaseModel):
    """HubSpot company data model"""
    id: str
    name: Optional[str] = None
    domain: Optional[str] = None
    industry: Optional[str] = None
    annualrevenue: Optional[float] = None
    numberofemployees: Optional[int] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    createdate: Optional[datetime] = None
    hs_lastmodifieddate: Optional[datetime] = None
    lifecyclestage: Optional[str] = None
    hs_lead_status: Optional[str] = None
    hubspotscore: Optional[float] = None
    total_money_raised: Optional[float] = None
    hs_analytics_source: Optional[str] = None
    hs_analytics_source_data_1: Optional[str] = None
    hs_num_child_companies: Optional[int] = 0
    associated_contact_ids: List[str] = Field(default_factory=list)
    associated_deal_ids: List[str] = Field(default_factory=list)


class HubSpotActivityData(BaseModel):
    """HubSpot engagement/activity data model"""
    id: str
    engagement_type: str  # EMAIL, CALL, MEETING, TASK, NOTE
    timestamp: datetime
    contact_id: Optional[str] = None
    deal_id: Optional[str] = None
    company_id: Optional[str] = None
    owner_id: Optional[str] = None
    subject: Optional[str] = None
    body: Optional[str] = None
    duration: Optional[int] = None  # in milliseconds
    email_status: Optional[str] = None  # SENT, DELIVERED, OPENED, CLICKED, BOUNCED
    call_disposition: Optional[str] = None
    meeting_outcome: Optional[str] = None
    source: Optional[str] = None
    source_id: Optional[str] = None


class HubSpotConnector:
    """HubSpot CRM API connector with advanced attribution analysis"""
    
    def __init__(self, access_token: str, portal_id: str):
        self.access_token = access_token
        self.portal_id = portal_id
        self.base_url = "https://api.hubapi.com"
        self.client = httpx.AsyncClient(timeout=30.0)
        self.rate_limit_delay = 0.1  # 100ms between requests (600 requests/minute)
    
    async def extract_causal_marketing_data(
        self,
        org_id: str,
        start_date: date,
        end_date: date,
        historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> List[CausalMarketingData]:
        """Extract causal marketing data from HubSpot CRM with advanced attribution"""
        
        logger.info(f"Starting HubSpot CRM data extraction for org {org_id}")
        
        try:
            # Get CRM data
            contacts = await self._get_contacts(start_date, end_date)
            deals = await self._get_deals(start_date, end_date)
            companies = await self._get_companies(start_date, end_date)
            activities = await self._get_activities(start_date, end_date)
            
            logger.info(f"Retrieved {len(contacts)} contacts, {len(deals)} deals, {len(companies)} companies, {len(activities)} activities")
            
            # Process deals as primary conversion events
            causal_data_list = []
            
            for deal in deals:
                try:
                    # Get associated contacts and companies
                    associated_contacts = [c for c in contacts if c.id in deal.associated_contact_ids]
                    associated_companies = [c for c in companies if c.id in deal.associated_company_ids]
                    related_activities = [a for a in activities if a.deal_id == deal.id]
                    
                    # Create causal marketing data
                    causal_data = await self._create_causal_marketing_data(
                        deal, associated_contacts, associated_companies, 
                        related_activities, org_id, historical_data
                    )
                    
                    # Enhance with KSE substrate
                    await self._enhance_with_kse_substrate(
                        causal_data, deal, associated_contacts, 
                        associated_companies, related_activities, org_id
                    )
                    
                    causal_data_list.append(causal_data)
                    
                except Exception as e:
                    logger.error(f"Failed to process deal {deal.id}: {str(e)}")
                    continue
            
            # Process MQL conversions (contacts that became MQLs)
            for contact in contacts:
                if (contact.lifecyclestage == HubSpotLifecycleStage.MARKETING_QUALIFIED_LEAD and
                    contact.recent_conversion_date and
                    start_date <= contact.recent_conversion_date.date() <= end_date):
                    
                    try:
                        # Create MQL conversion causal data
                        mql_causal_data = await self._create_mql_causal_data(
                            contact, companies, activities, org_id, historical_data
                        )
                        
                        # Enhance with KSE substrate
                        await self._enhance_mql_with_kse_substrate(
                            mql_causal_data, contact, companies, activities, org_id
                        )
                        
                        causal_data_list.append(mql_causal_data)
                        
                    except Exception as e:
                        logger.error(f"Failed to process MQL contact {contact.id}: {str(e)}")
                        continue
            
            logger.info(f"Generated {len(causal_data_list)} causal marketing records from HubSpot")
            return causal_data_list
            
        except Exception as e:
            logger.error(f"HubSpot data extraction failed: {str(e)}")
            raise
    
    async def _get_contacts(self, start_date: date, end_date: date) -> List[HubSpotContactData]:
        """Get contacts from HubSpot API"""
        contacts = []
        after = None
        
        while True:
            try:
                url = f"{self.base_url}/crm/v3/objects/contacts"
                params = {
                    "limit": 100,
                    "properties": [
                        "email", "firstname", "lastname", "company", "phone", "website",
                        "lifecyclestage", "lead_status", "hs_lead_status", "hubspotscore",
                        "createdate", "lastmodifieddate", "hs_analytics_source",
                        "hs_analytics_source_data_1", "hs_analytics_source_data_2",
                        "hs_latest_source", "hs_latest_source_data_1", "hs_latest_source_data_2",
                        "first_conversion_event_name", "first_conversion_date",
                        "recent_conversion_event_name", "recent_conversion_date",
                        "num_conversion_events", "hs_email_open_count", "hs_email_click_count",
                        "hs_email_bounce_count", "hs_social_num_broadcast_clicks",
                        "hs_time_to_move_from_lead_to_customer", "hs_time_to_move_from_mql_to_customer",
                        "hs_time_to_move_from_sql_to_customer"
                    ]
                }
                
                if after:
                    params["after"] = after
                
                headers = {
                    "Authorization": f"Bearer {self.access_token}",
                    "Content-Type": "application/json"
                }
                
                response = await self.client.get(url, params=params, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                
                for contact_data in data.get("results", []):
                    try:
                        # Parse contact properties
                        properties = contact_data.get("properties", {})
                        
                        # Convert date strings to datetime objects
                        for date_field in ["createdate", "lastmodifieddate", "first_conversion_date", "recent_conversion_date"]:
                            if properties.get(date_field):
                                try:
                                    properties[date_field] = datetime.fromisoformat(properties[date_field].replace('Z', '+00:00'))
                                except:
                                    properties[date_field] = None
                        
                        # Convert numeric fields
                        for numeric_field in ["hubspotscore", "num_conversion_events", "hs_email_open_count", 
                                            "hs_email_click_count", "hs_email_bounce_count", "hs_social_num_broadcast_clicks",
                                            "hs_time_to_move_from_lead_to_customer", "hs_time_to_move_from_mql_to_customer",
                                            "hs_time_to_move_from_sql_to_customer"]:
                            if properties.get(numeric_field):
                                try:
                                    properties[numeric_field] = float(properties[numeric_field])
                                except:
                                    properties[numeric_field] = None
                        
                        contact = HubSpotContactData(
                            id=contact_data["id"],
                            **properties
                        )
                        
                        # Filter by date range
                        if (contact.createdate and 
                            start_date <= contact.createdate.date() <= end_date):
                            contacts.append(contact)
                        elif (contact.recent_conversion_date and
                              start_date <= contact.recent_conversion_date.date() <= end_date):
                            contacts.append(contact)
                            
                    except Exception as e:
                        logger.warning(f"Failed to parse contact {contact_data.get('id', 'unknown')}: {str(e)}")
                        continue
                
                # Check for pagination
                paging = data.get("paging", {})
                if paging.get("next"):
                    after = paging["next"]["after"]
                else:
                    break
                
                # Rate limiting
                await asyncio.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.error(f"Failed to get contacts: {str(e)}")
                break
        
        return contacts
    
    async def _get_deals(self, start_date: date, end_date: date) -> List[HubSpotDealData]:
        """Get deals from HubSpot API"""
        deals = []
        after = None
        
        while True:
            try:
                url = f"{self.base_url}/crm/v3/objects/deals"
                params = {
                    "limit": 100,
                    "properties": [
                        "dealname", "amount", "dealstage", "pipeline", "closedate",
                        "createdate", "hs_lastmodifieddate", "hubspot_owner_id", "dealtype",
                        "hs_deal_stage_probability", "hs_analytics_source", "hs_analytics_source_data_1",
                        "hs_analytics_source_data_2", "hs_campaign", "hs_deal_amount_calculation_preference",
                        "hs_time_in_dealstage", "days_to_close", "num_contacted_notes",
                        "num_notes", "hs_num_times_contacted"
                    ],
                    "associations": ["contacts", "companies"]
                }
                
                if after:
                    params["after"] = after
                
                headers = {
                    "Authorization": f"Bearer {self.access_token}",
                    "Content-Type": "application/json"
                }
                
                response = await self.client.get(url, params=params, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                
                for deal_data in data.get("results", []):
                    try:
                        # Parse deal properties
                        properties = deal_data.get("properties", {})
                        
                        # Convert date strings to datetime objects
                        for date_field in ["closedate", "createdate", "hs_lastmodifieddate"]:
                            if properties.get(date_field):
                                try:
                                    properties[date_field] = datetime.fromisoformat(properties[date_field].replace('Z', '+00:00'))
                                except:
                                    properties[date_field] = None
                        
                        # Convert numeric fields
                        for numeric_field in ["amount", "hs_deal_stage_probability", "hs_time_in_dealstage", 
                                            "days_to_close", "num_contacted_notes", "num_notes", "hs_num_times_contacted"]:
                            if properties.get(numeric_field):
                                try:
                                    properties[numeric_field] = float(properties[numeric_field])
                                except:
                                    properties[numeric_field] = None
                        
                        # Get associated contact and company IDs
                        associations = deal_data.get("associations", {})
                        contact_ids = []
                        company_ids = []
                        
                        if "contacts" in associations:
                            contact_ids = [assoc["id"] for assoc in associations["contacts"]["results"]]
                        if "companies" in associations:
                            company_ids = [assoc["id"] for assoc in associations["companies"]["results"]]
                        
                        deal = HubSpotDealData(
                            id=deal_data["id"],
                            associated_contact_ids=contact_ids,
                            associated_company_ids=company_ids,
                            **properties
                        )
                        
                        # Filter by date range (created or closed in range)
                        if ((deal.createdate and start_date <= deal.createdate.date() <= end_date) or
                            (deal.closedate and start_date <= deal.closedate.date() <= end_date)):
                            deals.append(deal)
                            
                    except Exception as e:
                        logger.warning(f"Failed to parse deal {deal_data.get('id', 'unknown')}: {str(e)}")
                        continue
                
                # Check for pagination
                paging = data.get("paging", {})
                if paging.get("next"):
                    after = paging["next"]["after"]
                else:
                    break
                
                # Rate limiting
                await asyncio.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.error(f"Failed to get deals: {str(e)}")
                break
        
        return deals
    
    async def _get_companies(self, start_date: date, end_date: date) -> List[HubSpotCompanyData]:
        """Get companies from HubSpot API"""
        companies = []
        after = None
        
        while True:
            try:
                url = f"{self.base_url}/crm/v3/objects/companies"
                params = {
                    "limit": 100,
                    "properties": [
                        "name", "domain", "industry", "annualrevenue", "numberofemployees",
                        "city", "state", "country", "createdate", "hs_lastmodifieddate",
                        "lifecyclestage", "hs_lead_status", "hubspotscore", "total_money_raised",
                        "hs_analytics_source", "hs_analytics_source_data_1", "hs_num_child_companies"
                    ],
                    "associations": ["contacts", "deals"]
                }
                
                if after:
                    params["after"] = after
                
                headers = {
                    "Authorization": f"Bearer {self.access_token}",
                    "Content-Type": "application/json"
                }
                
                response = await self.client.get(url, params=params, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                
                for company_data in data.get("results", []):
                    try:
                        # Parse company properties
                        properties = company_data.get("properties", {})
                        
                        # Convert date strings to datetime objects
                        for date_field in ["createdate", "hs_lastmodifieddate"]:
                            if properties.get(date_field):
                                try:
                                    properties[date_field] = datetime.fromisoformat(properties[date_field].replace('Z', '+00:00'))
                                except:
                                    properties[date_field] = None
                        
                        # Convert numeric fields
                        for numeric_field in ["annualrevenue", "numberofemployees", "hubspotscore", 
                                            "total_money_raised", "hs_num_child_companies"]:
                            if properties.get(numeric_field):
                                try:
                                    properties[numeric_field] = float(properties[numeric_field])
                                except:
                                    properties[numeric_field] = None
                        
                        # Get associated contact and deal IDs
                        associations = company_data.get("associations", {})
                        contact_ids = []
                        deal_ids = []
                        
                        if "contacts" in associations:
                            contact_ids = [assoc["id"] for assoc in associations["contacts"]["results"]]
                        if "deals" in associations:
                            deal_ids = [assoc["id"] for assoc in associations["deals"]["results"]]
                        
                        company = HubSpotCompanyData(
                            id=company_data["id"],
                            associated_contact_ids=contact_ids,
                            associated_deal_ids=deal_ids,
                            **properties
                        )
                        
                        # Filter by date range
                        if (company.createdate and 
                            start_date <= company.createdate.date() <= end_date):
                            companies.append(company)
                            
                    except Exception as e:
                        logger.warning(f"Failed to parse company {company_data.get('id', 'unknown')}: {str(e)}")
                        continue
                
                # Check for pagination
                paging = data.get("paging", {})
                if paging.get("next"):
                    after = paging["next"]["after"]
                else:
                    break
                
                # Rate limiting
                await asyncio.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.error(f"Failed to get companies: {str(e)}")
                break
        
        return companies
    
    async def _get_activities(self, start_date: date, end_date: date) -> List[HubSpotActivityData]:
        """Get engagement activities from HubSpot API"""
        activities = []
        
        # Get different types of engagements
        engagement_types = ["emails", "calls", "meetings", "tasks", "notes"]
        
        for engagement_type in engagement_types:
            try:
                after = None
                
                while True:
                    url = f"{self.base_url}/crm/v3/objects/{engagement_type}"
                    params = {
                        "limit": 100,
                        "properties": [
                            "hs_timestamp", "hubspot_owner_id", "hs_engagement_source",
                            "hs_engagement_source_id"
                        ]
                    }
                    
                    if after:
                        params["after"] = after
                    
                    # Add engagement-specific properties
                    if engagement_type == "emails":
                        params["properties"].extend(["hs_email_subject", "hs_email_text", "hs_email_status"])
                    elif engagement_type == "calls":
                        params["properties"].extend(["hs_call_title", "hs_call_body", "hs_call_duration", "hs_call_disposition"])
                    elif engagement_type == "meetings":
                        params["properties"].extend(["hs_meeting_title", "hs_meeting_body", "hs_meeting_outcome"])
                    elif engagement_type == "tasks":
                        params["properties"].extend(["hs_task_subject", "hs_task_body"])
                    elif engagement_type == "notes":
                        params["properties"].extend(["hs_note_body"])
                    
                    headers = {
                        "Authorization": f"Bearer {self.access_token}",
                        "Content-Type": "application/json"
                    }
                    
                    response = await self.client.get(url, params=params, headers=headers)
                    response.raise_for_status()
                    
                    data = response.json()
                    
                    for activity_data in data.get("results", []):
                        try:
                            properties = activity_data.get("properties", {})
                            
                            # Parse timestamp
                            timestamp = None
                            if properties.get("hs_timestamp"):
                                try:
                                    timestamp = datetime.fromisoformat(properties["hs_timestamp"].replace('Z', '+00:00'))
                                except:
                                    continue
                            
                            if not timestamp:
                                continue
                            
                            # Filter by date range
                            if not (start_date <= timestamp.date() <= end_date):
                                continue
                            
                            # Extract engagement-specific data
                            subject = None
                            body = None
                            duration = None
                            email_status = None
                            call_disposition = None
                            meeting_outcome = None
                            
                            if engagement_type == "emails":
                                subject = properties.get("hs_email_subject")
                                body = properties.get("hs_email_text")
                                email_status = properties.get("hs_email_status")
                            elif engagement_type == "calls":
                                subject = properties.get("hs_call_title")
                                body = properties.get("hs_call_body")
                                duration = properties.get("hs_call_duration")
                                call_disposition = properties.get("hs_call_disposition")
                            elif engagement_type == "meetings":
                                subject = properties.get("hs_meeting_title")
                                body = properties.get("hs_meeting_body")
                                meeting_outcome = properties.get("hs_meeting_outcome")
                            elif engagement_type == "tasks":
                                subject = properties.get("hs_task_subject")
                                body = properties.get("hs_task_body")
                            elif engagement_type == "notes":
                                body = properties.get("hs_note_body")
                            
                            activity = HubSpotActivityData(
                                id=activity_data["id"],
                                engagement_type=engagement_type.upper().rstrip('S'),  # Remove 'S' and uppercase
                                timestamp=timestamp,
                                owner_id=properties.get("hubspot_owner_id"),
                                subject=subject,
                                body=body,
                                duration=int(duration) if duration else None,
                                email_status=email_status,
                                call_disposition=call_disposition,
                                meeting_outcome=meeting_outcome,
                                source=properties.get("hs_engagement_source"),
                                source_id=properties.get("hs_engagement_source_id")
                            )
                            
                            activities.append(activity)
                            
                        except Exception as e:
                            logger.warning(f"Failed to parse {engagement_type} activity {activity_data.get('id', 'unknown')}: {str(e)}")
                            continue
                    
                    # Check for pagination
                    paging = data.get("paging", {})
                    if paging.get("next"):
                        after = paging["next"]["after"]
                    else:
                        break
                    
                    # Rate limiting
                    await asyncio.sleep(self.rate_limit_delay)
                    
            except Exception as e:
                logger.error(f"Failed to get {engagement_type} activities: {str(e)}")
                continue
        
        return activities
    
    async def _create_causal_marketing_data(
        self,
        deal: HubSpotDealData,
        contacts: List[HubSpotContactData],
        companies: List[HubSpotCompanyData],
        activities: List[HubSpotActivityData],
        org_id: str,
        historical_data: Optional[List[Dict[str, Any]]]
    ) -> CausalMarketingData:
        """Create causal marketing data from HubSpot deal"""
        
        # Determine treatment assignment
        treatment_result = await self._determine_treatment_assignment(
            deal, contacts, companies, activities, historical_data
        )
        
        # Detect confounders
        confounders = await self._detect_hubspot_confounders(
            deal, contacts, companies, activities, historical_data
        )
        
        # Calculate data quality score
        data_quality_score = self._calculate_data_quality_score(deal, contacts, companies, activities)
        
        # Extract campaign information
        campaign_id, campaign_name = self._determine_campaign_info(deal, contacts)
        
        # Extract geographic and audience data
        geographic_data = self._extract_geographic_data(contacts, companies)
        audience_data = self._extract_audience_data(deal, contacts, companies, activities)
        
        # Create causal marketing data
        causal_data = CausalMarketingData(
            record_id=deal.id,
            org_id=org_id,
            data_source="hubspot",
            timestamp=deal.closedate or deal.createdate or datetime.now(),
            treatment_type=treatment_result.treatment_type,
            treatment_group=treatment_result.treatment_group,
            treatment_intensity=treatment_result.treatment_intensity,
            randomization_unit=treatment_result.randomization_unit,
            experiment_id=treatment_result.experiment_id,
            outcome_value=deal.amount or 0.0,
            outcome_type="deal_value",
            confounders=confounders,
            data_quality_score=data_quality_score,
            causal_metadata={
                "platform": "hubspot",
                "deal_stage": deal.dealstage.value if deal.dealstage else None,
                "pipeline": deal.pipeline,
                "deal_type": deal.dealtype,
                "stage_probability": deal.hs_deal_stage_probability,
                "days_to_close": deal.days_to_close,
                "num_contacts": len(contacts),
                "num_companies": len(companies),
                "num_activities": len(activities),
                "campaign_id": campaign_id,
                "campaign_name": campaign_name,
                "analytics_source": deal.hs_analytics_source,
                "analytics_source_data_1": deal.hs_analytics_source_data_1,
                "analytics_source_data_2": deal.hs_analytics_source_data_2,
                "hubspot_campaign": deal.hs_campaign,
                "time_in_stage": deal.hs_time_in_dealstage,
                "num_times_contacted": deal.hs_num_times_contacted,
                "crm_attribution": True,
                "lead_scoring_enabled": any(c.hubspotscore for c in contacts),
                "lifecycle_stage_tracking": True
            },
            geographic_data=geographic_data,
            audience_data=audience_data,
            platform_context={
                "source": "hubspot",
                "deal_id": deal.id,
                "deal_name": deal.dealname,
                "owner_id": deal.hubspot_owner_id,
                "associated_contacts": [c.id for c in contacts],
                "associated_companies": [c.id for c in companies],
                "primary_contact": contacts[0].id if contacts else None,
                "primary_company": companies[0].id if companies else None,
                "deal_analytics": {
                    "source": deal.hs_analytics_source,
                    "source_data_1": deal.hs_analytics_source_data_1,
                    "source_data_2": deal.hs_analytics_source_data_2,
                    "campaign": deal.hs_campaign
                },
                "engagement_summary": {
                    "total_activities": len(activities),
                    "emails": len([a for a in activities if a.engagement_type == "EMAIL"]),
                    "calls": len([a for a in activities if a.engagement_type == "CALL"]),
                    "meetings": len([a for a in activities if a.engagement_type == "MEETING"]),
                    "notes": len([a for a in activities if a.engagement_type == "NOTE"])
                }
            }
        )
        
        return causal_data
    
    async def _create_mql_causal_data(
        self,
        contact: HubSpotContactData,
        companies: List[HubSpotCompanyData],
        activities: List[HubSpotActivityData],
        org_id: str,
        historical_data: Optional[List[Dict[str, Any]]]
    ) -> CausalMarketingData:
        """Create causal marketing data for MQL conversion"""
        
        # Find associated company
        associated_companies = [c for c in companies if contact.id in c.associated_contact_ids]
        
        # Find related activities
        related_activities = [a for a in activities if a.contact_id == contact.id]
        
        # Determine treatment assignment for MQL
        treatment_result = await self._determine_mql_treatment_assignment(
            contact, associated_companies, related_activities, historical_data
        )
        
        # Detect MQL-specific confounders
        confounders = await self._detect_mql_confounders(
            contact, associated_companies, related_activities, historical_data
        )
        
        # Calculate MQL data quality score
        data_quality_score = self._calculate_mql_data_quality_score(contact, associated_companies, related_activities)
        
        # Extract campaign information
        campaign_id, campaign_name = self._determine_mql_campaign_info(contact)
        
        # Extract geographic and audience data
        geographic_data = self._extract_contact_geographic_data(contact, associated_companies)
        audience_data = self._extract_contact_audience_data(contact, associated_companies, related_activities)
        
        # Create MQL causal marketing data
        causal_data = CausalMarketingData(
            record_id=f"mql_{contact.id}",
            org_id=org_id,
            data_source="hubspot",
            timestamp=contact.recent_conversion_date or contact.createdate or datetime.now(),
            treatment_type=treatment_result.treatment_type,
            treatment_group=treatment_result.treatment_group,
            treatment_intensity=treatment_result.treatment_intensity,
            randomization_unit=treatment_result.randomization_unit,
            experiment_id=treatment_result.experiment_id,
            outcome_value=contact.hubspotscore or 0.0,
            outcome_type="mql_conversion",
            confounders=confounders,
            data_quality_score=data_quality_score,
            causal_metadata={
                "platform": "hubspot",
                "conversion_type": "mql",
                "lifecycle_stage": contact.lifecyclestage.value if contact.lifecyclestage else None,
                "lead_status": contact.lead_status,
                "hubspot_score": contact.hubspotscore,
                "first_conversion_event": contact.first_conversion_event_name,
                "recent_conversion_event": contact.recent_conversion_event_name,
                "num_conversion_events": contact.num_conversion_events,
                "time_to_mql": contact.hs_time_to_move_from_lead_to_customer,
                "email_engagement": {
                    "opens": contact.hs_email_open_count,
                    "clicks": contact.hs_email_click_count,
                    "bounces": contact.hs_email_bounce_count
                },
                "social_engagement": contact.hs_social_num_broadcast_clicks,
                "analytics_source": contact.hs_analytics_source,
                "latest_source": contact.hs_latest_source,
                "campaign_id": campaign_id,
                "campaign_name": campaign_name,
                "mql_attribution": True,
                "lead_scoring_enabled": contact.hubspotscore is not None
            },
            geographic_data=geographic_data,
            audience_data=audience_data,
            platform_context={
                "source": "hubspot",
                "contact_id": contact.id,
                "email": contact.email,
                "company": contact.company,
                "associated_companies": [c.id for c in associated_companies],
                "contact_analytics": {
                    "source": contact.hs_analytics_source,
                    "source_data_1": contact.hs_analytics_source_data_1,
                    "source_data_2": contact.hs_analytics_source_data_2,
                    "latest_source": contact.hs_latest_source,
                    "latest_source_data_1": contact.hs_latest_source_data_1,
                    "latest_source_data_2": contact.hs_latest_source_data_2
                },
                "engagement_summary": {
                    "total_activities": len(related_activities),
                    "emails": len([a for a in related_activities if a.engagement_type == "EMAIL"]),
                    "calls": len([a for a in related_activities if a.engagement_type == "CALL"]),
                    "meetings": len([a for a in related_activities if a.engagement_type == "MEETING"])
                }
            }
        )
        
        return causal_data
    
    async def _enhance_with_kse_substrate(
        self,
        causal_data: CausalMarketingData,
        deal: HubSpotDealData,
        contacts: List[HubSpotContactData],
        companies: List[HubSpotCompanyData],
        activities: List[HubSpotActivityData],
        org_id: str
    ):
        """Enhance causal data with KSE Universal Substrate integration"""
        
        try:
            # Initialize KSE client
            kse_client = LiftKSEClient()
            
            # Create neural content for embedding
            neural_content = self._create_neural_content(deal, contacts, companies, activities)
            
            # Map to conceptual space
            conceptual_space = self._map_to_conceptual_space(deal, contacts, companies, activities)
            
            # Create knowledge graph nodes
            knowledge_graph_nodes = self._create_knowledge_graph_nodes(deal, contacts, companies, activities, causal_data)
            
            # Create causal relationships
            causal_relationships = [
                CausalRelationship(
                    source_node=f"deal_{deal.id}",
                    target_node=f"revenue_{deal.amount or 0}",
                    relationship_type="generates_revenue",
                    strength=0.95,
                    confidence=0.9,
                    temporal_lag=0,
                    metadata={
                        "deal_stage": deal.dealstage.value if deal.dealstage else None,
                        "pipeline": deal.pipeline,
                        "close_probability": deal.hs_deal_stage_probability
                    }
                )
            ]
            
            # Add contact relationships
            for contact in contacts:
                if contact.hubspotscore:
                    causal_relationships.append(
                        CausalRelationship(
                            source_node=f"contact_{contact.id}",
                            target_node=f"deal_{deal.id}",
                            relationship_type="influences_deal",
                            strength=min(contact.hubspotscore / 100, 1.0),
                            confidence=0.8,
                            temporal_lag=0,
                            metadata={
                                "lifecycle_stage": contact.lifecyclestage.value if contact.lifecyclestage else None,
                                "lead_score": contact.hubspotscore,
                                "email_engagement": contact.hs_email_open_count + contact.hs_email_click_count
                            }
                        )
                    )
            
            # Add activity relationships
            if activities:
                activity_strength = min(len(activities) / 10, 1.0)  # Normalize by activity count
                causal_relationships.append(
                    CausalRelationship(
                        source_node=f"engagement_activities",
                        target_node=f"deal_{deal.id}",
                        relationship_type="drives_conversion",
                        strength=activity_strength,
                        confidence=0.85,
                        temporal_lag=0,
                        metadata={
                            "total_activities": len(activities),
                            "activity_types": list(set(a.engagement_type for a in activities)),
                            "engagement_intensity": activity_strength
                        }
                    )
                )
            
            # Create causal memory entry
            causal_memory = CausalMemoryEntry(
                content=neural_content,
                causal_relationships=causal_relationships,
                temporal_context={
                    "timestamp": (deal.closedate or deal.createdate or datetime.now()).isoformat(),
                    "day_of_week": (deal.closedate or deal.createdate or datetime.now()).weekday(),
                    "hour_of_day": (deal.closedate or deal.createdate or datetime.now()).hour,
                    "is_weekend": (deal.closedate or deal.createdate or datetime.now()).weekday() >= 5,
                    "deal_age_days": deal.days_to_close or 0
                },
                causal_metadata={
                    "platform": "hubspot",
                    "data_type": "crm_deal",
                    "treatment_type": causal_data.treatment_type.value,
                    "revenue_impact": deal.amount or 0.0,
                    "crm_attribution": True,
                    "lead_scoring_enabled": any(c.hubspotscore for c in contacts),
                    "multi_contact_deal": len(contacts) > 1,
                    "enterprise_deal": any(c.numberofemployees and c.numberofemployees > 1000 for c in companies)
                },
                platform_context={
                    "source": "hubspot",
                    "deal_id": deal.id,
                    "deal_name": deal.dealname,
                    "pipeline": deal.pipeline,
                    "stage": deal.dealstage.value if deal.dealstage else None,
                    "owner_id": deal.hubspot_owner_id,
                    "contact_count": len(contacts),
                    "company_count": len(companies),
                    "activity_count": len(activities),
                    "crm_intelligence": {
                        "stage_probability": deal.hs_deal_stage_probability,
                        "time_in_stage": deal.hs_time_in_dealstage,
                        "contact_frequency": deal.hs_num_times_contacted,
                        "lead_scores": [c.hubspotscore for c in contacts if c.hubspotscore]
                    }
                },
                experiment_id=causal_data.experiment_id
            )
            
            # Store in KSE
            memory_id = await kse_client.store_causal_memory(org_id, causal_memory)
            
            # Update causal data with KSE integration
            causal_data.conceptual_space = conceptual_space
            causal_data.knowledge_graph_nodes = knowledge_graph_nodes
            causal_data.causal_metadata["kse_memory_id"] = memory_id
            causal_data.causal_metadata["neural_embedding_generated"] = True
            causal_data.causal_metadata["conceptual_space_mapped"] = True
            causal_data.causal_metadata["knowledge_graph_integrated"] = True
            causal_data.causal_metadata["crm_intelligence_integrated"] = True
            
            logger.info(f"Enhanced HubSpot deal {deal.id} with KSE substrate integration")
            
        except Exception as e:
            logger.error(f"Failed to enhance deal {deal.id} with KSE substrate: {str(e)}")
            # Continue without KSE enhancement
            causal_data.causal_metadata["kse_integration_failed"] = True
            causal_data.causal_metadata["kse_error"] = str(e)
    
    async def _enhance_mql_with_kse_substrate(
        self,
        causal_data: CausalMarketingData,
        contact: HubSpotContactData,
        companies: List[HubSpotCompanyData],
        activities: List[HubSpotActivityData],
        org_id: str
    ):
        """Enhance MQL causal data with KSE Universal Substrate integration"""
        
        try:
            # Initialize KSE client
            kse_client = LiftKSEClient()
            
            # Create neural content for MQL embedding
            neural_content = self._create_mql_neural_content(contact, companies, activities)
            
            # Map to conceptual space
            conceptual_space = self._map_mql_to_conceptual_space(contact, companies, activities)
            
            # Create knowledge graph nodes
            knowledge_graph_nodes = self._create_mql_knowledge_graph_nodes(contact, companies, activities, causal_data)
            
            # Create MQL causal relationships
            causal_relationships = [
                CausalRelationship(
                    source_node=f"contact_{contact.id}",
                    target_node=f"mql_conversion",
                    relationship_type="achieves_mql_status",
                    strength=0.9,
                    confidence=0.85,
                    temporal_lag=0,
                    metadata={
                        "lifecycle_stage": contact.lifecyclestage.value if contact.lifecyclestage else None,
                        "lead_score": contact.hubspotscore,
                        "conversion_event": contact.recent_conversion_event_name
                    }
                )
            ]
            
            # Add engagement relationships
            if contact.hs_email_open_count or contact.hs_email_click_count:
                email_engagement = (contact.hs_email_open_count or 0) + (contact.hs_email_click_count or 0)
                causal_relationships.append(
                    CausalRelationship(
                        source_node=f"email_engagement",
                        target_node=f"contact_{contact.id}",
                        relationship_type="drives_engagement",
                        strength=min(email_engagement / 50, 1.0),  # Normalize
                        confidence=0.8,
                        temporal_lag=0,
                        metadata={
                            "opens": contact.hs_email_open_count,
                            "clicks": contact.hs_email_click_count,
                            "bounces": contact.hs_email_bounce_count
                        }
                    )
                )
            
            # Create causal memory entry
            causal_memory = CausalMemoryEntry(
                content=neural_content,
                causal_relationships=causal_relationships,
                temporal_context={
                    "timestamp": (contact.recent_conversion_date or contact.createdate or datetime.now()).isoformat(),
                    "day_of_week": (contact.recent_conversion_date or contact.createdate or datetime.now()).weekday(),
                    "hour_of_day": (contact.recent_conversion_date or contact.createdate or datetime.now()).hour,
                    "is_weekend": (contact.recent_conversion_date or contact.createdate or datetime.now()).weekday() >= 5
                },
                causal_metadata={
                    "platform": "hubspot",
                    "data_type": "mql_conversion",
                    "treatment_type": causal_data.treatment_type.value,
                    "lead_score": contact.hubspotscore or 0.0,
                    "mql_attribution": True,
                    "email_engagement_tracked": bool(contact.hs_email_open_count or contact.hs_email_click_count),
                    "multi_touch_attribution": contact.num_conversion_events > 1
                },
                platform_context={
                    "source": "hubspot",
                    "contact_id": contact.id,
                    "email": contact.email,
                    "lifecycle_stage": contact.lifecyclestage.value if contact.lifecyclestage else None,
                    "lead_score": contact.hubspotscore,
                    "conversion_events": contact.num_conversion_events,
                    "mql_intelligence": {
                        "first_conversion": contact.first_conversion_event_name,
                        "recent_conversion": contact.recent_conversion_event_name,
                        "email_opens": contact.hs_email_open_count,
                        "email_clicks": contact.hs_email_click_count,
                        "social_clicks": contact.hs_social_num_broadcast_clicks
                    }
                },
                experiment_id=causal_data.experiment_id
            )
            
            # Store in KSE
            memory_id = await kse_client.store_causal_memory(org_id, causal_memory)
            
            # Update causal data with KSE integration
            causal_data.conceptual_space = conceptual_space
            causal_data.knowledge_graph_nodes = knowledge_graph_nodes
            causal_data.causal_metadata["kse_memory_id"] = memory_id
            causal_data.causal_metadata["neural_embedding_generated"] = True
            causal_data.causal_metadata["conceptual_space_mapped"] = True
            causal_data.causal_metadata["knowledge_graph_integrated"] = True
            causal_data.causal_metadata["mql_intelligence_integrated"] = True
            
            logger.info(f"Enhanced HubSpot MQL contact {contact.id} with KSE substrate integration")
            
        except Exception as e:
            logger.error(f"Failed to enhance MQL contact {contact.id} with KSE substrate: {str(e)}")
            # Continue without KSE enhancement
            causal_data.causal_metadata["kse_integration_failed"] = True
            causal_data.causal_metadata["kse_error"] = str(e)
    
    def _create_neural_content(
        self, 
        deal: HubSpotDealData, 
        contacts: List[HubSpotContactData], 
        companies: List[HubSpotCompanyData], 
        activities: List[HubSpotActivityData]
    ) -> str:
        """Create rich text content for neural embedding"""
        content_parts = [
            f"HubSpot deal {deal.id}: {deal.dealname or 'Unnamed Deal'}",
            f"Value: ${deal.amount or 0:.2f}",
            f"Stage: {deal.dealstage.value if deal.dealstage else 'Unknown'}",
            f"Pipeline: {deal.pipeline or 'Default'}"
        ]
        
        # Add contact information
        if contacts:
            primary_contact = contacts[0]
            content_parts.append(f"Primary contact: {primary_contact.email or 'Unknown'}")
            if primary_contact.company:
                content_parts.append(f"Company: {primary_contact.company}")
            if primary_contact.hubspotscore:
                content_parts.append(f"Lead score: {primary_contact.hubspotscore}")
            if primary_contact.lifecyclestage:
                content_parts.append(f"Lifecycle: {primary_contact.lifecyclestage.value}")
        
        # Add company information
        if companies:
            primary_company = companies[0]
            if primary_company.name:
                content_parts.append(f"Company: {primary_company.name}")
            if primary_company.industry:
                content_parts.append(f"Industry: {primary_company.industry}")
            if primary_company.numberofemployees:
                content_parts.append(f"Employees: {primary_company.numberofemployees}")
        
        # Add attribution information
        if deal.hs_analytics_source:
            content_parts.append(f"Source: {deal.hs_analytics_source}")
        if deal.hs_campaign:
            content_parts.append(f"Campaign: {deal.hs_campaign}")
        
        # Add engagement summary
        if activities:
            content_parts.append(f"Activities: {len(activities)} total")
            activity_types = {}
            for activity in activities:
                activity_types[activity.engagement_type] = activity_types.get(activity.engagement_type, 0) + 1
            for activity_type, count in activity_types.items():
                content_parts.append(f"{activity_type.lower()}s: {count}")
        
        # Add timing information
        if deal.days_to_close:
            content_parts.append(f"Days to close: {deal.days_to_close}")
        if deal.hs_time_in_dealstage:
            content_parts.append(f"Time in stage: {deal.hs_time_in_dealstage:.1f} days")
        
        return ". ".join(content_parts)
    
    def _create_mql_neural_content(
        self, 
        contact: HubSpotContactData, 
        companies: List[HubSpotCompanyData], 
        activities: List[HubSpotActivityData]
    ) -> str:
        """Create rich text content for MQL neural embedding"""
        content_parts = [
            f"HubSpot MQL conversion: {contact.email or 'Unknown contact'}",
            f"Lead score: {contact.hubspotscore or 0}",
            f"Lifecycle stage: {contact.lifecyclestage.value if contact.lifecyclestage else 'Unknown'}"
        ]
        
        # Add contact details
        if contact.firstname or contact.lastname:
            name_parts = [part for part in [contact.firstname, contact.lastname] if part]
            content_parts.append(f"Name: {' '.join(name_parts)}")
        
        if contact.company:
            content_parts.append(f"Company: {contact.company}")
        
        # Add conversion information
        if contact.recent_conversion_event_name:
            content_parts.append(f"Conversion event: {contact.recent_conversion_event_name}")
        if contact.num_conversion_events:
            content_parts.append(f"Total conversions: {contact.num_conversion_events}")
        
        # Add engagement metrics
        if contact.hs_email_open_count:
            content_parts.append(f"Email opens: {contact.hs_email_open_count}")
        if contact.hs_email_click_count:
            content_parts.append(f"Email clicks: {contact.hs_email_click_count}")
        if contact.hs_social_num_broadcast_clicks:
            content_parts.append(f"Social clicks: {contact.hs_social_num_broadcast_clicks}")
        
        # Add attribution information
        if contact.hs_analytics_source:
            content_parts.append(f"Original source: {contact.hs_analytics_source}")
        if contact.hs_latest_source:
            content_parts.append(f"Latest source: {contact.hs_latest_source}")
        
        # Add company context
        if companies:
            company = companies[0]
            if company.industry:
                content_parts.append(f"Industry: {company.industry}")
            if company.numberofemployees:
                content_parts.append(f"Company size: {company.numberofemployees} employees")
        
        return ". ".join(content_parts)
    
    def _map_to_conceptual_space(
        self, 
        deal: HubSpotDealData, 
        contacts: List[HubSpotContactData], 
        companies: List[HubSpotCompanyData], 
        activities: List[HubSpotActivityData]
    ) -> str:
        """Map deal to conceptual space dimensions"""
        dimensions = []
        
        # Deal value dimension
        if deal.amount:
            if deal.amount < 1000:
                dimensions.append("low_value")
            elif deal.amount < 10000:
                dimensions.append("medium_value")
            elif deal.amount < 100000:
                dimensions.append("high_value")
            else:
                dimensions.append("enterprise_value")
        else:
            dimensions.append("unknown_value")
        
        # Deal stage dimension
        if deal.dealstage:
            if deal.dealstage in [HubSpotDealStage.CLOSED_WON]:
                dimensions.append("won_deal")
            elif deal.dealstage in [HubSpotDealStage.CLOSED_LOST]:
                dimensions.append("lost_deal")
            elif deal.dealstage in [HubSpotDealStage.CONTRACT_SENT, HubSpotDealStage.DECISION_MAKER_BOUGHT_IN]:
                dimensions.append("late_stage")
            else:
                dimensions.append("early_stage")
        
        # Contact engagement dimension
        if contacts:
            primary_contact = contacts[0]
            if primary_contact.hubspotscore:
                if primary_contact.hubspotscore > 80:
                    dimensions.append("high_engagement")
                elif primary_contact.hubspotscore > 40:
                    dimensions.append("medium_engagement")
                else:
                    dimensions.append("low_engagement")
            
            # Lifecycle stage dimension
            if primary_contact.lifecyclestage:
                if primary_contact.lifecyclestage == HubSpotLifecycleStage.CUSTOMER:
                    dimensions.append("existing_customer")
                elif primary_contact.lifecyclestage == HubSpotLifecycleStage.SALES_QUALIFIED_LEAD:
                    dimensions.append("sql_contact")
                elif primary_contact.lifecyclestage == HubSpotLifecycleStage.MARKETING_QUALIFIED_LEAD:
                    dimensions.append("mql_contact")
                else:
                    dimensions.append("early_stage_contact")
        
        # Company size dimension
        if companies:
            company = companies[0]
            if company.numberofemployees:
                if company.numberofemployees > 1000:
                    dimensions.append("enterprise_company")
                elif company.numberofemployees > 100:
                    dimensions.append("mid_market_company")
                else:
                    dimensions.append("small_company")
        
        # Activity intensity dimension
        if activities:
            if len(activities) > 20:
                dimensions.append("high_touch")
            elif len(activities) > 5:
                dimensions.append("medium_touch")
            else:
                dimensions.append("low_touch")
        else:
            dimensions.append("no_touch")
        
        # Source dimension
        if deal.hs_analytics_source:
            source = deal.hs_analytics_source.lower()
            if "organic" in source:
                dimensions.append("organic_source")
            elif "paid" in source or "ads" in source:
                dimensions.append("paid_source")
            elif "email" in source:
                dimensions.append("email_source")
            elif "social" in source:
                dimensions.append("social_source")
            else:
                dimensions.append("other_source")
        
        return "_".join(dimensions)
    
    def _map_mql_to_conceptual_space(
        self, 
        contact: HubSpotContactData, 
        companies: List[HubSpotCompanyData], 
        activities: List[HubSpotActivityData]
    ) -> str:
        """Map MQL to conceptual space dimensions"""
        dimensions = ["mql_conversion"]
        
        # Lead score dimension
        if contact.hubspotscore:
            if contact.hubspotscore > 80:
                dimensions.append("high_score")
            elif contact.hubspotscore > 40:
                dimensions.append("medium_score")
            else:
                dimensions.append("low_score")
        
        # Engagement dimension
        email_engagement = (contact.hs_email_open_count or 0) + (contact.hs_email_click_count or 0)
        if email_engagement > 20:
            dimensions.append("high_email_engagement")
        elif email_engagement > 5:
            dimensions.append("medium_email_engagement")
        else:
            dimensions.append("low_email_engagement")
        
        # Conversion velocity dimension
        if contact.num_conversion_events:
            if contact.num_conversion_events > 5:
                dimensions.append("multi_touch")
            else:
                dimensions.append("few_touch")
        
        # Company context dimension
        if companies:
            company = companies[0]
            if company.numberofemployees and company.numberofemployees > 1000:
                dimensions.append("enterprise_prospect")
            elif company.numberofemployees and company.numberofemployees > 100:
                dimensions.append("mid_market_prospect")
            else:
                dimensions.append("smb_prospect")
        
        # Source dimension
        if contact.hs_latest_source:
            source = contact.hs_latest_source.lower()
            if "organic" in source:
                dimensions.append("organic_mql")
            elif "paid" in source:
                dimensions.append("paid_mql")
            elif "email" in source:
                dimensions.append("email_mql")
            else:
                dimensions.append("other_mql")
        
        return "_".join(dimensions)
    
    def _create_knowledge_graph_nodes(
        self, 
        deal: HubSpotDealData, 
        contacts: List[HubSpotContactData], 
        companies: List[HubSpotCompanyData], 
        activities: List[HubSpotActivityData],
        causal_data: CausalMarketingData
    ) -> List[str]:
        """Create knowledge graph node connections"""
        nodes = [
            f"deal_{deal.id}",
            f"pipeline_{deal.pipeline or 'default'}",
            f"stage_{deal.dealstage.value if deal.dealstage else 'unknown'}",
            f"owner_{deal.hubspot_owner_id or 'unassigned'}"
        ]
        
        # Add contact nodes
        for contact in contacts:
            nodes.append(f"contact_{contact.id}")
            if contact.email:
                nodes.append(f"email_{contact.email.split('@')[1]}")  # Domain
            if contact.lifecyclestage:
                nodes.append(f"lifecycle_{contact.lifecyclestage.value}")
        
        # Add company nodes
        for company in companies:
            nodes.append(f"company_{company.id}")
            if company.industry:
                nodes.append(f"industry_{company.industry.lower().replace(' ', '_')}")
            if company.country:
                nodes.append(f"country_{company.country}")
        
        # Add activity nodes
        activity_types = set(activity.engagement_type for activity in activities)
        for activity_type in activity_types:
            nodes.append(f"activity_{activity_type.lower()}")
        
        # Add source nodes
        if deal.hs_analytics_source:
            nodes.append(f"source_{deal.hs_analytics_source.lower().replace(' ', '_')}")
        if deal.hs_campaign:
            nodes.append(f"campaign_{deal.hs_campaign.lower().replace(' ', '_')}")
        
        return nodes
    
    def _create_mql_knowledge_graph_nodes(
        self, 
        contact: HubSpotContactData, 
        companies: List[HubSpotCompanyData], 
        activities: List[HubSpotActivityData],
        causal_data: CausalMarketingData
    ) -> List[str]:
        """Create knowledge graph node connections for MQL"""
        nodes = [
            f"contact_{contact.id}",
            f"mql_conversion",
            f"lifecycle_{contact.lifecyclestage.value if contact.lifecyclestage else 'unknown'}"
        ]
        
        # Add email domain node
        if contact.email:
            nodes.append(f"email_domain_{contact.email.split('@')[1]}")
        
        # Add company nodes
        for company in companies:
            nodes.append(f"company_{company.id}")
            if company.industry:
                nodes.append(f"industry_{company.industry.lower().replace(' ', '_')}")
        
        # Add source nodes
        if contact.hs_analytics_source:
            nodes.append(f"original_source_{contact.hs_analytics_source.lower().replace(' ', '_')}")
        if contact.hs_latest_source:
            nodes.append(f"latest_source_{contact.hs_latest_source.lower().replace(' ', '_')}")
        
        # Add conversion event nodes
        if contact.first_conversion_event_name:
            nodes.append(f"first_conversion_{contact.first_conversion_event_name.lower().replace(' ', '_')}")
        if contact.recent_conversion_event_name:
            nodes.append(f"recent_conversion_{contact.recent_conversion_event_name.lower().replace(' ', '_')}")
        
        return nodes
    
    async def _determine_treatment_assignment(
        self, 
        deal: HubSpotDealData, 
        contacts: List[HubSpotContactData], 
        companies: List[HubSpotCompanyData], 
        activities: List[HubSpotActivityData],
        historical_data: Optional[List[Dict[str, Any]]]
    ) -> TreatmentAssignmentResult:
        """Determine treatment assignment for HubSpot deal"""
        
        # Determine treatment based on analytics source and campaign
        if deal.hs_analytics_source:
            source = deal.hs_analytics_source.lower()
            
            if "paid" in source or "ads" in source:
                treatment_type = TreatmentType.BUDGET_INCREASE
                treatment_group = "paid_advertising"
                treatment_intensity = 0.8
            elif "email" in source:
                treatment_type = TreatmentType.CREATIVE_CHANGE
                treatment_group = "email_marketing"
                treatment_intensity = 0.7
            elif "social" in source:
                treatment_type = TreatmentType.TARGETING_CHANGE
                treatment_group = "social_media"
                treatment_intensity = 0.6
            elif "organic" in source:
                treatment_type = TreatmentType.CONTROL
                treatment_group = "organic_crm"
                treatment_intensity = 0.0
            else:
                treatment_type = TreatmentType.CONTROL
                treatment_group = "unknown_source"
                treatment_intensity = 0.0
        else:
            treatment_type = TreatmentType.CONTROL
            treatment_group = "direct_crm"
            treatment_intensity = 0.0
        
        # Adjust intensity based on deal characteristics
        if deal.amount and deal.amount > 50000:
            treatment_intensity = min(treatment_intensity + 0.2, 1.0)  # Enterprise deals get higher intensity
        
        if len(activities) > 10:
            treatment_intensity = min(treatment_intensity + 0.1, 1.0)  # High-touch deals get higher intensity
        
        # Determine randomization unit
        if companies:
            randomization_unit = RandomizationUnit.COMPANY
        else:
            randomization_unit = RandomizationUnit.CUSTOMER
        
        # Generate experiment ID
        experiment_id = None
        if deal.hs_campaign:
            experiment_id = f"hubspot_campaign_{deal.hs_campaign}_{deal.createdate.strftime('%Y%m') if deal.createdate else 'unknown'}"
        elif deal.hs_analytics_source:
            experiment_id = f"hubspot_source_{deal.hs_analytics_source}_{deal.createdate.strftime('%Y%m') if deal.createdate else 'unknown'}"
        
        return TreatmentAssignmentResult(
            treatment_group=treatment_group,
            treatment_type=treatment_type,
            treatment_intensity=treatment_intensity,
            randomization_unit=randomization_unit,
            experiment_id=experiment_id,
            assignment_confidence=0.85 if deal.hs_analytics_source else 0.6
        )
    
    async def _determine_mql_treatment_assignment(
        self, 
        contact: HubSpotContactData, 
        companies: List[HubSpotCompanyData], 
        activities: List[HubSpotActivityData],
        historical_data: Optional[List[Dict[str, Any]]]
    ) -> TreatmentAssignmentResult:
        """Determine treatment assignment for MQL conversion"""
        
        # Determine treatment based on latest source
        if contact.hs_latest_source:
            source = contact.hs_latest_source.lower()
            
            if "email" in source:
                treatment_type = TreatmentType.CREATIVE_CHANGE
                treatment_group = "email_nurturing"
                treatment_intensity = 0.8
            elif "paid" in source or "ads" in source:
                treatment_type = TreatmentType.BUDGET_INCREASE
                treatment_group = "paid_lead_gen"
                treatment_intensity = 0.9
            elif "social" in source:
                treatment_type = TreatmentType.TARGETING_CHANGE
                treatment_group = "social_lead_gen"
                treatment_intensity = 0.7
            elif "organic" in source:
                treatment_type = TreatmentType.CONTROL
                treatment_group = "organic_mql"
                treatment_intensity = 0.0
            else:
                treatment_type = TreatmentType.CONTROL
                treatment_group = "unknown_mql_source"
                treatment_intensity = 0.0
        else:
            treatment_type = TreatmentType.CONTROL
            treatment_group = "direct_mql"
            treatment_intensity = 0.0
        
        # Adjust intensity based on engagement
        email_engagement = (contact.hs_email_open_count or 0) + (contact.hs_email_click_count or 0)
        if email_engagement > 20:
            treatment_intensity = min(treatment_intensity + 0.1, 1.0)
        
        # Determine randomization unit
        randomization_unit = RandomizationUnit.CUSTOMER
        
        # Generate experiment ID
        experiment_id = None
        if contact.hs_latest_source:
            experiment_id = f"hubspot_mql_{contact.hs_latest_source}_{contact.recent_conversion_date.strftime('%Y%m') if contact.recent_conversion_date else 'unknown'}"
        
        return TreatmentAssignmentResult(
            treatment_group=treatment_group,
            treatment_type=treatment_type,
            treatment_intensity=treatment_intensity,
            randomization_unit=randomization_unit,
            experiment_id=experiment_id,
            assignment_confidence=0.8 if contact.hs_latest_source else 0.5
        )
    
    async def _detect_hubspot_confounders(
        self, 
        deal: HubSpotDealData, 
        contacts: List[HubSpotContactData], 
        companies: List[HubSpotCompanyData], 
        activities: List[HubSpotActivityData],
        historical_data: Optional[List[Dict[str, Any]]]
    ) -> List[ConfounderVariable]:
        """Detect HubSpot-specific confounding variables"""
        confounders = []
        
        # Deal stage probability confounder
        if deal.hs_deal_stage_probability is not None:
            confounders.append(ConfounderVariable(
                variable_name="deal_stage_probability",
                variable_type="continuous",
                value=deal.hs_deal_stage_probability,
                importance_score=0.9,
                detection_method="crm_intelligence",
                control_strategy="include_stage_probability_as_covariate"
            ))
        
        # Lead scoring confounder
        if contacts:
            lead_scores = [c.hubspotscore for c in contacts if c.hubspotscore is not None]
            if lead_scores:
                avg_lead_score = sum(lead_scores) / len(lead_scores)
                confounders.append(ConfounderVariable(
                    variable_name="average_lead_score",
                    variable_type="continuous",
                    value=avg_lead_score,
                    importance_score=0.85,
                    detection_method="lead_scoring_analysis",
                    control_strategy="include_lead_score_as_covariate"
                ))
        
        # Company size confounder
        if companies:
            company_sizes = [c.numberofemployees for c in companies if c.numberofemployees is not None]
            if company_sizes:
                avg_company_size = sum(company_sizes) / len(company_sizes)
                confounders.append(ConfounderVariable(
                    variable_name="company_size",
                    variable_type="continuous",
                    value=avg_company_size,
                    importance_score=0.8,
                    detection_method="firmographic_analysis",
                    control_strategy="include_company_size_as_covariate"
                ))
        
        # Engagement intensity confounder
        if activities:
            confounders.append(ConfounderVariable(
                variable_name="engagement_intensity",
                variable_type="continuous",
                value=len(activities),
                importance_score=0.75,
                detection_method="activity_analysis",
                control_strategy="include_activity_count_as_covariate"
            ))
        
        # Time in stage confounder
        if deal.hs_time_in_dealstage is not None:
            confounders.append(ConfounderVariable(
                variable_name="time_in_stage",
                variable_type="continuous",
                value=deal.hs_time_in_dealstage,
                importance_score=0.7,
                detection_method="sales_velocity_analysis",
                control_strategy="include_time_in_stage_as_covariate"
            ))
        
        return confounders
    
    async def _detect_mql_confounders(
        self, 
        contact: HubSpotContactData, 
        companies: List[HubSpotCompanyData], 
        activities: List[HubSpotActivityData],
        historical_data: Optional[List[Dict[str, Any]]]
    ) -> List[ConfounderVariable]:
        """Detect MQL-specific confounding variables"""
        confounders = []
        
        # Lead score confounder
        if contact.hubspotscore is not None:
            confounders.append(ConfounderVariable(
                variable_name="lead_score",
                variable_type="continuous",
                value=contact.hubspotscore,
                importance_score=0.9,
                detection_method="lead_scoring_analysis",
                control_strategy="include_lead_score_as_covariate"
            ))
        
        # Email engagement confounder
        email_engagement = (contact.hs_email_open_count or 0) + (contact.hs_email_click_count or 0)
        if email_engagement > 0:
            confounders.append(ConfounderVariable(
                variable_name="email_engagement",
                variable_type="continuous",
                value=email_engagement,
                importance_score=0.8,
                detection_method="email_engagement_analysis",
                control_strategy="include_email_engagement_as_covariate"
            ))
        
        # Conversion velocity confounder
        if contact.num_conversion_events:
            confounders.append(ConfounderVariable(
                variable_name="conversion_velocity",
                variable_type="continuous",
                value=contact.num_conversion_events,
                importance_score=0.75,
                detection_method="conversion_analysis",
                control_strategy="include_conversion_count_as_covariate"
            ))
        
        # Company context confounder
        if companies:
            company = companies[0]
            if company.numberofemployees:
                confounders.append(ConfounderVariable(
                    variable_name="prospect_company_size",
                    variable_type="continuous",
                    value=company.numberofemployees,
                    importance_score=0.7,
                    detection_method="firmographic_analysis",
                    control_strategy="include_prospect_company_size_as_covariate"
                ))
        
        return confounders
    
    def _extract_geographic_data(
        self, 
        contacts: List[HubSpotContactData], 
        companies: List[HubSpotCompanyData]
    ) -> Dict[str, Any]:
        """Extract geographic information from contacts and companies"""
        geographic_data = {}
        
        # Get location from companies first (more reliable)
        if companies:
            company = companies[0]
            geographic_data.update({
                "city": company.city,
                "state": company.state,
                "country": company.country
            })
        
        return {k: v for k, v in geographic_data.items() if v is not None}
    
    def _extract_contact_geographic_data(
        self, 
        contact: HubSpotContactData, 
        companies: List[HubSpotCompanyData]
    ) -> Dict[str, Any]:
        """Extract geographic information for contact"""
        geographic_data = {}
        
        # Get location from associated companies
        if companies:
            company = companies[0]
            geographic_data.update({
                "city": company.city,
                "state": company.state,
                "country": company.country
            })
        
        return {k: v for k, v in geographic_data.items() if v is not None}
    
    def _extract_audience_data(
        self, 
        deal: HubSpotDealData, 
        contacts: List[HubSpotContactData], 
        companies: List[HubSpotCompanyData], 
        activities: List[HubSpotActivityData]
    ) -> Dict[str, Any]:
        """Extract audience characteristics from deal data"""
        audience_data = {
            "deal_value_segment": self._categorize_deal_value(deal.amount or 0),
            "deal_stage": deal.dealstage.value if deal.dealstage else None,
            "pipeline": deal.pipeline,
            "has_multiple_contacts": len(contacts) > 1,
            "has_company_association": len(companies) > 0,
            "engagement_level": self._categorize_engagement_level(activities)
        }
        
        # Add contact characteristics
        if contacts:
            primary_contact = contacts[0]
            audience_data.update({
                "contact_lifecycle_stage": primary_contact.lifecyclestage.value if primary_contact.lifecyclestage else None,
                "lead_score_segment": self._categorize_lead_score(primary_contact.hubspotscore),
                "email_engagement_level": self._categorize_email_engagement(primary_contact),
                "has_lead_score": primary_contact.hubspotscore is not None
            })
        
        # Add company characteristics
        if companies:
            primary_company = companies[0]
            audience_data.update({
                "company_size_segment": self._categorize_company_size(primary_company.numberofemployees),
                "industry": primary_company.industry,
                "has_revenue_data": primary_company.annualrevenue is not None
            })
        
        return audience_data
    
    def _extract_contact_audience_data(
        self, 
        contact: HubSpotContactData, 
        companies: List[HubSpotCompanyData], 
        activities: List[HubSpotActivityData]
    ) -> Dict[str, Any]:
        """Extract audience characteristics for contact"""
        audience_data = {
            "lifecycle_stage": contact.lifecyclestage.value if contact.lifecyclestage else None,
            "lead_score_segment": self._categorize_lead_score(contact.hubspotscore),
            "email_engagement_level": self._categorize_email_engagement(contact),
            "conversion_velocity": self._categorize_conversion_velocity(contact.num_conversion_events),
            "has_company_association": len(companies) > 0
        }
        
        # Add company context
        if companies:
            company = companies[0]
            audience_data.update({
                "company_size_segment": self._categorize_company_size(company.numberofemployees),
                "industry": company.industry
            })
        
        return audience_data
    
    def _calculate_data_quality_score(
        self, 
        deal: HubSpotDealData, 
        contacts: List[HubSpotContactData], 
        companies: List[HubSpotCompanyData], 
        activities: List[HubSpotActivityData]
    ) -> float:
        """Calculate data quality score for causal inference"""
        score_components = []
        
        # Required fields completeness
        required_fields = [deal.id, deal.dealname, deal.amount, deal.dealstage]
        completeness_score = sum(1 for field in required_fields if field is not None) / len(required_fields)
        score_components.append(completeness_score * 0.3)
        
        # Attribution data quality
        attribution_score = 0.2  # Base score
        if deal.hs_analytics_source:
            attribution_score += 0.4
        if deal.hs_campaign:
            attribution_score += 0.3
        if deal.hs_analytics_source_data_1:
            attribution_score += 0.1
        score_components.append(min(attribution_score, 1.0) * 0.25)
        
        # Contact data quality
        contact_score = 0.1 if contacts else 0.0
        if contacts:
            primary_contact = contacts[0]
            contact_score += 0.3 if primary_contact.email else 0.0
            contact_score += 0.3 if primary_contact.hubspotscore else 0.0
            contact_score += 0.3 if primary_contact.lifecyclestage else 0.0
        score_components.append(min(contact_score, 1.0) * 0.2)
        
        # Company data quality
        company_score = 0.1 if companies else 0.0
        if companies:
            primary_company = companies[0]
            company_score += 0.4 if primary_company.name else 0.0
            company_score += 0.3 if primary_company.industry else 0.0
            company_score += 0.2 if primary_company.numberofemployees else 0.0
        score_components.append(min(company_score, 1.0) * 0.15)
        
        # Activity data richness
        activity_score = min(len(activities) / 10, 1.0)  # Normalize to 10 activities
        score_components.append(activity_score * 0.1)
        
        return sum(score_components)
    
    def _categorize_deal_value(self, amount: float) -> str:
        """Categorize deal value into segments"""
        if amount < 1000:
            return "micro"
        elif amount < 10000:
            return "small"
        elif amount < 50000:
            return "medium"
        elif amount < 200000:
            return "large"
        else:
            return "enterprise"
    
    def _categorize_lead_score(self, score: Optional[int]) -> str:
        """Categorize lead score into segments"""
        if score is None:
            return "unscored"
        elif score < 20:
            return "cold"
        elif score < 40:
            return "warm"
        elif score < 70:
            return "hot"
        else:
            return "very_hot"
    
    def _categorize_email_engagement(self, contact: HubSpotContactData) -> str:
        """Categorize email engagement level"""
        opens = contact.hs_email_open_count or 0
        clicks = contact.hs_email_click_count or 0
        total_engagement = opens + (clicks * 2)  # Weight clicks more heavily
        
        if total_engagement == 0:
            return "no_engagement"
        elif total_engagement < 5:
            return "low_engagement"
        elif total_engagement < 20:
            return "medium_engagement"
        else:
            return "high_engagement"
    
    def _categorize_company_size(self, employees: Optional[int]) -> str:
        """Categorize company size"""
        if employees is None:
            return "unknown"
        elif employees < 10:
            return "startup"
        elif employees < 50:
            return "small"
        elif employees < 200:
            return "medium"
        elif employees < 1000:
            return "large"
        else:
            return "enterprise"
    
    def _categorize_engagement_level(self, activities: List[HubSpotActivityData]) -> str:
        """Categorize overall engagement level based on activities"""
        activity_count = len(activities)
        
        if activity_count == 0:
            return "no_touch"
        elif activity_count < 3:
            return "low_touch"
        elif activity_count < 10:
            return "medium_touch"
        else:
            return "high_touch"
    
    def _categorize_conversion_velocity(self, conversion_count: Optional[int]) -> str:
        """Categorize conversion velocity"""
        if conversion_count is None or conversion_count == 0:
            return "no_conversions"
        elif conversion_count == 1:
            return "single_conversion"
        elif conversion_count < 5:
            return "multiple_conversions"
        else:
            return "high_velocity"
    
    def _determine_campaign_info(
        self,
        deal: HubSpotDealData,
        contacts: List[HubSpotContactData],
        companies: List[HubSpotCompanyData]
    ) -> Dict[str, Any]:
        """Determine campaign information for deal"""
        campaign_info = {}
        
        # Primary campaign from deal
        if deal.hs_campaign:
            campaign_info["campaign_name"] = deal.hs_campaign
        
        # Source information
        if deal.hs_analytics_source:
            campaign_info["source"] = deal.hs_analytics_source
        
        # Additional source data
        if deal.hs_analytics_source_data_1:
            campaign_info["source_data_1"] = deal.hs_analytics_source_data_1
        if deal.hs_analytics_source_data_2:
            campaign_info["source_data_2"] = deal.hs_analytics_source_data_2
        
        # Contact attribution if available
        if contacts:
            primary_contact = contacts[0]
            if primary_contact.hs_analytics_source and not campaign_info.get("source"):
                campaign_info["source"] = primary_contact.hs_analytics_source
            if primary_contact.hs_latest_source:
                campaign_info["latest_source"] = primary_contact.hs_latest_source
        
        return campaign_info
    
    def _determine_mql_campaign_info(
        self,
        contact: HubSpotContactData,
        companies: List[HubSpotCompanyData]
    ) -> Dict[str, Any]:
        """Determine campaign information for MQL conversion"""
        campaign_info = {}
        
        # Source information
        if contact.hs_analytics_source:
            campaign_info["original_source"] = contact.hs_analytics_source
        
        if contact.hs_latest_source:
            campaign_info["latest_source"] = contact.hs_latest_source
        
        # Conversion event information
        if contact.first_conversion_event_name:
            campaign_info["first_conversion_event"] = contact.first_conversion_event_name
        
        if contact.recent_conversion_event_name:
            campaign_info["recent_conversion_event"] = contact.recent_conversion_event_name
        
        return campaign_info
    
    async def close(self):
        """Close HTTP client and cleanup resources"""
        if hasattr(self, 'session') and self.session:
            await self.session.close()