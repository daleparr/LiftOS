"""
Salesforce CRM Connector for LiftOS v1.3.0
Advanced CRM attribution with opportunity and lead tracking
"""

import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import json
import base64

from shared.models.marketing import DataSource
from shared.models.causal_marketing import (
    CausalMarketingData, ConfounderVariable,
    TreatmentType, RandomizationUnit
)
from shared.utils.causal_transforms import CausalDataTransformer, TreatmentAssignmentResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SalesforceLeadStatus(str, Enum):
    """Salesforce lead status values"""
    OPEN_NOT_CONTACTED = "Open - Not Contacted"
    WORKING_CONTACTED = "Working - Contacted"
    CLOSED_CONVERTED = "Closed - Converted"
    CLOSED_NOT_CONVERTED = "Closed - Not Converted"
    UNQUALIFIED = "Unqualified"

class SalesforceOpportunityStage(str, Enum):
    """Salesforce opportunity stage values"""
    PROSPECTING = "Prospecting"
    QUALIFICATION = "Qualification"
    NEEDS_ANALYSIS = "Needs Analysis"
    VALUE_PROPOSITION = "Value Proposition"
    ID_DECISION_MAKERS = "Id. Decision Makers"
    PERCEPTION_ANALYSIS = "Perception Analysis"
    PROPOSAL_PRICE_QUOTE = "Proposal/Price Quote"
    NEGOTIATION_REVIEW = "Negotiation/Review"
    CLOSED_WON = "Closed Won"
    CLOSED_LOST = "Closed Lost"

class SalesforceLeadSource(str, Enum):
    """Salesforce lead source values"""
    WEB = "Web"
    PHONE_INQUIRY = "Phone Inquiry"
    PARTNER_REFERRAL = "Partner Referral"
    PURCHASED_LIST = "Purchased List"
    OTHER = "Other"
    ADVERTISEMENT = "Advertisement"
    EMPLOYEE_REFERRAL = "Employee Referral"
    EXTERNAL_REFERRAL = "External Referral"
    TRADE_SHOW = "Trade Show"
    WEB_FORM = "Web Form"
    SOCIAL_MEDIA = "Social Media"
    EMAIL_CAMPAIGN = "Email Campaign"

# Salesforce Data Models
class SalesforceLeadData(BaseModel):
    """Salesforce Lead data model"""
    Id: str
    FirstName: Optional[str] = None
    LastName: Optional[str] = None
    Email: Optional[str] = None
    Company: Optional[str] = None
    Title: Optional[str] = None
    Phone: Optional[str] = None
    Status: Optional[SalesforceLeadStatus] = None
    LeadSource: Optional[SalesforceLeadSource] = None
    Rating: Optional[str] = None  # Hot, Warm, Cold
    Industry: Optional[str] = None
    AnnualRevenue: Optional[float] = None
    NumberOfEmployees: Optional[int] = None
    City: Optional[str] = None
    State: Optional[str] = None
    Country: Optional[str] = None
    CreatedDate: Optional[datetime] = None
    ConvertedDate: Optional[datetime] = None
    ConvertedAccountId: Optional[str] = None
    ConvertedContactId: Optional[str] = None
    ConvertedOpportunityId: Optional[str] = None
    IsConverted: Optional[bool] = None
    Campaign_Source__c: Optional[str] = None  # Custom field for campaign tracking
    UTM_Source__c: Optional[str] = None  # Custom UTM tracking
    UTM_Medium__c: Optional[str] = None
    UTM_Campaign__c: Optional[str] = None
    Lead_Score__c: Optional[int] = None  # Custom lead scoring field

class SalesforceOpportunityData(BaseModel):
    """Salesforce Opportunity data model"""
    Id: str
    Name: Optional[str] = None
    AccountId: Optional[str] = None
    Amount: Optional[float] = None
    StageName: Optional[SalesforceOpportunityStage] = None
    Probability: Optional[float] = None
    CloseDate: Optional[datetime] = None
    CreatedDate: Optional[datetime] = None
    LastModifiedDate: Optional[datetime] = None
    Type: Optional[str] = None  # New Business, Existing Customer - Upgrade, etc.
    LeadSource: Optional[SalesforceLeadSource] = None
    CampaignId: Optional[str] = None
    OwnerId: Optional[str] = None
    IsClosed: Optional[bool] = None
    IsWon: Optional[bool] = None
    ForecastCategory: Optional[str] = None
    NextStep: Optional[str] = None
    Description: Optional[str] = None
    Campaign_Source__c: Optional[str] = None
    UTM_Source__c: Optional[str] = None
    UTM_Medium__c: Optional[str] = None
    UTM_Campaign__c: Optional[str] = None
    Days_to_Close__c: Optional[int] = None  # Custom field
    Lead_Score_at_Creation__c: Optional[int] = None

class SalesforceAccountData(BaseModel):
    """Salesforce Account data model"""
    Id: str
    Name: Optional[str] = None
    Type: Optional[str] = None
    Industry: Optional[str] = None
    AnnualRevenue: Optional[float] = None
    NumberOfEmployees: Optional[int] = None
    BillingCity: Optional[str] = None
    BillingState: Optional[str] = None
    BillingCountry: Optional[str] = None
    Website: Optional[str] = None
    Phone: Optional[str] = None
    CreatedDate: Optional[datetime] = None
    LastModifiedDate: Optional[datetime] = None
    OwnerId: Optional[str] = None

class SalesforceContactData(BaseModel):
    """Salesforce Contact data model"""
    Id: str
    FirstName: Optional[str] = None
    LastName: Optional[str] = None
    Email: Optional[str] = None
    AccountId: Optional[str] = None
    Title: Optional[str] = None
    Phone: Optional[str] = None
    Department: Optional[str] = None
    LeadSource: Optional[SalesforceLeadSource] = None
    CreatedDate: Optional[datetime] = None
    LastModifiedDate: Optional[datetime] = None
    Email_Opt_Out: Optional[bool] = None
    HasOptedOutOfEmail: Optional[bool] = None

class SalesforceActivityData(BaseModel):
    """Salesforce Activity (Task/Event) data model"""
    Id: str
    Subject: Optional[str] = None
    ActivityDate: Optional[datetime] = None
    Status: Optional[str] = None
    Priority: Optional[str] = None
    Type: Optional[str] = None  # Call, Email, Meeting, etc.
    WhoId: Optional[str] = None  # Contact/Lead ID
    WhatId: Optional[str] = None  # Account/Opportunity ID
    OwnerId: Optional[str] = None
    CreatedDate: Optional[datetime] = None
    Description: Optional[str] = None

class SalesforceConnector:
    """
    Salesforce CRM Connector with advanced opportunity and lead attribution
    Integrates with Salesforce REST API for comprehensive CRM data extraction
    """
    
    def __init__(
        self,
        instance_url: str,
        client_id: str,
        client_secret: str,
        username: str,
        password: str,
        security_token: str,
        api_version: str = "v58.0"
    ):
        self.instance_url = instance_url.rstrip('/')
        self.client_id = client_id
        self.client_secret = client_secret
        self.username = username
        self.password = password
        self.security_token = security_token
        self.api_version = api_version
        self.access_token: Optional[str] = None
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting: Salesforce allows 100,000 API calls per 24 hours for most editions
        self.rate_limit_calls_per_hour = 4000  # Conservative rate limiting
        self.rate_limit_window = 3600  # 1 hour
        self.call_timestamps: List[datetime] = []
    
    async def authenticate(self) -> bool:
        """Authenticate with Salesforce using OAuth 2.0 Username-Password flow"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            auth_url = f"{self.instance_url}/services/oauth2/token"
            
            data = {
                'grant_type': 'password',
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'username': self.username,
                'password': f"{self.password}{self.security_token}"
            }
            
            async with self.session.post(auth_url, data=data) as response:
                if response.status == 200:
                    auth_data = await response.json()
                    self.access_token = auth_data['access_token']
                    self.instance_url = auth_data['instance_url']
                    logger.info("Successfully authenticated with Salesforce")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Salesforce authentication failed: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error during Salesforce authentication: {str(e)}")
            return False
    
    async def _check_rate_limit(self):
        """Check and enforce rate limiting"""
        now = datetime.now()
        
        # Remove timestamps older than the rate limit window
        self.call_timestamps = [
            ts for ts in self.call_timestamps 
            if (now - ts).total_seconds() < self.rate_limit_window
        ]
        
        # Check if we're at the rate limit
        if len(self.call_timestamps) >= self.rate_limit_calls_per_hour:
            sleep_time = self.rate_limit_window - (now - self.call_timestamps[0]).total_seconds()
            if sleep_time > 0:
                logger.warning(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)
        
        # Add current timestamp
        self.call_timestamps.append(now)
    
    async def _make_api_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make authenticated API request to Salesforce"""
        await self._check_rate_limit()
        
        if not self.access_token:
            if not await self.authenticate():
                return None
        
        url = f"{self.instance_url}/services/data/{self.api_version}/{endpoint}"
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }
        
        try:
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 401:
                    # Token expired, re-authenticate
                    logger.info("Access token expired, re-authenticating...")
                    if await self.authenticate():
                        headers['Authorization'] = f'Bearer {self.access_token}'
                        async with self.session.get(url, headers=headers, params=params) as retry_response:
                            if retry_response.status == 200:
                                return await retry_response.json()
                    return None
                else:
                    error_text = await response.text()
                    logger.error(f"Salesforce API request failed: {response.status} - {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error making Salesforce API request: {str(e)}")
            return None
    
    async def get_opportunities(
        self, 
        start_date: datetime, 
        end_date: datetime,
        limit: int = 1000
    ) -> List[SalesforceOpportunityData]:
        """Fetch opportunities from Salesforce"""
        opportunities = []
        
        # SOQL query for opportunities
        soql = f"""
        SELECT Id, Name, AccountId, Amount, StageName, Probability, CloseDate, 
               CreatedDate, LastModifiedDate, Type, LeadSource, CampaignId, 
               OwnerId, IsClosed, IsWon, ForecastCategory, NextStep, Description,
               Campaign_Source__c, UTM_Source__c, UTM_Medium__c, UTM_Campaign__c,
               Days_to_Close__c, Lead_Score_at_Creation__c
        FROM Opportunity 
        WHERE CreatedDate >= {start_date.strftime('%Y-%m-%dT%H:%M:%SZ')} 
        AND CreatedDate <= {end_date.strftime('%Y-%m-%dT%H:%M:%SZ')}
        ORDER BY CreatedDate DESC
        LIMIT {limit}
        """
        
        response = await self._make_api_request("query", {"q": soql})
        
        if response and 'records' in response:
            for record in response['records']:
                try:
                    # Convert date strings to datetime objects
                    if record.get('CreatedDate'):
                        record['CreatedDate'] = datetime.fromisoformat(record['CreatedDate'].replace('Z', '+00:00'))
                    if record.get('LastModifiedDate'):
                        record['LastModifiedDate'] = datetime.fromisoformat(record['LastModifiedDate'].replace('Z', '+00:00'))
                    if record.get('CloseDate'):
                        record['CloseDate'] = datetime.fromisoformat(record['CloseDate'] + 'T00:00:00+00:00')
                    
                    # Convert enum fields
                    if record.get('StageName'):
                        try:
                            record['StageName'] = SalesforceOpportunityStage(record['StageName'])
                        except ValueError:
                            record['StageName'] = None
                    
                    if record.get('LeadSource'):
                        try:
                            record['LeadSource'] = SalesforceLeadSource(record['LeadSource'])
                        except ValueError:
                            record['LeadSource'] = None
                    
                    opportunity = SalesforceOpportunityData(**record)
                    opportunities.append(opportunity)
                    
                except Exception as e:
                    logger.warning(f"Error parsing opportunity {record.get('Id', 'unknown')}: {str(e)}")
                    continue
        
        logger.info(f"Retrieved {len(opportunities)} opportunities from Salesforce")
        return opportunities
    
    async def get_leads(
        self, 
        start_date: datetime, 
        end_date: datetime,
        limit: int = 1000
    ) -> List[SalesforceLeadData]:
        """Fetch leads from Salesforce"""
        leads = []
        
        # SOQL query for leads
        soql = f"""
        SELECT Id, FirstName, LastName, Email, Company, Title, Phone, Status, 
               LeadSource, Rating, Industry, AnnualRevenue, NumberOfEmployees,
               City, State, Country, CreatedDate, ConvertedDate, ConvertedAccountId,
               ConvertedContactId, ConvertedOpportunityId, IsConverted,
               Campaign_Source__c, UTM_Source__c, UTM_Medium__c, UTM_Campaign__c,
               Lead_Score__c
        FROM Lead 
        WHERE CreatedDate >= {start_date.strftime('%Y-%m-%dT%H:%M:%SZ')} 
        AND CreatedDate <= {end_date.strftime('%Y-%m-%dT%H:%M:%SZ')}
        ORDER BY CreatedDate DESC
        LIMIT {limit}
        """
        
        response = await self._make_api_request("query", {"q": soql})
        
        if response and 'records' in response:
            for record in response['records']:
                try:
                    # Convert date strings to datetime objects
                    if record.get('CreatedDate'):
                        record['CreatedDate'] = datetime.fromisoformat(record['CreatedDate'].replace('Z', '+00:00'))
                    if record.get('ConvertedDate'):
                        record['ConvertedDate'] = datetime.fromisoformat(record['ConvertedDate'].replace('Z', '+00:00'))
                    
                    # Convert enum fields
                    if record.get('Status'):
                        try:
                            record['Status'] = SalesforceLeadStatus(record['Status'])
                        except ValueError:
                            record['Status'] = None
                    
                    if record.get('LeadSource'):
                        try:
                            record['LeadSource'] = SalesforceLeadSource(record['LeadSource'])
                        except ValueError:
                            record['LeadSource'] = None
                    
                    lead = SalesforceLeadData(**record)
                    leads.append(lead)
                    
                except Exception as e:
                    logger.warning(f"Error parsing lead {record.get('Id', 'unknown')}: {str(e)}")
                    continue
        
        logger.info(f"Retrieved {len(leads)} leads from Salesforce")
        return leads
    
    async def get_accounts(self, account_ids: List[str]) -> List[SalesforceAccountData]:
        """Fetch accounts by IDs"""
        if not account_ids:
            return []
        
        accounts = []
        
        # Process in batches of 100 (SOQL limit)
        for i in range(0, len(account_ids), 100):
            batch_ids = account_ids[i:i+100]
            ids_str = "', '".join(batch_ids)
            
            soql = f"""
            SELECT Id, Name, Type, Industry, AnnualRevenue, NumberOfEmployees,
                   BillingCity, BillingState, BillingCountry, Website, Phone,
                   CreatedDate, LastModifiedDate, OwnerId
            FROM Account 
            WHERE Id IN ('{ids_str}')
            """
            
            response = await self._make_api_request("query", {"q": soql})
            
            if response and 'records' in response:
                for record in response['records']:
                    try:
                        # Convert date strings to datetime objects
                        if record.get('CreatedDate'):
                            record['CreatedDate'] = datetime.fromisoformat(record['CreatedDate'].replace('Z', '+00:00'))
                        if record.get('LastModifiedDate'):
                            record['LastModifiedDate'] = datetime.fromisoformat(record['LastModifiedDate'].replace('Z', '+00:00'))
                        
                        account = SalesforceAccountData(**record)
                        accounts.append(account)
                        
                    except Exception as e:
                        logger.warning(f"Error parsing account {record.get('Id', 'unknown')}: {str(e)}")
                        continue
        
        logger.info(f"Retrieved {len(accounts)} accounts from Salesforce")
        return accounts
    
    async def get_contacts(self, contact_ids: List[str]) -> List[SalesforceContactData]:
        """Fetch contacts by IDs"""
        if not contact_ids:
            return []
        
        contacts = []
        
        # Process in batches of 100
        for i in range(0, len(contact_ids), 100):
            batch_ids = contact_ids[i:i+100]
            ids_str = "', '".join(batch_ids)
            
            soql = f"""
            SELECT Id, FirstName, LastName, Email, AccountId, Title, Phone,
                   Department, LeadSource, CreatedDate, LastModifiedDate,
                   Email_Opt_Out, HasOptedOutOfEmail
            FROM Contact 
            WHERE Id IN ('{ids_str}')
            """
            
            response = await self._make_api_request("query", {"q": soql})
            
            if response and 'records' in response:
                for record in response['records']:
                    try:
                        # Convert date strings to datetime objects
                        if record.get('CreatedDate'):
                            record['CreatedDate'] = datetime.fromisoformat(record['CreatedDate'].replace('Z', '+00:00'))
                        if record.get('LastModifiedDate'):
                            record['LastModifiedDate'] = datetime.fromisoformat(record['LastModifiedDate'].replace('Z', '+00:00'))
                        
                        # Convert enum fields
                        if record.get('LeadSource'):
                            try:
                                record['LeadSource'] = SalesforceLeadSource(record['LeadSource'])
                            except ValueError:
                                record['LeadSource'] = None
                        
                        contact = SalesforceContactData(**record)
                        contacts.append(contact)
                        
                    except Exception as e:
                        logger.warning(f"Error parsing contact {record.get('Id', 'unknown')}: {str(e)}")
                        continue
        
        logger.info(f"Retrieved {len(contacts)} contacts from Salesforce")
        return contacts
    
    async def get_activities(
        self, 
        related_ids: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> List[SalesforceActivityData]:
        """Fetch activities (tasks and events) for related records"""
        if not related_ids:
            return []
        
        activities = []
        
        # Process in batches
        for i in range(0, len(related_ids), 100):
            batch_ids = related_ids[i:i+100]
            ids_str = "', '".join(batch_ids)
            
            # Query Tasks
            task_soql = f"""
            SELECT Id, Subject, ActivityDate, Status, Priority, Type, WhoId, WhatId,
                   OwnerId, CreatedDate, Description
            FROM Task 
            WHERE (WhoId IN ('{ids_str}') OR WhatId IN ('{ids_str}'))
            AND CreatedDate >= {start_date.strftime('%Y-%m-%dT%H:%M:%SZ')}
            AND CreatedDate <= {end_date.strftime('%Y-%m-%dT%H:%M:%SZ')}
            """
            
            response = await self._make_api_request("query", {"q": task_soql})
            
            if response and 'records' in response:
                for record in response['records']:
                    try:
                        # Convert date strings to datetime objects
                        if record.get('CreatedDate'):
                            record['CreatedDate'] = datetime.fromisoformat(record['CreatedDate'].replace('Z', '+00:00'))
                        if record.get('ActivityDate'):
                            record['ActivityDate'] = datetime.fromisoformat(record['ActivityDate'] + 'T00:00:00+00:00')
                        
                        activity = SalesforceActivityData(**record)
                        activities.append(activity)
                        
                    except Exception as e:
                        logger.warning(f"Error parsing task {record.get('Id', 'unknown')}: {str(e)}")
                        continue
            
            # Query Events
            event_soql = f"""
            SELECT Id, Subject, ActivityDate, Type, WhoId, WhatId, OwnerId, CreatedDate, Description
            FROM Event 
            WHERE (WhoId IN ('{ids_str}') OR WhatId IN ('{ids_str}'))
            AND CreatedDate >= {start_date.strftime('%Y-%m-%dT%H:%M:%SZ')}
            AND CreatedDate <= {end_date.strftime('%Y-%m-%dT%H:%M:%SZ')}
            """
            
            response = await self._make_api_request("query", {"q": event_soql})
            
            if response and 'records' in response:
                for record in response['records']:
                    try:
                        # Convert date strings to datetime objects
                        if record.get('CreatedDate'):
                            record['CreatedDate'] = datetime.fromisoformat(record['CreatedDate'].replace('Z', '+00:00'))
                        if record.get('ActivityDate'):
                            record['ActivityDate'] = datetime.fromisoformat(record['ActivityDate'].replace('Z', '+00:00'))
                        
                        activity = SalesforceActivityData(**record)
                        activities.append(activity)
                        
                    except Exception as e:
                        logger.warning(f"Error parsing event {record.get('Id', 'unknown')}: {str(e)}")
                        continue
        
        logger.info(f"Retrieved {len(activities)} activities from Salesforce")
        return activities
    
    async def sync_salesforce_data(
        self,
        org_id: str,
        start_date: datetime,
        end_date: datetime,
        include_historical: bool = False
    ) -> List[CausalMarketingData]:
        """
        Sync Salesforce data and transform to causal marketing format
        Processes both opportunities and lead conversions as separate events
        """
        try:
            logger.info(f"Starting Salesforce data sync for org {org_id}")
            
            # Fetch opportunities and leads
            opportunities = await self.get_opportunities(start_date, end_date)
            leads = await self.get_leads(start_date, end_date)
            
            # Get related accounts and contacts
            account_ids = list(set([opp.AccountId for opp in opportunities if opp.AccountId]))
            contact_ids = []
            
            # Get contact IDs from converted leads
            for lead in leads:
                if lead.ConvertedContactId:
                    contact_ids.append(lead.ConvertedContactId)
            
            accounts = await self.get_accounts(account_ids)
            contacts = await self.get_contacts(contact_ids)
            
            # Get activities for all related records
            all_related_ids = account_ids + contact_ids + [opp.Id for opp in opportunities] + [lead.Id for lead in leads]
            activities = await self.get_activities(all_related_ids, start_date, end_date)
            
            # Create lookup dictionaries
            accounts_dict = {acc.Id: acc for acc in accounts}
            contacts_dict = {cont.Id: cont for cont in contacts}
            activities_by_record = {}
            
            for activity in activities:
                for related_id in [activity.WhoId, activity.WhatId]:
                    if related_id:
                        if related_id not in activities_by_record:
                            activities_by_record[related_id] = []
                        activities_by_record[related_id].append(activity)
            
            causal_data_list = []
            
            # Process opportunities as primary conversion events
            for opportunity in opportunities:
                try:
                    # Get related data
                    account = accounts_dict.get(opportunity.AccountId) if opportunity.AccountId else None
                    opp_activities = activities_by_record.get(opportunity.Id, [])
                    
                    # Process opportunity data
                    causal_data = await self._process_opportunity_data(
                        opportunity, account, opp_activities, org_id
                    )
                    
                    if causal_data:
                        causal_data_list.append(causal_data)
                        
                except Exception as e:
                    logger.error(f"Error processing opportunity {opportunity.Id}: {str(e)}")
                    continue
            
            # Process converted leads as secondary conversion events
            converted_leads = [lead for lead in leads if lead.IsConverted]
            for lead in converted_leads:
                try:
                    # Get related data
                    contact = contacts_dict.get(lead.ConvertedContactId) if lead.ConvertedContactId else None
                    account = accounts_dict.get(lead.ConvertedAccountId) if lead.ConvertedAccountId else None
                    lead_activities = activities_by_record.get(lead.Id, [])
                    
                    # Process lead conversion data
                    causal_data = await self._process_lead_conversion_data(
                        lead, contact, account, lead_activities, org_id
                    )
                    
                    if causal_data:
                        causal_data_list.append(causal_data)
                        
                except Exception as e:
                    logger.error(f"Error processing lead conversion {lead.Id}: {str(e)}")
                    continue
            
            logger.info(f"Successfully processed {len(causal_data_list)} Salesforce records")
            return causal_data_list
            
        except Exception as e:
            logger.error(f"Error in Salesforce data sync: {str(e)}")
            return []
    
    async def _process_opportunity_data(
        self,
        opportunity: SalesforceOpportunityData,
        account: Optional[SalesforceAccountData],
        activities: List[SalesforceActivityData],
        org_id: str
    ) -> Optional[CausalMarketingData]:
        """Process opportunity data into causal marketing format"""
        
        # Determine conversion value and status
        conversion_value = opportunity.Amount or 0.0
        is_conversion = opportunity.IsWon or False
        
        # Calculate conversion date
        conversion_date = opportunity.CloseDate if opportunity.IsClosed else opportunity.CreatedDate
        if not conversion_date:
            conversion_date = datetime.now()
        
        # Enhanced KSE integration
        kse_enhancement = await self._enhance_opportunity_with_kse(
            opportunity, account, activities
        )
        
        # Treatment assignment
        treatment_result = await self._determine_opportunity_treatment_assignment(
            opportunity, account, activities, None
        )
        
        # Confounder detection
        confounders = await self._detect_opportunity_confounders(
            opportunity, account, activities, None
        )
        
        # Extract geographic and audience data
        geographic_data = self._extract_opportunity_geographic_data(opportunity, account)
        audience_data = self._extract_opportunity_audience_data(opportunity, account, activities)
        
        # Calculate data quality score
        data_quality_score = self._calculate_opportunity_data_quality_score(
            opportunity, account, activities
        )
        
        # Determine campaign information
        campaign_info = self._determine_opportunity_campaign_info(opportunity, account)
        
        return CausalMarketingData(
            id=f"salesforce_opportunity_{opportunity.Id}",
            org_id=org_id,
            data_source=DataSource.SALESFORCE,
            conversion_date=conversion_date,
            conversion_value=conversion_value,
            is_conversion=is_conversion,
            customer_id=opportunity.AccountId or f"unknown_account_{opportunity.Id}",
            campaign_id=opportunity.CampaignId or campaign_info.get("campaign_name", "unknown_campaign"),
            ad_set_id=campaign_info.get("utm_campaign", "unknown_ad_set"),
            ad_id=campaign_info.get("utm_source", "unknown_ad"),
            treatment_assignment=treatment_result,
            confounding_variables=confounders,
            kse_enhancement=kse_enhancement,
            geographic_data=geographic_data,
            audience_data=audience_data,
            data_quality_score=data_quality_score,
            raw_data={
                "opportunity": opportunity.dict(),
                "account": account.dict() if account else None,
                "activities": [activity.dict() for activity in activities]
            }
        )
    
    def _extract_lead_audience_data(
        self,
        lead: SalesforceLeadData,
        contact: Optional[SalesforceContactData],
        account: Optional[SalesforceAccountData],
        activities: List[SalesforceActivityData]
    ) -> Dict[str, Any]:
        """Extract audience characteristics for lead"""
        audience_data = {
            "lead_status": lead.Status.value if lead.Status else None,
            "lead_rating": lead.Rating,
            "lead_score_segment": self._categorize_lead_score(lead.Lead_Score__c),
            "industry": lead.Industry,
            "company_size_segment": self._categorize_lead_company_size(lead.NumberOfEmployees),
            "has_account_association": account is not None,
            "is_converted": lead.IsConverted
        }
        
        return audience_data
    
    def _calculate_opportunity_data_quality_score(
        self,
        opportunity: SalesforceOpportunityData,
        account: Optional[SalesforceAccountData],
        activities: List[SalesforceActivityData]
    ) -> float:
        """Calculate data quality score for opportunity causal inference"""
        score_components = []
        
        # Required fields completeness
        required_fields = [opportunity.Id, opportunity.Name, opportunity.Amount, opportunity.StageName]
        completeness_score = sum(1 for field in required_fields if field is not None) / len(required_fields)
        score_components.append(completeness_score * 0.3)
        
        # Attribution data quality
        attribution_score = 0.2  # Base score
        if opportunity.LeadSource:
            attribution_score += 0.3
        if opportunity.UTM_Source__c:
            attribution_score += 0.3
        if opportunity.UTM_Campaign__c:
            attribution_score += 0.2
        score_components.append(min(attribution_score, 1.0) * 0.25)
        
        # Account data quality
        account_score = 0.1 if account else 0.0
        if account:
            account_score += 0.4 if account.Name else 0.0
            account_score += 0.3 if account.Industry else 0.0
            account_score += 0.2 if account.NumberOfEmployees else 0.0
        score_components.append(min(account_score, 1.0) * 0.2)
        
        # Probability and forecasting data
        forecast_score = 0.0
        if opportunity.Probability is not None:
            forecast_score += 0.5
        if opportunity.ForecastCategory:
            forecast_score += 0.3
        if opportunity.NextStep:
            forecast_score += 0.2
        score_components.append(min(forecast_score, 1.0) * 0.15)
        
        # Activity data richness
        activity_score = min(len(activities) / 15, 1.0)  # Normalize to 15 activities
        score_components.append(activity_score * 0.1)
        
        return sum(score_components)
    
    def _calculate_lead_data_quality_score(
        self,
        lead: SalesforceLeadData,
        contact: Optional[SalesforceContactData],
        account: Optional[SalesforceAccountData],
        activities: List[SalesforceActivityData]
    ) -> float:
        """Calculate data quality score for lead causal inference"""
        score_components = []
        
        # Required fields completeness
        required_fields = [lead.Id, lead.Email, lead.Company, lead.Status]
        completeness_score = sum(1 for field in required_fields if field is not None) / len(required_fields)
        score_components.append(completeness_score * 0.3)
        
        # Attribution data quality
        attribution_score = 0.2  # Base score
        if lead.LeadSource:
            attribution_score += 0.3
        if lead.UTM_Source__c:
            attribution_score += 0.3
        if lead.Campaign_Source__c:
            attribution_score += 0.2
        score_components.append(min(attribution_score, 1.0) * 0.25)
        
        # Lead scoring data
        scoring_score = 0.0
        if lead.Lead_Score__c is not None:
            scoring_score += 0.5
        if lead.Rating:
            scoring_score += 0.3
        if lead.Industry:
            scoring_score += 0.2
        score_components.append(min(scoring_score, 1.0) * 0.2)
        
        # Company data quality
        company_score = 0.0
        if lead.NumberOfEmployees:
            company_score += 0.4
        if lead.AnnualRevenue:
            company_score += 0.3
        if lead.Industry:
            company_score += 0.3
        score_components.append(min(company_score, 1.0) * 0.15)
        
        # Conversion data quality
        conversion_score = 0.0
        if lead.IsConverted:
            conversion_score += 0.5
            if lead.ConvertedDate:
                conversion_score += 0.3
            if lead.ConvertedOpportunityId:
                conversion_score += 0.2
        score_components.append(min(conversion_score, 1.0) * 0.1)
        
        return sum(score_components)
    
    def _categorize_opportunity_value(self, amount: float) -> str:
        """Categorize opportunity value into segments"""
        if amount < 5000:
            return "micro"
        elif amount < 25000:
            return "small"
        elif amount < 100000:
            return "medium"
        elif amount < 500000:
            return "large"
        else:
            return "enterprise"
    
    def _categorize_account_size(self, employees: Optional[int]) -> str:
        """Categorize account size"""
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
    
    def _categorize_lead_score(self, score: Optional[int]) -> str:
        """Categorize lead score into segments"""
        if score is None:
            return "unscored"
        elif score < 25:
            return "cold"
        elif score < 50:
            return "warm"
        elif score < 75:
            return "hot"
        else:
            return "very_hot"
    
    def _categorize_lead_company_size(self, employees: Optional[int]) -> str:
        """Categorize lead company size"""
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
    
    def _categorize_opportunity_engagement_level(self, activities: List[SalesforceActivityData]) -> str:
        """Categorize opportunity engagement level based on activities"""
        activity_count = len(activities)
        
        if activity_count == 0:
            return "no_touch"
        elif activity_count < 5:
            return "low_touch"
        elif activity_count < 15:
            return "medium_touch"
        else:
            return "high_touch"
    
    def _determine_opportunity_campaign_info(
        self,
        opportunity: SalesforceOpportunityData,
        account: Optional[SalesforceAccountData]
    ) -> Dict[str, Any]:
        """Determine campaign information for opportunity"""
        campaign_info = {}
        
        # UTM tracking information
        if opportunity.UTM_Source__c:
            campaign_info["utm_source"] = opportunity.UTM_Source__c
        if opportunity.UTM_Medium__c:
            campaign_info["utm_medium"] = opportunity.UTM_Medium__c
        if opportunity.UTM_Campaign__c:
            campaign_info["utm_campaign"] = opportunity.UTM_Campaign__c
        
        # Campaign source information
        if opportunity.Campaign_Source__c:
            campaign_info["campaign_source"] = opportunity.Campaign_Source__c
        
        # Lead source information
        if opportunity.LeadSource:
            campaign_info["lead_source"] = opportunity.LeadSource.value
        
        # Campaign ID
        if opportunity.CampaignId:
            campaign_info["campaign_id"] = opportunity.CampaignId
        
        return campaign_info
    
    def _determine_lead_campaign_info(
        self,
        lead: SalesforceLeadData,
        contact: Optional[SalesforceContactData],
        account: Optional[SalesforceAccountData]
    ) -> Dict[str, Any]:
        """Determine campaign information for lead conversion"""
        campaign_info = {}
        
        # UTM tracking information
        if lead.UTM_Source__c:
            campaign_info["utm_source"] = lead.UTM_Source__c
        if lead.UTM_Medium__c:
            campaign_info["utm_medium"] = lead.UTM_Medium__c
        if lead.UTM_Campaign__c:
            campaign_info["utm_campaign"] = lead.UTM_Campaign__c
        
        # Campaign source information
        if lead.Campaign_Source__c:
            campaign_info["campaign_source"] = lead.Campaign_Source__c
        
        # Lead source information
        if lead.LeadSource:
            campaign_info["lead_source"] = lead.LeadSource.value
        
        return campaign_info
    
    async def close(self):
        """Close HTTP client and cleanup resources"""
        if hasattr(self, 'session') and self.session:
            await self.session.close()