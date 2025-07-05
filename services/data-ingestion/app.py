"""
LiftOS Data Ingestion Service
Phase 2: API Connectors for Meta Business, Google Ads, and Klaviyo
"""
import asyncio
import time
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any, Union
from fastapi import FastAPI, HTTPException, Depends, Header, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uuid
import httpx
from pydantic import BaseModel, Field
import json

# Import shared modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.models.base import APIResponse, HealthCheck
from shared.models.marketing import (
    DataSource, MarketingDataIngestionRequest,
    MetaBusinessData, GoogleAdsData, KlaviyoData
)
from shared.models.causal_marketing import (
    CausalMarketingData, CausalAnalysisRequest, DataQualityAssessment
)
from shared.utils.causal_transforms import CausalDataTransformer
from shared.utils.config import get_service_config
from shared.utils.logging import setup_logging
from shared.health.health_checks import HealthChecker
from credential_manager import credential_manager

# Service configuration
config = get_service_config("data_ingestion", 8006)
logger = setup_logging("data_ingestion")

# Health checker
health_checker = HealthChecker("data_ingestion")

# Memory Service client
MEMORY_SERVICE_URL = os.getenv("MEMORY_SERVICE_URL", "http://localhost:8003")

# FastAPI app
app = FastAPI(
    title="LiftOS Data Ingestion Service",
    description="API Connectors for Meta Business, Google Ads, and Klaviyo",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if config.DEBUG else ["http://localhost:3000", "http://localhost:8501", "http://localhost:8502"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class APICredentials(BaseModel):
    """API credentials for external platforms"""
    platform: DataSource
    credentials: Dict[str, str]
    is_active: bool = True


class SyncJobRequest(BaseModel):
    """Data sync job request"""
    platform: DataSource
    date_range_start: date
    date_range_end: date
    sync_type: str = Field("full", pattern="^(full|incremental)$")
    campaigns: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class SyncJobStatus(BaseModel):
    """Data sync job status"""
    job_id: str
    platform: DataSource
    status: str  # pending, running, completed, failed
    progress: float = 0.0
    records_processed: int = 0
    records_total: Optional[int] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = {}


# Global state for tracking sync jobs
sync_jobs: Dict[str, SyncJobStatus] = {}

# Initialize causal data transformer
causal_transformer = CausalDataTransformer()


def get_user_context(
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
    x_memory_context: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """Extract user context from headers"""
    if not x_user_id or not x_org_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User context required"
        )
    
    return {
        "user_id": x_user_id,
        "org_id": x_org_id,
        "memory_context": x_memory_context or f"org_{x_org_id}_context",
        "roles": x_user_roles.split(",") if x_user_roles else []
    }


class MetaBusinessConnector:
    """Meta Business API connector"""
    
    def __init__(self, access_token: str, app_id: str, app_secret: str):
        self.access_token = access_token
        self.app_id = app_id
        self.app_secret = app_secret
        self.base_url = "https://graph.facebook.com/v18.0"
        self.client = httpx.AsyncClient(timeout=30.0)
    
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
            
            data = response.json()
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
            
            data = response.json()
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
            
            data = response.json()
            insights = data.get("data", [])
            
            return insights[0] if insights else {}
            
        except Exception as e:
            logger.error(f"Failed to get Meta campaign insights: {str(e)}")
            return {}
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


class GoogleAdsConnector:
    """Google Ads API connector"""
    
    def __init__(self, developer_token: str, client_id: str, client_secret: str, refresh_token: str):
        self.developer_token = developer_token
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self.access_token = None
        self.base_url = "https://googleads.googleapis.com/v14"
        self.client = httpx.AsyncClient(timeout=30.0)
    
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
            
            token_data = response.json()
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
            
            data = response.json()
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
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


class KlaviyoConnector:
    """Klaviyo API connector"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://a.klaviyo.com/api"
        self.client = httpx.AsyncClient(timeout=30.0)
    
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
            
            data = response.json()
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
        """Get campaign metrics from Klaviyo API"""
        try:
            url = f"{self.base_url}/campaign-recipient-estimation-jobs"
            headers = {
                "Authorization": f"Klaviyo-API-Key {self.api_key}",
                "revision": "2024-02-15"
            }
            
            # This is a simplified version - actual implementation would use
            # the proper Klaviyo metrics endpoints
            params = {
                "filter": f"equals(campaign_id,{campaign_id})"
            }
            
            response = await self.client.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            return data.get("data", {})
            
        except Exception as e:
            logger.error(f"Failed to get Klaviyo campaign metrics: {str(e)}")
            return {}
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


# API Endpoints
@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Data ingestion service health check"""
    memory_health = await check_memory_service_health()
    
    dependencies = {
        "memory_service": "healthy" if memory_health else "unhealthy",
        "active_sync_jobs": str(len([job for job in sync_jobs.values() if job.status == "running"]))
    }
    
    return HealthCheck(
        status="healthy" if memory_health else "degraded",
        dependencies=dependencies,
        uptime=time.time() - getattr(app.state, "start_time", time.time())
    )


async def check_memory_service_health() -> bool:
    """Check if Memory Service is healthy"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{MEMORY_SERVICE_URL}/health")
            return response.status_code == 200
    except:
        return False


@app.get("/", response_model=APIResponse)
async def root():
    """Data ingestion service root endpoint"""
    return APIResponse(
        message="LiftOS Data Ingestion Service",
        data={
            "version": "1.0.0",
            "supported_platforms": ["meta_business", "google_ads", "klaviyo"],
            "docs": "/docs"
        }
    )


@app.post("/sync/start", response_model=APIResponse)
async def start_sync_job(
    request: SyncJobRequest,
    background_tasks: BackgroundTasks,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Start a data sync job for specified platform"""
    job_id = str(uuid.uuid4())
    
    # Create sync job status
    sync_job = SyncJobStatus(
        job_id=job_id,
        platform=request.platform,
        status="pending",
        metadata={
            "org_id": user_context["org_id"],
            "user_id": user_context["user_id"],
            "sync_type": request.sync_type,
            "date_range": {
                "start": str(request.date_range_start),
                "end": str(request.date_range_end)
            }
        }
    )
    
    sync_jobs[job_id] = sync_job
    
    # Start background sync task
    background_tasks.add_task(
        execute_sync_job,
        job_id,
        request,
        user_context
    )
    
    logger.info(f"Started sync job {job_id} for {request.platform.value}")
    
    return APIResponse(
        message=f"Sync job started for {request.platform.value}",
        data={
            "job_id": job_id,
            "platform": request.platform.value,
            "status": "pending"
        }
    )


@app.get("/sync/status/{job_id}", response_model=APIResponse)
async def get_sync_status(
    job_id: str,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Get sync job status"""
    if job_id not in sync_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sync job not found"
        )
    
    sync_job = sync_jobs[job_id]
    
    # Check if user can access this job
    if sync_job.metadata.get("org_id") != user_context["org_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to sync job"
        )
    
    return APIResponse(
        message="Sync job status retrieved",
        data=sync_job.dict()
    )


@app.get("/sync/jobs", response_model=APIResponse)
async def list_sync_jobs(
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """List sync jobs for organization"""
    org_jobs = [
        job for job in sync_jobs.values()
        if job.metadata.get("org_id") == user_context["org_id"]
    ]
    
    return APIResponse(
        message="Sync jobs retrieved",
        data={
            "jobs": [job.dict() for job in org_jobs],
            "total": len(org_jobs)
        }
    )


async def execute_sync_job(job_id: str, request: SyncJobRequest, user_context: Dict[str, Any]):
    """Execute sync job in background with causal transformation"""
    sync_job = sync_jobs[job_id]
    
    try:
        sync_job.status = "running"
        sync_job.started_at = datetime.utcnow()
        
        # Get platform credentials (this would be stored securely in production)
        credentials = await get_platform_credentials(request.platform, user_context["org_id"])
        
        if not credentials:
            raise Exception(f"No credentials found for {request.platform.value}")
        
        # Execute platform-specific sync
        if request.platform == DataSource.META_BUSINESS:
            raw_data = await sync_meta_business_data(credentials, request)
        elif request.platform == DataSource.GOOGLE_ADS:
            raw_data = await sync_google_ads_data(credentials, request)
        elif request.platform == DataSource.KLAVIYO:
            raw_data = await sync_klaviyo_data(credentials, request)
        else:
            raise Exception(f"Unsupported platform: {request.platform.value}")
        
        # Transform raw data to causal format
        causal_data_list = []
        for raw_record in raw_data:
            try:
                # Get historical data for this campaign (for confounder detection)
                historical_data = await get_historical_data(
                    raw_record.get('campaign_id'),
                    request.platform,
                    user_context["org_id"]
                )
                
                # Transform to causal format
                causal_data = await causal_transformer.transform_to_causal_format(
                    raw_data=raw_record,
                    data_source=request.platform,
                    org_id=user_context["org_id"],
                    historical_data=historical_data
                )
                
                causal_data_list.append(causal_data)
                
            except Exception as transform_error:
                logger.warning(f"Failed to transform record {raw_record.get('id', 'unknown')}: {transform_error}")
                # Continue with other records
                continue
        
        # Send causal data to Memory Service
        await send_causal_data_to_memory_service(causal_data_list, request, user_context)
        
        sync_job.status = "completed"
        sync_job.records_processed = len(causal_data_list)
        sync_job.records_total = len(raw_data)
        sync_job.completed_at = datetime.utcnow()
        
        # Add causal transformation metadata
        sync_job.metadata.update({
            'causal_transformation': True,
            'raw_records': len(raw_data),
            'transformed_records': len(causal_data_list),
            'transformation_success_rate': len(causal_data_list) / len(raw_data) if raw_data else 0
        })
        
        logger.info(f"Sync job {job_id} completed successfully with {len(causal_data_list)}/{len(raw_data)} records transformed")
        
    except Exception as e:
        sync_job.status = "failed"
        sync_job.error_message = str(e)
        sync_job.completed_at = datetime.utcnow()
        
        logger.error(f"Sync job {job_id} failed: {str(e)}")


async def get_platform_credentials(platform: DataSource, org_id: str) -> Optional[Dict[str, str]]:
    """Get platform credentials for organization using credential manager"""
    try:
        if platform == DataSource.META_BUSINESS:
            return await credential_manager.get_meta_business_credentials(org_id)
        elif platform == DataSource.GOOGLE_ADS:
            return await credential_manager.get_google_ads_credentials(org_id)
        elif platform == DataSource.KLAVIYO:
            return await credential_manager.get_klaviyo_credentials(org_id)
        else:
            logger.error(f"Unsupported platform: {platform}")
            return None
    except Exception as e:
        logger.error(f"Error retrieving credentials for {platform}: {str(e)}")
        return None


async def sync_meta_business_data(credentials: Dict[str, str], request: SyncJobRequest) -> List[Dict[str, Any]]:
    """Sync data from Meta Business API"""
    connector = MetaBusinessConnector(
        access_token=credentials["access_token"],
        app_id=credentials["app_id"],
        app_secret=credentials["app_secret"]
    )
    
    try:
        # Get ad accounts
        accounts = await connector.get_ad_accounts()
        
        all_data = []
        for account in accounts:
            account_id = account["id"]
            
            # Get campaigns for this account
            campaigns = await connector.get_campaigns(
                account_id, request.date_range_start, request.date_range_end
            )
            
            for campaign in campaigns:
                # Transform to standard format
                campaign_data = {
                    "id": f"meta_{campaign['id']}",
                    "account_id": account_id,
                    "campaign_id": campaign["id"],
                    "campaign_name": campaign.get("name", ""),
                    "status": campaign.get("status", ""),
                    "objective": campaign.get("objective", ""),
                    **campaign.get("insights", {})
                }
                all_data.append(campaign_data)
        
        return all_data
        
    finally:
        await connector.close()


async def sync_google_ads_data(credentials: Dict[str, str], request: SyncJobRequest) -> List[Dict[str, Any]]:
    """Sync data from Google Ads API"""
    connector = GoogleAdsConnector(
        developer_token=credentials["developer_token"],
        client_id=credentials["client_id"],
        client_secret=credentials["client_secret"],
        refresh_token=credentials["refresh_token"]
    )
    
    try:
        # Get customers
        customers = await connector.get_customers()
        
        all_data = []
        for customer_resource in customers:
            # Extract customer ID from resource name
            customer_id = customer_resource.split("/")[-1]
            
            # Get campaigns for this customer
            campaigns = await connector.get_campaigns(
                customer_id, request.date_range_start, request.date_range_end
            )
            
            for campaign_result in campaigns:
                campaign = campaign_result.get("campaign", {})
                metrics = campaign_result.get("metrics", {})
                segments = campaign_result.get("segments", {})
                
                # Transform to standard format
                campaign_data = {
                    "id": f"google_{campaign.get('id', '')}",
                    "customer_id": customer_id,
                    "campaign_id": campaign.get("id", ""),
                    "campaign_name": campaign.get("name", ""),
                    "status": campaign.get("status", ""),
                    "advertising_channel_type": campaign.get("advertisingChannelType", ""),
                    "cost_micros": metrics.get("costMicros", 0),
                    "impressions": metrics.get("impressions", 0),
                    "clicks": metrics.get("clicks", 0),
                    "conversions": metrics.get("conversions", 0),
                    "conversions_value": metrics.get("conversionsValue", 0),
                    "date": segments.get("date", "")
                }
                all_data.append(campaign_data)
        
        return all_data
        
    finally:
        await connector.close()


async def sync_klaviyo_data(credentials: Dict[str, str], request: SyncJobRequest) -> List[Dict[str, Any]]:
    """Sync data from Klaviyo API"""
    connector = KlaviyoConnector(api_key=credentials["api_key"])
    
    try:
        # Get campaigns
        campaigns = await connector.get_campaigns(
            request.date_range_start, request.date_range_end
        )
        
        all_data = []
        for campaign in campaigns:
            attributes = campaign.get("attributes", {})
            metrics = campaign.get("metrics", {})
            
            # Transform to standard format
            campaign_data = {
                "id": f"klaviyo_{campaign.get('id', '')}",
                "campaign_id": campaign.get("id", ""),
                "campaign_name": attributes.get("name", ""),
                "status": attributes.get("status", ""),
                "send_time": attributes.get("send_time", ""),
                "delivered": metrics.get("delivered", 0),
                "opened": metrics.get("opened", 0),
                "clicked": metrics.get("clicked", 0),
                "unsubscribed": metrics.get("unsubscribed", 0),
                "bounced": metrics.get("bounced", 0)
            }
            all_data.append(campaign_data)
        
        return all_data
        
    finally:
        await connector.close()


async def send_to_memory_service(data: List[Dict[str, Any]], request: SyncJobRequest, user_context: Dict[str, Any]):
    """Send synced data to Memory Service"""
    if not data:
        return
    
    # Prepare ingestion request
    ingestion_request = {
        "data_source": request.platform.value,
        "data_entries": data,
        "date_range_start": str(request.date_range_start),
        "date_range_end": str(request.date_range_end),
        "metadata": {
            "sync_job_id": sync_jobs[list(sync_jobs.keys())[-1]].job_id,
            "sync_type": request.sync_type,
            "ingestion_source": "data_ingestion_service"
        }
    }
    
    # Send to Memory Service
    headers = {
        "X-User-Id": user_context["user_id"],
        "X-Org-Id": user_context["org_id"],
        "X-Memory-Context": user_context["memory_context"],
        "X-User-Roles": ",".join(user_context["roles"]),
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{MEMORY_SERVICE_URL}/marketing/ingest",
            json=ingestion_request,
            headers=headers
        )
        response.raise_for_status()
        
        logger.info(f"Successfully sent {len(data)} records to Memory Service")

async def get_historical_data(platform: str, user_context: Dict[str, Any], days_back: int = 30) -> List[Dict[str, Any]]:
    """Get historical data for causal analysis context"""
    try:
        headers = {
            "X-User-Id": user_context["user_id"],
            "X-Org-Id": user_context["org_id"],
            "X-Memory-Context": user_context["memory_context"],
            "X-User-Roles": ",".join(user_context["roles"]),
            "Content-Type": "application/json"
        }
        
        # Calculate date range for historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        params = {
            "data_source": platform,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(
                f"{MEMORY_SERVICE_URL}/marketing/data",
                params=params,
                headers=headers
            )
            
            if response.status_code == 200:
                historical_data = response.json().get("data", [])
                logger.info(f"Retrieved {len(historical_data)} historical records for {platform}")
                return historical_data
            else:
                logger.warning(f"Failed to retrieve historical data: {response.status_code}")
                return []
                
    except Exception as e:
        logger.error(f"Error retrieving historical data: {str(e)}")
        return []


async def send_causal_data_to_memory_service(causal_data: CausalMarketingData, user_context: Dict[str, Any]):
    """Send transformed causal data to Memory Service"""
    try:
        # Convert causal data to dict for transmission
        causal_data_dict = causal_data.model_dump()
        
        headers = {
            "X-User-Id": user_context["user_id"],
            "X-Org-Id": user_context["org_id"],
            "X-Memory-Context": user_context["memory_context"],
            "X-User-Roles": ",".join(user_context["roles"]),
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{MEMORY_SERVICE_URL}/api/v1/marketing/ingest/causal",
                json=causal_data_dict,
                headers=headers
            )
            
            if response.status_code == 201:
                logger.info(f"Successfully sent causal data to Memory Service")
            else:
                logger.error(f"Failed to send causal data to Memory Service: {response.status_code} - {response.text}")
                
    except Exception as e:
        logger.error(f"Error sending causal data to Memory Service: {str(e)}")
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize data ingestion service on startup"""
    app.state.start_time = time.time()
    logger.info("Data Ingestion Service started")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Data Ingestion Service stopped")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_level=config.LOG_LEVEL.lower()
    )