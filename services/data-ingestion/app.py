"""
LiftOS Data Ingestion Service
Phase 4: API Connectors for 17 platforms across 4 tiers including Tier 4 Extended Social/CRM platforms
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

# Import e-commerce connectors
from connectors.shopify_connector import ShopifyConnector
from connectors.woocommerce_connector import WooCommerceConnector
from connectors.amazon_connector import AmazonConnector

# Import Tier 2 connectors (CRM and Payment)
from connectors.hubspot_connector import HubSpotConnector
from connectors.salesforce_connector import SalesforceConnector
from connectors.stripe_connector import StripeConnector
from connectors.paypal_connector import PayPalConnector

# Import Tier 0 connectors (Legacy)
from connectors.meta_business_connector import MetaBusinessConnector, create_meta_business_connector
from connectors.google_ads_connector import GoogleAdsConnector, create_google_ads_connector
from connectors.klaviyo_connector import KlaviyoConnector, create_klaviyo_connector

# Import Tier 3 connectors (Social/Analytics/Data)
from connectors.tiktok_connector import TikTokConnector
from connectors.snowflake_connector import SnowflakeConnector
from connectors.databricks_connector import DatabricksConnector

# Import Tier 4 connectors (Extended Social/CRM)
from connectors.zoho_crm_connector import ZohoCRMConnector
from connectors.linkedin_ads_connector import LinkedInAdsConnector
from connectors.x_ads_connector import XAdsConnector

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
    description="API Connectors for 16 platforms across 4 tiers: Meta Business, Google Ads, Klaviyo, Shopify, WooCommerce, Amazon Seller Central, HubSpot, Salesforce, Stripe, PayPal, TikTok, Snowflake, Databricks, Zoho CRM, LinkedIn Ads, and X Ads",
    version="1.4.0",
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
            "version": "1.4.0",
            "supported_platforms": [
                # Tier 0 (Legacy)
                "meta_business", "google_ads", "klaviyo",
                # Tier 1 (E-commerce)
                "shopify", "woocommerce", "amazon_seller_central",
                # Tier 2 (CRM/Payment)
                "hubspot", "salesforce", "stripe", "paypal",
                # Tier 3 (Social/Analytics/Data)
                "tiktok", "snowflake", "databricks",
                # Tier 4 (Extended Social/CRM)
                "zoho_crm", "linkedin_ads", "x_ads"
            ],
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
        elif request.platform == DataSource.SHOPIFY:
            raw_data = await sync_shopify_data(credentials, request, user_context)
        elif request.platform == DataSource.WOOCOMMERCE:
            raw_data = await sync_woocommerce_data(credentials, request, user_context)
        elif request.platform == DataSource.AMAZON_SELLER_CENTRAL:
            raw_data = await sync_amazon_data(credentials, request, user_context)
        # Tier 2 platforms
        elif request.platform == DataSource.HUBSPOT:
            raw_data = await sync_hubspot_data(credentials, request, user_context)
        elif request.platform == DataSource.SALESFORCE:
            raw_data = await sync_salesforce_data(credentials, request, user_context)
        elif request.platform == DataSource.STRIPE:
            raw_data = await sync_stripe_data(credentials, request, user_context)
        elif request.platform == DataSource.PAYPAL:
            raw_data = await sync_paypal_data(credentials, request, user_context)
        # Tier 3 platforms
        elif request.platform == DataSource.TIKTOK:
            raw_data = await sync_tiktok_data(credentials, request, user_context)
        elif request.platform == DataSource.SNOWFLAKE:
            raw_data = await sync_snowflake_data(credentials, request, user_context)
        elif request.platform == DataSource.DATABRICKS:
            raw_data = await sync_databricks_data(credentials, request, user_context)
        # Tier 4 platforms
        elif request.platform == DataSource.ZOHO_CRM:
            raw_data = await sync_zoho_crm_data(credentials, request, user_context)
        elif request.platform == DataSource.LINKEDIN_ADS:
            raw_data = await sync_linkedin_ads_data(credentials, request, user_context)
        elif request.platform == DataSource.X_ADS:
            raw_data = await sync_x_ads_data(credentials, request, user_context)
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
        elif platform == DataSource.SHOPIFY:
            return await credential_manager.get_shopify_credentials(org_id)
        elif platform == DataSource.WOOCOMMERCE:
            return await credential_manager.get_woocommerce_credentials(org_id)
        elif platform == DataSource.AMAZON_SELLER_CENTRAL:
            return await credential_manager.get_amazon_credentials(org_id)
        # Tier 2 platforms
        elif platform == DataSource.HUBSPOT:
            return await credential_manager.get_hubspot_credentials(org_id)
        elif platform == DataSource.SALESFORCE:
            return await credential_manager.get_salesforce_credentials(org_id)
        elif platform == DataSource.STRIPE:
            return await credential_manager.get_stripe_credentials(org_id)
        elif platform == DataSource.PAYPAL:
            return await credential_manager.get_paypal_credentials(org_id)
        # Tier 3 platforms
        elif platform == DataSource.TIKTOK:
            return await credential_manager.get_tiktok_credentials(org_id)
        elif platform == DataSource.SNOWFLAKE:
            return await credential_manager.get_snowflake_credentials(org_id)
        elif platform == DataSource.DATABRICKS:
            return await credential_manager.get_databricks_credentials(org_id)
        # Tier 4 platforms
        elif platform == DataSource.ZOHO_CRM:
            return await credential_manager.get_zoho_crm_credentials(org_id)
        elif platform == DataSource.LINKEDIN_ADS:
            return await credential_manager.get_linkedin_ads_credentials(org_id)
        elif platform == DataSource.X_ADS:
            return await credential_manager.get_x_ads_credentials(org_id)
        else:
            logger.error(f"Unsupported platform: {platform}")
            return None
    except Exception as e:
        logger.error(f"Error retrieving credentials for {platform}: {str(e)}")
        return None


async def sync_meta_business_data(credentials: Dict[str, str], request: SyncJobRequest) -> List[Dict[str, Any]]:
    """Sync data from Meta Business API with KSE integration"""
    connector = await create_meta_business_connector(credentials)
    
    try:
        # Extract enhanced data with KSE integration
        meta_data = await connector.extract_data(
            request.date_range_start,
            request.date_range_end
        )
        
        # Transform to standard format for backward compatibility
        all_data = []
        for data in meta_data:
            campaign_data = {
                "id": f"meta_{data.campaign_id}",
                "account_id": data.account_id,
                "campaign_id": data.campaign_id,
                "campaign_name": data.campaign_name,
                "status": data.status,
                "objective": data.objective,
                "spend": data.spend,
                "impressions": data.impressions,
                "clicks": data.clicks,
                "reach": data.reach,
                "frequency": data.frequency,
                "cpm": data.cpm,
                "cpc": data.cpc,
                "ctr": data.ctr,
                "video_views": data.video_views,
                "actions": data.actions,
                "created_time": data.created_time,
                "updated_time": data.updated_time,
                # KSE enhanced fields
                "kse_enhanced": True,
                "data_quality_score": getattr(data, 'quality_score', None),
                "semantic_embeddings": getattr(data, 'embeddings', None)
            }
            all_data.append(campaign_data)
        
        return all_data
        
    finally:
        await connector.close()


async def sync_google_ads_data(credentials: Dict[str, str], request: SyncJobRequest) -> List[Dict[str, Any]]:
    """Sync data from Google Ads API with KSE integration"""
    connector = await create_google_ads_connector(credentials)
    
    try:
        # Extract enhanced data with KSE integration
        google_ads_data = await connector.extract_data(
            request.date_range_start,
            request.date_range_end
        )
        
        # Transform to standard format for backward compatibility
        all_data = []
        for data in google_ads_data:
            campaign_data = {
                "id": f"google_{data.campaign_id}",
                "customer_id": data.customer_id,
                "campaign_id": data.campaign_id,
                "campaign_name": data.campaign_name,
                "status": data.status,
                "advertising_channel_type": data.advertising_channel_type,
                "cost_micros": data.cost_micros,
                "impressions": data.impressions,
                "clicks": data.clicks,
                "conversions": data.conversions,
                "conversions_value": data.conversions_value,
                "date": data.date,
                # KSE enhanced fields
                "kse_enhanced": True,
                "data_quality_score": getattr(data, 'quality_score', None),
                "semantic_embeddings": getattr(data, 'embeddings', None)
            }
            all_data.append(campaign_data)
        
        return all_data
        
    finally:
        await connector.close()


async def sync_klaviyo_data(credentials: Dict[str, str], request: SyncJobRequest) -> List[Dict[str, Any]]:
    """Sync data from Klaviyo API with KSE integration"""
    connector = await create_klaviyo_connector(credentials)
    
    try:
        # Extract enhanced data with KSE integration
        klaviyo_data = await connector.extract_data(
            request.date_range_start,
            request.date_range_end
        )
        
        # Transform to standard format for backward compatibility
        all_data = []
        for data in klaviyo_data:
            campaign_data = {
                "id": f"klaviyo_{data.campaign_id}",
                "campaign_id": data.campaign_id,
                "campaign_name": data.campaign_name,
                "status": data.status,
                "send_time": data.send_time,
                "sent_count": data.sent_count,
                "delivered_count": data.delivered_count,
                "open_count": data.open_count,
                "click_count": data.click_count,
                "unsubscribe_count": data.unsubscribe_count,
                "bounce_count": data.bounce_count,
                "open_rate": data.open_rate,
                "click_rate": data.click_rate,
                "unsubscribe_rate": data.unsubscribe_rate,
                "bounce_rate": data.bounce_rate,
                "subject_line": data.subject_line,
                "campaign_type": data.campaign_type,
                # KSE enhanced fields
                "kse_enhanced": True,
                "data_quality_score": getattr(data, 'quality_score', None),
                "semantic_embeddings": getattr(data, 'embeddings', None)
            }
            all_data.append(campaign_data)
        
        return all_data
        
    finally:
        await connector.close()


async def sync_shopify_data(credentials: Dict[str, str], request: SyncJobRequest, user_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Sync data from Shopify API with KSE integration"""
    connector = ShopifyConnector(
        shop_domain=credentials["shop_domain"],
        access_token=credentials["access_token"]
    )
    
    try:
        # Get historical data for causal analysis
        historical_data = await get_historical_data(
            request.platform.value,
            user_context,
            days_back=30
        )
        
        # Extract causal marketing data
        causal_data_list = await connector.extract_causal_marketing_data(
            org_id=user_context["org_id"],
            start_date=request.date_range_start,
            end_date=request.date_range_end,
            historical_data=historical_data
        )
        
        # Convert to standard format for compatibility
        all_data = []
        for causal_data in causal_data_list:
            data_dict = causal_data.model_dump()
            data_dict["id"] = f"shopify_{data_dict.get('record_id', '')}"
            data_dict["platform"] = "shopify"
            all_data.append(data_dict)
        
        logger.info(f"Synced {len(all_data)} Shopify records with KSE integration")
        return all_data
        
    finally:
        await connector.close()


async def sync_woocommerce_data(credentials: Dict[str, str], request: SyncJobRequest, user_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Sync data from WooCommerce API with KSE integration"""
    connector = WooCommerceConnector(
        site_url=credentials["site_url"],
        consumer_key=credentials["consumer_key"],
        consumer_secret=credentials["consumer_secret"]
    )
    
    try:
        # Get historical data for causal analysis
        historical_data = await get_historical_data(
            request.platform.value,
            user_context,
            days_back=30
        )
        
        # Extract causal marketing data
        causal_data_list = await connector.extract_causal_marketing_data(
            org_id=user_context["org_id"],
            start_date=request.date_range_start,
            end_date=request.date_range_end,
            historical_data=historical_data
        )
        
        # Convert to standard format for compatibility
        all_data = []
        for causal_data in causal_data_list:
            data_dict = causal_data.model_dump()
            data_dict["id"] = f"woocommerce_{data_dict.get('record_id', '')}"
            data_dict["platform"] = "woocommerce"
            all_data.append(data_dict)
        
        logger.info(f"Synced {len(all_data)} WooCommerce records with KSE integration")
        return all_data
        
    finally:
        await connector.close()


async def sync_amazon_data(credentials: Dict[str, str], request: SyncJobRequest, user_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Sync data from Amazon Seller Central API with KSE integration"""
    connector = AmazonConnector(
        marketplace_id=credentials["marketplace_id"],
        seller_id=credentials["seller_id"],
        aws_access_key=credentials["aws_access_key"],
        aws_secret_key=credentials["aws_secret_key"],
        role_arn=credentials["role_arn"],
        client_id=credentials["client_id"],
        client_secret=credentials["client_secret"],
        refresh_token=credentials["refresh_token"]
    )
    
    try:
        # Get historical data for causal analysis
        historical_data = await get_historical_data(
            request.platform.value,
            user_context,
            days_back=30
        )
        
        # Extract causal marketing data
        causal_data_list = await connector.extract_causal_marketing_data(
            org_id=user_context["org_id"],
            start_date=request.date_range_start,
            end_date=request.date_range_end,
            historical_data=historical_data
        )
        
        # Convert to standard format for compatibility
        all_data = []
        for causal_data in causal_data_list:
            data_dict = causal_data.model_dump()
            data_dict["id"] = f"amazon_{data_dict.get('record_id', '')}"
            data_dict["platform"] = "amazon_seller_central"
            all_data.append(data_dict)
        
        logger.info(f"Synced {len(all_data)} Amazon records with KSE integration")
        return all_data
        
    finally:
        await connector.close()


async def sync_hubspot_data(credentials: Dict[str, str], request: SyncJobRequest, user_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Sync data from HubSpot CRM API with KSE integration"""
    connector = HubSpotConnector(api_key=credentials["api_key"])
    
    try:
        # Get historical data for causal analysis
        historical_data = await get_historical_data(
            request.platform.value,
            user_context,
            days_back=30
        )
        
        # Extract causal marketing data
        causal_data_list = await connector.extract_causal_marketing_data(
            org_id=user_context["org_id"],
            start_date=request.date_range_start,
            end_date=request.date_range_end,
            historical_data=historical_data
        )
        
        # Convert to standard format for compatibility
        all_data = []
        for causal_data in causal_data_list:
            data_dict = causal_data.model_dump()
            data_dict["id"] = f"hubspot_{data_dict.get('record_id', '')}"
            data_dict["platform"] = "hubspot"
            all_data.append(data_dict)
        
        logger.info(f"Synced {len(all_data)} HubSpot records with KSE integration")
        return all_data
        
    finally:
        await connector.close()


async def sync_salesforce_data(credentials: Dict[str, str], request: SyncJobRequest, user_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Sync data from Salesforce CRM API with KSE integration"""
    connector = SalesforceConnector(
        username=credentials["username"],
        password=credentials["password"],
        security_token=credentials["security_token"],
        client_id=credentials["client_id"],
        client_secret=credentials["client_secret"],
        is_sandbox=credentials.get("is_sandbox", "false").lower() == "true"
    )
    
    try:
        # Get historical data for causal analysis
        historical_data = await get_historical_data(
            request.platform.value,
            user_context,
            days_back=30
        )
        
        # Extract causal marketing data
        causal_data_list = await connector.extract_causal_marketing_data(
            org_id=user_context["org_id"],
            start_date=request.date_range_start,
            end_date=request.date_range_end,
            historical_data=historical_data
        )
        
        # Convert to standard format for compatibility
        all_data = []
        for causal_data in causal_data_list:
            data_dict = causal_data.model_dump()
            data_dict["id"] = f"salesforce_{data_dict.get('record_id', '')}"
            data_dict["platform"] = "salesforce"
            all_data.append(data_dict)
        
        logger.info(f"Synced {len(all_data)} Salesforce records with KSE integration")
        return all_data
        
    finally:
        await connector.close()


async def sync_stripe_data(credentials: Dict[str, str], request: SyncJobRequest, user_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Sync data from Stripe Payment API with KSE integration"""
    connector = StripeConnector(api_key=credentials["api_key"])
    
    try:
        # Get historical data for causal analysis
        historical_data = await get_historical_data(
            request.platform.value,
            user_context,
            days_back=30
        )
        
        # Extract causal marketing data
        causal_data_list = await connector.extract_causal_marketing_data(
            org_id=user_context["org_id"],
            start_date=request.date_range_start,
            end_date=request.date_range_end,
            historical_data=historical_data
        )
        
        # Convert to standard format for compatibility
        all_data = []
        for causal_data in causal_data_list:
            data_dict = causal_data.model_dump()
            data_dict["id"] = f"stripe_{data_dict.get('record_id', '')}"
            data_dict["platform"] = "stripe"
            all_data.append(data_dict)
        
        logger.info(f"Synced {len(all_data)} Stripe records with KSE integration")
        return all_data
        
    finally:
        await connector.close()


async def sync_paypal_data(credentials: Dict[str, str], request: SyncJobRequest, user_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Sync data from PayPal Payment API with KSE integration"""
    connector = PayPalConnector(
        client_id=credentials["client_id"],
        client_secret=credentials["client_secret"],
        is_sandbox=credentials.get("is_sandbox", "false").lower() == "true"
    )
    
    try:
        # Get historical data for causal analysis
        historical_data = await get_historical_data(
            request.platform.value,
            user_context,
            days_back=30
        )
        
        # Extract causal marketing data
        causal_data_list = await connector.extract_causal_marketing_data(
            org_id=user_context["org_id"],
            start_date=request.date_range_start,
            end_date=request.date_range_end,
            historical_data=historical_data
        )
        
        # Convert to standard format for compatibility
        all_data = []
        for causal_data in causal_data_list:
            data_dict = causal_data.model_dump()
            data_dict["id"] = f"paypal_{data_dict.get('record_id', '')}"
            data_dict["platform"] = "paypal"
            all_data.append(data_dict)
        
        logger.info(f"Synced {len(all_data)} PayPal records with KSE integration")
        return all_data
        
    finally:
        await connector.close()


async def sync_tiktok_data(credentials: Dict[str, str], request: SyncJobRequest, user_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Sync data from TikTok for Business API with KSE integration"""
    async with TikTokConnector(credentials) as connector:
        try:
            # Sync TikTok data with causal analysis
            sync_result = await connector.sync_data(
                str(request.date_range_start),
                str(request.date_range_end)
            )
            
            # Convert to standard format for compatibility
            all_data = []
            for campaign in sync_result.get("campaigns", []):
                data_dict = campaign.copy()
                data_dict["id"] = f"tiktok_{campaign.get('campaign_id', '')}"
                data_dict["platform"] = "tiktok"
                all_data.append(data_dict)
            
            logger.info(f"Synced {len(all_data)} TikTok campaigns with KSE integration")
            return all_data
            
        except Exception as e:
            logger.error(f"TikTok sync failed: {str(e)}")
            raise


async def sync_snowflake_data(credentials: Dict[str, str], request: SyncJobRequest, user_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Sync data from Snowflake Data Warehouse with KSE integration"""
    async with SnowflakeConnector(credentials) as connector:
        try:
            # Sync Snowflake data with quality analysis
            sync_result = await connector.sync_data(
                str(request.date_range_start),
                str(request.date_range_end)
            )
            
            # Convert to standard format for compatibility
            all_data = []
            for table in sync_result.get("tables", []):
                data_dict = table.copy()
                data_dict["id"] = f"snowflake_{table.get('table_name', '')}"
                data_dict["platform"] = "snowflake"
                all_data.append(data_dict)
            
            logger.info(f"Analyzed {len(all_data)} Snowflake tables with KSE integration")
            return all_data
            
        except Exception as e:
            logger.error(f"Snowflake sync failed: {str(e)}")
            raise


async def sync_databricks_data(credentials: Dict[str, str], request: SyncJobRequest, user_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Sync data from Databricks Analytics Platform with KSE integration"""
    async with DatabricksConnector(credentials) as connector:
        try:
            # Sync Databricks data with ML workflow analysis
            sync_result = await connector.sync_data(
                str(request.date_range_start),
                str(request.date_range_end)
            )
            
            # Convert to standard format for compatibility
            all_data = []
            for job in sync_result.get("jobs", []):
                data_dict = job.copy()
                data_dict["id"] = f"databricks_{job.get('job_id', '')}"
                data_dict["platform"] = "databricks"
                all_data.append(data_dict)
            
            logger.info(f"Analyzed {len(all_data)} Databricks jobs with KSE integration")
            return all_data
            
        except Exception as e:
            logger.error(f"Databricks sync failed: {str(e)}")
            raise


async def sync_zoho_crm_data(credentials: Dict[str, str], request: SyncJobRequest, user_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Sync data from Zoho CRM API with KSE integration"""
    # Import the credentials class
    from connectors.zoho_crm_connector import ZohoCRMCredentials
    
    # Create credentials object
    zoho_creds = ZohoCRMCredentials(
        client_id=credentials["client_id"],
        client_secret=credentials["client_secret"],
        refresh_token=credentials["refresh_token"],
        domain=credentials.get("domain", "com")
    )
    
    async with ZohoCRMConnector(zoho_creds) as connector:
        try:
            # Sync Zoho CRM data with pipeline analytics
            sync_result = await connector.sync_data(
                str(request.date_range_start),
                str(request.date_range_end)
            )
            
            # Convert to standard format for compatibility
            all_data = []
            for record in sync_result.get("records", []):
                data_dict = record.copy()
                data_dict["id"] = f"zoho_crm_{record.get('record_id', '')}"
                data_dict["platform"] = "zoho_crm"
                all_data.append(data_dict)
            
            logger.info(f"Synced {len(all_data)} Zoho CRM records with KSE integration")
            return all_data
            
        except Exception as e:
            logger.error(f"Zoho CRM sync failed: {str(e)}")
            raise


async def sync_linkedin_ads_data(credentials: Dict[str, str], request: SyncJobRequest, user_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Sync data from LinkedIn Ads API with KSE integration"""
    # Import the credentials class
    from connectors.linkedin_ads_connector import LinkedInAdsCredentials
    
    # Create credentials object
    linkedin_creds = LinkedInAdsCredentials(
        client_id=credentials["client_id"],
        client_secret=credentials["client_secret"],
        access_token=credentials["access_token"]
    )
    
    async with LinkedInAdsConnector(linkedin_creds) as connector:
        try:
            # Sync LinkedIn Ads data with professional targeting analytics
            sync_result = await connector.sync_data(
                str(request.date_range_start),
                str(request.date_range_end)
            )
            
            # Convert to standard format for compatibility
            all_data = []
            for campaign in sync_result.get("campaigns", []):
                data_dict = campaign.copy()
                data_dict["id"] = f"linkedin_ads_{campaign.get('campaign_id', '')}"
                data_dict["platform"] = "linkedin_ads"
                all_data.append(data_dict)
            
            logger.info(f"Synced {len(all_data)} LinkedIn Ads campaigns with KSE integration")
            return all_data
            
        except Exception as e:
            logger.error(f"LinkedIn Ads sync failed: {str(e)}")
            raise


async def sync_x_ads_data(credentials: Dict[str, str], request: SyncJobRequest, user_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Sync data from X (Twitter) Ads API with KSE integration"""
    # Import the credentials class
    from connectors.x_ads_connector import XAdsCredentials
    
    # Create credentials object
    x_ads_creds = XAdsCredentials(
        consumer_key=credentials["consumer_key"],
        consumer_secret=credentials["consumer_secret"],
        access_token=credentials["access_token"],
        access_token_secret=credentials["access_token_secret"]
    )
    
    async with XAdsConnector(x_ads_creds) as connector:
        try:
            # Sync X Ads data with viral amplification analytics
            sync_result = await connector.sync_data(
                str(request.date_range_start),
                str(request.date_range_end)
            )
            
            # Convert to standard format for compatibility
            all_data = []
            for campaign in sync_result.get("campaigns", []):
                data_dict = campaign.copy()
                data_dict["id"] = f"x_ads_{campaign.get('campaign_id', '')}"
                data_dict["platform"] = "x_ads"
                all_data.append(data_dict)
            
            logger.info(f"Synced {len(all_data)} X Ads campaigns with KSE integration")
            return all_data
            
        except Exception as e:
            logger.error(f"X Ads sync failed: {str(e)}")
            raise


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