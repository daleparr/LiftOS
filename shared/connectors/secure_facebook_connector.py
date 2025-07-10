"""
Secure Facebook Marketing API Connector
Enterprise-grade Facebook API integration with comprehensive security
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
import sys
import os

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.security.audit_logger import SecurityAuditLogger, SecurityEventType
from shared.connectors.base_secure_connector import BaseSecureConnector
from shared.utils.logging import setup_logging

logger = setup_logging("facebook_connector")

class SecureFacebookConnector(BaseSecureConnector):
    """Secure Facebook Marketing API connector with enterprise security"""
    
    def __init__(self):
        super().__init__(provider="facebook")
        self.base_url = "https://graph.facebook.com/v18.0"
        self.required_credentials = ["access_token", "app_secret", "app_id"]
        self.optional_credentials = ["business_id", "ad_account_id"]
        
        # Facebook-specific rate limits
        self.rate_limits = {
            "default": {"requests": 200, "window": 3600},  # 200 per hour
            "ads_insights": {"requests": 100, "window": 3600},  # 100 per hour
            "batch": {"requests": 50, "window": 3600}  # 50 per hour
        }
    
    async def validate_credentials(self, credentials: Dict[str, str]) -> Dict[str, Any]:
        """Validate Facebook API credentials"""
        try:
            validation_result = await super().validate_credentials(credentials)
            
            if not validation_result["valid"]:
                return validation_result
            
            # Facebook-specific validation
            access_token = credentials.get("access_token", "")
            app_id = credentials.get("app_id", "")
            
            # Validate access token format
            if not access_token.startswith("EAA"):
                validation_result["warnings"].append(
                    "Facebook access token should start with 'EAA'"
                )
            
            # Validate app ID format
            if not app_id.isdigit():
                validation_result["warnings"].append(
                    "Facebook app ID should be numeric"
                )
            
            # Test API connection
            test_result = await self._test_api_connection(credentials)
            if not test_result["success"]:
                validation_result["valid"] = False
                validation_result["warnings"].append(
                    f"API connection test failed: {test_result['error']}"
                )
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Facebook credential validation failed: {e}")
            return {
                "valid": False,
                "error": str(e),
                "missing_credentials": [],
                "unexpected_credentials": [],
                "warnings": []
            }
    
    async def _test_api_connection(self, credentials: Dict[str, str]) -> Dict[str, Any]:
        """Test Facebook API connection"""
        try:
            access_token = credentials["access_token"]
            
            # Test with a simple API call
            url = f"{self.base_url}/me"
            params = {
                "access_token": access_token,
                "fields": "id,name"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "success": True,
                            "user_id": data.get("id"),
                            "user_name": data.get("name")
                        }
                    else:
                        error_data = await response.json()
                        return {
                            "success": False,
                            "error": error_data.get("error", {}).get("message", "Unknown error"),
                            "status_code": response.status
                        }
                        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_ad_accounts(
        self,
        user_id: str,
        org_id: str,
        ip_address: str
    ) -> Dict[str, Any]:
        """Get Facebook ad accounts"""
        try:
            credentials = await self.get_credentials(user_id, org_id)
            
            url = f"{self.base_url}/me/adaccounts"
            params = {
                "access_token": credentials["access_token"],
                "fields": "id,name,account_status,currency,timezone_name"
            }
            
            # Apply rate limiting
            await self._check_rate_limit("default", ip_address)
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Log successful API access
                        await self.audit_logger.log_event(
                            event_type=SecurityEventType.API_ACCESS,
                            action="facebook_ad_accounts_retrieved",
                            user_id=user_id,
                            org_id=org_id,
                            ip_address=ip_address,
                            details={
                                "provider": "facebook",
                                "endpoint": "adaccounts",
                                "account_count": len(data.get("data", [])),
                                "success": True
                            }
                        )
                        
                        return {
                            "success": True,
                            "data": data.get("data", []),
                            "paging": data.get("paging", {})
                        }
                    else:
                        error_data = await response.json()
                        error_message = error_data.get("error", {}).get("message", "Unknown error")
                        
                        # Log API error
                        await self.audit_logger.log_event(
                            event_type=SecurityEventType.API_ACCESS,
                            action="facebook_api_error",
                            user_id=user_id,
                            org_id=org_id,
                            ip_address=ip_address,
                            details={
                                "provider": "facebook",
                                "endpoint": "adaccounts",
                                "error": error_message,
                                "status_code": response.status,
                                "success": False
                            }
                        )
                        
                        return {
                            "success": False,
                            "error": error_message,
                            "status_code": response.status
                        }
                        
        except Exception as e:
            await self.audit_logger.log_event(
                event_type=SecurityEventType.SECURITY_VIOLATION,
                action="facebook_api_exception",
                user_id=user_id,
                org_id=org_id,
                ip_address=ip_address,
                details={
                    "provider": "facebook",
                    "endpoint": "adaccounts",
                    "error": str(e),
                    "success": False
                }
            )
            logger.error(f"Facebook ad accounts retrieval failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_campaigns(
        self,
        ad_account_id: str,
        user_id: str,
        org_id: str,
        ip_address: str,
        date_range: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Get Facebook campaigns"""
        try:
            credentials = await self.get_credentials(user_id, org_id)
            
            url = f"{self.base_url}/{ad_account_id}/campaigns"
            params = {
                "access_token": credentials["access_token"],
                "fields": "id,name,status,objective,created_time,updated_time,start_time,stop_time"
            }
            
            # Add date filtering if provided
            if date_range:
                if "since" in date_range:
                    params["time_range"] = json.dumps({
                        "since": date_range["since"],
                        "until": date_range.get("until", datetime.now().strftime("%Y-%m-%d"))
                    })
            
            # Apply rate limiting
            await self._check_rate_limit("default", ip_address)
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Log successful API access
                        await self.audit_logger.log_event(
                            event_type=SecurityEventType.API_ACCESS,
                            action="facebook_campaigns_retrieved",
                            user_id=user_id,
                            org_id=org_id,
                            ip_address=ip_address,
                            details={
                                "provider": "facebook",
                                "endpoint": "campaigns",
                                "ad_account_id": ad_account_id,
                                "campaign_count": len(data.get("data", [])),
                                "date_range": date_range,
                                "success": True
                            }
                        )
                        
                        return {
                            "success": True,
                            "data": data.get("data", []),
                            "paging": data.get("paging", {})
                        }
                    else:
                        error_data = await response.json()
                        error_message = error_data.get("error", {}).get("message", "Unknown error")
                        
                        # Log API error
                        await self.audit_logger.log_event(
                            event_type=SecurityEventType.API_ACCESS,
                            action="facebook_api_error",
                            user_id=user_id,
                            org_id=org_id,
                            ip_address=ip_address,
                            details={
                                "provider": "facebook",
                                "endpoint": "campaigns",
                                "ad_account_id": ad_account_id,
                                "error": error_message,
                                "status_code": response.status,
                                "success": False
                            }
                        )
                        
                        return {
                            "success": False,
                            "error": error_message,
                            "status_code": response.status
                        }
                        
        except Exception as e:
            await self.audit_logger.log_event(
                event_type=SecurityEventType.SECURITY_VIOLATION,
                action="facebook_api_exception",
                user_id=user_id,
                org_id=org_id,
                ip_address=ip_address,
                details={
                    "provider": "facebook",
                    "endpoint": "campaigns",
                    "ad_account_id": ad_account_id,
                    "error": str(e),
                    "success": False
                }
            )
            logger.error(f"Facebook campaigns retrieval failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_insights(
        self,
        object_id: str,
        object_type: str,  # campaign, adset, ad
        user_id: str,
        org_id: str,
        ip_address: str,
        date_range: Dict[str, str],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get Facebook Ads insights"""
        try:
            credentials = await self.get_credentials(user_id, org_id)
            
            # Default metrics if not specified
            if not metrics:
                metrics = [
                    "impressions", "clicks", "spend", "reach", "frequency",
                    "cpm", "cpc", "ctr", "conversions", "cost_per_conversion"
                ]
            
            url = f"{self.base_url}/{object_id}/insights"
            params = {
                "access_token": credentials["access_token"],
                "fields": ",".join(metrics),
                "time_range": json.dumps({
                    "since": date_range["since"],
                    "until": date_range["until"]
                }),
                "level": object_type
            }
            
            # Apply stricter rate limiting for insights
            await self._check_rate_limit("ads_insights", ip_address)
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Log successful insights access
                        await self.audit_logger.log_event(
                            event_type=SecurityEventType.API_ACCESS,
                            action="facebook_insights_retrieved",
                            user_id=user_id,
                            org_id=org_id,
                            ip_address=ip_address,
                            details={
                                "provider": "facebook",
                                "endpoint": "insights",
                                "object_id": object_id,
                                "object_type": object_type,
                                "metrics": metrics,
                                "date_range": date_range,
                                "data_points": len(data.get("data", [])),
                                "success": True
                            }
                        )
                        
                        return {
                            "success": True,
                            "data": data.get("data", []),
                            "paging": data.get("paging", {})
                        }
                    else:
                        error_data = await response.json()
                        error_message = error_data.get("error", {}).get("message", "Unknown error")
                        
                        # Log insights error
                        await self.audit_logger.log_event(
                            event_type=SecurityEventType.API_ACCESS,
                            action="facebook_insights_error",
                            user_id=user_id,
                            org_id=org_id,
                            ip_address=ip_address,
                            details={
                                "provider": "facebook",
                                "endpoint": "insights",
                                "object_id": object_id,
                                "object_type": object_type,
                                "error": error_message,
                                "status_code": response.status,
                                "success": False
                            }
                        )
                        
                        return {
                            "success": False,
                            "error": error_message,
                            "status_code": response.status
                        }
                        
        except Exception as e:
            await self.audit_logger.log_event(
                event_type=SecurityEventType.SECURITY_VIOLATION,
                action="facebook_insights_exception",
                user_id=user_id,
                org_id=org_id,
                ip_address=ip_address,
                details={
                    "provider": "facebook",
                    "endpoint": "insights",
                    "object_id": object_id,
                    "object_type": object_type,
                    "error": str(e),
                    "success": False
                }
            )
            logger.error(f"Facebook insights retrieval failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def sync_data(
        self,
        user_id: str,
        org_id: str,
        ip_address: str,
        sync_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Sync Facebook data with comprehensive security logging"""
        try:
            sync_start = datetime.now(timezone.utc)
            
            # Log sync start
            await self.audit_logger.log_event(
                event_type=SecurityEventType.RESOURCE_ACCESSED,
                action="facebook_sync_started",
                user_id=user_id,
                org_id=org_id,
                ip_address=ip_address,
                details={
                    "provider": "facebook",
                    "sync_config": sync_config,
                    "sync_start": sync_start.isoformat()
                }
            )
            
            sync_results = {
                "ad_accounts": [],
                "campaigns": [],
                "insights": [],
                "errors": []
            }
            
            # Get ad accounts
            ad_accounts_result = await self.get_ad_accounts(user_id, org_id, ip_address)
            if ad_accounts_result["success"]:
                sync_results["ad_accounts"] = ad_accounts_result["data"]
            else:
                sync_results["errors"].append({
                    "step": "ad_accounts",
                    "error": ad_accounts_result["error"]
                })
            
            # Get campaigns for each ad account
            for ad_account in sync_results["ad_accounts"]:
                campaigns_result = await self.get_campaigns(
                    ad_account["id"], user_id, org_id, ip_address,
                    sync_config.get("date_range")
                )
                
                if campaigns_result["success"]:
                    sync_results["campaigns"].extend(campaigns_result["data"])
                else:
                    sync_results["errors"].append({
                        "step": "campaigns",
                        "ad_account_id": ad_account["id"],
                        "error": campaigns_result["error"]
                    })
            
            # Get insights if requested
            if sync_config.get("include_insights", False):
                for campaign in sync_results["campaigns"]:
                    insights_result = await self.get_insights(
                        campaign["id"], "campaign", user_id, org_id, ip_address,
                        sync_config.get("date_range", {
                            "since": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
                            "until": datetime.now().strftime("%Y-%m-%d")
                        })
                    )
                    
                    if insights_result["success"]:
                        sync_results["insights"].extend(insights_result["data"])
                    else:
                        sync_results["errors"].append({
                            "step": "insights",
                            "campaign_id": campaign["id"],
                            "error": insights_result["error"]
                        })
            
            sync_end = datetime.now(timezone.utc)
            sync_duration = (sync_end - sync_start).total_seconds()
            
            # Log sync completion
            await self.audit_logger.log_event(
                event_type=SecurityEventType.RESOURCE_ACCESSED,
                action="facebook_sync_completed",
                user_id=user_id,
                org_id=org_id,
                ip_address=ip_address,
                details={
                    "provider": "facebook",
                    "sync_duration": sync_duration,
                    "ad_accounts_count": len(sync_results["ad_accounts"]),
                    "campaigns_count": len(sync_results["campaigns"]),
                    "insights_count": len(sync_results["insights"]),
                    "errors_count": len(sync_results["errors"]),
                    "success": len(sync_results["errors"]) == 0
                }
            )
            
            return {
                "success": True,
                "data": sync_results,
                "sync_duration": sync_duration,
                "sync_start": sync_start.isoformat(),
                "sync_end": sync_end.isoformat()
            }
            
        except Exception as e:
            await self.audit_logger.log_event(
                event_type=SecurityEventType.SECURITY_VIOLATION,
                action="facebook_sync_exception",
                user_id=user_id,
                org_id=org_id,
                ip_address=ip_address,
                details={
                    "provider": "facebook",
                    "error": str(e),
                    "success": False
                }
            )
            logger.error(f"Facebook sync failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }