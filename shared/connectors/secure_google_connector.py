"""
Secure Google Ads API Connector
Enterprise-grade Google Ads API integration with comprehensive security
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

logger = setup_logging("google_connector")

class SecureGoogleConnector(BaseSecureConnector):
    """Secure Google Ads API connector with enterprise security"""
    
    def __init__(self):
        super().__init__(provider="google")
        self.base_url = "https://googleads.googleapis.com/v14"
        self.oauth_url = "https://oauth2.googleapis.com/token"
        self.required_credentials = ["client_id", "client_secret", "refresh_token"]
        self.optional_credentials = ["developer_token", "customer_id"]
        
        # Google Ads specific rate limits
        self.rate_limits = {
            "default": {"requests": 1000, "window": 3600},  # 1000 per hour
            "reports": {"requests": 100, "window": 3600},   # 100 per hour
            "mutations": {"requests": 50, "window": 3600},  # 50 per hour
            "oauth": {"requests": 10, "window": 3600}       # 10 per hour
        }
        
        # Access token cache
        self.access_token_cache: Dict[str, Dict[str, Any]] = {}
        self.token_cache_ttl = 3300  # 55 minutes (tokens expire in 1 hour)
    
    async def validate_credentials(self, credentials: Dict[str, str]) -> Dict[str, Any]:
        """Validate Google Ads API credentials"""
        try:
            validation_result = await super().validate_credentials(credentials)
            
            if not validation_result["valid"]:
                return validation_result
            
            # Google-specific validation
            client_id = credentials.get("client_id", "")
            client_secret = credentials.get("client_secret", "")
            refresh_token = credentials.get("refresh_token", "")
            
            # Validate client ID format
            if not client_id.endswith(".googleusercontent.com"):
                validation_result["warnings"].append(
                    "Google client ID should end with '.googleusercontent.com'"
                )
            
            # Validate refresh token format
            if not refresh_token.startswith("1//"):
                validation_result["warnings"].append(
                    "Google refresh token should start with '1//'"
                )
            
            # Test OAuth token refresh
            test_result = await self._test_token_refresh(credentials)
            if not test_result["success"]:
                validation_result["valid"] = False
                validation_result["warnings"].append(
                    f"OAuth token refresh failed: {test_result['error']}"
                )
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Google credential validation failed: {e}")
            return {
                "valid": False,
                "error": str(e),
                "missing_credentials": [],
                "unexpected_credentials": [],
                "warnings": []
            }
    
    async def _test_token_refresh(self, credentials: Dict[str, str]) -> Dict[str, Any]:
        """Test Google OAuth token refresh"""
        try:
            data = {
                "client_id": credentials["client_id"],
                "client_secret": credentials["client_secret"],
                "refresh_token": credentials["refresh_token"],
                "grant_type": "refresh_token"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.oauth_url, data=data) as response:
                    if response.status == 200:
                        token_data = await response.json()
                        return {
                            "success": True,
                            "access_token": token_data.get("access_token"),
                            "expires_in": token_data.get("expires_in", 3600)
                        }
                    else:
                        error_data = await response.json()
                        return {
                            "success": False,
                            "error": error_data.get("error_description", "Token refresh failed"),
                            "status_code": response.status
                        }
                        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _test_api_connection(self, credentials: Dict[str, str]) -> Dict[str, Any]:
        """Test Google Ads API connection"""
        try:
            # Get access token
            token_result = await self._get_access_token(credentials)
            if not token_result["success"]:
                return token_result
            
            access_token = token_result["access_token"]
            
            # Test with a simple API call (list accessible customers)
            url = f"{self.base_url}/customers:listAccessibleCustomers"
            headers = {
                "Authorization": f"Bearer {access_token}",
                "developer-token": credentials.get("developer_token", ""),
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "success": True,
                            "accessible_customers": data.get("resourceNames", [])
                        }
                    else:
                        error_data = await response.json()
                        return {
                            "success": False,
                            "error": error_data.get("error", {}).get("message", "API connection failed"),
                            "status_code": response.status
                        }
                        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _get_access_token(self, credentials: Dict[str, str]) -> Dict[str, Any]:
        """Get or refresh Google OAuth access token"""
        try:
            # Check cache first
            cache_key = f"google_token_{hash(credentials['refresh_token'])}"
            if cache_key in self.access_token_cache:
                cache_entry = self.access_token_cache[cache_key]
                if datetime.now(timezone.utc) < cache_entry["expires_at"]:
                    return {
                        "success": True,
                        "access_token": cache_entry["access_token"]
                    }
                else:
                    del self.access_token_cache[cache_key]
            
            # Refresh token
            token_result = await self._test_token_refresh(credentials)
            if not token_result["success"]:
                return token_result
            
            # Cache the new token
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=self.token_cache_ttl)
            self.access_token_cache[cache_key] = {
                "access_token": token_result["access_token"],
                "expires_at": expires_at
            }
            
            return {
                "success": True,
                "access_token": token_result["access_token"]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_accessible_customers(
        self,
        user_id: str,
        org_id: str,
        ip_address: str
    ) -> Dict[str, Any]:
        """Get Google Ads accessible customers"""
        try:
            credentials = await self.get_credentials(user_id, org_id)
            
            # Get access token
            token_result = await self._get_access_token(credentials)
            if not token_result["success"]:
                return token_result
            
            access_token = token_result["access_token"]
            
            # Apply rate limiting
            await self._check_rate_limit("default", ip_address)
            
            url = f"{self.base_url}/customers:listAccessibleCustomers"
            headers = {
                "Authorization": f"Bearer {access_token}",
                "developer-token": credentials.get("developer_token", ""),
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Log successful API access
                        await self.audit_logger.log_event(
                            event_type=SecurityEventType.API_ACCESS,
                            action="google_customers_retrieved",
                            user_id=user_id,
                            org_id=org_id,
                            ip_address=ip_address,
                            details={
                                "provider": "google",
                                "endpoint": "listAccessibleCustomers",
                                "customer_count": len(data.get("resourceNames", [])),
                                "success": True
                            }
                        )
                        
                        return {
                            "success": True,
                            "data": data.get("resourceNames", [])
                        }
                    else:
                        error_data = await response.json()
                        error_message = error_data.get("error", {}).get("message", "Unknown error")
                        
                        # Log API error
                        await self.audit_logger.log_event(
                            event_type=SecurityEventType.API_ACCESS,
                            action="google_api_error",
                            user_id=user_id,
                            org_id=org_id,
                            ip_address=ip_address,
                            details={
                                "provider": "google",
                                "endpoint": "listAccessibleCustomers",
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
                action="google_api_exception",
                user_id=user_id,
                org_id=org_id,
                ip_address=ip_address,
                details={
                    "provider": "google",
                    "endpoint": "listAccessibleCustomers",
                    "error": str(e),
                    "success": False
                }
            )
            logger.error(f"Google accessible customers retrieval failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_campaigns(
        self,
        customer_id: str,
        user_id: str,
        org_id: str,
        ip_address: str,
        date_range: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Get Google Ads campaigns"""
        try:
            credentials = await self.get_credentials(user_id, org_id)
            
            # Get access token
            token_result = await self._get_access_token(credentials)
            if not token_result["success"]:
                return token_result
            
            access_token = token_result["access_token"]
            
            # Apply rate limiting
            await self._check_rate_limit("default", ip_address)
            
            # Build GAQL query
            query = """
                SELECT 
                    campaign.id,
                    campaign.name,
                    campaign.status,
                    campaign.advertising_channel_type,
                    campaign.start_date,
                    campaign.end_date,
                    campaign.serving_status
                FROM campaign
                ORDER BY campaign.id
            """
            
            # Add date filtering if provided
            if date_range:
                if "since" in date_range:
                    query += f" WHERE segments.date >= '{date_range['since']}'"
                    if "until" in date_range:
                        query += f" AND segments.date <= '{date_range['until']}'"
            
            url = f"{self.base_url}/customers/{customer_id}/googleAds:searchStream"
            headers = {
                "Authorization": f"Bearer {access_token}",
                "developer-token": credentials.get("developer_token", ""),
                "Content-Type": "application/json"
            }
            
            payload = {"query": query}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Extract campaigns from response
                        campaigns = []
                        for result in data.get("results", []):
                            if "campaign" in result:
                                campaigns.append(result["campaign"])
                        
                        # Log successful API access
                        await self.audit_logger.log_event(
                            event_type=SecurityEventType.API_ACCESS,
                            action="google_campaigns_retrieved",
                            user_id=user_id,
                            org_id=org_id,
                            ip_address=ip_address,
                            details={
                                "provider": "google",
                                "endpoint": "searchStream",
                                "customer_id": customer_id,
                                "campaign_count": len(campaigns),
                                "date_range": date_range,
                                "success": True
                            }
                        )
                        
                        return {
                            "success": True,
                            "data": campaigns
                        }
                    else:
                        error_data = await response.json()
                        error_message = error_data.get("error", {}).get("message", "Unknown error")
                        
                        # Log API error
                        await self.audit_logger.log_event(
                            event_type=SecurityEventType.API_ACCESS,
                            action="google_api_error",
                            user_id=user_id,
                            org_id=org_id,
                            ip_address=ip_address,
                            details={
                                "provider": "google",
                                "endpoint": "searchStream",
                                "customer_id": customer_id,
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
                action="google_api_exception",
                user_id=user_id,
                org_id=org_id,
                ip_address=ip_address,
                details={
                    "provider": "google",
                    "endpoint": "searchStream",
                    "customer_id": customer_id,
                    "error": str(e),
                    "success": False
                }
            )
            logger.error(f"Google campaigns retrieval failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_performance_report(
        self,
        customer_id: str,
        user_id: str,
        org_id: str,
        ip_address: str,
        date_range: Dict[str, str],
        metrics: Optional[List[str]] = None,
        dimensions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get Google Ads performance report"""
        try:
            credentials = await self.get_credentials(user_id, org_id)
            
            # Get access token
            token_result = await self._get_access_token(credentials)
            if not token_result["success"]:
                return token_result
            
            access_token = token_result["access_token"]
            
            # Apply stricter rate limiting for reports
            await self._check_rate_limit("reports", ip_address)
            
            # Default metrics if not specified
            if not metrics:
                metrics = [
                    "metrics.impressions",
                    "metrics.clicks",
                    "metrics.cost_micros",
                    "metrics.conversions",
                    "metrics.ctr",
                    "metrics.average_cpc"
                ]
            
            # Default dimensions if not specified
            if not dimensions:
                dimensions = [
                    "campaign.id",
                    "campaign.name",
                    "segments.date"
                ]
            
            # Build GAQL query
            select_fields = dimensions + metrics
            query = f"""
                SELECT {', '.join(select_fields)}
                FROM campaign
                WHERE segments.date >= '{date_range['since']}'
                AND segments.date <= '{date_range['until']}'
                ORDER BY segments.date DESC
            """
            
            url = f"{self.base_url}/customers/{customer_id}/googleAds:searchStream"
            headers = {
                "Authorization": f"Bearer {access_token}",
                "developer-token": credentials.get("developer_token", ""),
                "Content-Type": "application/json"
            }
            
            payload = {"query": query}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Log successful report access
                        await self.audit_logger.log_event(
                            event_type=SecurityEventType.API_ACCESS,
                            action="google_report_retrieved",
                            user_id=user_id,
                            org_id=org_id,
                            ip_address=ip_address,
                            details={
                                "provider": "google",
                                "endpoint": "searchStream",
                                "customer_id": customer_id,
                                "metrics": metrics,
                                "dimensions": dimensions,
                                "date_range": date_range,
                                "data_points": len(data.get("results", [])),
                                "success": True
                            }
                        )
                        
                        return {
                            "success": True,
                            "data": data.get("results", [])
                        }
                    else:
                        error_data = await response.json()
                        error_message = error_data.get("error", {}).get("message", "Unknown error")
                        
                        # Log report error
                        await self.audit_logger.log_event(
                            event_type=SecurityEventType.API_ACCESS,
                            action="google_report_error",
                            user_id=user_id,
                            org_id=org_id,
                            ip_address=ip_address,
                            details={
                                "provider": "google",
                                "endpoint": "searchStream",
                                "customer_id": customer_id,
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
                action="google_report_exception",
                user_id=user_id,
                org_id=org_id,
                ip_address=ip_address,
                details={
                    "provider": "google",
                    "endpoint": "searchStream",
                    "customer_id": customer_id,
                    "error": str(e),
                    "success": False
                }
            )
            logger.error(f"Google performance report retrieval failed: {e}")
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
        """Sync Google Ads data with comprehensive security logging"""
        try:
            sync_start = datetime.now(timezone.utc)
            
            # Log sync start
            await self.audit_logger.log_event(
                event_type=SecurityEventType.RESOURCE_ACCESSED,
                action="google_sync_started",
                user_id=user_id,
                org_id=org_id,
                ip_address=ip_address,
                details={
                    "provider": "google",
                    "sync_config": sync_config,
                    "sync_start": sync_start.isoformat()
                }
            )
            
            sync_results = {
                "accessible_customers": [],
                "campaigns": [],
                "reports": [],
                "errors": []
            }
            
            # Get accessible customers
            customers_result = await self.get_accessible_customers(user_id, org_id, ip_address)
            if customers_result["success"]:
                sync_results["accessible_customers"] = customers_result["data"]
            else:
                sync_results["errors"].append({
                    "step": "accessible_customers",
                    "error": customers_result["error"]
                })
            
            # Get campaigns for each customer
            for customer_resource in sync_results["accessible_customers"]:
                # Extract customer ID from resource name
                customer_id = customer_resource.split("/")[-1]
                
                campaigns_result = await self.get_campaigns(
                    customer_id, user_id, org_id, ip_address,
                    sync_config.get("date_range")
                )
                
                if campaigns_result["success"]:
                    sync_results["campaigns"].extend(campaigns_result["data"])
                else:
                    sync_results["errors"].append({
                        "step": "campaigns",
                        "customer_id": customer_id,
                        "error": campaigns_result["error"]
                    })
            
            # Get performance reports if requested
            if sync_config.get("include_reports", False):
                for customer_resource in sync_results["accessible_customers"]:
                    customer_id = customer_resource.split("/")[-1]
                    
                    report_result = await self.get_performance_report(
                        customer_id, user_id, org_id, ip_address,
                        sync_config.get("date_range", {
                            "since": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
                            "until": datetime.now().strftime("%Y-%m-%d")
                        }),
                        sync_config.get("metrics"),
                        sync_config.get("dimensions")
                    )
                    
                    if report_result["success"]:
                        sync_results["reports"].extend(report_result["data"])
                    else:
                        sync_results["errors"].append({
                            "step": "reports",
                            "customer_id": customer_id,
                            "error": report_result["error"]
                        })
            
            sync_end = datetime.now(timezone.utc)
            sync_duration = (sync_end - sync_start).total_seconds()
            
            # Log sync completion
            await self.audit_logger.log_event(
                event_type=SecurityEventType.RESOURCE_ACCESSED,
                action="google_sync_completed",
                user_id=user_id,
                org_id=org_id,
                ip_address=ip_address,
                details={
                    "provider": "google",
                    "sync_duration": sync_duration,
                    "customers_count": len(sync_results["accessible_customers"]),
                    "campaigns_count": len(sync_results["campaigns"]),
                    "reports_count": len(sync_results["reports"]),
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
                action="google_sync_exception",
                user_id=user_id,
                org_id=org_id,
                ip_address=ip_address,
                details={
                    "provider": "google",
                    "error": str(e),
                    "success": False
                }
            )
            logger.error(f"Google sync failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }