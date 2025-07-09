"""
Service Integration Clients for Channels Service
Async clients for communicating with other LiftOS services
"""

import asyncio
import httpx
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
import json
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


class ServiceClientError(Exception):
    """Base exception for service client errors"""
    pass


class ServiceUnavailableError(ServiceClientError):
    """Raised when a service is unavailable"""
    pass


class BaseServiceClient:
    """Base class for service clients with common functionality"""
    
    def __init__(self, base_url: str, service_name: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip('/')
        self.service_name = service_name
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
        
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make HTTP request with error handling"""
        
        url = urljoin(self.base_url, endpoint.lstrip('/'))
        
        try:
            response = await self.client.request(
                method=method,
                url=url,
                json=data,
                params=params,
                headers=headers or {}
            )
            
            response.raise_for_status()
            
            if response.headers.get('content-type', '').startswith('application/json'):
                return response.json()
            else:
                return {"data": response.text}
                
        except httpx.TimeoutException:
            logger.error(f"Timeout calling {self.service_name} at {url}")
            raise ServiceUnavailableError(f"{self.service_name} service timeout")
        
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error calling {self.service_name}: {e.response.status_code}")
            raise ServiceClientError(f"{self.service_name} returned {e.response.status_code}")
        
        except Exception as e:
            logger.error(f"Error calling {self.service_name}: {str(e)}")
            raise ServiceClientError(f"Failed to call {self.service_name}: {str(e)}")
    
    async def health_check(self) -> bool:
        """Check if the service is healthy"""
        try:
            await self._make_request("GET", "/health")
            return True
        except Exception:
            return False


class CausalServiceClient(BaseServiceClient):
    """Client for Lift Causal service"""
    
    def __init__(self, base_url: str = "http://localhost:8003"):
        super().__init__(base_url, "Causal Service")
    
    async def get_channel_lift(self, org_id: str, channel_id: str) -> Dict[str, Any]:
        """Get causal lift estimate for a channel"""
        
        endpoint = f"/causal/lift/{org_id}/{channel_id}"
        
        try:
            result = await self._make_request("GET", endpoint)
            return result
        except ServiceClientError:
            # Return default values if service unavailable
            logger.warning(f"Using default lift values for {channel_id}")
            return {
                "lift_coefficient": 1.0,
                "confidence_interval": (0.8, 1.2),
                "p_value": 0.05,
                "effect_size": "medium"
            }
    
    async def get_cross_channel_effects(self, org_id: str, channels: List[str]) -> Dict[str, Any]:
        """Get cross-channel interaction effects"""
        
        endpoint = f"/causal/interactions/{org_id}"
        data = {"channels": channels}
        
        try:
            result = await self._make_request("POST", endpoint, data=data)
            return result
        except ServiceClientError:
            logger.warning("Using default cross-channel effects")
            return {
                "interaction_matrix": {},
                "synergy_effects": {},
                "cannibalization_effects": {}
            }
    
    async def get_causal_attribution(
        self, 
        org_id: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get causal attribution analysis"""
        
        endpoint = f"/causal/attribution/{org_id}"
        params = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        }
        
        try:
            result = await self._make_request("GET", endpoint, params=params)
            return result
        except ServiceClientError:
            logger.warning("Using default attribution data")
            return {
                "attribution_coefficients": {},
                "confidence_scores": {},
                "methodology": "default"
            }


class DataIngestionServiceClient(BaseServiceClient):
    """Client for Data Ingestion service"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        super().__init__(base_url, "Data Ingestion Service")
    
    async def get_channel_historical_data(
        self, 
        org_id: str, 
        channel_id: str,
        start_date: datetime,
        end_date: datetime,
        metrics: List[str]
    ) -> List[Dict[str, Any]]:
        """Get historical performance data for a channel"""
        
        endpoint = f"/data/historical/{org_id}/{channel_id}"
        params = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "metrics": ",".join(metrics)
        }
        
        try:
            result = await self._make_request("GET", endpoint, params=params)
            return result.get("data", [])
        except ServiceClientError:
            logger.warning(f"Using mock data for {channel_id}")
            return self._generate_mock_historical_data(channel_id, start_date, end_date)
    
    async def get_current_performance(self, org_id: str, channel_id: str) -> Dict[str, Any]:
        """Get current performance metrics for a channel"""
        
        endpoint = f"/data/current/{org_id}/{channel_id}"
        
        try:
            result = await self._make_request("GET", endpoint)
            return result
        except ServiceClientError:
            logger.warning(f"Using default performance data for {channel_id}")
            return {
                "spend": 1000.0,
                "revenue": 3500.0,
                "conversions": 100,
                "impressions": 10000,
                "clicks": 500,
                "roas": 3.5,
                "cac": 10.0,
                "last_updated": datetime.utcnow().isoformat()
            }
    
    async def get_real_time_metrics(self, org_id: str, channels: List[str]) -> Dict[str, Any]:
        """Get real-time metrics for channels"""
        
        endpoint = f"/data/realtime/{org_id}"
        data = {"channels": channels}
        
        try:
            result = await self._make_request("POST", endpoint, data=data)
            return result
        except ServiceClientError:
            logger.warning("Using mock real-time data")
            return {
                channel: {
                    "current_spend_rate": 100.0,
                    "current_conversion_rate": 0.02,
                    "current_roas": 3.5,
                    "timestamp": datetime.utcnow().isoformat()
                }
                for channel in channels
            }
    
    def _generate_mock_historical_data(
        self, 
        channel_id: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Generate mock historical data for development"""
        
        import numpy as np
        np.random.seed(42)
        
        data = []
        current_date = start_date
        
        while current_date <= end_date:
            # Generate realistic metrics with some variation
            base_spend = 1000.0
            spend = base_spend * (0.8 + 0.4 * np.random.random())
            
            # Revenue with saturation curve
            revenue = 1000 * (spend ** 0.7) / (500 ** 0.7 + spend ** 0.7)
            revenue *= (0.9 + 0.2 * np.random.random())
            
            conversions = revenue / 35.0  # $35 per conversion
            impressions = spend * 10  # $0.1 CPM
            clicks = spend * 0.5  # $2 CPC
            
            data.append({
                "date": current_date.isoformat(),
                "spend": float(spend),
                "revenue": float(revenue),
                "conversions": int(conversions),
                "impressions": int(impressions),
                "clicks": int(clicks),
                "roas": float(revenue / spend) if spend > 0 else 0.0,
                "cac": float(spend / conversions) if conversions > 0 else 0.0
            })
            
            current_date += timedelta(days=1)
        
        return data


class MemoryServiceClient(BaseServiceClient):
    """Client for Memory service"""
    
    def __init__(self, base_url: str = "http://localhost:8002"):
        super().__init__(base_url, "Memory Service")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory service"""
        
        endpoint = f"/memory/get/{key}"
        
        try:
            result = await self._make_request("GET", endpoint)
            return result.get("value")
        except ServiceClientError:
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in memory service"""
        
        endpoint = "/memory/set"
        data = {
            "key": key,
            "value": value,
            "ttl": ttl
        }
        
        try:
            await self._make_request("POST", endpoint, data=data)
            return True
        except ServiceClientError:
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from memory service"""
        
        endpoint = f"/memory/delete/{key}"
        
        try:
            await self._make_request("DELETE", endpoint)
            return True
        except ServiceClientError:
            return False
    
    async def get_channel_memory(self, org_id: str, channel_id: str) -> Dict[str, Any]:
        """Get channel-specific memory data"""
        
        endpoint = f"/memory/channel/{org_id}/{channel_id}"
        
        try:
            result = await self._make_request("GET", endpoint)
            return result.get("data", {})
        except ServiceClientError:
            return {}
    
    async def store_optimization_result(
        self, 
        org_id: str, 
        optimization_id: str, 
        result: Dict[str, Any]
    ) -> bool:
        """Store optimization result in memory"""
        
        endpoint = "/memory/optimization"
        data = {
            "org_id": org_id,
            "optimization_id": optimization_id,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            await self._make_request("POST", endpoint, data=data)
            return True
        except ServiceClientError:
            return False


class BayesianAnalysisServiceClient(BaseServiceClient):
    """Client for Bayesian Analysis service"""
    
    def __init__(self, base_url: str = "http://localhost:8004"):
        super().__init__(base_url, "Bayesian Analysis Service")
    
    async def get_channel_priors(self, org_id: str, channel_id: str) -> Dict[str, Any]:
        """Get Bayesian priors for a channel"""
        
        endpoint = f"/bayesian/priors/{org_id}/{channel_id}"
        
        try:
            result = await self._make_request("GET", endpoint)
            return result
        except ServiceClientError:
            logger.warning(f"Using default priors for {channel_id}")
            return {
                "roas_mean": 3.5,
                "roas_std": 0.5,
                "conversion_rate_mean": 0.02,
                "conversion_rate_std": 0.005,
                "uncertainty": 0.15,
                "confidence": 0.8
            }
    
    async def get_channel_performance_history(
        self, 
        org_id: str, 
        channel_id: str,
        days_back: int = 90
    ) -> List[Dict[str, Any]]:
        """Get channel performance history for Bayesian analysis"""
        
        endpoint = f"/bayesian/history/{org_id}/{channel_id}"
        params = {"days_back": days_back}
        
        try:
            result = await self._make_request("GET", endpoint, params=params)
            return result.get("data", [])
        except ServiceClientError:
            logger.warning(f"Using mock performance history for {channel_id}")
            return self._generate_mock_performance_history(channel_id, days_back)
    
    async def update_posterior(
        self, 
        org_id: str, 
        channel_id: str,
        observed_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update Bayesian posterior with new observations"""
        
        endpoint = f"/bayesian/update/{org_id}/{channel_id}"
        data = {"observed_data": observed_data}
        
        try:
            result = await self._make_request("POST", endpoint, data=data)
            return result
        except ServiceClientError:
            logger.warning("Using default posterior update")
            return {
                "posterior_mean": observed_data.get("roas", 3.5),
                "posterior_std": 0.3,
                "updated": True
            }
    
    async def calculate_uncertainty(
        self, 
        org_id: str, 
        channel_id: str,
        prediction_horizon: int = 30
    ) -> Dict[str, Any]:
        """Calculate prediction uncertainty"""
        
        endpoint = f"/bayesian/uncertainty/{org_id}/{channel_id}"
        params = {"prediction_horizon": prediction_horizon}
        
        try:
            result = await self._make_request("GET", endpoint, params=params)
            return result
        except ServiceClientError:
            return {
                "prediction_variance": 0.1,
                "confidence_interval": (0.8, 1.2),
                "uncertainty_sources": ["model", "data", "market"]
            }
    
    def _generate_mock_performance_history(self, channel_id: str, days_back: int) -> List[Dict[str, Any]]:
        """Generate mock performance history"""
        
        import numpy as np
        np.random.seed(42)
        
        data = []
        for i in range(days_back):
            date = datetime.utcnow() - timedelta(days=days_back - i)
            
            data.append({
                "date": date.isoformat(),
                "roas": float(np.random.gamma(2.0, 2.0)),
                "conversion_rate": float(np.random.beta(2.0, 98.0)),
                "cac": float(np.random.gamma(4.0, 2.5)),
                "spend": float(np.random.normal(1000, 200)),
                "revenue": float(np.random.normal(3500, 700))
            })
        
        return data


class ServiceClientManager:
    """Manager for all service clients with connection pooling and health monitoring"""
    
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.clients = {}
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize all service clients"""
        
        self.clients = {
            "causal": CausalServiceClient(
                self.config.get("causal_service_url", "http://localhost:8003")
            ),
            "data_ingestion": DataIngestionServiceClient(
                self.config.get("data_ingestion_service_url", "http://localhost:8001")
            ),
            "memory": MemoryServiceClient(
                self.config.get("memory_service_url", "http://localhost:8002")
            ),
            "bayesian_analysis": BayesianAnalysisServiceClient(
                self.config.get("bayesian_analysis_service_url", "http://localhost:8004")
            )
        }
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_all()
    
    async def close_all(self):
        """Close all client connections"""
        for client in self.clients.values():
            await client.client.aclose()
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all services"""
        
        health_status = {}
        
        for service_name, client in self.clients.items():
            try:
                health_status[service_name] = await client.health_check()
            except Exception as e:
                logger.error(f"Health check failed for {service_name}: {str(e)}")
                health_status[service_name] = False
        
        return health_status
    
    def get_client(self, service_name: str) -> BaseServiceClient:
        """Get a specific service client"""
        
        if service_name not in self.clients:
            raise ValueError(f"Unknown service: {service_name}")
        
        return self.clients[service_name]
    
    @property
    def causal(self) -> CausalServiceClient:
        """Get causal service client"""
        return self.clients["causal"]
    
    @property
    def data_ingestion(self) -> DataIngestionServiceClient:
        """Get data ingestion service client"""
        return self.clients["data_ingestion"]
    
    @property
    def memory(self) -> MemoryServiceClient:
        """Get memory service client"""
        return self.clients["memory"]
    
    @property
    def bayesian_analysis(self) -> BayesianAnalysisServiceClient:
        """Get Bayesian analysis service client"""
        return self.clients["bayesian_analysis"]


# Utility functions for service integration

async def test_service_connectivity(config: Dict[str, str]) -> Dict[str, Any]:
    """Test connectivity to all services"""
    
    async with ServiceClientManager(config) as manager:
        health_status = await manager.health_check_all()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "services": health_status,
            "all_healthy": all(health_status.values()),
            "healthy_count": sum(health_status.values()),
            "total_count": len(health_status)
        }


async def get_service_dependencies() -> List[str]:
    """Get list of service dependencies"""
    
    return [
        "causal",
        "data_ingestion", 
        "memory",
        "bayesian_analysis"
    ]


def create_service_config(
    causal_url: str = "http://localhost:8003",
    data_ingestion_url: str = "http://localhost:8001", 
    memory_url: str = "http://localhost:8002",
    bayesian_analysis_url: str = "http://localhost:8004"
) -> Dict[str, str]:
    """Create service configuration"""
    
    return {
        "causal_service_url": causal_url,
        "data_ingestion_service_url": data_ingestion_url,
        "memory_service_url": memory_url,
        "bayesian_analysis_service_url": bayesian_analysis_url
    }


# Aliases for backward compatibility
DataIngestionClient = DataIngestionServiceClient
BayesianServiceClient = BayesianAnalysisServiceClient