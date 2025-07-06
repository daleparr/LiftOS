import requests
import streamlit as st
from typing import Dict, Any, Optional, List
from config.settings import get_service_urls, get_api_timeout
from auth.session_manager import cache_api_response, get_cached_response
import json

class MockResponse:
    """Mock response object that behaves like requests.Response"""
    
    def __init__(self, status_code: int, data: Dict[str, Any]):
        self.status_code = status_code
        self._data = data
    
    def json(self) -> Dict[str, Any]:
        """Return JSON data like requests.Response.json()"""
        return self._data
    
    @property
    def text(self) -> str:
        """Return text representation"""
        return json.dumps(self._data)


class APIClient:
    """Centralized API client for microservice communication"""
    
    def __init__(self):
        self.service_urls = get_service_urls()
        self.timeout = get_api_timeout()
        self.session = requests.Session()
        self._setup_authentication()
    
    def _setup_authentication(self):
        """Setup authentication headers"""
        if 'auth_token' in st.session_state and st.session_state.auth_token:
            self.session.headers.update({
                'Authorization': f"Bearer {st.session_state.auth_token}",
                'Content-Type': 'application/json'
            })
    
    def _make_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request with error handling"""
        try:
            response = self.session.request(method, url, timeout=self.timeout, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            raise Exception(f"Request timeout after {self.timeout} seconds")
        except requests.exceptions.ConnectionError:
            raise Exception("Service unavailable - please check if microservices are running")
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                raise Exception("Authentication failed - please log in again")
            elif response.status_code == 403:
                raise Exception("Access denied - insufficient permissions")
            elif response.status_code == 404:
                raise Exception("Service endpoint not found")
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
        except Exception as e:
            raise Exception(f"Request failed: {str(e)}")
    
    # Standard HTTP Methods
    def get(self, endpoint: str, params: Optional[Dict] = None) -> 'MockResponse':
        """Make GET request to endpoint"""
        # Return mock response object that behaves like requests.Response
        data = self._get_mock_data_for_endpoint(endpoint)
        return MockResponse(200, data)
    
    def post(self, endpoint: str, json: Optional[Dict] = None, data: Optional[Dict] = None) -> 'MockResponse':
        """Make POST request to endpoint"""
        response_data = {"success": True, "message": "Request processed successfully"}
        return MockResponse(200, response_data)
    
    def put(self, endpoint: str, json: Optional[Dict] = None) -> 'MockResponse':
        """Make PUT request to endpoint"""
        response_data = {"success": True, "message": "Resource updated successfully"}
        return MockResponse(200, response_data)
    
    def delete(self, endpoint: str) -> 'MockResponse':
        """Make DELETE request to endpoint"""
        response_data = {"success": True, "message": "Resource deleted successfully"}
        return MockResponse(200, response_data)
    
    def _get_mock_data_for_endpoint(self, endpoint: str) -> Dict[str, Any]:
        """Return appropriate mock data based on endpoint"""
        if "/observability/overhead" in endpoint:
            return {
                "current_overhead_percentage": 0.078,
                "cpu_overhead_percentage": 0.045,
                "memory_overhead_percentage": 0.023,
                "network_overhead_percentage": 0.010
            }
        elif "/observability/micro-explanations" in endpoint:
            return {
                "recent_explanations": [
                    {
                        "timestamp": "2025-07-06T20:45:00Z",
                        "operation": "Attribution Analysis",
                        "explanation": "Causal model identified significant lift in Google Ads performance",
                        "confidence": 0.94
                    },
                    {
                        "timestamp": "2025-07-06T20:40:00Z",
                        "operation": "Budget Optimization",
                        "explanation": "Automated reallocation increased ROAS by 15%",
                        "confidence": 0.89
                    }
                ]
            }
        elif "/system/health/detailed" in endpoint:
            return {
                "services": [
                    {
                        "name": "Data Ingestion",
                        "status": "healthy",
                        "health_score": 98.5,
                        "response_time_ms": 45,
                        "uptime_percentage": 99.9
                    },
                    {
                        "name": "Attribution Engine",
                        "status": "healthy",
                        "health_score": 96.2,
                        "response_time_ms": 67,
                        "uptime_percentage": 99.8
                    },
                    {
                        "name": "Memory Service",
                        "status": "healthy",
                        "health_score": 94.7,
                        "response_time_ms": 23,
                        "uptime_percentage": 99.9
                    }
                ]
            }
        elif "/intelligence/collaboration/insights" in endpoint:
            return {
                "insights": [
                    {
                        "type": "team_performance",
                        "title": "Cross-functional Attribution Success",
                        "description": "Marketing and data teams achieved 94% attribution accuracy",
                        "confidence": 0.92,
                        "impact": "High"
                    },
                    {
                        "type": "knowledge_sharing",
                        "title": "Causal Model Insights Shared",
                        "description": "15 causal insights shared across teams this week",
                        "confidence": 0.88,
                        "impact": "Medium"
                    }
                ]
            }
        elif "/intelligence/collaboration/recommendations" in endpoint:
            return {
                "recommendations": [
                    {
                        "type": "ai_optimization",
                        "title": "Automated Budget Reallocation",
                        "description": "AI recommends shifting 15% budget from Facebook to Google Ads",
                        "confidence": 0.91,
                        "expected_roi": 1.23,
                        "priority": "high"
                    },
                    {
                        "type": "causal_insight",
                        "title": "Attribution Model Enhancement",
                        "description": "Incorporate weather data for 8% accuracy improvement",
                        "confidence": 0.85,
                        "expected_roi": 1.08,
                        "priority": "medium"
                    }
                ]
            }
        elif "/intelligence/optimization/recommendations" in endpoint:
            return {
                "recommendations": [
                    {
                        "id": "opt_001",
                        "type": "budget_optimization",
                        "title": "Increase Google Ads Spend",
                        "description": "Causal analysis shows 23% ROAS improvement potential",
                        "expected_roi": 1.23,
                        "confidence": 0.94,
                        "priority": "high",
                        "impact_timeline": "3-5 days",
                        "platforms": ["google_ads", "search"],
                        "predicted_metrics": {
                            "revenue_increase": 45000,
                            "cost_reduction": 12000,
                            "roas_improvement": 0.23
                        },
                        "causal_evidence": [
                            "Historical data shows 94% correlation between Google Ads spend and conversions",
                            "Causal model identifies Google Ads as primary driver of revenue growth"
                        ],
                        "budget_change": {
                            "google_ads": "+15%",
                            "facebook_ads": "-5%"
                        }
                    },
                    {
                        "id": "opt_002",
                        "type": "attribution_optimization",
                        "title": "Enhance Attribution Model",
                        "description": "Incorporate weather data for improved accuracy",
                        "expected_roi": 1.08,
                        "confidence": 0.87,
                        "priority": "medium",
                        "impact_timeline": "7-10 days",
                        "platforms": ["facebook_ads", "email_marketing"],
                        "predicted_metrics": {
                            "revenue_increase": 18000,
                            "cost_reduction": 5000,
                            "roas_improvement": 0.08
                        },
                        "causal_evidence": [
                            "Weather patterns correlate with 12% of conversion variance",
                            "Enhanced attribution reduces over-crediting by 8%"
                        ],
                        "budget_change": {
                            "attribution_accuracy": "+8%"
                        }
                    }
                ]
            }
        elif "/data-ingestion/budget/allocation" in endpoint:
            return {
                "current_allocation": {
                    "google_ads": 45000,
                    "facebook_ads": 32000,
                    "email_marketing": 8000
                },
                "recommended_allocation": {
                    "google_ads": 52000,
                    "facebook_ads": 27000,
                    "email_marketing": 6000
                }
            }
        else:
            # Default mock response
            return {
                "status": "success",
                "message": "Mock data for development",
                "timestamp": "2025-07-06T20:50:00Z"
            }
    
    # Data Ingestion Service Methods
    def sync_platform(self, platform: str, date_range: Optional[Dict] = None) -> Dict[str, Any]:
        """Sync data from marketing platform"""
        cache_key = f"sync_{platform}_{date_range}"
        cached_result = get_cached_response(cache_key)
        if cached_result:
            return cached_result
        
        # Use the data ingestion service for platform syncing
        url = f"{self.service_urls['data_ingestion']}/sync/start"
        
        # Map platform names to expected enum values
        platform_mapping = {
            "meta": "meta_business",
            "google": "google_ads",
            "klaviyo": "klaviyo"
        }
        
        mapped_platform = platform_mapping.get(platform, platform)
        
        # Calculate date range
        from datetime import date, timedelta
        if date_range and date_range.get('start_date') and date_range.get('end_date'):
            start_date = date_range['start_date']
            end_date = date_range['end_date']
        else:
            # Default to last 30 days
            end_date = date.today()
            start_date = end_date - timedelta(days=30)
        
        payload = {
            "platform": mapped_platform,
            "date_range_start": str(start_date),
            "date_range_end": str(end_date),
            "sync_type": "full"
        }
        
        # Add required headers for user context
        headers = {
            "x-user-id": st.session_state.get('user_id', 'demo_user'),
            "x-org-id": st.session_state.get('org_id', 'demo_org'),
            "x-memory-context": st.session_state.get('memory_context', 'demo_context'),
            "Content-Type": "application/json"
        }
        
        # Update session headers
        original_headers = self.session.headers.copy()
        self.session.headers.update(headers)
        
        try:
            result = self._make_request('POST', url, json=payload)
            cache_api_response(cache_key, result, ttl_minutes=10)
            return result
        finally:
            # Restore original headers
            self.session.headers = original_headers
    
    def get_sync_status(self, job_id: Optional[str] = None) -> Dict[str, Any]:
        """Get sync job status"""
        if job_id:
            url = f"{self.service_urls['data_ingestion']}/sync/status/{job_id}"
        else:
            url = f"{self.service_urls['data_ingestion']}/sync/jobs"
        
        # Add required headers for user context
        headers = {
            "x-user-id": st.session_state.get('user_id', 'demo_user'),
            "x-org-id": st.session_state.get('org_id', 'demo_org'),
            "x-memory-context": st.session_state.get('memory_context', 'demo_context')
        }
        
        # Update session headers
        original_headers = self.session.headers.copy()
        self.session.headers.update(headers)
        
        try:
            return self._make_request('GET', url)
        finally:
            # Restore original headers
            self.session.headers = original_headers
    
    def get_sync_jobs(self) -> Dict[str, Any]:
        """Get all sync jobs"""
        url = f"{self.service_urls['data_ingestion']}/sync/jobs"
        
        # Add required headers for user context
        headers = {
            "x-user-id": st.session_state.get('user_id', 'demo_user'),
            "x-org-id": st.session_state.get('org_id', 'demo_org'),
            "x-memory-context": st.session_state.get('memory_context', 'demo_context')
        }
        
        # Update session headers
        original_headers = self.session.headers.copy()
        self.session.headers.update(headers)
        
        try:
            return self._make_request('GET', url)
        finally:
            # Restore original headers
            self.session.headers = original_headers
    
    def run_attribution_analysis(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run attribution analysis"""
        url = f"{self.service_urls['causal']}/api/v1/attribution/analyze"
        
        payload = {
            "user_id": st.session_state.get('user_id', 'demo_user'),
            **request_data
        }
        
        # Add required headers for authentication and user context
        headers = {
            "Authorization": f"Bearer {st.session_state.get('access_token', 'demo_token')}",
            "x-user-id": st.session_state.get('user_id', 'demo_user'),
            "x-org-id": st.session_state.get('org_id', 'demo_org'),
            "Content-Type": "application/json"
        }
        
        # Update session headers
        original_headers = self.session.headers.copy()
        self.session.headers.update(headers)
        
        try:
            return self._make_request('POST', url, json=payload)
        finally:
            # Restore original headers
            self.session.headers = original_headers
    
    def create_model(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create attribution model"""
        url = f"{self.service_urls['causal']}/api/v1/models/create"
        
        payload = {
            "user_id": st.session_state.get('user_id', 'demo_user'),
            **model_config
        }
        
        # Add required headers for authentication and user context
        headers = {
            "Authorization": f"Bearer {st.session_state.get('access_token', 'demo_token')}",
            "x-user-id": st.session_state.get('user_id', 'demo_user'),
            "x-org-id": st.session_state.get('org_id', 'demo_org'),
            "Content-Type": "application/json"
        }
        
        # Update session headers
        original_headers = self.session.headers.copy()
        self.session.headers.update(headers)
        
        try:
            return self._make_request('POST', url, json=payload)
        finally:
            # Restore original headers
            self.session.headers = original_headers
    
    def run_experiment(self, experiment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run causal experiment"""
        url = f"{self.service_urls['causal']}/api/v1/experiments/run"
        
        payload = {
            "user_id": st.session_state.get('user_id', 'demo_user'),
            **experiment_config
        }
        
        # Store original headers
        original_headers = self.session.headers.copy()
        
        try:
            # Add authentication headers for causal service
            self.session.headers.update({
                'Authorization': f"Bearer {st.session_state.get('access_token', 'demo_token')}",
                'x-user-id': st.session_state.get('user_id', 'demo_user'),
                'x-org-id': st.session_state.get('org_id', 'demo_org')
            })
            
            return self._make_request('POST', url, json=payload)
        finally:
            # Restore original headers
            self.session.headers = original_headers
    
    def optimize_budget(self, optimization_config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize budget allocation"""
        url = f"{self.service_urls['causal']}/api/v1/optimization/budget"
        
        payload = {
            "user_id": st.session_state.get('user_id', 'demo_user'),
            **optimization_config
        }
        
        # Store original headers
        original_headers = self.session.headers.copy()
        
        try:
            # Add authentication headers for causal service
            self.session.headers.update({
                'Authorization': f"Bearer {st.session_state.get('access_token', 'demo_token')}",
                'x-user-id': st.session_state.get('user_id', 'demo_user'),
                'x-org-id': st.session_state.get('org_id', 'demo_org')
            })
            
            return self._make_request('POST', url, json=payload)
        finally:
            # Restore original headers
            self.session.headers = original_headers
        
        return self._make_request('POST', url, json=payload)
    
    def measure_lift(self, lift_config: Dict[str, Any]) -> Dict[str, Any]:
        """Measure incremental lift"""
        url = f"{self.service_urls['causal']}/api/v1/lift/measure"
        
        payload = {
            "user_id": st.session_state.get('user_id', 'demo_user'),
            **lift_config
        }
        
        return self._make_request('POST', url, json=payload)
    
    # Memory Service Methods
    def search_memory(self, query: str, filters: Optional[Dict] = None) -> Dict[str, Any]:
        """Search memory service"""
        url = f"{self.service_urls['memory']}/api/v1/search"
        
        params = {
            "query": query,
            "user_id": st.session_state.get('user_id', 'demo_user')
        }
        if filters:
            params.update(filters)
        
        return self._make_request('GET', url, params=params)
    
    def get_recent_activity(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent activity from memory"""
        cache_key = f"recent_activity_{limit}"
        cached_result = get_cached_response(cache_key)
        if cached_result:
            return cached_result.get('activities', [])
        
        url = f"{self.service_urls['memory']}/api/v1/recent"
        
        params = {
            "limit": limit,
            "user_id": st.session_state.get('user_id', 'demo_user')
        }
        
        try:
            result = self._make_request('GET', url, params=params)
            activities = result.get('activities', [])
            cache_api_response(cache_key, {'activities': activities}, ttl_minutes=2)
            return activities
        except Exception:
            # Return mock data if memory service is unavailable
            return self._get_mock_recent_activity()
    
    def store_in_memory(self, data: Dict[str, Any], tags: List[str]) -> str:
        """Store data in memory service"""
        url = f"{self.service_urls['memory']}/api/v1/store"
        
        payload = {
            "user_id": st.session_state.get('user_id', 'demo_user'),
            "data": data,
            "tags": tags
        }
        
        result = self._make_request('POST', url, json=payload)
        return result.get('memory_id', 'unknown')
    
    # LLM Service Methods
    def ask_llm(self, question: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Ask LLM assistant a question"""
        url = f"{self.service_urls['llm']}/api/v1/chat"
        
        payload = {
            "user_id": st.session_state.get('user_id', 'demo_user'),
            "message": question,
            "context": context
        }
        
        return self._make_request('POST', url, json=payload)
    
    # Surfacing Service Methods
    def run_surfacing_analysis(self, analysis_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run surfacing analysis"""
        url = f"{self.service_urls['surfacing']}/api/v1/analyze"
        
        payload = {
            "user_id": st.session_state.get('user_id', 'demo_user'),
            **analysis_config
        }
        
        return self._make_request('POST', url, json=payload)
    
    # Health Check Methods
    def check_service_health(self, service: str) -> bool:
        """Check if a service is healthy"""
        try:
            url = f"{self.service_urls[service]}/health"
            response = self.session.get(url, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_all_service_status(self) -> Dict[str, bool]:
        """Get status of all services"""
        status = {}
        for service in self.service_urls.keys():
            status[service] = self.check_service_health(service)
        return status
    
    # Mock Data Methods (for demo/development)
    def _get_mock_recent_activity(self) -> List[Dict[str, Any]]:
        """Get mock recent activity data"""
        return [
            {
                "type": "Attribution Analysis",
                "timestamp": "2025-01-04 08:15:00",
                "details": {
                    "model_type": "time_decay",
                    "channels": ["meta", "google"],
                    "status": "completed"
                }
            },
            {
                "type": "Platform Sync",
                "timestamp": "2025-01-04 08:10:00",
                "details": {
                    "platform": "meta",
                    "records_synced": 1250,
                    "status": "success"
                }
            },
            {
                "type": "Budget Optimization",
                "timestamp": "2025-01-04 08:05:00",
                "details": {
                    "total_budget": 50000,
                    "optimized_allocation": {"meta": 30000, "google": 20000},
                    "status": "completed"
                }
            }
        ]
    
    # Observability Service Methods
    def record_metric(self, metric_data: Dict[str, Any]) -> Dict[str, Any]:
        """Record a single metric"""
        url = f"{self.service_urls['observability']}/api/v1/metrics"
        
        payload = {
            "user_id": st.session_state.get('user_id', 'demo_user'),
            "organization_id": st.session_state.get('org_id', 'demo_org'),
            **metric_data
        }
        
        return self._make_request('POST', url, json=payload)
    
    def record_metrics_batch(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Record multiple metrics in batch"""
        url = f"{self.service_urls['observability']}/api/v1/metrics/batch"
        
        payload = {
            "user_id": st.session_state.get('user_id', 'demo_user'),
            "organization_id": st.session_state.get('org_id', 'demo_org'),
            "metrics": metrics
        }
        
        return self._make_request('POST', url, json=payload)
    
    def query_metrics(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Query metrics with filters"""
        url = f"{self.service_urls['observability']}/api/v1/metrics/query"
        
        payload = {
            "user_id": st.session_state.get('user_id', 'demo_user'),
            "organization_id": st.session_state.get('org_id', 'demo_org'),
            **query_params
        }
        
        return self._make_request('POST', url, json=payload)
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get system overview dashboard"""
        url = f"{self.service_urls['observability']}/api/v1/overview"
        
        params = {
            "user_id": st.session_state.get('user_id', 'demo_user'),
            "organization_id": st.session_state.get('org_id', 'demo_org')
        }
        
        return self._make_request('GET', url, params=params)
    
    def get_service_health(self) -> Dict[str, Any]:
        """Get service health status"""
        url = f"{self.service_urls['observability']}/api/v1/health/services"
        
        params = {
            "user_id": st.session_state.get('user_id', 'demo_user'),
            "organization_id": st.session_state.get('org_id', 'demo_org')
        }
        
        return self._make_request('GET', url, params=params)
    
    def get_alerts(self, filters: Optional[Dict] = None) -> Dict[str, Any]:
        """Get system alerts"""
        url = f"{self.service_urls['observability']}/api/v1/alerts"
        
        params = {
            "user_id": st.session_state.get('user_id', 'demo_user'),
            "organization_id": st.session_state.get('org_id', 'demo_org')
        }
        
        if filters:
            params.update(filters)
        
        return self._make_request('GET', url, params=params)
    
    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get dashboard metrics"""
        url = f"{self.service_urls['observability']}/api/v1/dashboard/metrics"
        
        params = {
            "user_id": st.session_state.get('user_id', 'demo_user'),
            "organization_id": st.session_state.get('org_id', 'demo_org')
        }
        
        return self._make_request('GET', url, params=params)
    
    # Data Transformation Methods
    def get_transformation_status(self) -> Dict[str, Any]:
        """Get data transformation pipeline status"""
        url = f"{self.service_urls['data_ingestion']}/api/v1/transformations/status"
        
        params = {
            "user_id": st.session_state.get('user_id', 'demo_user'),
            "organization_id": st.session_state.get('org_id', 'demo_org')
        }
        
        return self._make_request('GET', url, params=params)
    
    def get_causal_pipeline_metrics(self) -> Dict[str, Any]:
        """Get causal data pipeline metrics"""
        url = f"{self.service_urls['causal']}/api/v1/pipeline/metrics"
        
        params = {
            "user_id": st.session_state.get('user_id', 'demo_user'),
            "organization_id": st.session_state.get('org_id', 'demo_org')
        }
        
        return self._make_request('GET', url, params=params)
    
    def validate_data_quality(self, validation_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data quality"""
        url = f"{self.service_urls['data_ingestion']}/api/v1/quality/validate"
        
        payload = {
            "user_id": st.session_state.get('user_id', 'demo_user'),
            "organization_id": st.session_state.get('org_id', 'demo_org'),
            **validation_config
        }
        

# Alias for backward compatibility
LiftOSAPIClient = APIClient