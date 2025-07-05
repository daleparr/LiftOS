"""
Health check utilities for Lift OS Core services
Provides standardized health and readiness endpoints
"""

from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from fastapi import HTTPException
import asyncio
import logging

logger = logging.getLogger(__name__)

class HealthChecker:
    """Centralized health checking for all services"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.readiness_checks: List[Callable] = []
        self.health_checks: List[Callable] = []
    
    def add_readiness_check(self, check_func: Callable, name: str = None):
        """Add a readiness check function"""
        check_func._check_name = name or check_func.__name__
        self.readiness_checks.append(check_func)
    
    def add_health_check(self, check_func: Callable, name: str = None):
        """Add a health check function"""
        check_func._check_name = name or check_func.__name__
        self.health_checks.append(check_func)
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get basic health status - service is running"""
        return {
            "status": "healthy",
            "service": self.service_name,
            "timestamp": datetime.utcnow().isoformat(),
            "uptime": "running"
        }
    
    async def get_readiness_status(self) -> Dict[str, Any]:
        """Get readiness status - service is ready to handle requests"""
        results = {
            "status": "ready",
            "service": self.service_name,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {}
        }
        
        overall_ready = True
        
        # Run all readiness checks
        for check in self.readiness_checks:
            check_name = getattr(check, '_check_name', check.__name__)
            try:
                result = await check() if asyncio.iscoroutinefunction(check) else check()
                results["checks"][check_name] = {
                    "status": "ready",
                    "result": result
                }
            except Exception as e:
                logger.error(f"Readiness check {check_name} failed: {e}")
                results["checks"][check_name] = {
                    "status": "not_ready",
                    "error": str(e)
                }
                overall_ready = False
        
        if not overall_ready:
            results["status"] = "not_ready"
            raise HTTPException(
                status_code=503, 
                detail=f"Service {self.service_name} not ready"
            )
        
        return results

# Database connection check
async def check_database_connection():
    """Check if database is accessible"""
    try:
        from shared.database.connection import get_database_session
        
        async with get_database_session() as session:
            # Simple query to test connection
            result = await session.execute("SELECT 1")
            return {"database": "connected"}
    except Exception as e:
        raise Exception(f"Database connection failed: {e}")

# Memory service specific checks
async def check_kse_memory_sdk():
    """Check if KSE Memory SDK is accessible"""
    try:
        # This would be implemented based on actual KSE Memory SDK
        # For now, return a placeholder
        return {"kse_memory": "available"}
    except Exception as e:
        raise Exception(f"KSE Memory SDK check failed: {e}")

# External service check template
async def check_external_service(url: str, service_name: str, timeout: int = 5):
    """Generic external service health check"""
    try:
        import aiohttp
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async with session.get(f"{url}/health") as response:
                if response.status == 200:
                    return {service_name: "available"}
                else:
                    raise Exception(f"Service returned status {response.status}")
    except Exception as e:
        raise Exception(f"External service {service_name} check failed: {e}")

# Registry service check
async def check_service_registry():
    """Check if service registry is accessible"""
    return await check_external_service(
        "http://localhost:8005", 
        "service_registry"
    )

# Auth service check
async def check_auth_service():
    """Check if auth service is accessible"""
    return await check_external_service(
        "http://localhost:8001", 
        "auth_service"
    )