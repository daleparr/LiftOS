"""
Lift OS Core - Registry Service
Module Registration and Discovery
"""
import time
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import httpx

# Import shared modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
# KSE-SDK Integration
from shared.kse_sdk.client import LiftKSEClient
from shared.kse_sdk.core import Entity, SearchQuery, SearchResult
from shared.kse_sdk.models import EntityType, Domain


from shared.models.base import (
    APIResponse, HealthCheck, Module, ModuleStatus, PaginationParams, PaginatedResponse
)
from shared.utils.config import get_service_config
from shared.utils.logging import setup_logging
from shared.health.health_checks import HealthChecker

# Service configuration
config = get_service_config("registry", 8005)
logger = setup_logging("registry")

# Health checker
health_checker = HealthChecker("registry")

# FastAPI app

# KSE Client for intelligence integration
kse_client = None

async def initialize_kse_client():
    """Initialize KSE client for intelligence integration"""
    global kse_client
    try:
        kse_client = LiftKSEClient()
        print("KSE Client initialized successfully")
        return True
    except Exception as e:
        print(f"KSE Client initialization failed: {e}")
        kse_client = None
        return False

app = FastAPI(
    title="Lift OS Core - Registry Service",
    description="Module Registration and Discovery Service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if config.DEBUG else ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class ModuleRegistrationRequest(BaseModel):
    name: str
    version: str
    base_url: HttpUrl
    health_endpoint: str = "/health"
    api_prefix: str = "/api/v1"
    features: List[str] = []
    permissions: List[str] = []
    memory_requirements: Dict[str, Any] = {}
    ui_components: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}


class ModuleUpdateRequest(BaseModel):
    name: Optional[str] = None
    version: Optional[str] = None
    base_url: Optional[HttpUrl] = None
    health_endpoint: Optional[str] = None
    api_prefix: Optional[str] = None
    features: Optional[List[str]] = None
    permissions: Optional[List[str]] = None
    memory_requirements: Optional[Dict[str, Any]] = None
    ui_components: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    status: Optional[ModuleStatus] = None


# In-memory storage (replace with database in production)
modules_db: Dict[str, Dict] = {}
module_health_cache: Dict[str, Dict] = {}

# HTTP client for health checks
http_client = httpx.AsyncClient(timeout=10.0)


def get_user_context(
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
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
        "roles": x_user_roles.split(",") if x_user_roles else []
    }


def require_admin_role(user_context: Dict[str, Any]) -> bool:
    """Check if user has admin role"""
    return "admin" in user_context.get("roles", []) or "developer" in user_context.get("roles", [])


async def check_module_health(module: Dict[str, Any]) -> Dict[str, Any]:
    """Check module health status"""
    module_id = module["id"]
    base_url = module["base_url"]
    health_endpoint = module["health_endpoint"]
    
    try:
        health_url = f"{base_url.rstrip('/')}{health_endpoint}"
        response = await http_client.get(health_url, timeout=5.0)
        
        if response.status_code == 200:
            health_data = response.json()
            status_info = {
                "status": "healthy",
                "last_check": datetime.utcnow().isoformat(),
                "response_time": response.elapsed.total_seconds(),
                "details": health_data
            }
        else:
            status_info = {
                "status": "unhealthy",
                "last_check": datetime.utcnow().isoformat(),
                "error": f"HTTP {response.status_code}"
            }
    except Exception as e:
        status_info = {
            "status": "unreachable",
            "last_check": datetime.utcnow().isoformat(),
            "error": str(e)
        }
    
    # Cache health status
    module_health_cache[module_id] = status_info
    return status_info


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Registry service health check"""
    return HealthCheck(
        status="healthy",
        dependencies={
            "registered_modules": str(len(modules_db)),
            "active_modules": str(len([m for m in modules_db.values() if m["status"] == "active"]))
        },
        uptime=time.time() - getattr(app.state, "start_time", time.time())
    )


@app.get("/ready", tags=["health"])
async def readiness_check():
    """Registry service readiness probe - service is ready to handle requests"""
    return await health_checker.get_readiness_status()


@app.get("/", response_model=APIResponse)
async def root():
    """Registry service root endpoint"""
    return APIResponse(
        message="Lift OS Core Registry Service",
        data={
            "version": "1.0.0",
            "registered_modules": len(modules_db),
            "docs": "/docs"
        }
    )


@app.post("/modules", response_model=APIResponse)
async def register_module(
    request: ModuleRegistrationRequest,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Register a new module"""
    # Check admin permissions
    if not require_admin_role(user_context):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required to register modules"
        )
    
    try:
        # Generate module ID
        module_id = str(uuid.uuid4())
        
        # Create module record
        module_data = {
            "id": module_id,
            "name": request.name,
            "version": request.version,
            "base_url": str(request.base_url),
            "health_endpoint": request.health_endpoint,
            "api_prefix": request.api_prefix,
            "status": ModuleStatus.ACTIVE.value,
            "features": request.features,
            "permissions": request.permissions,
            "memory_requirements": request.memory_requirements,
            "ui_components": request.ui_components,
            "metadata": request.metadata,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": None,
            "registered_by": user_context["user_id"]
        }
        
        # Check module health
        health_status = await check_module_health(module_data)
        
        # Store module
        modules_db[module_id] = module_data
        
        logger.info(
            f"Module registered: {request.name} ({module_id})",
            extra={
                "module_id": module_id,
                "module_name": request.name,
                "user_id": user_context["user_id"],
                "health_status": health_status["status"]
            }
        )
        
        return APIResponse(
            message="Module registered successfully",
            data={
                "module_id": module_id,
                "module": module_data,
                "health": health_status
            }
        )
        
    except Exception as e:
        logger.error(f"Module registration failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Module registration failed: {str(e)}"
        )


@app.get("/modules", response_model=APIResponse)
async def list_modules(
    pagination: PaginationParams = Depends(),
    status_filter: Optional[ModuleStatus] = None,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """List registered modules"""
    try:
        # Filter modules
        filtered_modules = list(modules_db.values())
        
        if status_filter:
            filtered_modules = [m for m in filtered_modules if m["status"] == status_filter.value]
        
        # Sort modules
        if pagination.sort_by:
            reverse = pagination.sort_order == "desc"
            filtered_modules.sort(
                key=lambda x: x.get(pagination.sort_by, ""),
                reverse=reverse
            )
        
        # Paginate
        total = len(filtered_modules)
        start = (pagination.page - 1) * pagination.size
        end = start + pagination.size
        items = filtered_modules[start:end]
        
        # Add health status to each module
        for module in items:
            module_id = module["id"]
            if module_id in module_health_cache:
                module["health"] = module_health_cache[module_id]
        
        pages = (total + pagination.size - 1) // pagination.size
        
        response_data = PaginatedResponse(
            items=items,
            total=total,
            page=pagination.page,
            size=pagination.size,
            pages=pages,
            has_next=pagination.page < pages,
            has_prev=pagination.page > 1
        )
        
        return APIResponse(
            message="Modules retrieved successfully",
            data=response_data.dict()
        )
        
    except Exception as e:
        logger.error(f"Module listing failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Module listing failed: {str(e)}"
        )


@app.get("/modules/{module_id}", response_model=APIResponse)
async def get_module(
    module_id: str,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Get module details"""
    try:
        module = modules_db.get(module_id)
        if not module:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Module not found"
            )
        
        # Add current health status
        health_status = await check_module_health(module)
        module["health"] = health_status
        
        return APIResponse(
            message="Module retrieved successfully",
            data=module
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Module retrieval failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Module retrieval failed: {str(e)}"
        )


@app.put("/modules/{module_id}", response_model=APIResponse)
async def update_module(
    module_id: str,
    request: ModuleUpdateRequest,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Update module configuration"""
    # Check admin permissions
    if not require_admin_role(user_context):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required to update modules"
        )
    
    try:
        module = modules_db.get(module_id)
        if not module:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Module not found"
            )
        
        # Update module fields
        update_data = request.dict(exclude_unset=True)
        for field, value in update_data.items():
            if field == "base_url" and value:
                module[field] = str(value)
            elif value is not None:
                module[field] = value
        
        module["updated_at"] = datetime.utcnow().isoformat()
        
        # Check health if URL changed
        if "base_url" in update_data or "health_endpoint" in update_data:
            health_status = await check_module_health(module)
        else:
            health_status = module_health_cache.get(module_id, {"status": "unknown"})
        
        logger.info(
            f"Module updated: {module['name']} ({module_id})",
            extra={
                "module_id": module_id,
                "user_id": user_context["user_id"],
                "updated_fields": list(update_data.keys())
            }
        )
        
        return APIResponse(
            message="Module updated successfully",
            data={
                "module": module,
                "health": health_status
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Module update failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Module update failed: {str(e)}"
        )


@app.delete("/modules/{module_id}", response_model=APIResponse)
async def unregister_module(
    module_id: str,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Unregister a module"""
    # Check admin permissions
    if not require_admin_role(user_context):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required to unregister modules"
        )
    
    try:
        module = modules_db.get(module_id)
        if not module:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Module not found"
            )
        
        # Remove module
        del modules_db[module_id]
        
        # Remove health cache
        if module_id in module_health_cache:
            del module_health_cache[module_id]
        
        logger.info(
            f"Module unregistered: {module['name']} ({module_id})",
            extra={
                "module_id": module_id,
                "user_id": user_context["user_id"]
            }
        )
        
        return APIResponse(
            message="Module unregistered successfully",
            data={"module_id": module_id}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Module unregistration failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Module unregistration failed: {str(e)}"
        )


@app.get("/modules/{module_id}/health", response_model=APIResponse)
async def check_module_health_endpoint(
    module_id: str,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Check module health status"""
    try:
        module = modules_db.get(module_id)
        if not module:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Module not found"
            )
        
        health_status = await check_module_health(module)
        
        return APIResponse(
            message="Module health checked",
            data={
                "module_id": module_id,
                "health": health_status
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Module health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Module health check failed: {str(e)}"
        )


@app.get("/health-summary", response_model=APIResponse)
async def get_health_summary(
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Get health summary of all modules"""
    try:
        health_summary = {
            "total_modules": len(modules_db),
            "healthy": 0,
            "unhealthy": 0,
            "unreachable": 0,
            "unknown": 0,
            "modules": []
        }
        
        for module_id, module in modules_db.items():
            health_status = await check_module_health(module)
            status = health_status["status"]
            
            health_summary["modules"].append({
                "module_id": module_id,
                "name": module["name"],
                "status": status,
                "last_check": health_status["last_check"]
            })
            
            if status == "healthy":
                health_summary["healthy"] += 1
            elif status == "unhealthy":
                health_summary["unhealthy"] += 1
            elif status == "unreachable":
                health_summary["unreachable"] += 1
            else:
                health_summary["unknown"] += 1
        
        return APIResponse(
            message="Health summary retrieved",
            data=health_summary
        )
        
    except Exception as e:
        logger.error(f"Health summary failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health summary failed: {str(e)}"
        )


@app.on_event("startup")
async def startup_event():
    """Initialize registry service on startup"""
    app.state.start_time = time.time()
    logger.info("Registry service started")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await http_client.aclose()
    logger.info("Registry service stopped")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_level=config.LOG_LEVEL.lower()
    )