"""
Lift OS Core - Module Template
Template for creating new Lift modules
"""
import time
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Header, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# Module configuration
MODULE_NAME = "template_module"
MODULE_VERSION = "1.0.0"
MODULE_PORT = 9000

# FastAPI app
app = FastAPI(
    title=f"Lift Module - {MODULE_NAME.title()}",
    description=f"Lift OS Core Module: {MODULE_NAME}",
    version=MODULE_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Memory service client
memory_service_url = "http://memory:8003"
http_client = httpx.AsyncClient(timeout=30.0)


class APIResponse(BaseModel):
    success: bool = True
    message: str = "Success"
    data: Optional[Any] = None
    errors: Optional[list] = None


class HealthCheck(BaseModel):
    status: str = "healthy"
    timestamp: datetime
    version: str
    uptime: Optional[float] = None


def get_user_context(
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
    x_memory_context: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
    x_module_id: Optional[str] = Header(None)
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
        "memory_context": x_memory_context,
        "roles": x_user_roles.split(",") if x_user_roles else [],
        "module_id": x_module_id
    }


async def search_memory(query: str, search_type: str = "hybrid", user_context: Dict[str, Any] = None):
    """Search memory using the memory service"""
    try:
        headers = {
            "X-User-ID": user_context["user_id"],
            "X-Org-ID": user_context["org_id"],
            "X-Memory-Context": user_context["memory_context"]
        }
        
        response = await http_client.post(
            f"{memory_service_url}/search",
            json={
                "query": query,
                "search_type": search_type,
                "limit": 10
            },
            headers=headers
        )
        
        if response.status_code == 200:
            return response.json()["data"]["results"]
        else:
            return []
            
    except Exception as e:
        print(f"Memory search failed: {str(e)}")
        return []


async def store_memory(content: str, memory_type: str = "general", metadata: Dict = None, user_context: Dict[str, Any] = None):
    """Store content in memory using the memory service"""
    try:
        headers = {
            "X-User-ID": user_context["user_id"],
            "X-Org-ID": user_context["org_id"],
            "X-Memory-Context": user_context["memory_context"]
        }
        
        response = await http_client.post(
            f"{memory_service_url}/store",
            json={
                "content": content,
                "memory_type": memory_type,
                "metadata": metadata or {}
            },
            headers=headers
        )
        
        if response.status_code == 200:
            return response.json()["data"]["memory_id"]
        else:
            return None
            
    except Exception as e:
        print(f"Memory storage failed: {str(e)}")
        return None


@app.get("/health")
async def health_check():
    """Module health check"""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.utcnow(),
        version=MODULE_VERSION,
        uptime=time.time() - getattr(app.state, "start_time", time.time())
    )


@app.get("/")
async def root():
    """Module root endpoint"""
    return APIResponse(
        message=f"Lift Module: {MODULE_NAME}",
        data={
            "name": MODULE_NAME,
            "version": MODULE_VERSION,
            "features": ["example_feature"],
            "docs": "/docs"
        }
    )


@app.get("/api/v1/info")
async def get_module_info(user_context: Dict[str, Any] = Header()):
    """Get module information"""
    return APIResponse(
        message="Module information",
        data={
            "name": MODULE_NAME,
            "version": MODULE_VERSION,
            "user_context": user_context,
            "capabilities": [
                "memory_integration",
                "user_context_aware",
                "health_monitoring"
            ]
        }
    )


@app.post("/api/v1/example")
async def example_endpoint(
    query: str,
    user_context: Dict[str, Any] = Header()
):
    """Example endpoint that uses memory integration"""
    correlation_id = str(uuid.uuid4())
    
    try:
        # Search memory for relevant information
        memory_results = await search_memory(
            query=query,
            search_type="hybrid",
            user_context=user_context
        )
        
        # Process the query (example logic)
        result = {
            "query": query,
            "processed_at": datetime.utcnow().isoformat(),
            "memory_results_count": len(memory_results),
            "memory_results": memory_results[:3],  # Top 3 results
            "correlation_id": correlation_id
        }
        
        # Store the result in memory for future reference
        await store_memory(
            content=f"Processed query: {query}",
            memory_type=f"{MODULE_NAME}_results",
            metadata={
                "correlation_id": correlation_id,
                "query": query,
                "results_count": len(memory_results)
            },
            user_context=user_context
        )
        
        return APIResponse(
            message="Query processed successfully",
            data=result
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {str(e)}"
        )


@app.get("/api/v1/memory/search")
async def search_module_memory(
    query: str,
    search_type: str = "hybrid",
    user_context: Dict[str, Any] = Header()
):
    """Search memory through the module"""
    try:
        results = await search_memory(
            query=query,
            search_type=search_type,
            user_context=user_context
        )
        
        return APIResponse(
            message="Memory search completed",
            data={
                "query": query,
                "search_type": search_type,
                "results": results
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Memory search failed: {str(e)}"
        )


@app.on_event("startup")
async def startup_event():
    """Initialize module on startup"""
    app.state.start_time = time.time()
    print(f"Module {MODULE_NAME} started on port {MODULE_PORT}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await http_client.aclose()
    print(f"Module {MODULE_NAME} stopped")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=MODULE_PORT,
        reload=True
    )