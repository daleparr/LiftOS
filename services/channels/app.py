"""
Lift Channels Service - Cross-Channel Budget Optimizer
Advanced multi-objective optimization with Bayesian inference
"""

import asyncio
import time
import uuid
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from fastapi import FastAPI, HTTPException, Depends, Header, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
import numpy as np
from contextlib import asynccontextmanager

# Import shared modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.models.base import APIResponse, HealthCheck
from shared.kse_sdk.client import LiftKSEClient
from shared.utils.config import get_service_config
from shared.utils.logging import setup_logging
from shared.health.health_checks import HealthChecker

# Import Channels-specific models and engines
from models.channels import (
    ChannelPerformance, BudgetOptimizationRequest, OptimizationResult,
    SimulationRequest, SimulationResult, ChannelConfiguration,
    SaturationModel, OptimizationConstraint, RecommendationResponse
)
from engines.optimization_engine import OptimizationEngine
from engines.simulation_engine import SimulationEngine
from engines.saturation_engine import SaturationEngine
from engines.bayesian_engine import BayesianEngine
from engines.recommendation_engine import RecommendationEngine
from integrations.service_clients import (
    CausalServiceClient, DataIngestionClient, MemoryServiceClient,
    BayesianServiceClient
)

# Service configuration
config = get_service_config("channels", 8003)
logger = setup_logging("channels")

# Health checker
health_checker = HealthChecker("channels")

# Global service clients
causal_client = None
data_client = None
memory_client = None
bayesian_client = None
kse_client = None

# Global engines
optimization_engine = None
simulation_engine = None
saturation_engine = None
bayesian_engine = None
recommendation_engine = None


async def initialize_services():
    """Initialize all service clients and engines"""
    global causal_client, data_client, memory_client, bayesian_client, kse_client
    global optimization_engine, simulation_engine, saturation_engine, bayesian_engine, recommendation_engine
    
    try:
        # Initialize service clients
        causal_client = CausalServiceClient(
            base_url=os.getenv("CAUSAL_SERVICE_URL", "http://lift-causal:9001")
        )
        data_client = DataIngestionClient(
            base_url=os.getenv("DATA_INGESTION_URL", "http://data-ingestion:8006")
        )
        memory_client = MemoryServiceClient(
            base_url=os.getenv("MEMORY_SERVICE_URL", "http://memory:8002")
        )
        bayesian_client = BayesianServiceClient(
            base_url=os.getenv("BAYESIAN_SERVICE_URL", "http://bayesian-analysis:8010")
        )
        
        # Initialize KSE client
        kse_client = LiftKSEClient()
        await kse_client.initialize()
        
        # Initialize engines in correct order
        saturation_engine = SaturationEngine(memory_client, data_client)
        bayesian_engine = BayesianEngine(memory_client, bayesian_client)
        optimization_engine = OptimizationEngine(
            saturation_engine, bayesian_engine, causal_client
        )
        simulation_engine = SimulationEngine(saturation_engine, bayesian_engine, optimization_engine)
        recommendation_engine = RecommendationEngine(
            optimization_engine, simulation_engine, saturation_engine, bayesian_engine
        )
        
        logger.info("All services and engines initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("Starting Lift Channels Service...")
    
    try:
        await initialize_services()
        app.state.start_time = time.time()
        logger.info("Lift Channels Service started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start Lift Channels Service: {str(e)}")
        raise
    finally:
        logger.info("Shutting down Lift Channels Service...")


# Custom JSON encoder for datetime serialization
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# Custom JSON response class
class CustomJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            cls=CustomJSONEncoder,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")

# FastAPI app
app = FastAPI(
    title="Lift Channels Service",
    description="Cross-Channel Budget Optimizer with Advanced Multi-Objective Optimization",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    default_response_class=CustomJSONResponse
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if config.DEBUG else ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Channels service health check"""
    try:
        # Check service dependencies
        dependencies = {}
        
        # Check KSE client
        if kse_client:
            kse_health = await kse_client.health_check()
            dependencies["kse_sdk"] = "healthy" if kse_health.get("healthy") else "unhealthy"
        else:
            dependencies["kse_sdk"] = "not_initialized"
        
        # Check service clients
        for name, client in [
            ("causal_service", causal_client),
            ("data_ingestion", data_client),
            ("memory_service", memory_client),
            ("bayesian_service", bayesian_client)
        ]:
            if client:
                try:
                    health_status = await client.health_check()
                    dependencies[name] = "healthy" if health_status else "unhealthy"
                except:
                    dependencies[name] = "unreachable"
            else:
                dependencies[name] = "not_initialized"
        
        # Check engines
        engine_status = {
            "optimization_engine": "ready" if optimization_engine else "not_initialized",
            "simulation_engine": "ready" if simulation_engine else "not_initialized",
            "saturation_engine": "ready" if saturation_engine else "not_initialized",
            "bayesian_engine": "ready" if bayesian_engine else "not_initialized",
            "recommendation_engine": "ready" if recommendation_engine else "not_initialized"
        }
        dependencies.update(engine_status)
        
        # Determine overall health
        critical_services = ["kse_sdk", "optimization_engine", "simulation_engine"]
        is_healthy = all(
            dependencies.get(service) in ["healthy", "ready"] 
            for service in critical_services
        )
        
        return HealthCheck(
            status="healthy" if is_healthy else "degraded",
            dependencies=dependencies,
            uptime=time.time() - getattr(app.state, "start_time", time.time())
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthCheck(
            status="unhealthy",
            dependencies={"error": str(e)},
            uptime=0
        )


@app.get("/ready", tags=["health"])
async def readiness_check():
    """Channels service readiness probe"""
    return await health_checker.get_readiness_status()


@app.get("/", response_model=APIResponse)
async def root():
    """Channels service root endpoint"""
    return APIResponse(
        message="Lift Channels Service - Cross-Channel Budget Optimizer",
        data={
            "version": "1.0.0",
            "capabilities": [
                "multi_objective_optimization",
                "bayesian_inference",
                "saturation_modeling",
                "scenario_simulation",
                "constraint_handling"
            ],
            "docs": "/docs"
        }
    )


# Core Optimization Endpoints

@app.post("/api/v1/optimize/budget", response_model=APIResponse)
async def optimize_budget(
    request: BudgetOptimizationRequest,
    background_tasks: BackgroundTasks,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Optimize budget allocation across channels"""
    try:
        optimization_id = str(uuid.uuid4())
        
        logger.info(
            f"Starting budget optimization for org {user_context['org_id']}",
            extra={
                "optimization_id": optimization_id,
                "org_id": user_context["org_id"],
                "total_budget": request.total_budget,
                "channels": request.channels
            }
        )
        
        # Run optimization
        result = await optimization_engine.optimize_budget(
            org_id=user_context["org_id"],
            request=request,
            optimization_id=optimization_id
        )
        
        # Store optimization result in memory
        background_tasks.add_task(
            store_optimization_result,
            user_context["org_id"],
            optimization_id,
            result
        )
        
        return APIResponse(
            message="Budget optimization completed",
            data=result.dict(),
            metadata={
                "optimization_id": optimization_id,
                "org_id": user_context["org_id"]
            }
        )
        
    except Exception as e:
        logger.error(f"Budget optimization failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Budget optimization failed: {str(e)}"
        )


@app.post("/api/v1/simulate/scenarios", response_model=APIResponse)
async def simulate_scenarios(
    request: SimulationRequest,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Simulate what-if scenarios for budget allocation"""
    try:
        simulation_id = str(uuid.uuid4())
        
        logger.info(
            f"Starting scenario simulation for org {user_context['org_id']}",
            extra={
                "simulation_id": simulation_id,
                "org_id": user_context["org_id"],
                "scenarios": len(request.scenarios)
            }
        )
        
        # Run simulation
        result = await simulation_engine.simulate_scenarios(
            org_id=user_context["org_id"],
            request=request,
            simulation_id=simulation_id
        )
        
        return APIResponse(
            message="Scenario simulation completed",
            data=result.dict(),
            metadata={
                "simulation_id": simulation_id,
                "org_id": user_context["org_id"]
            }
        )
        
    except Exception as e:
        logger.error(f"Scenario simulation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scenario simulation failed: {str(e)}"
        )


@app.get("/api/v1/recommendations/{org_id}", response_model=APIResponse)
async def get_recommendations(
    org_id: str,
    limit: int = 10,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Get budget reallocation recommendations"""
    try:
        # Verify org access
        if user_context["org_id"] != org_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to organization data"
            )
        
        # Get recommendations
        recommendations = await recommendation_engine.get_recommendations(
            org_id=org_id,
            limit=limit
        )
        
        return APIResponse(
            message="Recommendations retrieved successfully",
            data=recommendations,
            metadata={
                "org_id": org_id,
                "count": len(recommendations)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get recommendations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get recommendations: {str(e)}"
        )


@app.get("/api/v1/saturation/{channel_id}", response_model=APIResponse)
async def get_saturation_curve(
    channel_id: str,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Get saturation curve for a specific channel"""
    try:
        # Get saturation model
        saturation_model = await saturation_engine.get_channel_saturation(
            org_id=user_context["org_id"],
            channel_id=channel_id
        )
        
        return APIResponse(
            message="Saturation curve retrieved successfully",
            data=saturation_model.dict(),
            metadata={
                "channel_id": channel_id,
                "org_id": user_context["org_id"]
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get saturation curve: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get saturation curve: {str(e)}"
        )


@app.post("/api/v1/constraints/validate", response_model=APIResponse)
async def validate_constraints(
    constraints: List[OptimizationConstraint],
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Validate optimization constraints"""
    try:
        # Validate constraints
        validation_result = await optimization_engine.validate_constraints(
            org_id=user_context["org_id"],
            constraints=constraints
        )
        
        return APIResponse(
            message="Constraints validated successfully",
            data=validation_result,
            metadata={
                "org_id": user_context["org_id"],
                "constraints_count": len(constraints)
            }
        )
        
    except Exception as e:
        logger.error(f"Constraint validation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Constraint validation failed: {str(e)}"
        )


# Channel Management Endpoints

@app.get("/api/v1/channels", response_model=APIResponse)
async def list_channels(
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """List all configured channels for organization"""
    try:
        # Get channel configurations
        channels = await data_client.get_channels(user_context["org_id"])
        
        return APIResponse(
            message="Channels retrieved successfully",
            data=channels,
            metadata={
                "org_id": user_context["org_id"],
                "count": len(channels)
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to list channels: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list channels: {str(e)}"
        )


@app.post("/api/v1/channels/{channel_id}/calibrate", response_model=APIResponse)
async def calibrate_channel(
    channel_id: str,
    background_tasks: BackgroundTasks,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Calibrate saturation model for a specific channel"""
    try:
        calibration_id = str(uuid.uuid4())
        
        # Start calibration process
        background_tasks.add_task(
            calibrate_channel_saturation,
            user_context["org_id"],
            channel_id,
            calibration_id
        )
        
        return APIResponse(
            message="Channel calibration started",
            data={
                "calibration_id": calibration_id,
                "channel_id": channel_id,
                "status": "in_progress"
            },
            metadata={
                "org_id": user_context["org_id"]
            }
        )
        
    except Exception as e:
        logger.error(f"Channel calibration failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Channel calibration failed: {str(e)}"
        )


# Background Tasks

async def store_optimization_result(org_id: str, optimization_id: str, result: OptimizationResult):
    """Store optimization result in memory service"""
    try:
        await memory_client.store_optimization_result(org_id, optimization_id, result)
        logger.info(f"Stored optimization result {optimization_id} for org {org_id}")
    except Exception as e:
        logger.error(f"Failed to store optimization result: {str(e)}")


async def calibrate_channel_saturation(org_id: str, channel_id: str, calibration_id: str):
    """Calibrate saturation model for channel"""
    try:
        await saturation_engine.calibrate_channel(org_id, channel_id, calibration_id)
        logger.info(f"Completed calibration {calibration_id} for channel {channel_id}")
    except Exception as e:
        logger.error(f"Channel calibration failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)