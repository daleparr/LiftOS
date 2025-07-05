"""
Lift Causal Module - Example integration with Lift OS Core

This module demonstrates causal modeling and analysis capabilities
integrated with the Lift OS Core memory system.
"""

import os
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
import json
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx

# Import shared utilities
import sys
sys.path.append('/app/shared')

from models.base import User
from auth.jwt_utils import verify_token, require_permissions
from utils.logging import get_logger
from utils.config import get_config
from kse_sdk.client import KSEMemoryClient

# Initialize logging
logger = get_logger(__name__)
config = get_config()

app = FastAPI(
    title="Lift Causal Module",
    description="Causal modeling and analysis for Lift OS Core",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize KSE Memory client
memory_client = KSEMemoryClient(
    api_key=config.kse_api_key,
    environment=config.kse_environment
)

# In-memory storage for causal models (replace with database in production)
causal_models: Dict[str, Dict[str, Any]] = {}
causal_analyses: Dict[str, List[Dict[str, Any]]] = {}

# Request/Response Models
class CausalModelRequest(BaseModel):
    name: str
    description: str
    variables: List[str]
    relationships: List[Dict[str, Any]]
    data_source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class CausalAnalysisRequest(BaseModel):
    model_id: str
    intervention: Dict[str, Any]
    target_variables: List[str]
    confidence_level: float = Field(0.95, ge=0.5, le=0.99)

class CausalInsightRequest(BaseModel):
    query: str
    context: Optional[str] = None
    include_models: Optional[List[str]] = None

class CausalModelResponse(BaseModel):
    model_id: str
    name: str
    description: str
    variables: List[str]
    relationships: List[Dict[str, Any]]
    created_at: datetime
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any]

class CausalAnalysisResponse(BaseModel):
    analysis_id: str
    model_id: str
    intervention: Dict[str, Any]
    results: Dict[str, Any]
    confidence_intervals: Dict[str, Any]
    created_at: datetime

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the causal module"""
    logger.info("Starting Lift Causal Module")
    
    # Register with the core registry
    await register_with_core()
    
    logger.info("Lift Causal Module initialized")

async def register_with_core():
    """Register this module with the Lift OS Core registry"""
    try:
        registry_url = os.getenv('REGISTRY_SERVICE_URL', 'http://registry:8004')
        
        module_config = {
            "module_id": "lift_causal",
            "name": "Lift Causal",
            "version": "1.0.0",
            "base_url": "http://lift-causal:9001",
            "health_endpoint": "/health",
            "api_prefix": "/api/v1",
            "features": [
                "causal_modeling",
                "causal_analysis",
                "causal_insights",
                "data_visualization"
            ],
            "memory_requirements": {
                "read_access": True,
                "write_access": True,
                "search_types": ["hybrid", "conceptual", "knowledge"],
                "memory_types": ["causal_models", "causal_analyses", "general"]
            },
            "ui_components": [
                {
                    "name": "CausalModelingDashboard",
                    "path": "/dashboard",
                    "permissions": ["causal:read"]
                },
                {
                    "name": "CausalAnalysisView",
                    "path": "/analysis",
                    "permissions": ["causal:read", "causal:analyze"]
                }
            ],
            "permissions": [
                "causal:read",
                "causal:write",
                "causal:analyze",
                "causal:delete",
                "memory:read",
                "memory:write"
            ],
            "metadata": {
                "description": "Advanced causal modeling and analysis capabilities",
                "author": "Lift Team",
                "category": "analytics",
                "tags": ["causal", "modeling", "analysis", "statistics"]
            }
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{registry_url}/api/v1/modules",
                json=module_config,
                timeout=10.0
            )
            
            if response.status_code == 200:
                logger.info("Successfully registered with Lift OS Core registry")
            else:
                logger.warning(f"Failed to register with registry: {response.status_code}")
                
    except Exception as e:
        logger.error(f"Error registering with core registry: {e}")

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "lift-causal",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "features": ["causal_modeling", "causal_analysis", "causal_insights"]
    }

# Causal Model Management
@app.post("/api/v1/models", response_model=CausalModelResponse)
async def create_causal_model(
    request: CausalModelRequest,
    current_user: User = Depends(verify_token)
):
    """Create a new causal model"""
    
    # Verify user has causal modeling permissions
    if not require_permissions(current_user, ["causal:write"]):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions for causal modeling"
        )
    
    model_id = f"causal_model_{int(datetime.utcnow().timestamp())}"
    
    # Create causal model
    causal_model = {
        "model_id": model_id,
        "name": request.name,
        "description": request.description,
        "variables": request.variables,
        "relationships": request.relationships,
        "data_source": request.data_source,
        "created_at": datetime.utcnow(),
        "updated_at": None,
        "created_by": current_user.id,
        "organization_id": current_user.org_id,
        "metadata": request.metadata or {}
    }
    
    causal_models[model_id] = causal_model
    
    # Store in memory system for future retrieval and insights
    try:
        memory_content = f"""
        Causal Model: {request.name}
        Description: {request.description}
        Variables: {', '.join(request.variables)}
        Relationships: {json.dumps(request.relationships, indent=2)}
        Created by: {current_user.name}
        Organization: {current_user.org_id}
        """
        
        await memory_client.store_memory(
            content=memory_content,
            memory_type="causal_models",
            metadata={
                "model_id": model_id,
                "model_name": request.name,
                "variables": request.variables,
                "created_by": current_user.id,
                "organization_id": current_user.org_id
            },
            org_id=current_user.org_id
        )
        
    except Exception as e:
        logger.warning(f"Failed to store model in memory system: {e}")
    
    logger.info(f"Created causal model {model_id} for user {current_user.id}")
    
    return CausalModelResponse(**causal_model)

@app.get("/api/v1/models")
async def get_causal_models(
    current_user: User = Depends(verify_token)
):
    """Get all causal models for the user's organization"""
    
    if not require_permissions(current_user, ["causal:read"]):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions for causal model access"
        )
    
    # Filter models by organization
    org_models = [
        model for model in causal_models.values()
        if model.get("organization_id") == current_user.org_id
    ]
    
    return {"models": org_models, "count": len(org_models)}

@app.get("/api/v1/models/{model_id}", response_model=CausalModelResponse)
async def get_causal_model(
    model_id: str,
    current_user: User = Depends(verify_token)
):
    """Get a specific causal model"""
    
    if not require_permissions(current_user, ["causal:read"]):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions for causal model access"
        )
    
    if model_id not in causal_models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Causal model {model_id} not found"
        )
    
    model = causal_models[model_id]
    
    # Check organization access
    if model.get("organization_id") != current_user.org_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this causal model"
        )
    
    return CausalModelResponse(**model)

# Causal Analysis
@app.post("/api/v1/analysis", response_model=CausalAnalysisResponse)
async def perform_causal_analysis(
    request: CausalAnalysisRequest,
    current_user: User = Depends(verify_token)
):
    """Perform causal analysis using a model"""
    
    if not require_permissions(current_user, ["causal:analyze"]):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions for causal analysis"
        )
    
    if request.model_id not in causal_models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Causal model {request.model_id} not found"
        )
    
    model = causal_models[request.model_id]
    
    # Check organization access
    if model.get("organization_id") != current_user.org_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this causal model"
        )
    
    analysis_id = f"analysis_{int(datetime.utcnow().timestamp())}"
    
    # Simulate causal analysis (in real implementation, this would use actual causal inference)
    results = simulate_causal_analysis(model, request.intervention, request.target_variables)
    confidence_intervals = calculate_confidence_intervals(results, request.confidence_level)
    
    analysis = {
        "analysis_id": analysis_id,
        "model_id": request.model_id,
        "intervention": request.intervention,
        "target_variables": request.target_variables,
        "results": results,
        "confidence_intervals": confidence_intervals,
        "created_at": datetime.utcnow(),
        "created_by": current_user.id,
        "organization_id": current_user.org_id
    }
    
    # Store analysis
    if current_user.org_id not in causal_analyses:
        causal_analyses[current_user.org_id] = []
    causal_analyses[current_user.org_id].append(analysis)
    
    # Store analysis results in memory system
    try:
        memory_content = f"""
        Causal Analysis Results
        Model: {model['name']}
        Intervention: {json.dumps(request.intervention, indent=2)}
        Target Variables: {', '.join(request.target_variables)}
        Results: {json.dumps(results, indent=2)}
        Confidence Level: {request.confidence_level}
        Performed by: {current_user.name}
        """
        
        await memory_client.store_memory(
            content=memory_content,
            memory_type="causal_analyses",
            metadata={
                "analysis_id": analysis_id,
                "model_id": request.model_id,
                "intervention": request.intervention,
                "target_variables": request.target_variables,
                "created_by": current_user.id,
                "organization_id": current_user.org_id
            },
            org_id=current_user.org_id
        )
        
    except Exception as e:
        logger.warning(f"Failed to store analysis in memory system: {e}")
    
    logger.info(f"Performed causal analysis {analysis_id} for user {current_user.id}")
    
    return CausalAnalysisResponse(**analysis)

@app.get("/api/v1/analysis")
async def get_causal_analyses(
    model_id: Optional[str] = None,
    current_user: User = Depends(verify_token)
):
    """Get causal analyses for the user's organization"""
    
    if not require_permissions(current_user, ["causal:read"]):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions for causal analysis access"
        )
    
    org_analyses = causal_analyses.get(current_user.org_id, [])
    
    # Filter by model_id if specified
    if model_id:
        org_analyses = [a for a in org_analyses if a["model_id"] == model_id]
    
    return {"analyses": org_analyses, "count": len(org_analyses)}

# Causal Insights
@app.post("/api/v1/insights")
async def get_causal_insights(
    request: CausalInsightRequest,
    current_user: User = Depends(verify_token)
):
    """Get causal insights using memory search"""
    
    if not require_permissions(current_user, ["causal:read", "memory:read"]):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions for causal insights"
        )
    
    try:
        # Search memory for relevant causal information
        search_results = await memory_client.hybrid_search(
            query=request.query,
            limit=10,
            memory_types=["causal_models", "causal_analyses"],
            org_id=current_user.org_id
        )
        
        # Process and format insights
        insights = []
        for result in search_results:
            insight = {
                "content": result["content"],
                "relevance_score": result["score"],
                "source_type": result["metadata"].get("memory_type", "unknown"),
                "model_id": result["metadata"].get("model_id"),
                "created_at": result["timestamp"]
            }
            insights.append(insight)
        
        return {
            "query": request.query,
            "insights": insights,
            "total_found": len(insights)
        }
        
    except Exception as e:
        logger.error(f"Error retrieving causal insights: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve causal insights"
        )

# Helper functions
def simulate_causal_analysis(model: Dict[str, Any], intervention: Dict[str, Any], target_variables: List[str]) -> Dict[str, Any]:
    """Simulate causal analysis (replace with actual causal inference)"""
    results = {}
    
    for target in target_variables:
        # Simulate effect estimation
        baseline_effect = np.random.normal(0, 1)
        intervention_effect = np.random.normal(2, 0.5)  # Positive effect
        
        results[target] = {
            "baseline_value": baseline_effect,
            "intervention_effect": intervention_effect,
            "total_effect": baseline_effect + intervention_effect,
            "effect_size": intervention_effect / abs(baseline_effect) if baseline_effect != 0 else float('inf')
        }
    
    return results

def calculate_confidence_intervals(results: Dict[str, Any], confidence_level: float) -> Dict[str, Any]:
    """Calculate confidence intervals for analysis results"""
    confidence_intervals = {}
    
    for target, result in results.items():
        # Simulate confidence interval calculation
        effect = result["intervention_effect"]
        margin_of_error = 1.96 * 0.5  # Simplified calculation
        
        confidence_intervals[target] = {
            "lower_bound": effect - margin_of_error,
            "upper_bound": effect + margin_of_error,
            "confidence_level": confidence_level
        }
    
    return confidence_intervals

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9001)