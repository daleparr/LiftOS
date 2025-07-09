"""
LiftOS Surfacing Service
Product Analysis and Surfacing Capabilities with KSE Integration
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager
import httpx

# Import shared modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import KSE SDK for universal intelligence substrate
from shared.kse_sdk.client import LiftKSEClient
from shared.kse_sdk.core import Entity, SearchQuery, SearchResult, ConceptualSpace, KSEConfig
from shared.kse_sdk.core.models import SearchType

from shared.models.base import APIResponse, HealthCheck
from shared.models.causal_marketing import (
    CausalOptimizationRequest, CausalOptimizationResponse,
    TreatmentRecommendationRequest, TreatmentRecommendationResponse,
    ExperimentDesignRequest, ExperimentDesignResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize KSE client for universal intelligence substrate
kse_client = LiftKSEClient()

# Global KSE integration
surfacing_kse = None

# Service configuration
SURFACING_MODULE_URL = os.getenv("SURFACING_MODULE_URL", "http://localhost:9005")


class SurfacingServiceKSEIntegration:
    """KSE integration for surfacing service"""
    
    def __init__(self, kse_client: LiftKSEClient):
        self.kse_client = kse_client
        self.logger = logger
    
    async def initialize(self):
        """Initialize KSE client connection"""
        try:
            await self.kse_client.initialize()
            self.logger.info("Surfacing Service KSE integration initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Surfacing Service KSE integration: {str(e)}")
            raise
    
    async def proxy_with_kse_enrichment(self, endpoint: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Proxy request to surfacing module with KSE enrichment"""
        try:
            org_id = request_data.get('org_id', 'default')
            
            # Retrieve optimization context from KSE
            optimization_context = await self.retrieve_optimization_context(request_data)
            
            # Add KSE context to request
            enriched_request = {
                **request_data,
                "kse_context": optimization_context
            }
            
            # Call surfacing module
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{SURFACING_MODULE_URL}/{endpoint}",
                    json=enriched_request,
                    timeout=60.0
                )
                response.raise_for_status()
                result = response.json()
            
            # Enrich KSE with results
            if result.get('recommendations'):
                await self.enrich_treatment_recommendations(result['recommendations'], org_id)
            
            if result.get('optimization_results'):
                await self.enrich_optimization_results(result['optimization_results'], org_id)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to proxy request with KSE enrichment: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def retrieve_optimization_context(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve optimization context from KSE"""
        try:
            org_id = request_data.get('org_id', 'default')
            campaign_type = request_data.get('campaign_type', 'general')
            
            search_results = await self.kse_client.hybrid_search(
                org_id=org_id,
                query=f"optimization patterns {campaign_type}",
                search_type="hybrid",
                limit=10,
                filters={
                    "entity_type": "optimization_pattern",
                    "success_rate": {"$gte": 0.7}
                }
            )
            
            context = {
                "historical_patterns": [],
                "performance_benchmarks": {},
                "optimization_strategies": []
            }
            
            for result in search_results:
                context["historical_patterns"].append({
                    "pattern_id": result.id,
                    "strategy": result.content.get("strategy"),
                    "success_rate": result.metadata.get("success_rate"),
                    "confidence_score": result.score
                })
            
            return context
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve optimization context: {str(e)}")
            return {}
    
    async def enrich_treatment_recommendations(self, recommendations: List[Dict], org_id: str) -> None:
        """Enrich KSE with treatment recommendations"""
        try:
            for rec in recommendations:
                entity = Entity(
                    id=f"treatment_rec_{rec.get('id', 'unknown')}",
                    type="treatment_recommendation",
                    content=rec,
                    metadata={
                        "org_id": org_id,
                        "confidence_score": rec.get('confidence', 0.0),
                        "treatment_type": rec.get('type', 'unknown'),
                        "expected_lift": rec.get('expected_lift', 0.0),
                        "service_source": "surfacing_service",
                        "entity_type": "treatment_recommendation"
                    }
                )
                await self.kse_client.store_entity(org_id, entity)
            
            self.logger.info(f"Enriched KSE with {len(recommendations)} treatment recommendations")
            
        except Exception as e:
            self.logger.error(f"Failed to enrich treatment recommendations: {str(e)}")
    
    async def enrich_optimization_results(self, optimization_results: Dict[str, Any], org_id: str) -> None:
        """Enrich KSE with optimization results"""
        try:
            entity = Entity(
                id=f"optimization_result_{optimization_results.get('id', 'unknown')}",
                type="optimization_result",
                content=optimization_results,
                metadata={
                    "org_id": org_id,
                    "optimization_type": optimization_results.get("type", "general"),
                    "performance_improvement": optimization_results.get("improvement", 0.0),
                    "confidence_level": optimization_results.get("confidence", 0.0),
                    "service_source": "surfacing_service",
                    "entity_type": "optimization_result"
                }
            )
            
            await self.kse_client.store_entity(org_id, entity)
            self.logger.info(f"Enriched KSE with optimization results for org {org_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to enrich optimization results: {str(e)}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global surfacing_kse
    
    logger.info("Starting Surfacing Service...")
    
    try:
        # Initialize KSE integration
        surfacing_kse = SurfacingServiceKSEIntegration(kse_client)
        await surfacing_kse.initialize()
        logger.info("Surfacing Service started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start Surfacing Service: {str(e)}")
        raise
    finally:
        logger.info("Shutting down Surfacing Service...")


# FastAPI app
app = FastAPI(
    title="LiftOS Surfacing Service",
    description="Product Analysis and Surfacing Capabilities with KSE Integration",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        service="surfacing",
        status="healthy",
        timestamp=datetime.utcnow(),
        version="1.0.0"
    )


@app.post("/optimize", response_model=CausalOptimizationResponse)
async def optimize_campaign(request: CausalOptimizationRequest):
    """Optimize campaign with KSE-enhanced intelligence"""
    try:
        result = await surfacing_kse.proxy_with_kse_enrichment("optimize", request.dict())
        return CausalOptimizationResponse(**result)
    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend-treatment", response_model=TreatmentRecommendationResponse)
async def recommend_treatment(request: TreatmentRecommendationRequest):
    """Generate treatment recommendations with KSE intelligence"""
    try:
        result = await surfacing_kse.proxy_with_kse_enrichment("recommend-treatment", request.dict())
        return TreatmentRecommendationResponse(**result)
    except Exception as e:
        logger.error(f"Treatment recommendation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/design-experiment", response_model=ExperimentDesignResponse)
async def design_experiment(request: ExperimentDesignRequest):
    """Design experiments with KSE-enhanced insights"""
    try:
        result = await surfacing_kse.proxy_with_kse_enrichment("design-experiment", request.dict())
        return ExperimentDesignResponse(**result)
    except Exception as e:
        logger.error(f"Experiment design failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/kse-status")
async def kse_status():
    """Check KSE integration status"""
    try:
        if surfacing_kse and surfacing_kse.kse_client:
            return {
                "kse_integration": "active",
                "status": "healthy",
                "timestamp": datetime.utcnow()
            }
        else:
            return {
                "kse_integration": "inactive",
                "status": "error",
                "timestamp": datetime.utcnow()
            }
    except Exception as e:
        return {
            "kse_integration": "error",
            "status": str(e),
            "timestamp": datetime.utcnow()
        }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8020)