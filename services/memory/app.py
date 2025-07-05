"""
Lift OS Core - Memory Service
KSE Memory SDK Integration
"""
import asyncio
import time
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware
import uuid

# Import shared modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.models.base import (
    APIResponse, HealthCheck, MemorySearchRequest, MemorySearchResult,
    MemoryStoreRequest, MemoryInsights
)
from shared.models.marketing import (
    MarketingDataIngestionRequest, MarketingDataSearchRequest,
    MarketingDataAggregationRequest, MarketingInsights,
    DataSource, MarketingDataEntry, CampaignData, AdSetData, AdData,
    MetaBusinessData, GoogleAdsData, KlaviyoData, CalendarDimension,
    PandasTransformationConfig
)
from shared.models.causal_marketing import (
    CausalMarketingData, CausalExperiment, ConfounderVariable,
    ExternalFactor, CausalGraph, CausalInsight, AttributionModel,
    CausalAnalysisRequest, CausalAnalysisResponse,
    CausalExperimentRequest, CausalExperimentResponse,
    ConfounderAnalysisRequest, ConfounderAnalysisResponse
)
from shared.kse_sdk.client import kse_client
from shared.utils.config import get_service_config
from shared.utils.logging import setup_logging, get_memory_logger
from shared.utils.marketing_transforms import MarketingDataTransformer, create_marketing_data_entry
from shared.utils.causal_transforms import CausalDataTransformer
from shared.health.health_checks import HealthChecker

# Import MMM Spine integration components
from shared.mmm_spine_integration.memory import (
    EnhancedKSEIntegration, EmbeddingManager, MemoryInterface
)
from shared.mmm_spine_integration.observability import (
    ObservabilityManager, TraceLevel
)
from shared.mmm_spine_integration.orchestration import (
    UltraFastOrchestrator, WorkflowOrchestrator, TaskPriority
)

# Service configuration
config = get_service_config("memory", 8003)
logger = setup_logging("memory")
memory_logger = get_memory_logger("memory")

# Health checker
health_checker = HealthChecker("memory")

# Marketing data transformer
marketing_transformer = MarketingDataTransformer()

# Causal data transformer
causal_transformer = CausalDataTransformer()

# Initialize MMM Spine integration components
embedding_manager = EmbeddingManager(cache_size=10000, cache_ttl=3600)
enhanced_kse = EnhancedKSEIntegration(embedding_manager)
memory_interface = MemoryInterface(enhanced_kse)
observability = ObservabilityManager("memory", "default_org")
orchestrator = UltraFastOrchestrator(
    max_workers=10,
    cache_size=10000,
    observability=observability
)
workflow_orchestrator = WorkflowOrchestrator(orchestrator)

# Register core memory functions for orchestration
orchestrator.register_function("enhanced_search", memory_interface.search)
orchestrator.register_function("enhanced_store", memory_interface.store)
orchestrator.register_function("create_conceptual_space", memory_interface.create_conceptual_space)

# FastAPI app
app = FastAPI(
    title="Lift OS Core - Memory Service",
    description="KSE Memory SDK Integration Service",
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


def get_user_context(
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
    x_memory_context: Optional[str] = Header(None),
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
        "memory_context": x_memory_context or f"org_{x_org_id}_context",
        "roles": x_user_roles.split(",") if x_user_roles else []
    }


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Memory service health check with MMM Spine integration metrics"""
    kse_health = await kse_client.health_check()
    context_stats = await kse_client.get_context_stats()
    
    # Get MMM Spine integration metrics
    embedding_stats = embedding_manager.get_cache_stats()
    memory_performance = memory_interface.get_performance_metrics()
    orchestration_stats = orchestrator.get_orchestration_stats()
    observability_stats = observability.get_comprehensive_stats()
    
    dependencies = {
        "kse_sdk": "healthy" if kse_health.get("healthy") else "unhealthy",
        "active_contexts": str(context_stats.get("active_contexts", 0)),
        "embedding_cache_hit_rate": f"{embedding_stats.get('hit_rate', 0):.2%}",
        "embedding_cache_size": str(embedding_stats.get('cache_size', 0)),
        "memory_operations": str(memory_performance.get('total_operations', 0)),
        "memory_success_rate": f"{memory_performance.get('success_rate', 0):.2%}",
        "orchestration_cache_hit_rate": f"{orchestration_stats.get('cache_hit_rate', 0):.2%}",
        "total_workflows": str(orchestration_stats.get('total_workflows', 0)),
        "observability_traces": str(observability_stats.get('tracing', {}).get('total_spans', 0))
    }
    
    # Determine overall health status
    is_healthy = (
        kse_health.get("healthy", False) and
        embedding_stats.get('hit_rate', 0) > 0.5 and  # Good cache performance
        memory_performance.get('success_rate', 0) > 0.95  # High success rate
    )
    
    return HealthCheck(
        status="healthy" if is_healthy else "degraded",
        dependencies=dependencies,
        uptime=time.time() - getattr(app.state, "start_time", time.time())
    )


@app.get("/ready", tags=["health"])
async def readiness_check():
    """Memory service readiness probe - service is ready to handle requests"""
    return await health_checker.get_readiness_status()


@app.get("/", response_model=APIResponse)
async def root():
    """Memory service root endpoint"""
    return APIResponse(
        message="Lift OS Core Memory Service",
        data={
            "version": "1.0.0",
            "kse_integration": "active",
            "docs": "/docs"
        }
    )


@app.post("/search", response_model=APIResponse)
async def search_memory(
    request: MemorySearchRequest,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Search organization memory using enhanced KSE capabilities with MMM Spine integration"""
    start_time = datetime.utcnow()
    correlation_id = str(uuid.uuid4())
    
    try:
        # Use enhanced memory interface with observability
        async with observability.observe_async_operation(
            "memory_search",
            tags={
                "org_id": user_context["org_id"],
                "search_type": request.search_type,
                "query_length": len(request.query),
                "correlation_id": correlation_id
            },
            level=TraceLevel.INFO
        ) as span:
            
            # Perform enhanced search with conceptual space support
            results = await memory_interface.search(
                org_id=user_context["org_id"],
                query=request.query,
                search_type=request.search_type,
                limit=request.limit,
                filters=request.filters
            )
            
            # Add span metadata
            if span:
                observability.tracer.add_span_tag(span, "results_count", len(results))
                observability.tracer.add_span_tag(span, "cache_performance",
                                                embedding_manager.get_cache_stats())
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        # Record accountability decision
        observability.accountability.record_decision(
            operation="memory_search",
            decision_point="search_algorithm_selection",
            reasoning=f"Selected {request.search_type} search for query: {request.query[:50]}...",
            inputs={
                "query": request.query,
                "search_type": request.search_type,
                "org_id": user_context["org_id"]
            },
            outputs={
                "results_count": len(results),
                "duration_ms": duration * 1000
            },
            confidence=0.95,
            trace_id=span.trace_id if span else None,
            span_id=span.span_id if span else None
        )
        
        # Log the search operation
        memory_logger.log_search(
            org_id=user_context["org_id"],
            query=request.query,
            search_type=request.search_type,
            results_count=len(results),
            duration=duration,
            user_id=user_context["user_id"],
            correlation_id=correlation_id
        )
        
        # Get performance metrics for response
        performance_metrics = memory_interface.get_performance_metrics()
        
        return APIResponse(
            message="Enhanced memory search completed",
            data={
                "results": results,
                "count": len(results),
                "search_type": request.search_type,
                "duration": duration,
                "performance_metrics": {
                    "cache_hit_rate": performance_metrics.get("embedding_cache_stats", {}).get("hit_rate", 0),
                    "average_operation_time_ms": performance_metrics.get("recent_average_time_ms", 0),
                    "conceptual_spaces_available": len(performance_metrics.get("conceptual_spaces_count", 0))
                }
            },
            metadata={
                "correlation_id": correlation_id,
                "org_id": user_context["org_id"],
                "query_length": len(request.query),
                "enhanced_features": ["embedding_cache", "conceptual_spaces", "observability"]
            }
        )
        
    except Exception as e:
        logger.error(f"Enhanced memory search failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Enhanced memory search failed: {str(e)}"
        )


@app.post("/store", response_model=APIResponse)
async def store_memory(
    request: MemoryStoreRequest,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Store content in organization memory with enhanced KSE integration"""
    correlation_id = str(uuid.uuid4())
    
    try:
        # Use enhanced memory interface with observability
        async with observability.observe_async_operation(
            "memory_store",
            tags={
                "org_id": user_context["org_id"],
                "memory_type": request.memory_type,
                "content_length": len(request.content),
                "correlation_id": correlation_id
            },
            level=TraceLevel.INFO
        ) as span:
            
            # Store using enhanced interface with conceptual mapping
            memory_id = await memory_interface.store(
                org_id=user_context["org_id"],
                content=request.content,
                metadata=request.metadata,
                memory_type=request.memory_type
            )
            
            # Add span metadata
            if span:
                observability.tracer.add_span_tag(span, "memory_id", memory_id)
                observability.tracer.add_span_tag(span, "embedding_generated", True)
        
        # Record accountability decision
        observability.accountability.record_decision(
            operation="memory_store",
            decision_point="storage_strategy_selection",
            reasoning=f"Stored {request.memory_type} memory with enhanced embedding and conceptual mapping",
            inputs={
                "content_length": len(request.content),
                "memory_type": request.memory_type,
                "org_id": user_context["org_id"]
            },
            outputs={
                "memory_id": memory_id,
                "embedding_cached": True
            },
            confidence=0.98,
            trace_id=span.trace_id if span else None,
            span_id=span.span_id if span else None
        )
        
        # Log the storage operation
        memory_logger.log_storage(
            org_id=user_context["org_id"],
            memory_type=request.memory_type,
            content_length=len(request.content),
            memory_id=memory_id,
            user_id=user_context["user_id"],
            correlation_id=correlation_id
        )
        
        # Get performance metrics
        performance_metrics = memory_interface.get_performance_metrics()
        
        return APIResponse(
            message="Enhanced memory storage completed",
            data={
                "memory_id": memory_id,
                "memory_type": request.memory_type,
                "content_length": len(request.content),
                "enhanced_features": {
                    "embedding_generated": True,
                    "conceptual_mapping": True,
                    "cache_optimization": True
                },
                "performance_metrics": {
                    "storage_time_ms": performance_metrics.get("recent_average_time_ms", 0),
                    "success_rate": performance_metrics.get("success_rate", 0)
                }
            },
            metadata={
                "correlation_id": correlation_id,
                "org_id": user_context["org_id"],
                "enhanced_storage": True
            }
        )
        
    except Exception as e:
        logger.error(f"Enhanced memory storage failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Enhanced memory storage failed: {str(e)}"
        )


@app.get("/insights", response_model=APIResponse)
async def get_memory_insights(
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Get memory analytics and insights for organization"""
    correlation_id = str(uuid.uuid4())
    
    try:
        # Get insights from KSE
        insights = await kse_client.get_memory_insights(user_context["org_id"])
        
        logger.info(
            f"Memory insights generated for org {user_context['org_id']}",
            extra={
                "org_id": user_context["org_id"],
                "user_id": user_context["user_id"],
                "correlation_id": correlation_id
            }
        )
        
        return APIResponse(
            message="Memory insights generated",
            data=insights.dict(),
            metadata={
                "correlation_id": correlation_id,
                "org_id": user_context["org_id"]
            }
        )
        
    except Exception as e:
        logger.error(f"Memory insights failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Memory insights failed: {str(e)}"
        )


@app.post("/context/initialize", response_model=APIResponse)
async def initialize_memory_context(
    domain: str = "general",
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Initialize memory context for organization"""
    correlation_id = str(uuid.uuid4())
    
    try:
        # Initialize org memory context
        context_id = await kse_client.initialize_org_memory(
            org_id=user_context["org_id"],
            domain=domain
        )
        
        # Log context initialization
        memory_logger.log_context_init(
            org_id=user_context["org_id"],
            context_id=context_id,
            domain=domain,
            user_id=user_context["user_id"],
            correlation_id=correlation_id
        )
        
        return APIResponse(
            message="Memory context initialized",
            data={
                "context_id": context_id,
                "org_id": user_context["org_id"],
                "domain": domain
            },
            metadata={
                "correlation_id": correlation_id
            }
        )
        
    except Exception as e:
        logger.error(f"Memory context initialization failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Memory context initialization failed: {str(e)}"
        )



# Enhanced MMM Spine Integration Endpoints

@app.post("/conceptual-spaces/create", response_model=APIResponse)
async def create_conceptual_space(
    name: str,
    dimensions: List[str],
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Create a conceptual space for knowledge organization"""
    correlation_id = str(uuid.uuid4())
    
    try:
        async with observability.observe_async_operation(
            "create_conceptual_space",
            tags={
                "org_id": user_context["org_id"],
                "space_name": name,
                "dimensions_count": len(dimensions)
            },
            level=TraceLevel.INFO
        ) as span:
            
            space_id = await memory_interface.create_conceptual_space(
                org_id=user_context["org_id"],
                name=name,
                dimensions=dimensions
            )
            
            if span:
                observability.tracer.add_span_tag(span, "space_id", space_id)
        
        # Record accountability decision
        observability.accountability.record_decision(
            operation="create_conceptual_space",
            decision_point="space_design",
            reasoning=f"Created conceptual space '{name}' with {len(dimensions)} dimensions for knowledge organization",
            inputs={
                "name": name,
                "dimensions": dimensions,
                "org_id": user_context["org_id"]
            },
            outputs={
                "space_id": space_id
            },
            confidence=0.95,
            trace_id=span.trace_id if span else None,
            span_id=span.span_id if span else None
        )
        
        logger.info(f"Created conceptual space {space_id} for org {user_context['org_id']}")
        
        return APIResponse(
            message="Conceptual space created successfully",
            data={
                "space_id": space_id,
                "name": name,
                "dimensions": dimensions,
                "org_id": user_context["org_id"]
            },
            metadata={
                "correlation_id": correlation_id,
                "enhanced_feature": "conceptual_spaces"
            }
        )
        
    except Exception as e:
        logger.error(f"Conceptual space creation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Conceptual space creation failed: {str(e)}"
        )


@app.get("/conceptual-spaces", response_model=APIResponse)
async def list_conceptual_spaces(
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """List all conceptual spaces for the organization"""
    try:
        spaces_info = memory_interface.get_conceptual_spaces_info()
        
        return APIResponse(
            message="Conceptual spaces retrieved",
            data={
                "conceptual_spaces": spaces_info,
                "count": len(spaces_info)
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to retrieve conceptual spaces: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve conceptual spaces: {str(e)}"
        )


@app.get("/performance/metrics", response_model=APIResponse)
async def get_performance_metrics(
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Get comprehensive performance metrics for MMM Spine integration"""
    try:
        # Gather all performance metrics
        memory_metrics = memory_interface.get_performance_metrics()
        embedding_metrics = embedding_manager.get_cache_stats()
        orchestration_metrics = orchestrator.get_orchestration_stats()
        observability_metrics = observability.get_comprehensive_stats()
        
        comprehensive_metrics = {
            "memory_performance": memory_metrics,
            "embedding_cache": embedding_metrics,
            "orchestration": orchestration_metrics,
            "observability": observability_metrics,
            "summary": {
                "total_operations": memory_metrics.get("total_operations", 0),
                "success_rate": memory_metrics.get("success_rate", 0),
                "cache_hit_rate": embedding_metrics.get("hit_rate", 0),
                "average_response_time_ms": memory_metrics.get("recent_average_time_ms", 0),
                "conceptual_spaces_count": memory_metrics.get("conceptual_spaces_count", 0),
                "active_workflows": orchestration_metrics.get("active_workflows", 0)
            }
        }
        
        return APIResponse(
            message="Performance metrics retrieved",
            data=comprehensive_metrics
        )
        
    except Exception as e:
        logger.error(f"Failed to retrieve performance metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve performance metrics: {str(e)}"
        )


@app.post("/workflows/create", response_model=APIResponse)
async def create_workflow(
    name: str,
    metadata: Optional[Dict[str, Any]] = None,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Create a new workflow for orchestrated operations"""
    correlation_id = str(uuid.uuid4())
    
    try:
        workflow_id = orchestrator.create_workflow(
            name=name,
            org_id=user_context["org_id"],
            metadata=metadata or {}
        )
        
        logger.info(f"Created workflow {workflow_id} for org {user_context['org_id']}")
        
        return APIResponse(
            message="Workflow created successfully",
            data={
                "workflow_id": workflow_id,
                "name": name,
                "org_id": user_context["org_id"],
                "metadata": metadata
            },
            metadata={
                "correlation_id": correlation_id,
                "enhanced_feature": "orchestration"
            }
        )
        
    except Exception as e:
        logger.error(f"Workflow creation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow creation failed: {str(e)}"
        )


@app.get("/workflows/{workflow_id}/status", response_model=APIResponse)
async def get_workflow_status(
    workflow_id: str,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Get workflow execution status"""
    try:
        status_info = orchestrator.get_workflow_status(workflow_id)
        
        return APIResponse(
            message="Workflow status retrieved",
            data=status_info
        )
        
    except Exception as e:
        logger.error(f"Failed to get workflow status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workflow status: {str(e)}"
        )


@app.post("/search/batch", response_model=APIResponse)
async def batch_search_memory(
    queries: List[str],
    search_type: str = "hybrid",
    limit: int = 10,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Perform batch search operations with enhanced performance"""
    correlation_id = str(uuid.uuid4())
    start_time = datetime.utcnow()
    
    try:
        async with observability.observe_async_operation(
            "batch_memory_search",
            tags={
                "org_id": user_context["org_id"],
                "queries_count": len(queries),
                "search_type": search_type
            },
            level=TraceLevel.INFO
        ) as span:
            
            # Use orchestrator for batch processing
            workflow_id = orchestrator.create_workflow(
                name="batch_search",
                org_id=user_context["org_id"],
                metadata={"correlation_id": correlation_id}
            )
            
            # Add search tasks for each query
            task_ids = []
            for i, query in enumerate(queries):
                task_id = orchestrator.add_task(
                    workflow_id=workflow_id,
                    task_name=f"search_query_{i}",
                    function_name="enhanced_search",
                    args=(user_context["org_id"], query),
                    kwargs={
                        "search_type": search_type,
                        "limit": limit
                    },
                    priority=TaskPriority.HIGH
                )
                task_ids.append(task_id)
            
            # Execute workflow
            workflow_results = await orchestrator.execute_workflow(
                workflow_id,
                context={"batch_operation": True}
            )
            
            if span:
                observability.tracer.add_span_tag(span, "workflow_id", workflow_id)
                observability.tracer.add_span_tag(span, "tasks_executed", len(task_ids))
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        return APIResponse(
            message="Batch search completed",
            data={
                "workflow_id": workflow_id,
                "results": workflow_results.get("task_results", {}),
                "queries_processed": len(queries),
                "execution_time_ms": workflow_results.get("execution_time_ms", 0),
                "duration": duration
            },
            metadata={
                "correlation_id": correlation_id,
                "enhanced_feature": "batch_processing"
            }
        )
        
    except Exception as e:
        logger.error(f"Batch search failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch search failed: {str(e)}"
        )
@app.get("/context/stats", response_model=APIResponse)
async def get_context_stats(
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Get memory context statistics"""
    try:
        stats = await kse_client.get_context_stats()
        
        return APIResponse(
            message="Context statistics retrieved",
            data=stats
        )
        
    except Exception as e:
        logger.error(f"Context stats failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Context stats failed: {str(e)}"
        )


@app.delete("/context/{org_id}", response_model=APIResponse)
async def cleanup_memory_context(
    org_id: str,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Cleanup memory context for organization"""
    # Check if user can access this org
    if user_context["org_id"] != org_id and "admin" not in user_context["roles"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to organization context"
        )
    
    try:
        await kse_client.cleanup_org_context(org_id)
        
        return APIResponse(
            message="Memory context cleaned up",
            data={"org_id": org_id}
        )
        
    except Exception as e:
        logger.error(f"Memory context cleanup failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Memory context cleanup failed: {str(e)}"
        )


# Neural search endpoint
@app.post("/search/neural", response_model=APIResponse)
async def neural_search(
    request: MemorySearchRequest,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Perform neural embeddings search"""
    request.search_type = "neural"
    return await search_memory(request, user_context)


# Conceptual search endpoint
@app.post("/search/conceptual", response_model=APIResponse)
async def conceptual_search(
    request: MemorySearchRequest,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Perform conceptual spaces search"""
    request.search_type = "conceptual"
    return await search_memory(request, user_context)


# Knowledge graph search endpoint
@app.post("/search/knowledge", response_model=APIResponse)
async def knowledge_search(
    request: MemorySearchRequest,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Perform knowledge graph search"""
    request.search_type = "knowledge"
    return await search_memory(request, user_context)


# Marketing Data Endpoints
@app.post("/marketing/ingest", response_model=APIResponse)
async def ingest_marketing_data(
    request: MarketingDataIngestionRequest,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Ingest marketing data from various sources into KSE memory"""
    correlation_id = str(uuid.uuid4())
    start_time = datetime.utcnow()
    
    try:
        # Transform raw data using pandas
        df_transformed = marketing_transformer.transform_raw_data(
            raw_data=request.data_entries,
            data_source=request.data_source,
            date_range_start=request.date_range_start,
            date_range_end=request.date_range_end,
            org_id=user_context["org_id"]
        )
        
        stored_entries = []
        
        # Process each transformed row
        for _, row in df_transformed.iterrows():
            # Create marketing data entry
            entry = create_marketing_data_entry(
                raw_data=row.to_dict(),
                data_source=request.data_source,
                org_id=user_context["org_id"],
                date_range_start=request.date_range_start,
                date_range_end=request.date_range_end
            )
            
            # Enhanced content with transformed data
            content_data = {
                "data_source": request.data_source.value,
                "metrics": {
                    "spend": row.get("spend", 0),
                    "impressions": row.get("impressions", 0),
                    "clicks": row.get("clicks", 0),
                    "conversions": row.get("conversions", 0),
                    "revenue": row.get("revenue", 0),
                    "cpm": row.get("cpm", 0),
                    "cpc": row.get("cpc", 0),
                    "ctr": row.get("ctr", 0),
                    "conversion_rate": row.get("conversion_rate", 0),
                    "roas": row.get("roas", 0)
                },
                "calendar": {
                    "date": str(row.get("date", request.date_range_start)),
                    "year": row.get("year"),
                    "quarter": row.get("quarter"),
                    "month": row.get("month"),
                    "week": row.get("week"),
                    "day_of_week": row.get("day_of_week"),
                    "is_weekend": row.get("is_weekend"),
                    "season": row.get("season")
                },
                "identifiers": {
                    "campaign_id": row.get("campaign_id"),
                    "adset_id": row.get("adset_id"),
                    "ad_id": row.get("ad_id")
                }
            }
            
            content = f"Marketing data from {request.data_source.value}: {content_data}"
            
            # Enhanced metadata for better searchability
            memory_metadata = {
                "data_source": request.data_source.value,
                "source_id": entry.source_id,
                "date_range_start": str(request.date_range_start),
                "date_range_end": str(request.date_range_end),
                "entry_id": entry.id,
                "campaign_id": row.get("campaign_id"),
                "spend": float(row.get("spend", 0)),
                "impressions": int(row.get("impressions", 0)),
                "clicks": int(row.get("clicks", 0)),
                "conversions": int(row.get("conversions", 0)),
                "revenue": float(row.get("revenue", 0)),
                "year": row.get("year"),
                "quarter": row.get("quarter"),
                "month": row.get("month"),
                "season": row.get("season"),
                **(request.metadata or {})
            }
            
            memory_id = await kse_client.store_memory(
                org_id=user_context["org_id"],
                content=content,
                metadata=memory_metadata,
                memory_type=entry.memory_type
            )
            
            stored_entries.append({
                "entry_id": entry.id,
                "memory_id": memory_id,
                "data_source": request.data_source.value,
                "metrics": content_data["metrics"]
            })
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        # Log the ingestion operation
        memory_logger.log_storage(
            org_id=user_context["org_id"],
            memory_type="marketing_data_batch",
            content_length=len(request.data_entries),
            memory_id=f"batch_{correlation_id}",
            user_id=user_context["user_id"],
            correlation_id=correlation_id
        )
        
        return APIResponse(
            message=f"Successfully ingested {len(stored_entries)} marketing data entries",
            data={
                "stored_entries": stored_entries,
                "data_source": request.data_source.value,
                "date_range": {
                    "start": str(request.date_range_start),
                    "end": str(request.date_range_end)
                },
                "duration": duration
            },
            metadata={
                "correlation_id": correlation_id,
                "org_id": user_context["org_id"]
            }
        )
        
    except Exception as e:
        logger.error(f"Marketing data ingestion failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Marketing data ingestion failed: {str(e)}"
        )


@app.post("/marketing/search", response_model=APIResponse)
async def search_marketing_data(
    request: MarketingDataSearchRequest,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Search marketing data with advanced filtering"""
    correlation_id = str(uuid.uuid4())
    start_time = datetime.utcnow()
    
    try:
        # Build search filters
        filters = {}
        
        if request.data_sources:
            filters["data_source"] = [source.value for source in request.data_sources]
        
        if request.date_range_start:
            filters["date_range_start"] = {"$gte": str(request.date_range_start)}
        
        if request.date_range_end:
            filters["date_range_end"] = {"$lte": str(request.date_range_end)}
        
        if request.campaign_ids:
            filters["campaign_id"] = {"$in": request.campaign_ids}
        
        # Perform search using KSE hybrid capabilities
        results = await kse_client.hybrid_search(
            query=request.query,
            org_id=user_context["org_id"],
            limit=request.limit,
            filters=filters,
            search_type=request.search_type
        )
        
        # Filter results to marketing data only
        marketing_results = [
            result for result in results
            if result.metadata.get("data_source") in [source.value for source in DataSource]
        ]
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        # Log the search operation
        memory_logger.log_search(
            org_id=user_context["org_id"],
            query=request.query,
            search_type=request.search_type,
            results_count=len(marketing_results),
            duration=duration,
            user_id=user_context["user_id"],
            correlation_id=correlation_id
        )
        
        return APIResponse(
            message="Marketing data search completed",
            data={
                "results": [result.dict() for result in marketing_results],
                "count": len(marketing_results),
                "search_type": request.search_type,
                "filters_applied": filters,
                "duration": duration
            },
            metadata={
                "correlation_id": correlation_id,
                "org_id": user_context["org_id"]
            }
        )
        
    except Exception as e:
        logger.error(f"Marketing data search failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Marketing data search failed: {str(e)}"
        )


@app.post("/marketing/insights", response_model=APIResponse)
async def get_marketing_insights(
    request: MarketingDataAggregationRequest,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Generate marketing insights and analytics"""
    correlation_id = str(uuid.uuid4())
    start_time = datetime.utcnow()
    
    try:
        # Search for marketing data within date range
        search_filters = {
            "data_source": [source.value for source in request.data_sources],
            "date_range_start": {"$gte": str(request.date_range_start)},
            "date_range_end": {"$lte": str(request.date_range_end)}
        }
        
        if request.filters:
            search_filters.update(request.filters)
        
        # Get all marketing data for the period
        results = await kse_client.hybrid_search(
            query="marketing data analytics insights",
            org_id=user_context["org_id"],
            limit=1000,  # Large limit to get comprehensive data
            filters=search_filters,
            search_type="hybrid"
        )
        
        # Process results to generate insights
        total_spend = 0.0
        total_impressions = 0
        total_clicks = 0
        total_conversions = 0
        total_revenue = 0.0
        
        data_sources_summary = {}
        top_campaigns = []
        
        for result in results:
            metadata = result.metadata
            raw_data = metadata.get("raw_data", {})
            
            # Extract metrics (this would be enhanced with proper data parsing)
            spend = float(raw_data.get("spend", 0))
            impressions = int(raw_data.get("impressions", 0))
            clicks = int(raw_data.get("clicks", 0))
            conversions = int(raw_data.get("conversions", 0))
            revenue = float(raw_data.get("revenue", 0))
            
            total_spend += spend
            total_impressions += impressions
            total_clicks += clicks
            total_conversions += conversions
            total_revenue += revenue
            
            # Track by data source
            source = metadata.get("data_source", "unknown")
            if source not in data_sources_summary:
                data_sources_summary[source] = {
                    "spend": 0.0,
                    "impressions": 0,
                    "clicks": 0,
                    "conversions": 0,
                    "revenue": 0.0
                }
            
            data_sources_summary[source]["spend"] += spend
            data_sources_summary[source]["impressions"] += impressions
            data_sources_summary[source]["clicks"] += clicks
            data_sources_summary[source]["conversions"] += conversions
            data_sources_summary[source]["revenue"] += revenue
        
        # Calculate derived metrics
        average_cpc = total_spend / total_clicks if total_clicks > 0 else 0.0
        average_cpm = (total_spend / total_impressions) * 1000 if total_impressions > 0 else 0.0
        average_ctr = (total_clicks / total_impressions) * 100 if total_impressions > 0 else 0.0
        conversion_rate = (total_conversions / total_clicks) * 100 if total_clicks > 0 else 0.0
        roas = total_revenue / total_spend if total_spend > 0 else 0.0
        
        insights = MarketingInsights(
            total_spend=total_spend,
            total_impressions=total_impressions,
            total_clicks=total_clicks,
            total_conversions=total_conversions,
            total_revenue=total_revenue,
            average_cpc=average_cpc,
            average_cpm=average_cpm,
            average_ctr=average_ctr,
            conversion_rate=conversion_rate,
            roas=roas,
            data_sources_summary=data_sources_summary,
            top_campaigns=top_campaigns,
            performance_trends={},
            calendar_insights={}
        )
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(
            f"Marketing insights generated for org {user_context['org_id']}",
            extra={
                "org_id": user_context["org_id"],
                "user_id": user_context["user_id"],
                "correlation_id": correlation_id,
                "data_sources": [source.value for source in request.data_sources],
                "duration": duration
            }
        )
        
        return APIResponse(
            message="Marketing insights generated successfully",
            data=insights.dict(),
            metadata={
                "correlation_id": correlation_id,
                "org_id": user_context["org_id"],
                "duration": duration
            }
        )
        
    except Exception as e:
        logger.error(f"Marketing insights generation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Marketing insights generation failed: {str(e)}"
        )


@app.get("/marketing/calendar/{year}", response_model=APIResponse)
async def get_calendar_dimensions(
    year: int,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Get calendar dimensions for causal modeling"""
    try:
        from datetime import date, timedelta
        import calendar
        
        calendar_data = []
        start_date = date(year, 1, 1)
        end_date = date(year, 12, 31)
        
        current_date = start_date
        while current_date <= end_date:
            # Calculate calendar dimensions
            day_of_year = current_date.timetuple().tm_yday
            week_number = current_date.isocalendar()[1]
            quarter = (current_date.month - 1) // 3 + 1
            
            # Determine season
            if current_date.month in [12, 1, 2]:
                season = "winter"
            elif current_date.month in [3, 4, 5]:
                season = "spring"
            elif current_date.month in [6, 7, 8]:
                season = "summer"
            else:
                season = "fall"
            
            calendar_dim = CalendarDimension(
                date=current_date,
                year=current_date.year,
                quarter=quarter,
                month=current_date.month,
                week=week_number,
                day_of_year=day_of_year,
                day_of_month=current_date.day,
                day_of_week=current_date.weekday(),
                is_weekend=current_date.weekday() >= 5,
                season=season
            )
            
            calendar_data.append(calendar_dim.dict())
            current_date += timedelta(days=1)
        
        return APIResponse(
            message=f"Calendar dimensions generated for {year}",
            data={
                "year": year,
                "calendar_dimensions": calendar_data,
                "total_days": len(calendar_data)
            }
        )
        
    except Exception as e:
        logger.error(f"Calendar dimensions generation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Calendar dimensions generation failed: {str(e)}"
        )


@app.post("/marketing/export/causal", response_model=APIResponse)
async def export_marketing_data_for_causal_analysis(
    request: MarketingDataAggregationRequest,
    target_metric: str = "conversions",
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Export marketing data in format optimized for causal analysis"""
    correlation_id = str(uuid.uuid4())
    start_time = datetime.utcnow()
    
    try:
        # Search for marketing data within date range
        search_filters = {
            "data_source": [source.value for source in request.data_sources],
            "date_range_start": {"$gte": str(request.date_range_start)},
            "date_range_end": {"$lte": str(request.date_range_end)}
        }
        
        if request.filters:
            search_filters.update(request.filters)
        
        # Get comprehensive marketing data
        results = await kse_client.hybrid_search(
            query="marketing data causal analysis export",
            org_id=user_context["org_id"],
            limit=10000,  # Large limit for comprehensive analysis
            filters=search_filters,
            search_type="hybrid"
        )
        
        # Convert results to DataFrame for processing
        data_records = []
        for result in results:
            metadata = result.metadata
            
            # Extract structured data from metadata
            record = {
                "date": metadata.get("date_range_start"),
                "data_source": metadata.get("data_source"),
                "campaign_id": metadata.get("campaign_id"),
                "spend": float(metadata.get("spend", 0)),
                "impressions": int(metadata.get("impressions", 0)),
                "clicks": int(metadata.get("clicks", 0)),
                "conversions": int(metadata.get("conversions", 0)),
                "revenue": float(metadata.get("revenue", 0)),
                "year": metadata.get("year"),
                "quarter": metadata.get("quarter"),
                "month": metadata.get("month"),
                "season": metadata.get("season")
            }
            data_records.append(record)
        
        if not data_records:
            return APIResponse(
                message="No marketing data found for the specified criteria",
                data={"causal_data": {}, "records_count": 0}
            )
        
        # Create DataFrame and process with transformer
        import pandas as pd
        df = pd.DataFrame(data_records)
        
        # Aggregate data if requested
        if request.group_by and request.metrics:
            df = marketing_transformer.aggregate_data(
                df=df,
                group_by=request.group_by,
                metrics=request.metrics
            )
        
        # Export for causal analysis
        causal_data = marketing_transformer.export_for_causal_analysis(
            df=df,
            target_metric=target_metric
        )
        
        # Add additional metadata for causal modeling
        causal_data["export_metadata"] = {
            "org_id": user_context["org_id"],
            "date_range": {
                "start": str(request.date_range_start),
                "end": str(request.date_range_end)
            },
            "data_sources": [source.value for source in request.data_sources],
            "target_metric": target_metric,
            "records_count": len(df),
            "export_timestamp": datetime.utcnow().isoformat(),
            "correlation_id": correlation_id
        }
        
        # Add statistical summary
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            causal_data["statistical_summary"] = {
                "descriptive_stats": df[numeric_cols].describe().to_dict(),
                "missing_values": df[numeric_cols].isnull().sum().to_dict(),
                "data_types": df.dtypes.astype(str).to_dict()
            }
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(
            f"Causal analysis export completed for org {user_context['org_id']}",
            extra={
                "org_id": user_context["org_id"],
                "user_id": user_context["user_id"],
                "correlation_id": correlation_id,
                "records_exported": len(df),
                "target_metric": target_metric,
                "duration": duration
            }
        )
        
        return APIResponse(
            message="Marketing data exported for causal analysis",
            data={
                "causal_data": causal_data,
                "records_count": len(df),
                "target_metric": target_metric,
                "duration": duration
            },
            metadata={
                "correlation_id": correlation_id,
                "org_id": user_context["org_id"]
            }
        )
        
    except Exception as e:
        logger.error(f"Causal analysis export failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Causal analysis export failed: {str(e)}"
        )


@app.post("/marketing/transform", response_model=APIResponse)
async def transform_marketing_data(
    request: MarketingDataIngestionRequest,
    transformation_config: Optional[PandasTransformationConfig] = None,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Transform marketing data using pandas without storing (for testing/preview)"""
    correlation_id = str(uuid.uuid4())
    start_time = datetime.utcnow()
    
    try:
        # Transform data using pandas
        df_transformed = marketing_transformer.transform_raw_data(
            raw_data=request.data_entries,
            data_source=request.data_source,
            date_range_start=request.date_range_start,
            date_range_end=request.date_range_end,
            org_id=user_context["org_id"]
        )
        
        # Convert to records for response
        transformed_records = df_transformed.to_dict('records')
        
        # Generate transformation summary
        transformation_summary = {
            "original_records": len(request.data_entries),
            "transformed_records": len(transformed_records),
            "columns_added": list(set(df_transformed.columns) - set(request.data_entries[0].keys() if request.data_entries else [])),
            "data_source": request.data_source.value,
            "transformation_applied": True
        }
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        return APIResponse(
            message="Marketing data transformation completed",
            data={
                "transformed_data": transformed_records,
                "transformation_summary": transformation_summary,
                "sample_record": transformed_records[0] if transformed_records else {},
                "duration": duration
            },
            metadata={
                "correlation_id": correlation_id,
                "org_id": user_context["org_id"]
            }
        )
        
    except Exception as e:
        logger.error(f"Marketing data transformation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Marketing data transformation failed: {str(e)}"
        )


# Causal Marketing Data Endpoints

@app.post("/api/v1/marketing/ingest/causal", response_model=APIResponse)
async def ingest_causal_marketing_data(
    causal_data: CausalMarketingData,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Ingest causal marketing data with KSE integration"""
    correlation_id = str(uuid.uuid4())
    start_time = datetime.utcnow()
    
    try:
        async with observability.observe_async_operation(
            "causal_data_ingestion",
            tags={
                "org_id": user_context["org_id"],
                "platform": causal_data.platform,
                "campaign_id": causal_data.campaign_id,
                "correlation_id": correlation_id
            },
            level=TraceLevel.INFO
        ) as span:
            
            # Convert causal data to enhanced content for KSE storage
            content_data = {
                "causal_marketing_data": causal_data.model_dump(),
                "platform": causal_data.platform,
                "campaign_id": causal_data.campaign_id,
                "timestamp": causal_data.timestamp.isoformat(),
                "treatment_assignment": causal_data.treatment_assignment.model_dump() if causal_data.treatment_assignment else None,
                "confounders": [c.model_dump() for c in causal_data.confounders],
                "external_factors": [f.model_dump() for f in causal_data.external_factors],
                "causal_graph": causal_data.causal_graph.model_dump() if causal_data.causal_graph else None,
                "quality_score": causal_data.quality_score
            }
            
            content = f"Causal marketing data from {causal_data.platform}: {content_data}"
            
            # Enhanced metadata for causal analysis
            causal_metadata = {
                "data_type": "causal_marketing",
                "platform": causal_data.platform,
                "campaign_id": causal_data.campaign_id,
                "timestamp": causal_data.timestamp.isoformat(),
                "quality_score": causal_data.quality_score,
                "has_treatment": causal_data.treatment_assignment is not None,
                "confounder_count": len(causal_data.confounders),
                "external_factor_count": len(causal_data.external_factors),
                "has_causal_graph": causal_data.causal_graph is not None,
                "org_id": user_context["org_id"],
                "correlation_id": correlation_id
            }
            
            # Store in KSE with causal-aware embeddings
            memory_id = await enhanced_kse.store_with_causal_context(
                org_id=user_context["org_id"],
                content=content,
                metadata=causal_metadata,
                causal_data=causal_data
            )
            
            if span:
                observability.tracer.add_span_tag(span, "memory_id", memory_id)
                observability.tracer.add_span_tag(span, "quality_score", causal_data.quality_score)
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        # Record accountability decision
        observability.accountability.record_decision(
            operation="causal_data_ingestion",
            decision_point="causal_storage_strategy",
            reasoning=f"Stored causal marketing data with quality score {causal_data.quality_score}",
            inputs={
                "platform": causal_data.platform,
                "campaign_id": causal_data.campaign_id,
                "quality_score": causal_data.quality_score
            },
            outputs={
                "memory_id": memory_id,
                "storage_duration_ms": duration * 1000
            },
            confidence=0.95,
            trace_id=span.trace_id if span else None,
            span_id=span.span_id if span else None
        )
        
        logger.info(f"Successfully ingested causal marketing data for org {user_context['org_id']}")
        
        return APIResponse(
            message="Causal marketing data ingested successfully",
            data={
                "memory_id": memory_id,
                "platform": causal_data.platform,
                "campaign_id": causal_data.campaign_id,
                "quality_score": causal_data.quality_score,
                "duration": duration
            },
            metadata={
                "correlation_id": correlation_id,
                "org_id": user_context["org_id"],
                "causal_features": ["treatment_assignment", "confounders", "external_factors", "causal_graph"]
            }
        )
        
    except Exception as e:
        logger.error(f"Causal marketing data ingestion failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Causal marketing data ingestion failed: {str(e)}"
        )


@app.get("/api/v1/marketing/causal/{experiment_id}", response_model=APIResponse)
async def get_causal_experiment_data(
    experiment_id: str,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Retrieve causal experiment data"""
    correlation_id = str(uuid.uuid4())
    
    try:
        # Search for causal experiment data
        search_filters = {
            "data_type": "causal_marketing",
            "experiment_id": experiment_id,
            "org_id": user_context["org_id"]
        }
        
        results = await kse_client.hybrid_search(
            query=f"causal experiment {experiment_id}",
            org_id=user_context["org_id"],
            limit=100,
            filters=search_filters,
            search_type="hybrid"
        )
        
        if not results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Causal experiment {experiment_id} not found"
            )
        
        # Process and structure the causal experiment data
        experiment_data = []
        for result in results:
            metadata = result.metadata
            experiment_data.append({
                "memory_id": result.id,
                "platform": metadata.get("platform"),
                "campaign_id": metadata.get("campaign_id"),
                "timestamp": metadata.get("timestamp"),
                "quality_score": metadata.get("quality_score"),
                "has_treatment": metadata.get("has_treatment"),
                "confounder_count": metadata.get("confounder_count"),
                "external_factor_count": metadata.get("external_factor_count")
            })
        
        logger.info(f"Retrieved causal experiment data for {experiment_id}")
        
        return APIResponse(
            message=f"Causal experiment data retrieved for {experiment_id}",
            data={
                "experiment_id": experiment_id,
                "data_points": experiment_data,
                "count": len(experiment_data)
            },
            metadata={
                "correlation_id": correlation_id,
                "org_id": user_context["org_id"]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve causal experiment data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve causal experiment data: {str(e)}"
        )


@app.get("/api/v1/marketing/confounders", response_model=APIResponse)
async def get_confounder_analysis(
    platform: str,
    date_range_start: str,
    date_range_end: str,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Get confounder analysis for platform"""
    correlation_id = str(uuid.uuid4())
    
    try:
        # Search for causal data with confounders
        search_filters = {
            "data_type": "causal_marketing",
            "platform": platform,
            "org_id": user_context["org_id"]
        }
        
        results = await kse_client.hybrid_search(
            query=f"confounders {platform} marketing data",
            org_id=user_context["org_id"],
            limit=1000,
            filters=search_filters,
            search_type="hybrid"
        )
        
        # Analyze confounders across the dataset
        confounder_analysis = {}
        total_data_points = len(results)
        
        for result in results:
            metadata = result.metadata
            confounder_count = metadata.get("confounder_count", 0)
            
            if confounder_count > 0:
                # Extract confounder information from content
                # This would be enhanced with actual confounder data parsing
                platform_key = metadata.get("platform", "unknown")
                if platform_key not in confounder_analysis:
                    confounder_analysis[platform_key] = {
                        "total_occurrences": 0,
                        "average_confounders_per_datapoint": 0,
                        "quality_scores": []
                    }
                
                confounder_analysis[platform_key]["total_occurrences"] += 1
                confounder_analysis[platform_key]["quality_scores"].append(
                    metadata.get("quality_score", 0)
                )
        
        # Calculate summary statistics
        for platform_key in confounder_analysis:
            data = confounder_analysis[platform_key]
            if data["quality_scores"]:
                data["average_quality_score"] = sum(data["quality_scores"]) / len(data["quality_scores"])
                data["confounder_detection_rate"] = data["total_occurrences"] / total_data_points
            else:
                data["average_quality_score"] = 0
                data["confounder_detection_rate"] = 0
        
        logger.info(f"Confounder analysis completed for {platform}")
        
        return APIResponse(
            message=f"Confounder analysis completed for {platform}",
            data={
                "platform": platform,
                "date_range": {
                    "start": date_range_start,
                    "end": date_range_end
                },
                "confounder_analysis": confounder_analysis,
                "total_data_points": total_data_points
            },
            metadata={
                "correlation_id": correlation_id,
                "org_id": user_context["org_id"]
            }
        )
        
    except Exception as e:
        logger.error(f"Confounder analysis failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Confounder analysis failed: {str(e)}"
        )


@app.post("/api/v1/marketing/causal/search", response_model=APIResponse)
async def search_causal_marketing_data(
    request: CausalAnalysisRequest,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Search causal marketing data with advanced filtering"""
    correlation_id = str(uuid.uuid4())
    start_time = datetime.utcnow()
    
    try:
        async with observability.observe_async_operation(
            "causal_data_search",
            tags={
                "org_id": user_context["org_id"],
                "platform": request.platform,
                "analysis_type": request.analysis_type,
                "correlation_id": correlation_id
            },
            level=TraceLevel.INFO
        ) as span:
            
            # Build search filters for causal data
            search_filters = {
                "data_type": "causal_marketing",
                "platform": request.platform,
                "org_id": user_context["org_id"]
            }
            
            if request.date_range_start:
                search_filters["date_range_start"] = str(request.date_range_start)
            if request.date_range_end:
                search_filters["date_range_end"] = str(request.date_range_end)
            
            # Perform causal-aware search
            results = await enhanced_kse.search_with_causal_context(
                org_id=user_context["org_id"],
                query=f"causal {request.analysis_type} {request.platform}",
                filters=search_filters,
                limit=request.limit or 100
            )
            
            if span:
                observability.tracer.add_span_tag(span, "results_count", len(results))
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(f"Causal marketing data search completed for org {user_context['org_id']}")
        
        return APIResponse(
            message="Causal marketing data search completed",
            data={
                "results": results,
                "count": len(results),
                "platform": request.platform,
                "analysis_type": request.analysis_type,
                "duration": duration
            },
            metadata={
                "correlation_id": correlation_id,
                "org_id": user_context["org_id"],
                "causal_search": True
            }
        )
        
    except Exception as e:
        logger.error(f"Causal marketing data search failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Causal marketing data search failed: {str(e)}"
        )


@app.on_event("startup")
async def startup_event():
    """Initialize memory service with MMM Spine integration on startup"""
    app.state.start_time = time.time()
    
    # Initialize KSE client
    try:
        await kse_client.initialize()
        logger.info("KSE client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize KSE client: {str(e)}")
        # Continue startup even if KSE fails (graceful degradation)
    
    # Initialize MMM Spine components
    try:
        # Register workflow templates for common operations
        search_template = {
            "name": "enhanced_search_workflow",
            "tasks": [
                {
                    "name": "search_memory",
                    "function": "enhanced_search",
                    "args": ["{org_id}", "{query}"],
                    "kwargs": {
                        "search_type": "{search_type}",
                        "limit": "{limit}"
                    },
                    "priority": 2,
                    "timeout_seconds": 30.0
                }
            ]
        }
        
        batch_processing_template = {
            "name": "batch_processing_workflow",
            "tasks": [
                {
                    "name": "process_batch_{batch_id}",
                    "function": "enhanced_search",
                    "args": ["{org_id}", "{query}"],
                    "kwargs": {
                        "search_type": "hybrid",
                        "limit": 10
                    },
                    "priority": 3,
                    "timeout_seconds": 60.0
                }
            ]
        }
        
        workflow_orchestrator.register_workflow_template("enhanced_search", search_template)
        workflow_orchestrator.register_workflow_template("batch_processing", batch_processing_template)
        
        logger.info("MMM Spine workflow templates registered successfully")
        
        # Log initialization metrics
        logger.info(
            "Memory service started with enhanced MMM Spine integration",
            extra={
                "embedding_cache_size": embedding_manager.cache_size,
                "orchestrator_workers": orchestrator.max_workers,
                "observability_enabled": True,
                "conceptual_spaces_enabled": True,
                "workflow_templates": ["enhanced_search", "batch_processing"]
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to initialize MMM Spine components: {str(e)}")
        # Continue startup even if MMM Spine initialization fails


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Memory service stopped")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_level=config.LOG_LEVEL.lower()
    )