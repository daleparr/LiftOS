"""
LiftOS LLM Module - Advanced LLM Evaluation and Content Generation Platform
Provides comprehensive LLM capabilities including model evaluation, prompt engineering,
content generation, and multi-provider integration.
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from contextlib import asynccontextmanager

import httpx
import openai
import cohere
from transformers import pipeline, AutoTokenizer
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import structlog
from prometheus_client import Counter, Histogram, generate_latest
import nltk
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from langdetect import detect
import redis
from jinja2 import Template

# Import causal models and utilities
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import KSE SDK for universal intelligence substrate
from shared.kse_sdk.client import LiftKSEClient
from shared.kse_sdk.core import Entity, SearchQuery, SearchResult, ConceptualSpace, KSEConfig
from shared.kse_sdk.core.models import SearchType

# Import Phase 2 Advanced Intelligence Flow
from shared.kse_sdk.intelligence.orchestrator import (
    IntelligenceOrchestrator,
    IntelligenceEvent,
    IntelligenceEventType,
    IntelligencePriority
)
from shared.kse_sdk.intelligence.flow_manager import AdvancedIntelligenceFlowManager

from shared.models.causal_marketing import (
    CausalMarketingData, CausalExperiment, CausalInsight,
    AttributionModel, ConfounderVariable, ExternalFactor,
    CausalInsightRequest, CausalInsightResponse,
    CounterfactualAnalysisRequest, CounterfactualAnalysisResponse,
    CausalExplanationRequest, CausalExplanationResponse
)
from shared.utils.causal_transforms import CausalDataTransformer

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Metrics
llm_requests_total = Counter('llm_requests_total', 'Total LLM requests', ['endpoint', 'provider', 'model'])
llm_request_duration = Histogram('llm_request_duration_seconds', 'LLM request duration', ['endpoint', 'provider'])
llm_tokens_used = Counter('llm_tokens_used_total', 'Total tokens used', ['provider', 'model', 'type'])

# Configuration
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://llm-service:3004")
MEMORY_SERVICE_URL = os.getenv("MEMORY_SERVICE_URL", "http://memory:8003")
AUTH_SERVICE_URL = os.getenv("AUTH_SERVICE_URL", "http://auth:8001")
REGISTRY_SERVICE_URL = os.getenv("REGISTRY_SERVICE_URL", "http://registry:8005")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# API Keys (should be in environment variables)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Initialize KSE client for universal intelligence substrate
kse_client = LiftKSEClient()

# Initialize KSE integration for LLM intelligence
llm_kse = None
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Global clients
redis_client = None
openai_client = None
cohere_client = None
sentence_model = None
causal_transformer = None


# KSE Integration for LLM Intelligence
class LLMKSEIntegration:
    """KSE integration for LLM capabilities and intelligence with Phase 2 Advanced Intelligence Flow"""
    
    def __init__(self, kse_client: LiftKSEClient):
        self.kse_client = kse_client
        self.logger = logger
        
        # Phase 2: Advanced Intelligence Flow Components
        self.intelligence_orchestrator = None
        self.flow_manager = None
    
    async def initialize(self):
        """Initialize KSE client connection and Phase 2 advanced intelligence flow"""
        try:
            await self.kse_client.initialize()
            
            # Initialize Phase 2 Advanced Intelligence Flow
            self.intelligence_orchestrator = IntelligenceOrchestrator(self.kse_client)
            self.flow_manager = AdvancedIntelligenceFlowManager(self.kse_client)
            
            await self.intelligence_orchestrator.initialize()
            await self.flow_manager.initialize()
            
            # Register for cross-service intelligence events
            await self._setup_intelligence_flows()
            
            self.logger.info("LLM KSE integration with Phase 2 Advanced Intelligence Flow initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM KSE integration: {str(e)}")
            raise
    
    async def _setup_intelligence_flows(self):
        """Setup Phase 2 cross-service intelligence flows for LLM capabilities"""
        try:
            # Subscribe to intelligence events from other services
            await self.intelligence_orchestrator.subscribe_to_event(
                IntelligenceEventType.PATTERN_DISCOVERY,
                self._handle_pattern_discovery
            )
            
            await self.intelligence_orchestrator.subscribe_to_event(
                IntelligenceEventType.INSIGHT_GENERATION,
                self._handle_insight_generation
            )
            
            # Register service capabilities for intelligent routing
            await self.flow_manager.register_service_capabilities(
                service_name="llm",
                capabilities={
                    "content_generation": 0.95,
                    "text_analysis": 0.9,
                    "language_understanding": 0.85,
                    "prompt_optimization": 0.8,
                    "semantic_analysis": 0.9
                },
                input_types=["text_content", "generation_request", "analysis_request"],
                output_types=["generated_content", "text_insights", "semantic_analysis"]
            )
            
            self.logger.info("Phase 2 intelligence flows setup completed for LLM service")
            
        except Exception as e:
            self.logger.error(f"Failed to setup intelligence flows: {str(e)}")
    
    async def _handle_pattern_discovery(self, event: IntelligenceEvent):
        """Handle pattern discovery events from other services"""
        try:
            if event.data.get("service") != "llm":
                # Process patterns from other services for content optimization
                pattern_type = event.data.get("pattern_type")
                if pattern_type in ["optimization", "causal", "behavioral"]:
                    # Use patterns to enhance content generation
                    await self._enhance_content_generation(event.data)
                    
        except Exception as e:
            self.logger.error(f"Failed to handle pattern discovery event: {str(e)}")
    
    async def _handle_insight_generation(self, event: IntelligenceEvent):
        """Handle insight generation events from other services"""
        try:
            insight_type = event.data.get("insight_type")
            if insight_type in ["causal_insights", "treatment_recommendations", "optimization_patterns"]:
                # Use insights to improve content relevance and effectiveness
                await self._improve_content_relevance(event.data)
                
        except Exception as e:
            self.logger.error(f"Failed to handle insight generation event: {str(e)}")
    
    async def _enhance_content_generation(self, pattern_data: Dict[str, Any]):
        """Enhance content generation with patterns from other services"""
        try:
            # Create content enhancement strategy based on external patterns
            enhancement_strategy = {
                "pattern_source": pattern_data.get("service"),
                "pattern_type": pattern_data.get("pattern_type"),
                "enhancement_focus": "content_optimization",
                "confidence": pattern_data.get("confidence", 0.0),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Store enhancement strategy for future content generation
            entity = Entity(
                id=f"content_enhancement_{uuid.uuid4()}",
                type="content_enhancement_strategy",
                content=enhancement_strategy,
                metadata={
                    "source_service": pattern_data.get("service"),
                    "pattern_type": pattern_data.get("pattern_type"),
                    "entity_type": "content_enhancement_strategy"
                }
            )
            
            await self.kse_client.store_entity("global", entity)
            self.logger.info(f"Enhanced content generation strategy from {pattern_data.get('service')} patterns")
            
        except Exception as e:
            self.logger.error(f"Failed to enhance content generation: {str(e)}")
    
    async def _improve_content_relevance(self, insight_data: Dict[str, Any]):
        """Improve content relevance using insights from other services"""
        try:
            # Generate content relevance improvement
            relevance_improvement = {
                "insight_source": insight_data.get("service"),
                "insight_type": insight_data.get("insight_type"),
                "improvement_focus": "content_relevance",
                "confidence": insight_data.get("confidence", 0.0),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Publish improved content strategy
            improvement_event = IntelligenceEvent(
                event_type=IntelligenceEventType.INSIGHT_GENERATION,
                source_service="llm",
                target_service="all",
                priority=IntelligencePriority.MEDIUM,
                data={
                    "insight_type": "content_relevance_improvement",
                    "improvement_strategy": relevance_improvement,
                    "original_insight": insight_data.get("insight_type"),
                    "confidence": min(insight_data.get("confidence", 0.0) * 0.9, 1.0),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            await self.intelligence_orchestrator.publish_event(improvement_event)
            self.logger.info(f"Improved content relevance with {insight_data.get('insight_type')} insight")
            
        except Exception as e:
            self.logger.error(f"Failed to improve content relevance: {str(e)}")
    
    async def retrieve_content_context(self, content_request: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant content patterns and templates from KSE"""
        try:
            org_id = content_request.get('org_id', 'default')
            content_type = content_request.get('content_type', 'general')
            
            # Search for content patterns and high-performing templates
            search_results = await self.kse_client.hybrid_search(
                org_id=org_id,
                query=f"content patterns {content_type}",
                search_type="conceptual",
                limit=8,
                filters={
                    "entity_type": "content_template",
                    "content_type": content_type,
                    "performance_score": {"$gte": 0.8}
                }
            )
            
            # Process content context
            content_context = {
                "successful_templates": [],
                "performance_patterns": [],
                "style_guidelines": {},
                "optimization_insights": []
            }
            
            for result in search_results:
                if result.metadata.get("entity_type") == "content_template":
                    content_context["successful_templates"].append({
                        "template_id": result.id,
                        "template_content": result.content.get("template"),
                        "performance_score": result.metadata.get("performance_score"),
                        "engagement_metrics": result.content.get("engagement_metrics"),
                        "confidence_score": result.score
                    })
            
            self.logger.info(f"Retrieved {len(search_results)} content patterns for {content_type}")
            return content_context
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve content context: {str(e)}")
            return {"error": str(e)}
    
    async def enrich_generated_content(self, content: Dict[str, Any], performance_metrics: Dict[str, Any], org_id: str) -> None:
        """Enrich KSE with generated content and performance data"""
        try:
            entity = Entity(
                id=f"generated_content_{content.get('id', uuid.uuid4())}",
                type="generated_content",
                content=content,
                metadata={
                    "org_id": org_id,
                    "content_type": content.get('content_type', 'unknown'),
                    "generation_model": content.get('model', 'unknown'),
                    "performance_metrics": performance_metrics,
                    "quality_score": performance_metrics.get('quality_score', 0.0),
                    "engagement_score": performance_metrics.get('engagement_score', 0.0),
                    "causal_insights": content.get('causal_insights', []),
                    "generation_timestamp": datetime.utcnow().isoformat(),
                    "entity_type": "generated_content"
                }
            )
            
            await self.kse_client.store_entity(org_id, entity)
            self.logger.info(f"Enriched KSE with generated content for org {org_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to enrich generated content: {str(e)}")
    
    async def retrieve_causal_context(self, org_id: str, content_type: str) -> Dict[str, Any]:
        """Retrieve causal insights for content generation"""
        try:
            search_results = await self.kse_client.hybrid_search(
                org_id=org_id,
                query=f"causal insights {content_type}",
                search_type="hybrid",
                limit=5,
                filters={
                    "entity_type": "causal_insight",
                    "content_type": content_type
                }
            )
            
            causal_context = {
                "causal_relationships": [],
                "performance_drivers": [],
                "optimization_opportunities": []
            }
            
            for result in search_results:
                content = result.content
                if content.get("causal_relationship"):
                    causal_context["causal_relationships"].append(content["causal_relationship"])
                if content.get("performance_drivers"):
                    causal_context["performance_drivers"].extend(content["performance_drivers"])
            
            return causal_context
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve causal context: {str(e)}")
            return {}
    
    async def enrich_llm_insights(self, insights: List[Dict], org_id: str) -> None:
        """Enrich KSE with LLM-generated insights and analysis"""
        try:
            for insight in insights:
                entity = Entity(
                    id=f"llm_insight_{insight.get('id', uuid.uuid4())}",
                    type="llm_insight",
                    content=insight,
                    metadata={
                        "org_id": org_id,
                        "insight_type": insight.get('insight_type', 'general'),
                        "confidence_score": insight.get('confidence_score', 0.0),
                        "model_used": insight.get('model_used', 'unknown'),
                        "insight_timestamp": datetime.utcnow().isoformat(),
                        "entity_type": "llm_insight"
                    }
                )
                
                await self.kse_client.store_entity(org_id, entity)
            
            self.logger.info(f"Enriched KSE with {len(insights)} LLM insights")
            
        except Exception as e:
            self.logger.error(f"Failed to enrich LLM insights: {str(e)}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global redis_client, openai_client, cohere_client, sentence_model, causal_transformer, llm_kse
    
    logger.info("Starting LLM module...")
    
    # Initialize KSE integration
    try:
        llm_kse = LLMKSEIntegration(kse_client)
        await llm_kse.initialize()
        logger.info("LLM KSE integration initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize LLM KSE integration: {str(e)}")
        # Continue startup even if KSE integration fails
    
    # Initialize Redis
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        await asyncio.to_thread(redis_client.ping)
        logger.info("Redis connection established")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        redis_client = None
    
    # Initialize LLM clients
    if OPENAI_API_KEY:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI client initialized")
    
    if COHERE_API_KEY:
        cohere_client = cohere.Client(COHERE_API_KEY)
        logger.info("Cohere client initialized")
    
    # Initialize sentence transformer for embeddings
    try:
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Sentence transformer model loaded")
    except Exception as e:
        logger.warning(f"Failed to load sentence transformer: {e}")
    
    # Initialize causal transformer
    try:
        causal_transformer = CausalDataTransformer()
        logger.info("Causal data transformer initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize causal transformer: {e}")
        causal_transformer = None
    
    # Initialize causal reasoning components
    global causal_insight_generator, counterfactual_analyzer, causal_explanation_engine
    try:
        causal_insight_generator = CausalInsightGenerator(openai_client)
        counterfactual_analyzer = CounterfactualAnalyzer(openai_client)
        causal_explanation_engine = CausalExplanationEngine(openai_client)
        logger.info("Causal reasoning components initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize causal reasoning components: {e}")
    
    logger.info("LLM module startup complete")
    
    yield
    
    logger.info("Shutting down LLM module...")
    if redis_client:
        redis_client.close()

# FastAPI app
app = FastAPI(
    title="LiftOS LLM Module",
    description="Advanced LLM evaluation and content generation platform",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ModelEvaluationRequest(BaseModel):
    models: List[str] = Field(..., description="List of models to evaluate")
    prompts: List[str] = Field(..., description="Test prompts")
    reference_outputs: Optional[List[str]] = Field(None, description="Reference outputs for comparison")
    metrics: List[str] = Field(default=["bleu", "rouge", "bert_score"], description="Evaluation metrics")
    language: str = Field(default="en", description="Language code")

class ContentGenerationRequest(BaseModel):
    template: str = Field(..., description="Template name or custom template")
    variables: Dict[str, Any] = Field(default={}, description="Template variables")
    provider: str = Field(default="openai", description="LLM provider")
    model: str = Field(default="gpt-3.5-turbo", description="Model name")
    max_tokens: int = Field(default=1000, description="Maximum tokens")
    temperature: float = Field(default=0.7, description="Generation temperature")
    context: Optional[str] = Field(None, description="Additional context")

class PromptOptimizationRequest(BaseModel):
    base_prompt: str = Field(..., description="Base prompt to optimize")
    target_metrics: Dict[str, float] = Field(..., description="Target metric scores")
    test_cases: List[Dict[str, str]] = Field(..., description="Test cases for optimization")
    max_iterations: int = Field(default=10, description="Maximum optimization iterations")

class MetricsCalculationRequest(BaseModel):
    predictions: List[str] = Field(..., description="Model predictions")
    references: List[str] = Field(..., description="Reference texts")
    metrics: List[str] = Field(default=["bleu", "rouge", "bert_score"], description="Metrics to calculate")
    language: str = Field(default="en", description="Language code")

class RLHFScoringRequest(BaseModel):
    responses: List[str] = Field(..., description="Model responses to score")
    criteria: List[str] = Field(..., description="Evaluation criteria")
    human_preferences: Optional[List[Dict[str, Any]]] = Field(None, description="Human preference data")

class MultilingualTestRequest(BaseModel):
    text: str = Field(..., description="Text to test")
    target_languages: List[str] = Field(..., description="Target languages")
    test_type: str = Field(default="translation", description="Type of multilingual test")

# Utility functions
async def get_user_context(request: Request) -> Dict[str, Any]:
    """Extract user context from request headers"""
    return {
        "user_id": request.headers.get("X-User-ID"),
        "organization_id": request.headers.get("X-Organization-ID"),
        "request_id": request.headers.get("X-Request-ID"),
    }

async def call_llm_service(endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Call the external LLM service"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{LLM_SERVICE_URL}{endpoint}",
                json=data,
                timeout=60.0
            )
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            logger.error(f"LLM service request failed: {e}")
            raise HTTPException(status_code=503, detail="LLM service unavailable")
        except httpx.HTTPStatusError as e:
            logger.error(f"LLM service error: {e.response.status_code}")
            raise HTTPException(status_code=e.response.status_code, detail="LLM service error")

async def store_in_memory(data: Dict[str, Any], memory_type: str, user_context: Dict[str, Any]) -> bool:
    """Store results in memory service"""
    try:
        async with httpx.AsyncClient() as client:
            memory_data = {
                "type": memory_type,
                "data": data,
                "metadata": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "user_id": user_context.get("user_id"),
                    "organization_id": user_context.get("organization_id"),
                    "module": "llm"
                }
            }
            
            response = await client.post(
                f"{MEMORY_SERVICE_URL}/store",
                json=memory_data,
                timeout=30.0
            )
            return response.status_code in [200, 201]
    except Exception as e:
        logger.error(f"Failed to store in memory: {e}")
        return False

def calculate_bleu_score(predictions: List[str], references: List[str]) -> float:
    """Calculate BLEU score"""
    try:
        from nltk.translate.bleu_score import corpus_bleu
        references_tokenized = [[ref.split()] for ref in references]
        predictions_tokenized = [pred.split() for pred in predictions]
        return corpus_bleu(references_tokenized, predictions_tokenized)
    except Exception as e:
        logger.error(f"BLEU calculation failed: {e}")
        return 0.0

def calculate_rouge_scores(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate ROUGE scores"""
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for pred, ref in zip(predictions, references):
            score = scorer.score(ref, pred)
            for metric in scores:
                scores[metric].append(score[metric].fmeasure)
        
        return {metric: np.mean(values) for metric, values in scores.items()}
    except Exception as e:
        logger.error(f"ROUGE calculation failed: {e}")
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

def calculate_bert_scores(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate BERTScore"""
    try:
        P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
        return {
            'bert_precision': P.mean().item(),
            'bert_recall': R.mean().item(),
            'bert_f1': F1.mean().item()
        }
    except Exception as e:
        logger.error(f"BERTScore calculation failed: {e}")
        return {'bert_precision': 0.0, 'bert_recall': 0.0, 'bert_f1': 0.0}

async def get_cached_result(cache_key: str) -> Optional[Dict[str, Any]]:
    """Get cached result from Redis"""
    if not redis_client:
        return None
    
    try:
        cached = await asyncio.to_thread(redis_client.get, cache_key)
        if cached:
            return json.loads(cached)
    except Exception as e:
        logger.error(f"Cache retrieval failed: {e}")
    
    return None

async def cache_result(cache_key: str, result: Dict[str, Any], ttl: int = 3600):
    """Cache result in Redis"""
    if not redis_client:
        return
    
# Causal Reasoning Classes

class CausalInsightGenerator:
    """Generate natural language insights from causal analysis results"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client or openai_client
    
    async def generate_causal_narrative(self, causal_data: CausalMarketingData, analysis_results: Dict[str, Any]) -> str:
        """Generate a natural language narrative explaining causal findings"""
        try:
            # Extract key insights from causal data
            treatment_effects = analysis_results.get("treatment_effects", {})
            confounders = analysis_results.get("confounders", [])
            external_factors = analysis_results.get("external_factors", [])
            
            # Build context for LLM
            context = f"""
            Causal Analysis Results:
            - Campaign: {causal_data.campaign_name}
            - Platform: {causal_data.platform}
            - Date Range: {causal_data.date_start} to {causal_data.date_end}
            - Treatment Effects: {treatment_effects}
            - Confounding Variables: {[c.get('variable_name') for c in confounders]}
            - External Factors: {[f.get('factor_name') for f in external_factors]}
            """
            
            prompt = f"""
            Based on the following causal analysis results, generate a clear, business-focused narrative that explains:
            1. What marketing interventions caused what outcomes
            2. Which confounding factors influenced the results
            3. How external factors may have impacted performance
            4. Actionable recommendations based on causal findings
            
            {context}
            
            Write this as a professional marketing analysis report that non-technical stakeholders can understand.
            Focus on causation, not correlation. Be specific about confidence levels and limitations.
            """
            
            if self.llm_client:
                response = await asyncio.to_thread(
                    self.llm_client.chat.completions.create,
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.3
                )
                return response.choices[0].message.content
            else:
                return "Causal analysis completed. LLM client not available for narrative generation."
                
        except Exception as e:
            logger.error(f"Failed to generate causal narrative: {e}")
            return f"Error generating narrative: {str(e)}"
    
    async def explain_treatment_effect(self, treatment: str, effect_size: float, confidence: float, method: str) -> str:
        """Explain a specific treatment effect in natural language"""
        try:
            prompt = f"""
            Explain the following causal finding in simple business terms:
            
            Treatment: {treatment}
            Effect Size: {effect_size}
            Confidence Level: {confidence}%
            Analysis Method: {method}
            
            Provide a clear explanation that includes:
            1. What the treatment was
            2. What impact it had (positive/negative/neutral)
            3. How confident we are in this finding
            4. What this means for future marketing decisions
            
            Keep it concise and actionable.
            """
            
            if self.llm_client:
                response = await asyncio.to_thread(
                    self.llm_client.chat.completions.create,
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.2
                )
                return response.choices[0].message.content
            else:
                return f"Treatment '{treatment}' showed an effect size of {effect_size} with {confidence}% confidence using {method} analysis."
                
        except Exception as e:
            logger.error(f"Failed to explain treatment effect: {e}")
            return f"Error explaining treatment effect: {str(e)}"

class CounterfactualAnalyzer:
    """Analyze counterfactual scenarios for marketing decisions"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client or openai_client
    
    async def analyze_counterfactual(self, scenario: str, causal_data: CausalMarketingData) -> Dict[str, Any]:
        """Analyze what would have happened under different conditions"""
        try:
            # Build counterfactual context
            context = f"""
            Original Scenario:
            - Campaign: {causal_data.campaign_name}
            - Platform: {causal_data.platform}
            - Spend: ${causal_data.spend}
            - Impressions: {causal_data.impressions}
            - Clicks: {causal_data.clicks}
            - Conversions: {causal_data.conversions}
            
            Counterfactual Scenario: {scenario}
            """
            
            prompt = f"""
            Based on the causal analysis data, analyze this counterfactual scenario:
            
            {context}
            
            Provide analysis on:
            1. Likely impact on key metrics (impressions, clicks, conversions, ROI)
            2. Confidence level in predictions
            3. Key assumptions and limitations
            4. Risk factors to consider
            5. Recommended next steps
            
            Be specific about causal mechanisms and avoid correlation-based reasoning.
            """
            
            if self.llm_client:
                response = await asyncio.to_thread(
                    self.llm_client.chat.completions.create,
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=800,
                    temperature=0.4
                )
                
                analysis_text = response.choices[0].message.content
                
                # Extract structured insights
                return {
                    "scenario": scenario,
                    "analysis": analysis_text,
                    "confidence_level": "medium",  # Could be enhanced with more sophisticated extraction
                    "key_assumptions": ["Causal relationships remain stable", "External factors unchanged"],
                    "risk_factors": ["Market conditions", "Competitive response"],
                    "recommendations": ["Test with small budget", "Monitor key metrics closely"]
                }
            else:
                return {
                    "scenario": scenario,
                    "analysis": "Counterfactual analysis requires LLM client for detailed insights.",
                    "confidence_level": "unknown",
                    "key_assumptions": [],
                    "risk_factors": [],
                    "recommendations": []
                }
                
        except Exception as e:
            logger.error(f"Failed to analyze counterfactual: {e}")
            return {
                "scenario": scenario,
                "analysis": f"Error in counterfactual analysis: {str(e)}",
                "confidence_level": "unknown",
                "key_assumptions": [],
                "risk_factors": [],
                "recommendations": []
            }

class CausalExplanationEngine:
    """Provide detailed explanations of causal analysis methods and results"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client or openai_client
    
    async def explain_causal_method(self, method: str, results: Dict[str, Any]) -> str:
        """Explain how a causal analysis method works and what the results mean"""
        try:
            method_descriptions = {
                "difference_in_differences": "Difference-in-Differences (DiD) compares changes over time between treatment and control groups to isolate causal effects.",
                "instrumental_variables": "Instrumental Variables (IV) uses external instruments to identify causal effects when randomization isn't possible.",
                "synthetic_control": "Synthetic Control creates a synthetic comparison unit from weighted combinations of control units.",
                "regression_discontinuity": "Regression Discontinuity exploits arbitrary cutoffs in treatment assignment to identify causal effects.",
                "propensity_score_matching": "Propensity Score Matching pairs similar units that received different treatments to estimate causal effects."
            }
            
            method_desc = method_descriptions.get(method, f"Analysis using {method} method")
            
            prompt = f"""
            Explain the following causal analysis method and results in business-friendly terms:
            
            Method: {method}
            Description: {method_desc}
            Results: {results}
            
            Provide an explanation that covers:
            1. How this method works (in simple terms)
            2. Why it's appropriate for this analysis
            3. What the results tell us
            4. Limitations and caveats
            5. How to interpret the findings for business decisions
            
            Avoid technical jargon and focus on practical implications.
            """
            
            if self.llm_client:
                response = await asyncio.to_thread(
                    self.llm_client.chat.completions.create,
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=600,
                    temperature=0.3
                )
                return response.choices[0].message.content
            else:
                return f"Method: {method}\nDescription: {method_desc}\nResults require LLM client for detailed explanation."
                
        except Exception as e:
            logger.error(f"Failed to explain causal method: {e}")
            return f"Error explaining method: {str(e)}"

# Initialize causal reasoning components
causal_insight_generator = None
counterfactual_analyzer = None
causal_explanation_engine = None

async def get_causal_service_data(endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get data from the causal service"""
    try:
        causal_service_url = os.getenv("CAUSAL_SERVICE_URL", "http://causal:8007")
        async with httpx.AsyncClient() as client:
            if params:
                response = await client.post(f"{causal_service_url}{endpoint}", json=params, timeout=30.0)
            else:
                response = await client.get(f"{causal_service_url}{endpoint}", timeout=30.0)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Failed to get causal service data: {e}")
        return {}

    try:
        await asyncio.to_thread(
            redis_client.setex, 
            cache_key, 
            ttl, 
            json.dumps(result, default=str)
        )
    except Exception as e:
        logger.error(f"Cache storage failed: {e}")

# API Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "module": "llm",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "llm_service": LLM_SERVICE_URL,
            "memory_service": MEMORY_SERVICE_URL,
            "redis": "connected" if redis_client else "disconnected",
            "openai": "configured" if OPENAI_API_KEY else "not_configured",
            "cohere": "configured" if COHERE_API_KEY else "not_configured"
        }
    }

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.post("/api/v1/models/evaluate")
async def evaluate_models(
    request: ModelEvaluationRequest,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Evaluate multiple models on given prompts"""
    llm_requests_total.labels(endpoint="evaluate", provider="multiple", model="multiple").inc()
    
    try:
        # Generate cache key
        cache_key = f"eval:{hash(str(request.dict()))}"
        cached_result = await get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        results = {}
        
        for model in request.models:
            model_results = []
            
            for prompt in request.prompts:
                # Call LLM service for generation
                generation_data = {
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": 1000,
                    "temperature": 0.7
                }
                
                try:
                    response = await call_llm_service("/generate", generation_data)
                    generated_text = response.get("text", "")
                    
                    model_results.append({
                        "prompt": prompt,
                        "generated_text": generated_text,
                        "tokens_used": response.get("tokens_used", 0)
                    })
                    
                    llm_tokens_used.labels(provider="llm_service", model=model, type="generation").inc(
                        response.get("tokens_used", 0)
                    )
                    
                except Exception as e:
                    logger.error(f"Generation failed for model {model}: {e}")
                    model_results.append({
                        "prompt": prompt,
                        "generated_text": "",
                        "error": str(e)
                    })
            
            results[model] = model_results
        
        # Calculate metrics if reference outputs provided
        if request.reference_outputs:
            for model in request.models:
                predictions = [r.get("generated_text", "") for r in results[model]]
                
                metrics = {}
                if "bleu" in request.metrics:
                    metrics["bleu"] = calculate_bleu_score(predictions, request.reference_outputs)
                
                if "rouge" in request.metrics:
                    metrics.update(calculate_rouge_scores(predictions, request.reference_outputs))
                
                if "bert_score" in request.metrics:
                    metrics.update(calculate_bert_scores(predictions, request.reference_outputs))
                
                results[model + "_metrics"] = metrics
        
        # Store results in memory
        await store_in_memory(results, "llm_evaluation", user_context)
        
        # Cache results
        await cache_result(cache_key, results)
        
        return {
            "evaluation_results": results,
            "metadata": {
                "models_evaluated": len(request.models),
                "prompts_tested": len(request.prompts),
                "metrics_calculated": request.metrics,
                "language": request.language,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.get("/api/v1/models/leaderboard")
async def get_model_leaderboard(
    metric: str = "overall",
    language: str = "en",
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Get model performance leaderboard"""
    try:
        # Call LLM service for leaderboard data
        leaderboard_data = await call_llm_service("/leaderboard", {
            "metric": metric,
            "language": language
        })
        
        return {
            "leaderboard": leaderboard_data.get("rankings", []),
            "metadata": {
                "metric": metric,
                "language": language,
                "last_updated": leaderboard_data.get("last_updated"),
                "total_models": len(leaderboard_data.get("rankings", []))
            }
        }
        
    except Exception as e:
        logger.error(f"Leaderboard retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Leaderboard failed: {str(e)}")

@app.post("/api/v1/models/compare")
async def compare_models(
    models: List[str],
    prompt: str,
    metrics: List[str] = ["quality", "speed", "cost"],
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Compare multiple models on a single prompt"""
    llm_requests_total.labels(endpoint="compare", provider="multiple", model="multiple").inc()
    
    try:
        comparison_data = {
            "models": models,
            "prompt": prompt,
            "metrics": metrics
        }
        
        results = await call_llm_service("/compare", comparison_data)
        
        # Store comparison results
        await store_in_memory(results, "llm_comparison", user_context)
        
        return {
            "comparison_results": results,
            "metadata": {
                "models_compared": len(models),
                "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "metrics": metrics,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Model comparison failed: {e}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

@app.post("/api/v1/prompts/generate")
async def generate_content(
    request: ContentGenerationRequest,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Generate content using prompt templates"""
    llm_requests_total.labels(endpoint="generate", provider=request.provider, model=request.model).inc()
    
    try:
        # Load template
        template_content = await load_template(request.template)
        
        # Render template with variables
        template = Template(template_content)
        rendered_prompt = template.render(**request.variables)
        
        if request.context:
            rendered_prompt = f"{request.context}\n\n{rendered_prompt}"
        
        # Generate content
        generation_data = {
            "prompt": rendered_prompt,
            "provider": request.provider,
            "model": request.model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature
        }
        
        result = await call_llm_service("/generate", generation_data)
        
        # Store generation results
        generation_record = {
            "template": request.template,
            "variables": request.variables,
            "generated_content": result.get("text"),
            "provider": request.provider,
            "model": request.model,
            "tokens_used": result.get("tokens_used", 0)
        }
        
        await store_in_memory(generation_record, "llm_generation", user_context)
        
        llm_tokens_used.labels(
            provider=request.provider,
            model=request.model,
            type="generation"
        ).inc(result.get("tokens_used", 0))
        
        return {
            "generated_content": result.get("text"),
            "metadata": {
                "template": request.template,
                "provider": request.provider,
                "model": request.model,
                "tokens_used": result.get("tokens_used", 0),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Content generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

# Causal Reasoning API Endpoints

@app.post("/api/v1/llm/causal/insights")
async def generate_causal_insights(
    request: CausalInsightRequest,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Generate natural language insights from causal analysis results"""
    llm_requests_total.labels(endpoint="causal_insights", provider="openai", model="gpt-3.5-turbo").inc()
    
    try:
        if not causal_insight_generator:
            raise HTTPException(status_code=503, detail="Causal insight generator not available")
        
        # Get causal analysis results from causal service
        causal_results = await get_causal_service_data(
            f"/api/v1/causal/experiments/{request.experiment_id}",
            {"include_analysis": True}
        )
        
        if not causal_results:
            raise HTTPException(status_code=404, detail="Causal experiment not found")
        
        # Extract causal data and analysis results
        causal_data = CausalMarketingData(**causal_results.get("causal_data", {}))
        analysis_results = causal_results.get("analysis_results", {})
        
        # Generate insights based on request type
        insights = []
        
        if request.insight_type in ["narrative", "all"]:
            narrative = await causal_insight_generator.generate_causal_narrative(causal_data, analysis_results)
            insights.append({
                "type": "narrative",
                "content": narrative,
                "confidence": "high"
            })
        
        if request.insight_type in ["treatment_effects", "all"]:
            treatment_effects = analysis_results.get("treatment_effects", {})
            for treatment, effect_data in treatment_effects.items():
                explanation = await causal_insight_generator.explain_treatment_effect(
                    treatment,
                    effect_data.get("effect_size", 0),
                    effect_data.get("confidence", 0),
                    effect_data.get("method", "unknown")
                )
                insights.append({
                    "type": "treatment_effect",
                    "treatment": treatment,
                    "content": explanation,
                    "effect_size": effect_data.get("effect_size", 0),
                    "confidence": effect_data.get("confidence", 0)
                })
        
        # Store insights in memory
        insight_record = {
            "experiment_id": request.experiment_id,
            "insight_type": request.insight_type,
            "insights": insights,
            "context": request.context
        }
        
        await store_in_memory(insight_record, "causal_insights", user_context)
        
        return CausalInsightResponse(
            experiment_id=request.experiment_id,
            insights=insights,
            metadata={
                "insight_type": request.insight_type,
                "total_insights": len(insights),
                "generated_at": datetime.utcnow().isoformat(),
                "context": request.context
            }
        )
        
    except Exception as e:
        logger.error(f"Causal insight generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Insight generation failed: {str(e)}")

@app.post("/api/v1/llm/causal/counterfactual")
async def analyze_counterfactual_scenario(
    request: CounterfactualAnalysisRequest,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Analyze counterfactual scenarios for marketing decisions"""
    llm_requests_total.labels(endpoint="counterfactual", provider="openai", model="gpt-3.5-turbo").inc()
    
    try:
        if not counterfactual_analyzer:
            raise HTTPException(status_code=503, detail="Counterfactual analyzer not available")
        
        # Get original causal data
        causal_results = await get_causal_service_data(
            f"/api/v1/causal/experiments/{request.experiment_id}"
        )
        
        if not causal_results:
            raise HTTPException(status_code=404, detail="Original experiment not found")
        
        causal_data = CausalMarketingData(**causal_results.get("causal_data", {}))
        
        # Analyze counterfactual scenario
        analysis = await counterfactual_analyzer.analyze_counterfactual(request.scenario, causal_data)
        
        # Store counterfactual analysis
        counterfactual_record = {
            "experiment_id": request.experiment_id,
            "scenario": request.scenario,
            "analysis": analysis,
            "baseline_metrics": request.baseline_metrics
        }
        
        await store_in_memory(counterfactual_record, "counterfactual_analysis", user_context)
        
        return CounterfactualAnalysisResponse(
            experiment_id=request.experiment_id,
            scenario=request.scenario,
            analysis=analysis["analysis"],
            predicted_outcomes=analysis.get("predicted_outcomes", {}),
            confidence_level=analysis["confidence_level"],
            key_assumptions=analysis["key_assumptions"],
            risk_factors=analysis["risk_factors"],
            recommendations=analysis["recommendations"],
            metadata={
                "analyzed_at": datetime.utcnow().isoformat(),
                "baseline_metrics": request.baseline_metrics
            }
        )
        
    except Exception as e:
        logger.error(f"Counterfactual analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Counterfactual analysis failed: {str(e)}")

@app.post("/api/v1/llm/causal/explain")
async def explain_causal_analysis(
    request: CausalExplanationRequest,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Provide detailed explanations of causal analysis methods and results"""
    llm_requests_total.labels(endpoint="causal_explain", provider="openai", model="gpt-3.5-turbo").inc()
    
    try:
        if not causal_explanation_engine:
            raise HTTPException(status_code=503, detail="Causal explanation engine not available")
        
        # Get analysis results
        if request.experiment_id:
            causal_results = await get_causal_service_data(
                f"/api/v1/causal/experiments/{request.experiment_id}",
                {"include_analysis": True}
            )
            analysis_results = causal_results.get("analysis_results", {})
        else:
            analysis_results = request.analysis_results or {}
        
        # Generate explanation
        explanation = await causal_explanation_engine.explain_causal_method(
            request.method,
            analysis_results
        )
        
        # Store explanation
        explanation_record = {
            "experiment_id": request.experiment_id,
            "method": request.method,
            "explanation": explanation,
            "analysis_results": analysis_results
        }
        
        await store_in_memory(explanation_record, "causal_explanations", user_context)
        
        return CausalExplanationResponse(
            method=request.method,
            explanation=explanation,
            key_concepts=_extract_key_concepts(explanation),
            limitations=_extract_limitations(explanation),
            business_implications=_extract_business_implications(explanation),
            metadata={
                "explained_at": datetime.utcnow().isoformat(),
                "experiment_id": request.experiment_id
            }
        )
        
    except Exception as e:
        logger.error(f"Causal explanation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Causal explanation failed: {str(e)}")

@app.get("/api/v1/llm/causal/templates")
async def get_causal_prompt_templates():
    """Get available causal analysis prompt templates"""
    try:
        templates = {
            "causal_narrative": {
                "name": "Causal Analysis Narrative",
                "description": "Generate comprehensive narrative explaining causal findings",
                "template": "Based on the causal analysis showing {{treatment_effects}} with confounders {{confounders}}, explain the business implications and actionable insights.",
                "variables": ["treatment_effects", "confounders", "external_factors"]
            },
            "treatment_explanation": {
                "name": "Treatment Effect Explanation",
                "description": "Explain specific treatment effects in business terms",
                "template": "The {{treatment}} intervention resulted in {{effect_size}} change with {{confidence}}% confidence. Explain what this means for marketing strategy.",
                "variables": ["treatment", "effect_size", "confidence", "method"]
            },
            "counterfactual_scenario": {
                "name": "Counterfactual Analysis",
                "description": "Analyze what-if scenarios based on causal models",
                "template": "If we had {{scenario}} instead of the original approach, predict the likely outcomes based on causal relationships identified in the data.",
                "variables": ["scenario", "baseline_metrics", "causal_relationships"]
            },
            "method_explanation": {
                "name": "Causal Method Explanation",
                "description": "Explain causal analysis methods in business-friendly terms",
                "template": "Explain how {{method}} analysis works and why the results showing {{findings}} are reliable for business decision-making.",
                "variables": ["method", "findings", "assumptions", "limitations"]
            }
        }
        
        return {
            "templates": templates,
            "categories": ["narrative", "explanation", "counterfactual", "methodology"],
            "total_count": len(templates)
        }
        
    except Exception as e:
        logger.error(f"Failed to get causal templates: {e}")
        raise HTTPException(status_code=500, detail=f"Template retrieval failed: {str(e)}")

def _extract_key_concepts(explanation: str) -> List[str]:
    """Extract key concepts from explanation text"""
    # Simple keyword extraction - could be enhanced with NLP
    key_terms = [
        "causation", "correlation", "treatment effect", "confounding", 
        "randomization", "bias", "counterfactual", "instrumental variable",
        "difference-in-differences", "synthetic control", "regression discontinuity"
    ]
    
    found_concepts = []
    explanation_lower = explanation.lower()
    
    for term in key_terms:
        if term in explanation_lower:
            found_concepts.append(term)
    
    return found_concepts

def _extract_limitations(explanation: str) -> List[str]:
    """Extract limitations mentioned in explanation"""
    limitation_indicators = [
        "limitation", "caveat", "assumption", "cannot", "unable", 
        "requires", "depends on", "may not", "might not"
    ]
    
    limitations = []
    sentences = explanation.split('.')
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(indicator in sentence_lower for indicator in limitation_indicators):
            limitations.append(sentence.strip())
    
    return limitations[:5]  # Return top 5 limitations

def _extract_business_implications(explanation: str) -> List[str]:
    """Extract business implications from explanation"""
    business_indicators = [
        "recommend", "suggest", "should", "strategy", "decision", 
        "optimize", "improve", "increase", "decrease", "roi", "budget"
    ]
    
    implications = []
    sentences = explanation.split('.')
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(indicator in sentence_lower for indicator in business_indicators):
            implications.append(sentence.strip())
    
    return implications[:5]  # Return top 5 implications

        

@app.get("/api/v1/prompts/templates")
async def list_templates():
    """List available prompt templates"""
    try:
        templates = await call_llm_service("/templates", {})
        
        return {
            "templates": templates.get("templates", []),
            "categories": templates.get("categories", []),
            "total_count": len(templates.get("templates", []))
        }
        
    except Exception as e:
        logger.error(f"Template listing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Template listing failed: {str(e)}")

@app.post("/api/v1/evaluation/metrics")
async def calculate_metrics(
    request: MetricsCalculationRequest,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Calculate evaluation metrics for predictions vs references"""
    try:
        metrics_results = {}
        
        if "bleu" in request.metrics:
            metrics_results["bleu"] = calculate_bleu_score(request.predictions, request.references)
        
        if "rouge" in request.metrics:
            metrics_results.update(calculate_rouge_scores(request.predictions, request.references))
        
        if "bert_score" in request.metrics:
            metrics_results.update(calculate_bert_scores(request.predictions, request.references))
        
        # Store metrics calculation
        await store_in_memory(metrics_results, "llm_metrics", user_context)
        
        return {
            "metrics": metrics_results,
            "metadata": {
                "predictions_count": len(request.predictions),
                "references_count": len(request.references),
                "language": request.language,
                "calculated_metrics": request.metrics,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Metrics calculation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics calculation failed: {str(e)}")

async def load_template(template_name: str) -> str:
    """Load prompt template"""
    try:
        # Try to load from LLM service first
        result = await call_llm_service(f"/templates/{template_name}", {})
        return result.get("template", "")
    except:
        # Fallback to default templates
        default_templates = {
            "ad_copy": "Create compelling advertising copy for {{product}} targeting {{audience}}. Highlight {{benefits}} and include a strong call-to-action.",
            "seo_content": "Write SEO-optimized content about {{topic}} for {{target_keywords}}. Include {{word_count}} words and maintain {{tone}} tone.",
            "email_marketing": "Create an engaging email for {{campaign_type}} about {{subject}}. Target audience: {{audience}}. Include {{cta}}.",
            "chatbot_responses": "Generate a helpful chatbot response for: {{user_query}}. Be {{tone}} and provide {{response_type}}."
        }
        return default_templates.get(template_name, "{{content}}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8009)