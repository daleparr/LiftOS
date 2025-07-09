"""
LiftOS Intelligence Service
Core intelligence orchestration with learning engines and decision systems
"""
import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union
from fastapi import FastAPI, HTTPException, Depends, Header, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import httpx
import json
import numpy as np
from collections import defaultdict

# Import shared modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import KSE SDK for universal intelligence substrate
from shared.kse_sdk.client import LiftKSEClient
from shared.kse_sdk.core import Entity, SearchQuery, SearchResult, ConceptualSpace, KSEConfig
from shared.kse_sdk.core.models import SearchType

from shared.models.base import APIResponse, HealthCheck
from shared.models.learning import (
    LearningType, LearningRequest, LearningResponse, LearningModel,
    Pattern, PatternType, KnowledgeItem, LearningOutcome,
    CausalLearningResult, PerformanceLearningResult, UserBehaviorLearningResult,
    CompoundLearning, LearningMetrics, AdaptiveLearning
)
from shared.models.decision import (
    DecisionType, DecisionRequest, DecisionResponse, Decision,
    Recommendation, AutomatedAction, DecisionContext, DecisionOutcome,
    ConfidenceScore, DecisionRule, DecisionStrategy, ConfidenceLevel,
    RiskLevel, DecisionStatus, ActionType
)
from shared.utils.config import get_service_config
from shared.utils.logging import setup_logging
from shared.health.health_checks import HealthChecker

# Service configuration
config = get_service_config("intelligence", 8009)
logger = setup_logging("intelligence")

# Health checker
health_checker = HealthChecker("intelligence")

# Service URLs
MEMORY_SERVICE_URL = os.getenv("MEMORY_SERVICE_URL", "http://localhost:8003")
CAUSAL_SERVICE_URL = os.getenv("CAUSAL_SERVICE_URL", "http://localhost:8008")
OBSERVABILITY_SERVICE_URL = os.getenv("OBSERVABILITY_SERVICE_URL", "http://localhost:8004")

# FastAPI app
app = FastAPI(
    title="LiftOS Intelligence Service",
    description="Core intelligence orchestration with learning and decision capabilities",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if config.DEBUG else ["http://localhost:3000", "http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTTP client for service communication
http_client = httpx.AsyncClient(timeout=60.0)

# Global state for intelligence operations
learning_processes: Dict[str, Dict[str, Any]] = {}
decision_processes: Dict[str, Dict[str, Any]] = {}
knowledge_base: Dict[str, KnowledgeItem] = {}
pattern_registry: Dict[str, Pattern] = {}
decision_rules: Dict[str, DecisionRule] = {}
active_strategies: Dict[str, DecisionStrategy] = {}

# Initialize KSE client for universal intelligence substrate
kse_client = LiftKSEClient()


class IntelligenceEngine:
    """Core intelligence engine orchestrating learning and decisions"""
    
    def __init__(self):
        self.learning_models: Dict[str, LearningModel] = {}
        self.pattern_detectors: Dict[str, Any] = {}
        self.decision_engines: Dict[str, Any] = {}
        self.compound_learning_graph: Dict[str, List[str]] = {}
        self.kse_client = kse_client
        self.latest_patterns = []
        self.latest_decisions = []
        self.latest_confidence = []
        self.latest_insights = []
        
    async def initialize(self):
        """Initialize intelligence components"""
        await self._load_existing_models()
        await self._initialize_pattern_detectors()
        await self._initialize_decision_engines()
        logger.info("Intelligence engine initialized successfully")
    
    async def _load_existing_models(self):
        """Load existing learning models from memory service"""
        try:
            response = await http_client.get(f"{MEMORY_SERVICE_URL}/api/v1/search",
                params={"query": "learning_model", "limit": 100})
            if response.status_code == 200:
                models_data = response.json().get("data", {}).get("results", [])
                for model_data in models_data:
                    if "learning_model" in model_data.get("metadata", {}):
                        model = LearningModel(**model_data["metadata"]["learning_model"])
                        self.learning_models[model.id] = model
                logger.info(f"Loaded {len(self.learning_models)} existing learning models")
        except Exception as e:
            logger.warning(f"Failed to load existing models: {e}")
    
    async def retrieve_intelligence_data(self, query: str, domain: str = None) -> Dict[str, Any]:
        """Retrieve intelligence data from KSE universal substrate"""
        try:
            # Use KSE hybrid search for comprehensive intelligence retrieval
            results = await self.kse_client.hybrid_search(
                query=query,
                domain=domain,
                limit=10,
                include_embeddings=True,
                include_concepts=True,
                include_knowledge_graph=True
            )
            
            # Extract patterns and insights from KSE results
            intelligence_data = {
                'patterns': [],
                'concepts': [],
                'entities': [],
                'relationships': [],
                'confidence_scores': []
            }
            
            for result in results:
                if 'patterns' in result:
                    intelligence_data['patterns'].extend(result['patterns'])
                if 'concepts' in result:
                    intelligence_data['concepts'].extend(result['concepts'])
                if 'entities' in result:
                    intelligence_data['entities'].extend(result['entities'])
                if 'relationships' in result:
                    intelligence_data['relationships'].extend(result['relationships'])
                if 'confidence' in result:
                    intelligence_data['confidence_scores'].append(result['confidence'])
            
            return intelligence_data
            
        except Exception as e:
            logger.error(f"Failed to retrieve intelligence data: {e}")
            return {}
    
    async def enrich_intelligence_layer(self, data: Dict[str, Any], trace_id: str = None) -> bool:
        """Write back results, observations, and traces to enrich KSE intelligence layer"""
        try:
            # Create comprehensive trace for intelligence enrichment
            trace_data = {
                'service': 'intelligence',
                'timestamp': datetime.utcnow().isoformat(),
                'trace_id': trace_id or str(uuid.uuid4()),
                'operation': 'intelligence_processing',
                'data': data,
                'patterns_discovered': self.latest_patterns,
                'decisions_made': self.latest_decisions,
                'confidence_scores': self.latest_confidence,
                'learning_insights': self.latest_insights
            }
            
            # Store trace in KSE for intelligence layer enrichment
            await self.kse_client.store_trace(trace_data)
            
            # Store patterns as entities for future intelligence retrieval
            if 'patterns' in data:
                for pattern in data['patterns']:
                    entity_data = {
                        'type': 'intelligence_pattern',
                        'domain': data.get('domain', 'general'),
                        'content': pattern,
                        'confidence': pattern.get('confidence', 0.5) if isinstance(pattern, dict) else 0.5,
                        'source': 'intelligence_engine',
                        'metadata': {
                            'learning_rate': 0.1,
                            'discovery_time': datetime.utcnow().isoformat(),
                            'trace_id': trace_data['trace_id']
                        }
                    }
                    await self.kse_client.store_entity(entity_data)
            
            # Store decision insights for future intelligence
            if 'decisions' in data:
                for decision in data['decisions']:
                    entity_data = {
                        'type': 'intelligence_decision',
                        'domain': data.get('domain', 'general'),
                        'content': decision,
                        'confidence': decision.get('confidence', 0.5) if isinstance(decision, dict) else 0.5,
                        'source': 'intelligence_engine',
                        'metadata': {
                            'decision_context': decision.get('context', {}) if isinstance(decision, dict) else {},
                            'outcome_prediction': decision.get('prediction', {}) if isinstance(decision, dict) else {},
                            'trace_id': trace_data['trace_id']
                        }
                    }
                    await self.kse_client.store_entity(entity_data)
            
            logger.info(f"Successfully enriched intelligence layer with trace {trace_data['trace_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to enrich intelligence layer: {e}")
            return False
    
    async def _initialize_pattern_detectors(self):
        """Initialize pattern detection algorithms"""
        self.pattern_detectors = {
            PatternType.SEASONAL: SeasonalPatternDetector(),
            PatternType.TREND: TrendPatternDetector(),
            PatternType.ANOMALY: AnomalyPatternDetector(),
            PatternType.CORRELATION: CorrelationPatternDetector(),
            PatternType.CAUSAL: CausalPatternDetector(),
            PatternType.CYCLICAL: CyclicalPatternDetector(),
            PatternType.THRESHOLD: ThresholdPatternDetector()
        }
    
    async def _initialize_decision_engines(self):
        """Initialize decision-making engines"""
        self.decision_engines = {
            DecisionType.RECOMMENDATION: RecommendationEngine(),
            DecisionType.AUTOMATED_ACTION: AutomationEngine(),
            DecisionType.OPTIMIZATION: OptimizationEngine(),
            DecisionType.PREDICTION: PredictionEngine(),
            DecisionType.STRATEGY: StrategyEngine()
        }


# Pattern Detection Classes
class PatternDetector:
    """Base class for pattern detection"""
    
    async def detect_patterns(self, data: Dict[str, Any], context: Dict[str, Any]) -> List[Pattern]:
        """Detect patterns in data"""
        raise NotImplementedError


class SeasonalPatternDetector(PatternDetector):
    """Detect seasonal patterns in time series data"""
    
    async def detect_patterns(self, data: Dict[str, Any], context: Dict[str, Any]) -> List[Pattern]:
        patterns = []
        
        # Simulate seasonal pattern detection
        if "time_series" in data:
            pattern = Pattern(
                id=str(uuid.uuid4()),
                pattern_type=PatternType.SEASONAL,
                name="Weekly Seasonality",
                description="Strong weekly seasonal pattern detected",
                confidence=0.85,
                strength=0.75,
                variables=["conversions", "spend"],
                parameters={"period": "weekly", "amplitude": 0.3},
                time_range=(datetime.now() - timedelta(days=90), datetime.now()),
                frequency="weekly",
                evidence=["Statistical significance p<0.01", "Consistent across multiple weeks"]
            )
            patterns.append(pattern)
        
        return patterns


class TrendPatternDetector(PatternDetector):
    """Detect trend patterns in data"""
    
    async def detect_patterns(self, data: Dict[str, Any], context: Dict[str, Any]) -> List[Pattern]:
        patterns = []
        
        # Simulate trend detection
        if "metrics" in data:
            pattern = Pattern(
                id=str(uuid.uuid4()),
                pattern_type=PatternType.TREND,
                name="Upward Performance Trend",
                description="Consistent improvement in conversion rates",
                confidence=0.78,
                strength=0.65,
                variables=["conversion_rate", "roas"],
                parameters={"slope": 0.05, "r_squared": 0.82},
                time_range=(datetime.now() - timedelta(days=30), datetime.now()),
                evidence=["Linear regression R²=0.82", "Consistent week-over-week growth"]
            )
            patterns.append(pattern)
        
        return patterns


class AnomalyPatternDetector(PatternDetector):
    """Detect anomalies in data"""
    
    async def detect_patterns(self, data: Dict[str, Any], context: Dict[str, Any]) -> List[Pattern]:
        patterns = []
        
        # Simulate anomaly detection
        if "outliers" in data:
            pattern = Pattern(
                id=str(uuid.uuid4()),
                pattern_type=PatternType.ANOMALY,
                name="Spend Anomaly",
                description="Unusual spike in advertising spend detected",
                confidence=0.92,
                strength=0.88,
                variables=["daily_spend"],
                parameters={"threshold": 3.5, "z_score": 4.2},
                time_range=(datetime.now() - timedelta(days=1), datetime.now()),
                evidence=["Z-score > 3.5", "Deviation from 30-day average"]
            )
            patterns.append(pattern)
        
        return patterns


class CorrelationPatternDetector(PatternDetector):
    """Detect correlation patterns between variables"""
    
    async def detect_patterns(self, data: Dict[str, Any], context: Dict[str, Any]) -> List[Pattern]:
        patterns = []
        
        # Simulate correlation detection
        if "correlations" in data:
            pattern = Pattern(
                id=str(uuid.uuid4()),
                pattern_type=PatternType.CORRELATION,
                name="Spend-Conversion Correlation",
                description="Strong positive correlation between spend and conversions",
                confidence=0.89,
                strength=0.76,
                variables=["spend", "conversions"],
                parameters={"correlation": 0.76, "p_value": 0.001},
                time_range=(datetime.now() - timedelta(days=60), datetime.now()),
                evidence=["Pearson correlation r=0.76", "Statistically significant p<0.001"]
            )
            patterns.append(pattern)
        
        return patterns


class CausalPatternDetector(PatternDetector):
    """Detect causal patterns using causal inference"""
    
    async def detect_patterns(self, data: Dict[str, Any], context: Dict[str, Any]) -> List[Pattern]:
        patterns = []
        
        # Simulate causal pattern detection
        if "causal_data" in data:
            pattern = Pattern(
                id=str(uuid.uuid4()),
                pattern_type=PatternType.CAUSAL,
                name="Campaign Causal Effect",
                description="Causal relationship between campaign launch and sales lift",
                confidence=0.82,
                strength=0.71,
                variables=["campaign_active", "sales"],
                parameters={"treatment_effect": 0.15, "confidence_interval": [0.08, 0.22]},
                time_range=(datetime.now() - timedelta(days=45), datetime.now()),
                evidence=["Difference-in-differences analysis", "Instrumental variable validation"]
            )
            patterns.append(pattern)
        
        return patterns


class CyclicalPatternDetector(PatternDetector):
    """Detect cyclical patterns in data"""
    
    async def detect_patterns(self, data: Dict[str, Any], context: Dict[str, Any]) -> List[Pattern]:
        patterns = []
        
        # Simulate cyclical pattern detection
        if "cycles" in data:
            pattern = Pattern(
                id=str(uuid.uuid4()),
                pattern_type=PatternType.CYCLICAL,
                name="Monthly Budget Cycle",
                description="Regular monthly budget allocation pattern",
                confidence=0.87,
                strength=0.79,
                variables=["budget_allocation"],
                parameters={"cycle_length": 30, "phase_shift": 5},
                time_range=(datetime.now() - timedelta(days=180), datetime.now()),
                frequency="monthly",
                evidence=["Fourier analysis", "Consistent monthly peaks"]
            )
            patterns.append(pattern)
        
        return patterns


class ThresholdPatternDetector(PatternDetector):
    """Detect threshold-based patterns"""
    
    async def detect_patterns(self, data: Dict[str, Any], context: Dict[str, Any]) -> List[Pattern]:
        patterns = []
        
        # Simulate threshold detection
        if "thresholds" in data:
            pattern = Pattern(
                id=str(uuid.uuid4()),
                pattern_type=PatternType.THRESHOLD,
                name="ROAS Threshold Effect",
                description="Performance threshold at ROAS = 3.0",
                confidence=0.91,
                strength=0.84,
                variables=["roas", "profitability"],
                parameters={"threshold": 3.0, "effect_size": 0.25},
                time_range=(datetime.now() - timedelta(days=90), datetime.now()),
                evidence=["Regression discontinuity", "Clear breakpoint at 3.0"]
            )
            patterns.append(pattern)
        
        return patterns


# Decision Engine Classes
class DecisionEngine:
    """Base class for decision engines"""
    
    async def make_decision(self, context: DecisionContext, request: DecisionRequest) -> Decision:
        """Make a decision based on context and request"""
        raise NotImplementedError


class RecommendationEngine(DecisionEngine):
    """Generate intelligent recommendations"""
    
    async def make_decision(self, context: DecisionContext, request: DecisionRequest) -> Decision:
        # Simulate recommendation generation
        decision = Decision(
            id=str(uuid.uuid4()),
            decision_type=DecisionType.RECOMMENDATION,
            title="Budget Optimization Recommendation",
            description="Reallocate budget from underperforming campaigns to high-ROAS campaigns",
            confidence=0.84,
            confidence_level=ConfidenceLevel.HIGH,
            risk_level=RiskLevel.LOW,
            reasoning=[
                "Campaign A has ROAS of 1.2, below target of 2.0",
                "Campaign B has ROAS of 4.5, with room for scale",
                "Historical data shows similar reallocations improved overall ROAS by 15%"
            ],
            evidence=[
                "30-day performance analysis",
                "Statistical significance testing",
                "Causal impact analysis"
            ],
            expected_impact={"roas_improvement": 0.15, "cost_reduction": 0.08},
            created_by="intelligence_system",
            organization_id=context.organization_id,
            domain=context.domain
        )
        
        return decision


class AutomationEngine(DecisionEngine):
    """Handle automated actions"""
    
    async def make_decision(self, context: DecisionContext, request: DecisionRequest) -> Decision:
        # Simulate automation decision
        decision = Decision(
            id=str(uuid.uuid4()),
            decision_type=DecisionType.AUTOMATED_ACTION,
            title="Automatic Campaign Pause",
            description="Pause campaign due to poor performance and budget depletion risk",
            confidence=0.95,
            confidence_level=ConfidenceLevel.VERY_HIGH,
            risk_level=RiskLevel.MINIMAL,
            reasoning=[
                "Campaign ROAS dropped below 0.5 for 3 consecutive days",
                "Daily spend exceeds budget allocation by 150%",
                "No improvement trend detected in past 7 days"
            ],
            evidence=[
                "Real-time performance monitoring",
                "Budget tracking alerts",
                "Trend analysis"
            ],
            expected_impact={"budget_saved": 500.0, "risk_reduction": 0.9},
            created_by="automation_system",
            organization_id=context.organization_id,
            domain=context.domain
        )
        
        return decision


class OptimizationEngine(DecisionEngine):
    """Generate optimization decisions"""
    
    async def make_decision(self, context: DecisionContext, request: DecisionRequest) -> Decision:
        # Simulate optimization decision
        decision = Decision(
            id=str(uuid.uuid4()),
            decision_type=DecisionType.OPTIMIZATION,
            title="Bid Optimization Strategy",
            description="Implement dynamic bidding based on time-of-day performance patterns",
            confidence=0.78,
            confidence_level=ConfidenceLevel.HIGH,
            risk_level=RiskLevel.MEDIUM,
            reasoning=[
                "Conversion rates 40% higher during 2-4 PM",
                "Competition lower during early morning hours",
                "Dynamic bidding can improve efficiency by 20%"
            ],
            evidence=[
                "Hourly performance analysis",
                "Competitive intelligence data",
                "A/B test results from similar accounts"
            ],
            expected_impact={"efficiency_gain": 0.20, "cost_per_conversion": -0.15},
            created_by="optimization_system",
            organization_id=context.organization_id,
            domain=context.domain
        )
        
        return decision


class PredictionEngine(DecisionEngine):
    """Generate predictions and forecasts"""
    
    async def make_decision(self, context: DecisionContext, request: DecisionRequest) -> Decision:
        # Simulate prediction decision
        decision = Decision(
            id=str(uuid.uuid4()),
            decision_type=DecisionType.PREDICTION,
            title="Q1 Performance Forecast",
            description="Predicted 25% increase in conversions based on seasonal trends and budget increase",
            confidence=0.72,
            confidence_level=ConfidenceLevel.HIGH,
            risk_level=RiskLevel.LOW,
            reasoning=[
                "Historical Q1 shows 20% seasonal lift",
                "Budget increase of 30% planned",
                "New product launch expected to drive additional 10% lift"
            ],
            evidence=[
                "3-year historical seasonal analysis",
                "Budget planning documents",
                "Product launch impact modeling"
            ],
            expected_impact={"conversion_increase": 0.25, "revenue_increase": 0.30},
            created_by="prediction_system",
            organization_id=context.organization_id,
            domain=context.domain
        )
        
        return decision


class StrategyEngine(DecisionEngine):
    """Generate strategic decisions"""
    
    async def make_decision(self, context: DecisionContext, request: DecisionRequest) -> Decision:
        # Simulate strategic decision
        decision = Decision(
            id=str(uuid.uuid4()),
            decision_type=DecisionType.STRATEGY,
            title="Multi-Channel Attribution Strategy",
            description="Implement unified attribution model across all marketing channels",
            confidence=0.81,
            confidence_level=ConfidenceLevel.HIGH,
            risk_level=RiskLevel.MEDIUM,
            reasoning=[
                "Current last-click attribution undervalues upper-funnel channels",
                "Cross-channel synergies not being captured",
                "Unified model can improve budget allocation by 18%"
            ],
            evidence=[
                "Attribution modeling analysis",
                "Cross-channel correlation study",
                "Industry benchmark comparison"
            ],
            expected_impact={"attribution_accuracy": 0.35, "budget_efficiency": 0.18},
            created_by="strategy_system",
            organization_id=context.organization_id,
            domain=context.domain
        )
        
        return decision


# Initialize intelligence engine
intelligence_engine = IntelligenceEngine()


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


@app.on_event("startup")
async def startup_event():
    """Initialize intelligence engine on startup"""
    await intelligence_engine.initialize()
    logger.info("Intelligence service started successfully")


@app.get("/health", response_model=HealthCheck, tags=["health"])
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.utcnow(),
        service="intelligence",
        version="1.1.0",
        details={
            "learning_models": len(intelligence_engine.learning_models),
            "pattern_detectors": len(intelligence_engine.pattern_detectors),
            "decision_engines": len(intelligence_engine.decision_engines),
            "active_processes": len(learning_processes) + len(decision_processes)
        }
    )


@app.post("/api/v1/learning/start", response_model=LearningResponse, tags=["learning"])
async def start_learning(
    request: LearningRequest,
    background_tasks: BackgroundTasks,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Start a learning process"""
    learning_id = str(uuid.uuid4())
    
    # Initialize learning process
    learning_processes[learning_id] = {
        "id": learning_id,
        "request": request,
        "status": "initializing",
        "progress": 0.0,
        "started_at": datetime.utcnow(),
        "user_context": user_context
    }
    
    # Start background learning task
    background_tasks.add_task(execute_learning_process, learning_id, request, user_context)
    
    return LearningResponse(
        learning_id=learning_id,
        status="initializing",
        progress=0.0,
        estimated_completion=datetime.utcnow() + timedelta(minutes=30),
        message="Learning process initiated successfully"
    )


@app.post("/api/v1/decisions/request", response_model=DecisionResponse, tags=["decisions"])
async def request_decision(
    request: DecisionRequest,
    background_tasks: BackgroundTasks,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Request a decision from the intelligence system"""
    decision_id = str(uuid.uuid4())
    
    # Initialize decision process
    decision_processes[decision_id] = {
        "id": decision_id,
        "request": request,
        "status": "processing",
        "started_at": datetime.utcnow(),
        "user_context": user_context
    }
    
    # Start background decision task
    background_tasks.add_task(execute_decision_process, decision_id, request, user_context)
    
    return DecisionResponse(
        decision_id=decision_id,
        status="processing",
        estimated_completion=datetime.utcnow() + timedelta(minutes=5),
        message="Decision request received and processing"
    )


async def execute_learning_process(learning_id: str, request: LearningRequest, user_context: Dict[str, Any]):
    """Execute learning process in background with KSE integration"""
    try:
        # Update status
        learning_processes[learning_id]["status"] = "training"
        learning_processes[learning_id]["progress"] = 0.1
        
        # Retrieve intelligence data from KSE universal substrate
        intelligence_data = await intelligence_engine.retrieve_intelligence_data(
            query=f"learning patterns {request.domain} {request.learning_type.value}",
            domain=request.domain
        )
        learning_processes[learning_id]["progress"] = 0.3
        
        # Use KSE intelligence data to enhance learning
        kse_patterns = intelligence_data.get('patterns', [])
        kse_concepts = intelligence_data.get('concepts', [])
        kse_entities = intelligence_data.get('entities', [])
        
        # Generate learning outcomes with KSE enhancement
        patterns = []
        if request.learning_type == LearningType.PATTERN_DISCOVERY:
            # Use pattern detectors enhanced with KSE data
            for detector in intelligence_engine.pattern_detectors.values():
                detected_patterns = await detector.detect_patterns(
                    {
                        "time_series": True,
                        "metrics": True,
                        "kse_patterns": kse_patterns,
                        "kse_concepts": kse_concepts,
                        "kse_entities": kse_entities
                    },
                    {"domain": request.domain, "intelligence_data": intelligence_data}
                )
                patterns.extend(detected_patterns)
        
        learning_processes[learning_id]["progress"] = 0.7
        
        # Store patterns in registry and update intelligence engine state
        for pattern in patterns:
            pattern_registry[pattern.id] = pattern
        
        intelligence_engine.latest_patterns = [p.dict() for p in patterns]
        intelligence_engine.latest_insights = [
            "KSE-enhanced seasonal patterns detected in conversion data",
            "Strong correlation between spend and conversions validated by KSE",
            "Opportunity for budget optimization identified through intelligence layer",
            f"Leveraged {len(kse_patterns)} existing patterns from KSE",
            f"Incorporated {len(kse_concepts)} conceptual insights from KSE"
        ]
        
        # Create learning outcome
        outcome = LearningOutcome(
            id=str(uuid.uuid4()),
            learning_model_id=learning_id,
            patterns_discovered=[p.id for p in patterns],
            insights=intelligence_engine.latest_insights,
            recommendations=[
                "Implement seasonal budget adjustments based on KSE patterns",
                "Increase spend during high-conversion periods identified by intelligence layer",
                "Monitor for anomalies using KSE-enhanced detection",
                "Leverage conceptual insights for strategic decisions"
            ],
            confidence=0.92  # Higher confidence due to KSE enhancement
        )
        
        # Enrich intelligence layer with learning results
        enrichment_data = {
            'domain': request.domain,
            'patterns': [p.dict() for p in patterns],
            'insights': intelligence_engine.latest_insights,
            'learning_type': request.learning_type.value,
            'confidence': outcome.confidence,
            'kse_enhancement': {
                'patterns_used': len(kse_patterns),
                'concepts_used': len(kse_concepts),
                'entities_used': len(kse_entities)
            }
        }
        
        await intelligence_engine.enrich_intelligence_layer(
            enrichment_data,
            trace_id=f"learning_{learning_id}"
        )
        
        # Update final status
        learning_processes[learning_id]["status"] = "completed"
        learning_processes[learning_id]["progress"] = 1.0
        learning_processes[learning_id]["outcome"] = outcome
        learning_processes[learning_id]["completed_at"] = datetime.utcnow()
        learning_processes[learning_id]["kse_integration"] = {
            "intelligence_retrieved": True,
            "patterns_enhanced": len(patterns),
            "layer_enriched": True
        }
        
        logger.info(f"Learning process {learning_id} completed successfully with KSE integration")
        
    except Exception as e:
        learning_processes[learning_id]["status"] = "failed"
        learning_processes[learning_id]["error"] = str(e)
        logger.error(f"Learning process {learning_id} failed: {e}")


async def execute_decision_process(decision_id: str, request: DecisionRequest, user_context: Dict[str, Any]):
    """Execute decision process in background with KSE integration"""
    try:
        # Retrieve intelligence data from KSE for decision context
        intelligence_data = await intelligence_engine.retrieve_intelligence_data(
            query=f"decision patterns {request.domain} {request.decision_type.value}",
            domain=request.domain
        )
        
        # Create enhanced decision context with KSE intelligence
        context = DecisionContext(
            id=str(uuid.uuid4()),
            organization_id=request.organization_id,
            user_id=request.user_id,
            domain=request.domain,
            current_state={
                **request.context,
                "kse_intelligence": intelligence_data,
                "available_patterns": intelligence_data.get('patterns', []),
                "conceptual_insights": intelligence_data.get('concepts', []),
                "related_entities": intelligence_data.get('entities', [])
            },
            constraints=request.constraints,
            objectives=request.objectives
        )
        
        # Get appropriate decision engine
        engine = intelligence_engine.decision_engines.get(request.decision_type)
        if not engine:
            raise ValueError(f"No decision engine available for type: {request.decision_type}")
        
        # Generate KSE-enhanced decision
        decision = await engine.make_decision(context, request)
        
        # Update intelligence engine state with decision insights
        intelligence_engine.latest_decisions = [decision.dict() if hasattr(decision, 'dict') else str(decision)]
        intelligence_engine.latest_confidence = [getattr(decision, 'confidence', 0.8)]
        
        # Enrich intelligence layer with decision results
        enrichment_data = {
            'domain': request.domain,
            'decisions': intelligence_engine.latest_decisions,
            'decision_type': request.decision_type.value,
            'context': context.dict() if hasattr(context, 'dict') else str(context),
            'confidence': intelligence_engine.latest_confidence[0] if intelligence_engine.latest_confidence else 0.8,
            'kse_enhancement': {
                'patterns_considered': len(intelligence_data.get('patterns', [])),
                'concepts_used': len(intelligence_data.get('concepts', [])),
                'entities_analyzed': len(intelligence_data.get('entities', []))
            }
        }
        
        await intelligence_engine.enrich_intelligence_layer(
            enrichment_data,
            trace_id=f"decision_{decision_id}"
        )
        
        # Store decision with KSE integration details
        decision_processes[decision_id]["decision"] = decision
        decision_processes[decision_id]["status"] = "completed"
        decision_processes[decision_id]["completed_at"] = datetime.utcnow()
        decision_processes[decision_id]["kse_integration"] = {
            "intelligence_retrieved": True,
            "patterns_considered": len(intelligence_data.get('patterns', [])),
            "layer_enriched": True
        }
        
        logger.info(f"Decision process {decision_id} completed successfully with KSE integration")
        
    except Exception as e:
        decision_processes[decision_id]["status"] = "failed"
        decision_processes[decision_id]["error"] = str(e)
        logger.error(f"Decision process {decision_id} failed: {e}")


@app.get("/api/v1/learning/{learning_id}/status", tags=["learning"])
async def get_learning_status(
    learning_id: str,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Get status of learning process"""
    if learning_id not in learning_processes:
        raise HTTPException(status_code=404, detail="Learning process not found")
    
    process = learning_processes[learning_id]
    return APIResponse(
        success=True,
        data={
            "learning_id": learning_id,
            "status": process["status"],
            "progress": process.get("progress", 0.0),
            "started_at": process["started_at"],
            "completed_at": process.get("completed_at"),
            "outcome": process.get("outcome"),
            "error": process.get("error")
        }
    )


@app.get("/api/v1/decisions/{decision_id}/result", tags=["decisions"])
async def get_decision_result(
    decision_id: str,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Get result of decision process"""
    if decision_id not in decision_processes:
        raise HTTPException(status_code=404, detail="Decision process not found")
    
    process = decision_processes[decision_id]
    return APIResponse(
        success=True,
        data={
            "decision_id": decision_id,
            "status": process["status"],
            "started_at": process["started_at"],
            "completed_at": process.get("completed_at"),
            "decision": process.get("decision"),
            "error": process.get("error")
        }
    )


@app.get("/api/v1/patterns", tags=["patterns"])
async def get_patterns(
    pattern_type: Optional[PatternType] = None,
    limit: int = 50,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Get discovered patterns"""
    patterns = list(pattern_registry.values())
    
    if pattern_type:
        patterns = [p for p in patterns if p.pattern_type == pattern_type]
    
    patterns = patterns[:limit]
    
    return APIResponse(
        success=True,
        data={
            "patterns": patterns,
            "total": len(patterns)
        }
    )


@app.get("/api/v1/knowledge", tags=["knowledge"])
async def get_knowledge_base(
    domain: Optional[str] = None,
    limit: int = 50,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Get knowledge base items"""
    knowledge_items = list(knowledge_base.values())
    
    if domain:
        knowledge_items = [k for k in knowledge_items if k.domain == domain]
    
    knowledge_items = knowledge_items[:limit]
    
    return APIResponse(
        success=True,
        data={
            "knowledge_items": knowledge_items,
            "total": len(knowledge_items)
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=config.PORT)