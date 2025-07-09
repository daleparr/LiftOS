"""
LiftOS Feedback Service
Continuous learning through outcome tracking and model improvement
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
from collections import defaultdict, deque

# Import shared modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
# KSE-SDK Integration
from shared.kse_sdk.client import LiftKSEClient
from shared.kse_sdk.core import Entity, SearchQuery, SearchResult
from shared.kse_sdk.models import EntityType, Domain


from shared.models.base import APIResponse, HealthCheck
from shared.models.decision import (
    DecisionOutcome, DecisionLearning, DecisionMetrics,
    Decision, DecisionStatus, ConfidenceLevel
)
from shared.models.learning import (
    LearningMetrics, AdaptiveLearning, LearningModel, LearningStatus
)
from shared.utils.config import get_service_config
from shared.utils.logging import setup_logging
from shared.health.health_checks import HealthChecker
from pydantic import BaseModel, Field

# Service configuration
config = get_service_config("feedback", 8010)
logger = setup_logging("feedback")

# Health checker
health_checker = HealthChecker("feedback")

# Service URLs
INTELLIGENCE_SERVICE_URL = os.getenv("INTELLIGENCE_SERVICE_URL", "http://localhost:8009")
MEMORY_SERVICE_URL = os.getenv("MEMORY_SERVICE_URL", "http://localhost:8003")
OBSERVABILITY_SERVICE_URL = os.getenv("OBSERVABILITY_SERVICE_URL", "http://localhost:8004")

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
    title="LiftOS Feedback Service",
    description="Continuous learning through outcome tracking and model improvement",
    version="1.0.0",
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

# Global state for feedback tracking
outcome_tracking: Dict[str, DecisionOutcome] = {}
learning_feedback: Dict[str, Dict[str, Any]] = {}
performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
model_performance: Dict[str, Dict[str, float]] = defaultdict(dict)


# Request/Response Models
class OutcomeReportRequest(BaseModel):
    """Request to report decision outcome"""
    decision_id: str
    actual_impact: Dict[str, float]
    success_metrics: Dict[str, float] = {}
    unexpected_consequences: List[str] = []
    user_satisfaction: Optional[float] = Field(None, ge=1.0, le=5.0)
    business_value: Optional[float] = None
    measurement_period: Optional[str] = None
    notes: Optional[str] = None


class LearningFeedbackRequest(BaseModel):
    """Request to provide learning feedback"""
    learning_id: str
    model_id: str
    accuracy_feedback: Dict[str, float] = {}
    pattern_validation: Dict[str, bool] = {}
    insight_usefulness: Dict[str, float] = {}
    recommendation_adoption: Dict[str, bool] = {}
    improvement_suggestions: List[str] = []


class ModelPerformanceRequest(BaseModel):
    """Request to update model performance"""
    model_id: str
    performance_metrics: Dict[str, float]
    validation_results: Dict[str, Any] = {}
    drift_indicators: Dict[str, float] = {}
    retraining_needed: bool = False
    notes: Optional[str] = None


class FeedbackAnalysisRequest(BaseModel):
    """Request for feedback analysis"""
    time_period: Optional[str] = "30d"
    model_ids: Optional[List[str]] = None
    decision_types: Optional[List[str]] = None
    domains: Optional[List[str]] = None


class ContinuousLearningEngine:
    """Engine for continuous learning from feedback"""
    
    def __init__(self):
        self.learning_rate = 0.01
        self.adaptation_threshold = 0.1
        self.confidence_decay = 0.95
        self.performance_window = 100
        
    async def process_outcome_feedback(self, outcome: DecisionOutcome) -> DecisionLearning:
        """Process outcome feedback and generate learning"""
        try:
            # Calculate accuracy score
            accuracy_score = await self._calculate_accuracy_score(outcome)
            
            # Identify learning opportunities
            insights = await self._extract_insights(outcome)
            
            # Generate model updates
            model_updates = await self._generate_model_updates(outcome, accuracy_score)
            
            # Create decision learning record
            learning = DecisionLearning(
                id=str(uuid.uuid4()),
                decision_id=outcome.decision_id,
                outcome_id=outcome.id,
                learning_type="outcome_feedback",
                insights=insights,
                model_updates=model_updates,
                performance_improvement=accuracy_score - 0.5,  # Baseline comparison
                created_at=datetime.utcnow()
            )
            
            # Store learning
            await self._store_learning(learning)
            
            # Update model performance tracking
            await self._update_performance_tracking(outcome, accuracy_score)
            
            return learning
            
        except Exception as e:
            logger.error(f"Failed to process outcome feedback: {e}")
            raise
    
    async def _calculate_accuracy_score(self, outcome: DecisionOutcome) -> float:
        """Calculate accuracy score based on predicted vs actual impact"""
        try:
            # Get original decision to compare predictions
            decision_response = await http_client.get(
                f"{INTELLIGENCE_SERVICE_URL}/api/v1/decisions/{outcome.decision_id}/result"
            )
            
            if decision_response.status_code != 200:
                return 0.5  # Default neutral score
            
            decision_data = decision_response.json().get("data", {}).get("decision", {})
            expected_impact = decision_data.get("expected_impact", {})
            actual_impact = outcome.actual_impact
            
            # Calculate accuracy for each metric
            accuracies = []
            for metric, expected_value in expected_impact.items():
                if metric in actual_impact:
                    actual_value = actual_impact[metric]
                    # Calculate relative accuracy (1 - relative error)
                    if expected_value != 0:
                        relative_error = abs(actual_value - expected_value) / abs(expected_value)
                        accuracy = max(0, 1 - relative_error)
                    else:
                        accuracy = 1.0 if actual_value == 0 else 0.0
                    accuracies.append(accuracy)
            
            # Return average accuracy or default
            return np.mean(accuracies) if accuracies else 0.5
            
        except Exception as e:
            logger.warning(f"Failed to calculate accuracy score: {e}")
            return 0.5
    
    async def _extract_insights(self, outcome: DecisionOutcome) -> List[str]:
        """Extract insights from outcome feedback"""
        insights = []
        
        # Analyze unexpected consequences
        if outcome.unexpected_consequences:
            insights.append(f"Unexpected consequences detected: {len(outcome.unexpected_consequences)} items")
            insights.extend([f"Unexpected: {cons}" for cons in outcome.unexpected_consequences[:3]])
        
        # Analyze performance vs expectations
        if outcome.actual_impact:
            for metric, value in outcome.actual_impact.items():
                if value > 0:
                    insights.append(f"Positive impact on {metric}: {value:.2f}")
                elif value < 0:
                    insights.append(f"Negative impact on {metric}: {value:.2f}")
        
        # Analyze user satisfaction
        if outcome.satisfaction_score:
            if outcome.satisfaction_score >= 4.0:
                insights.append("High user satisfaction with decision")
            elif outcome.satisfaction_score <= 2.0:
                insights.append("Low user satisfaction - investigate decision quality")
        
        # Add lessons learned
        insights.extend(outcome.lessons_learned)
        
        return insights
    
    async def _generate_model_updates(self, outcome: DecisionOutcome, accuracy_score: float) -> Dict[str, Any]:
        """Generate model updates based on feedback"""
        updates = {}
        
        # Confidence calibration updates
        if accuracy_score < 0.7:
            updates["confidence_adjustment"] = -0.1
            updates["reason"] = "Lower than expected accuracy"
        elif accuracy_score > 0.9:
            updates["confidence_adjustment"] = 0.05
            updates["reason"] = "Higher than expected accuracy"
        
        # Feature importance updates
        if outcome.unexpected_consequences:
            updates["feature_review_needed"] = True
            updates["review_reason"] = "Unexpected consequences suggest missing features"
        
        # Model retraining triggers
        if accuracy_score < 0.5:
            updates["retraining_recommended"] = True
            updates["retraining_priority"] = "high"
        
        return updates
    
    async def _store_learning(self, learning: DecisionLearning):
        """Store learning in memory service"""
        try:
            await http_client.post(
                f"{MEMORY_SERVICE_URL}/api/v1/store",
                json={
                    "content": f"Decision learning: {learning.learning_type}",
                    "metadata": {
                        "type": "decision_learning",
                        "decision_id": learning.decision_id,
                        "learning": learning.dict(),
                        "domain": "feedback"
                    },
                    "user_id": "system",
                    "organization_id": "system"
                }
            )
        except Exception as e:
            logger.warning(f"Failed to store learning in memory: {e}")
    
    async def _update_performance_tracking(self, outcome: DecisionOutcome, accuracy_score: float):
        """Update performance tracking metrics"""
        decision_id = outcome.decision_id
        
        # Add to performance history
        performance_history[decision_id].append({
            "timestamp": datetime.utcnow(),
            "accuracy_score": accuracy_score,
            "satisfaction_score": outcome.satisfaction_score,
            "business_value": outcome.business_value
        })
        
        # Update rolling averages
        recent_scores = [p["accuracy_score"] for p in list(performance_history[decision_id])[-10:]]
        model_performance[decision_id]["recent_accuracy"] = np.mean(recent_scores)
        
        if outcome.satisfaction_score:
            recent_satisfaction = [p["satisfaction_score"] for p in list(performance_history[decision_id])[-10:] 
                                 if p["satisfaction_score"] is not None]
            if recent_satisfaction:
                model_performance[decision_id]["recent_satisfaction"] = np.mean(recent_satisfaction)


# Initialize continuous learning engine
learning_engine = ContinuousLearningEngine()


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


@app.get("/health", response_model=HealthCheck, tags=["health"])
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.utcnow(),
        service="feedback",
        version="1.0.0",
        details={
            "tracked_outcomes": len(outcome_tracking),
            "learning_records": len(learning_feedback),
            "performance_histories": len(performance_history),
            "model_performance_records": len(model_performance)
        }
    )


@app.post("/api/v1/outcomes/report", tags=["outcomes"])
async def report_outcome(
    request: OutcomeReportRequest,
    background_tasks: BackgroundTasks,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Report decision outcome for learning"""
    try:
        # Create outcome record
        outcome = DecisionOutcome(
            id=str(uuid.uuid4()),
            decision_id=request.decision_id,
            actual_impact=request.actual_impact,
            success_metrics=request.success_metrics,
            unexpected_consequences=request.unexpected_consequences,
            satisfaction_score=request.user_satisfaction,
            business_value=request.business_value,
            measurement_period=request.measurement_period
        )
        
        # Store outcome
        outcome_tracking[outcome.id] = outcome
        
        # Process feedback in background
        background_tasks.add_task(process_outcome_feedback, outcome)
        
        return APIResponse(
            success=True,
            message="Outcome reported successfully",
            data={"outcome_id": outcome.id}
        )
        
    except Exception as e:
        logger.error(f"Failed to report outcome: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/learning/feedback", tags=["learning"])
async def provide_learning_feedback(
    request: LearningFeedbackRequest,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Provide feedback on learning results"""
    try:
        # Store learning feedback
        learning_feedback[request.learning_id] = {
            "model_id": request.model_id,
            "accuracy_feedback": request.accuracy_feedback,
            "pattern_validation": request.pattern_validation,
            "insight_usefulness": request.insight_usefulness,
            "recommendation_adoption": request.recommendation_adoption,
            "improvement_suggestions": request.improvement_suggestions,
            "provided_at": datetime.utcnow(),
            "provided_by": user_context["user_id"]
        }
        
        # Calculate feedback scores
        accuracy_scores = list(request.accuracy_feedback.values())
        usefulness_scores = list(request.insight_usefulness.values())
        adoption_rate = sum(request.recommendation_adoption.values()) / len(request.recommendation_adoption) if request.recommendation_adoption else 0
        
        # Update model performance
        if request.model_id not in model_performance:
            model_performance[request.model_id] = {}
        
        model_performance[request.model_id].update({
            "avg_accuracy_feedback": np.mean(accuracy_scores) if accuracy_scores else 0,
            "avg_usefulness": np.mean(usefulness_scores) if usefulness_scores else 0,
            "recommendation_adoption_rate": adoption_rate,
            "last_feedback": datetime.utcnow()
        })
        
        return APIResponse(
            success=True,
            message="Learning feedback recorded successfully",
            data={
                "feedback_id": request.learning_id,
                "summary": {
                    "avg_accuracy": np.mean(accuracy_scores) if accuracy_scores else 0,
                    "avg_usefulness": np.mean(usefulness_scores) if usefulness_scores else 0,
                    "adoption_rate": adoption_rate
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to record learning feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/models/performance", tags=["models"])
async def update_model_performance(
    request: ModelPerformanceRequest,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Update model performance metrics"""
    try:
        # Update model performance tracking
        if request.model_id not in model_performance:
            model_performance[request.model_id] = {}
        
        model_performance[request.model_id].update({
            **request.performance_metrics,
            "validation_results": request.validation_results,
            "drift_indicators": request.drift_indicators,
            "retraining_needed": request.retraining_needed,
            "last_updated": datetime.utcnow(),
            "updated_by": user_context["user_id"]
        })
        
        # Check for retraining triggers
        retraining_triggers = []
        if request.retraining_needed:
            retraining_triggers.append("Manual flag set")
        
        # Check drift indicators
        for indicator, value in request.drift_indicators.items():
            if value > 0.1:  # Threshold for significant drift
                retraining_triggers.append(f"Drift detected in {indicator}: {value:.3f}")
        
        # Check performance degradation
        if "accuracy" in request.performance_metrics and request.performance_metrics["accuracy"] < 0.7:
            retraining_triggers.append(f"Low accuracy: {request.performance_metrics['accuracy']:.3f}")
        
        return APIResponse(
            success=True,
            message="Model performance updated successfully",
            data={
                "model_id": request.model_id,
                "retraining_triggers": retraining_triggers,
                "retraining_recommended": len(retraining_triggers) > 0
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to update model performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/analysis/feedback", tags=["analysis"])
async def analyze_feedback(
    request: FeedbackAnalysisRequest = Depends(),
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Analyze feedback patterns and trends"""
    try:
        # Parse time period
        if request.time_period == "7d":
            cutoff_date = datetime.utcnow() - timedelta(days=7)
        elif request.time_period == "30d":
            cutoff_date = datetime.utcnow() - timedelta(days=30)
        elif request.time_period == "90d":
            cutoff_date = datetime.utcnow() - timedelta(days=90)
        else:
            cutoff_date = datetime.utcnow() - timedelta(days=30)
        
        # Analyze outcomes
        recent_outcomes = [
            outcome for outcome in outcome_tracking.values()
            if outcome.measured_at >= cutoff_date
        ]
        
        # Calculate metrics
        total_outcomes = len(recent_outcomes)
        avg_satisfaction = np.mean([o.satisfaction_score for o in recent_outcomes if o.satisfaction_score]) if recent_outcomes else 0
        avg_business_value = np.mean([o.business_value for o in recent_outcomes if o.business_value]) if recent_outcomes else 0
        
        # Analyze learning feedback
        recent_learning = [
            feedback for feedback in learning_feedback.values()
            if feedback["provided_at"] >= cutoff_date
        ]
        
        total_learning_feedback = len(recent_learning)
        avg_accuracy_feedback = np.mean([
            np.mean(list(f["accuracy_feedback"].values())) 
            for f in recent_learning if f["accuracy_feedback"]
        ]) if recent_learning else 0
        
        # Model performance summary
        model_summary = {}
        for model_id, performance in model_performance.items():
            if request.model_ids and model_id not in request.model_ids:
                continue
            
            model_summary[model_id] = {
                "recent_accuracy": performance.get("recent_accuracy", 0),
                "recent_satisfaction": performance.get("recent_satisfaction", 0),
                "recommendation_adoption_rate": performance.get("recommendation_adoption_rate", 0),
                "retraining_needed": performance.get("retraining_needed", False)
            }
        
        # Identify trends
        trends = []
        if avg_satisfaction > 4.0:
            trends.append("High user satisfaction trend")
        elif avg_satisfaction < 2.5:
            trends.append("Declining user satisfaction")
        
        if avg_accuracy_feedback > 0.8:
            trends.append("High accuracy feedback")
        elif avg_accuracy_feedback < 0.6:
            trends.append("Accuracy concerns reported")
        
        return APIResponse(
            success=True,
            data={
                "analysis_period": request.time_period,
                "outcome_metrics": {
                    "total_outcomes": total_outcomes,
                    "avg_satisfaction": avg_satisfaction,
                    "avg_business_value": avg_business_value
                },
                "learning_metrics": {
                    "total_feedback": total_learning_feedback,
                    "avg_accuracy_feedback": avg_accuracy_feedback
                },
                "model_performance": model_summary,
                "trends": trends,
                "recommendations": [
                    "Continue monitoring satisfaction scores",
                    "Focus on models with low accuracy feedback",
                    "Investigate unexpected consequences patterns"
                ]
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to analyze feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_outcome_feedback(outcome: DecisionOutcome):
    """Process outcome feedback in background"""
    try:
        learning = await learning_engine.process_outcome_feedback(outcome)
        logger.info(f"Processed outcome feedback for decision {outcome.decision_id}")
        
        # Send metrics to observability service
        await http_client.post(
            f"{OBSERVABILITY_SERVICE_URL}/api/v1/metrics",
            json={
                "name": "feedback_outcome_processed",
                "value": 1,
                "labels": {
                    "decision_id": outcome.decision_id,
                    "accuracy_score": str(learning.performance_improvement or 0)
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to process outcome feedback: {e}")


@app.get("/api/v1/outcomes/{outcome_id}", tags=["outcomes"])
async def get_outcome(
    outcome_id: str,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Get specific outcome details"""
    if outcome_id not in outcome_tracking:
        raise HTTPException(status_code=404, detail="Outcome not found")
    
    outcome = outcome_tracking[outcome_id]
    return APIResponse(
        success=True,
        data=outcome.dict()
    )


@app.get("/api/v1/models/{model_id}/performance", tags=["models"])
async def get_model_performance(
    model_id: str,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Get model performance metrics"""
    if model_id not in model_performance:
        raise HTTPException(status_code=404, detail="Model performance data not found")
    
    performance = model_performance[model_id]
    
    # Get performance history
    history = list(performance_history.get(model_id, []))
    
    return APIResponse(
        success=True,
        data={
            "model_id": model_id,
            "current_performance": performance,
            "performance_history": history[-50:],  # Last 50 records
            "trend_analysis": {
                "improving": len(history) > 1 and history[-1]["accuracy_score"] > history[-2]["accuracy_score"],
                "stable": len([h for h in history[-10:] if abs(h["accuracy_score"] - np.mean([p["accuracy_score"] for p in history[-10:]])) < 0.1]) > 7
            }
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=config.PORT)