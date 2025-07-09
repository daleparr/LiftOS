"""
LiftOS Impact Monitoring Service
Tracks decision outcomes and measures actual business impact.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager
from collections import defaultdict
import statistics
import json

# Import models
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
# KSE-SDK Integration
from shared.kse_sdk.client import LiftKSEClient
from shared.kse_sdk.core import Entity, SearchQuery, SearchResult
from shared.kse_sdk.models import EntityType, Domain


from shared.models.business import BusinessImpactAssessment, BusinessMetric, MetricType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DecisionOutcome:
    """Track decision outcome and impact"""
    def __init__(self, decision_id: str, user_id: str, decision_data: Dict[str, Any]):
        self.decision_id = decision_id
        self.user_id = user_id
        self.decision_timestamp = datetime.utcnow()
        self.decision_data = decision_data
        
        # Predictions
        self.predicted_impact = decision_data.get('predicted_impact', 0.0)
        self.predicted_revenue = decision_data.get('predicted_revenue', 0.0)
        self.predicted_cost_savings = decision_data.get('predicted_cost_savings', 0.0)
        self.confidence_score = decision_data.get('confidence_score', 0.5)
        
        # Actual outcomes (to be measured)
        self.actual_impact: Optional[float] = None
        self.actual_revenue: Optional[float] = None
        self.actual_cost_savings: Optional[float] = None
        self.outcome_measured_at: Optional[datetime] = None
        
        # Impact breakdown
        self.direct_impact = 0.0
        self.indirect_impact = 0.0
        self.long_term_impact = 0.0
        
        # Context
        self.business_context = decision_data.get('business_context', {})
        self.external_factors: List[str] = []
        
        # Status
        self.is_measured = False
        self.measurement_confidence = 0.0

class ImpactMeasurement:
    """Represents a measured impact"""
    def __init__(self, outcome_id: str, measurement_data: Dict[str, Any]):
        self.id = f"impact_{datetime.utcnow().timestamp()}"
        self.outcome_id = outcome_id
        self.measurement_timestamp = datetime.utcnow()
        self.measurement_data = measurement_data
        
        # Impact values
        self.revenue_impact = measurement_data.get('revenue_impact', 0.0)
        self.cost_impact = measurement_data.get('cost_impact', 0.0)
        self.efficiency_impact = measurement_data.get('efficiency_impact', 0.0)
        self.customer_impact = measurement_data.get('customer_impact', 0.0)
        
        # Attribution
        self.attribution_method = measurement_data.get('attribution_method', 'direct')
        self.attribution_confidence = measurement_data.get('attribution_confidence', 0.5)
        
        # Quality
        self.data_quality_score = measurement_data.get('data_quality_score', 1.0)
        self.measurement_method = measurement_data.get('measurement_method', 'manual')

class ImpactMonitoringEngine:
    """Core impact monitoring and measurement engine"""
    
    def __init__(self):
        self.decision_outcomes: Dict[str, DecisionOutcome] = {}
        self.impact_measurements: Dict[str, ImpactMeasurement] = {}
        self.impact_assessments: List[BusinessImpactAssessment] = []
        
        # Analytics
        self.accuracy_history: List[float] = []
        self.impact_trends: Dict[str, List[float]] = defaultdict(list)
        self.attribution_models: Dict[str, Dict[str, Any]] = {}
        
        # Monitoring
        self.pending_measurements: List[str] = []
        self.measurement_schedule: Dict[str, datetime] = {}
        
    async def track_decision_outcome(self, decision_data: Dict[str, Any]) -> DecisionOutcome:
        """Track a decision for outcome monitoring"""
        try:
            decision_id = decision_data.get('decision_id')
            user_id = decision_data.get('user_id')
            
            if not decision_id or not user_id:
                raise ValueError("decision_id and user_id are required")
            
            outcome = DecisionOutcome(decision_id, user_id, decision_data)
            self.decision_outcomes[decision_id] = outcome
            
            # Schedule impact measurement
            measurement_delay = decision_data.get('measurement_delay_days', 7)
            measurement_date = datetime.utcnow() + timedelta(days=measurement_delay)
            self.measurement_schedule[decision_id] = measurement_date
            self.pending_measurements.append(decision_id)
            
            logger.info(f"Tracking decision outcome: {decision_id}, predicted impact: ${outcome.predicted_impact:,.2f}")
            return outcome
            
        except Exception as e:
            logger.error(f"Error tracking decision outcome: {e}")
            raise
    
    async def measure_impact(self, decision_id: str, measurement_data: Dict[str, Any]) -> ImpactMeasurement:
        """Measure actual impact of a decision"""
        try:
            if decision_id not in self.decision_outcomes:
                raise ValueError(f"Decision outcome {decision_id} not found")
            
            outcome = self.decision_outcomes[decision_id]
            measurement = ImpactMeasurement(decision_id, measurement_data)
            
            # Update outcome with actual measurements
            outcome.actual_impact = measurement_data.get('total_impact', 0.0)
            outcome.actual_revenue = measurement.revenue_impact
            outcome.actual_cost_savings = abs(measurement.cost_impact) if measurement.cost_impact < 0 else 0
            outcome.outcome_measured_at = datetime.utcnow()
            outcome.is_measured = True
            outcome.measurement_confidence = measurement.attribution_confidence
            
            # Calculate impact breakdown
            outcome.direct_impact = measurement_data.get('direct_impact', outcome.actual_impact * 0.7)
            outcome.indirect_impact = measurement_data.get('indirect_impact', outcome.actual_impact * 0.2)
            outcome.long_term_impact = measurement_data.get('long_term_impact', outcome.actual_impact * 0.1)
            
            # Store measurement
            self.impact_measurements[measurement.id] = measurement
            
            # Calculate accuracy
            accuracy = await self._calculate_prediction_accuracy(outcome)
            self.accuracy_history.append(accuracy)
            
            # Create business impact assessment
            assessment = await self._create_impact_assessment(outcome, measurement)
            self.impact_assessments.append(assessment)
            
            # Remove from pending measurements
            if decision_id in self.pending_measurements:
                self.pending_measurements.remove(decision_id)
            
            # Update trends
            self.impact_trends['total_impact'].append(outcome.actual_impact)
            self.impact_trends['revenue_impact'].append(measurement.revenue_impact)
            self.impact_trends['cost_impact'].append(measurement.cost_impact)
            
            logger.info(f"Measured impact for decision {decision_id}: ${outcome.actual_impact:,.2f} actual vs ${outcome.predicted_impact:,.2f} predicted")
            return measurement
            
        except Exception as e:
            logger.error(f"Error measuring impact: {e}")
            raise
    
    async def _calculate_prediction_accuracy(self, outcome: DecisionOutcome) -> float:
        """Calculate accuracy of impact prediction"""
        try:
            if outcome.predicted_impact == 0:
                return 0.5  # Neutral accuracy for zero predictions
            
            # Calculate percentage error
            error = abs(outcome.actual_impact - outcome.predicted_impact) / abs(outcome.predicted_impact)
            
            # Convert to accuracy (1 - error, capped at 0)
            accuracy = max(0.0, 1.0 - error)
            
            return accuracy
            
        except Exception as e:
            logger.error(f"Error calculating prediction accuracy: {e}")
            return 0.0
    
    async def _create_impact_assessment(self, outcome: DecisionOutcome, measurement: ImpactMeasurement) -> BusinessImpactAssessment:
        """Create a business impact assessment"""
        try:
            assessment = BusinessImpactAssessment(
                decision_id=outcome.decision_id,
                predicted_impact=outcome.predicted_impact,
                actual_impact=outcome.actual_impact,
                impact_accuracy=await self._calculate_prediction_accuracy(outcome),
                revenue_impact=measurement.revenue_impact,
                cost_impact=measurement.cost_impact,
                efficiency_impact=measurement.efficiency_impact,
                customer_impact=measurement.customer_impact,
                direct_impact=outcome.direct_impact,
                indirect_impact=outcome.indirect_impact,
                attribution_confidence=measurement.attribution_confidence,
                impact_date=outcome.decision_timestamp,
                business_context=outcome.business_context,
                external_factors=outcome.external_factors
            )
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error creating impact assessment: {e}")
            raise
    
    async def calculate_roi_impact(self, time_period_days: int = 30) -> Dict[str, Any]:
        """Calculate ROI impact over a time period"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=time_period_days)
            
            # Filter recent measured outcomes
            recent_outcomes = [
                outcome for outcome in self.decision_outcomes.values()
                if outcome.is_measured and outcome.decision_timestamp >= cutoff_date
            ]
            
            if not recent_outcomes:
                return {"message": "No measured outcomes in the specified period"}
            
            # Calculate totals
            total_predicted_impact = sum(o.predicted_impact for o in recent_outcomes)
            total_actual_impact = sum(o.actual_impact for o in recent_outcomes)
            total_revenue_impact = sum(o.actual_revenue or 0 for o in recent_outcomes)
            total_cost_savings = sum(o.actual_cost_savings or 0 for o in recent_outcomes)
            
            # Calculate ROI metrics
            total_investment = sum(o.business_context.get('investment_cost', 0) for o in recent_outcomes)
            roi_percentage = 0.0
            if total_investment > 0:
                roi_percentage = ((total_actual_impact - total_investment) / total_investment) * 100
            
            # Calculate accuracy
            avg_accuracy = statistics.mean(await self._calculate_prediction_accuracy(o) for o in recent_outcomes)
            
            roi_impact = {
                "time_period_days": time_period_days,
                "total_decisions": len(recent_outcomes),
                "total_predicted_impact": total_predicted_impact,
                "total_actual_impact": total_actual_impact,
                "total_revenue_impact": total_revenue_impact,
                "total_cost_savings": total_cost_savings,
                "total_investment": total_investment,
                "roi_percentage": roi_percentage,
                "prediction_accuracy": avg_accuracy,
                "impact_variance": total_actual_impact - total_predicted_impact
            }
            
            return roi_impact
            
        except Exception as e:
            logger.error(f"Error calculating ROI impact: {e}")
            raise
    
    async def get_impact_attribution_analysis(self) -> Dict[str, Any]:
        """Analyze impact attribution across different factors"""
        try:
            measured_outcomes = [o for o in self.decision_outcomes.values() if o.is_measured]
            
            if not measured_outcomes:
                return {"message": "No measured outcomes available"}
            
            # Attribution by decision type
            attribution_by_type = defaultdict(list)
            for outcome in measured_outcomes:
                decision_type = outcome.business_context.get('decision_type', 'unknown')
                attribution_by_type[decision_type].append(outcome.actual_impact)
            
            # Attribution by user
            attribution_by_user = defaultdict(list)
            for outcome in measured_outcomes:
                attribution_by_user[outcome.user_id].append(outcome.actual_impact)
            
            # Attribution by confidence level
            high_confidence = [o.actual_impact for o in measured_outcomes if o.confidence_score >= 0.8]
            medium_confidence = [o.actual_impact for o in measured_outcomes if 0.5 <= o.confidence_score < 0.8]
            low_confidence = [o.actual_impact for o in measured_outcomes if o.confidence_score < 0.5]
            
            # Calculate averages
            attribution_analysis = {
                "total_measured_decisions": len(measured_outcomes),
                "attribution_by_type": {
                    decision_type: {
                        "count": len(impacts),
                        "total_impact": sum(impacts),
                        "average_impact": statistics.mean(impacts) if impacts else 0
                    }
                    for decision_type, impacts in attribution_by_type.items()
                },
                "attribution_by_confidence": {
                    "high_confidence": {
                        "count": len(high_confidence),
                        "average_impact": statistics.mean(high_confidence) if high_confidence else 0
                    },
                    "medium_confidence": {
                        "count": len(medium_confidence),
                        "average_impact": statistics.mean(medium_confidence) if medium_confidence else 0
                    },
                    "low_confidence": {
                        "count": len(low_confidence),
                        "average_impact": statistics.mean(low_confidence) if low_confidence else 0
                    }
                },
                "top_performing_users": sorted(
                    [(user_id, sum(impacts)) for user_id, impacts in attribution_by_user.items()],
                    key=lambda x: x[1], reverse=True
                )[:10]
            }
            
            return attribution_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing impact attribution: {e}")
            raise
    
    async def get_prediction_accuracy_trends(self) -> Dict[str, Any]:
        """Get prediction accuracy trends over time"""
        try:
            if not self.accuracy_history:
                return {"message": "No accuracy data available"}
            
            # Calculate trends
            recent_accuracy = self.accuracy_history[-10:] if len(self.accuracy_history) >= 10 else self.accuracy_history
            overall_accuracy = statistics.mean(self.accuracy_history)
            recent_avg_accuracy = statistics.mean(recent_accuracy)
            
            # Trend direction
            trend_direction = "stable"
            if len(self.accuracy_history) >= 5:
                early_avg = statistics.mean(self.accuracy_history[:len(self.accuracy_history)//2])
                late_avg = statistics.mean(self.accuracy_history[len(self.accuracy_history)//2:])
                
                if late_avg > early_avg * 1.05:
                    trend_direction = "improving"
                elif late_avg < early_avg * 0.95:
                    trend_direction = "declining"
            
            accuracy_trends = {
                "total_predictions": len(self.accuracy_history),
                "overall_accuracy": overall_accuracy,
                "recent_accuracy": recent_avg_accuracy,
                "trend_direction": trend_direction,
                "accuracy_variance": statistics.stdev(self.accuracy_history) if len(self.accuracy_history) > 1 else 0,
                "accuracy_history": self.accuracy_history[-20:]  # Last 20 predictions
            }
            
            return accuracy_trends
            
        except Exception as e:
            logger.error(f"Error getting prediction accuracy trends: {e}")
            raise
    
    async def get_pending_measurements(self) -> List[Dict[str, Any]]:
        """Get decisions pending impact measurement"""
        try:
            pending = []
            current_time = datetime.utcnow()
            
            for decision_id in self.pending_measurements:
                if decision_id in self.decision_outcomes:
                    outcome = self.decision_outcomes[decision_id]
                    scheduled_date = self.measurement_schedule.get(decision_id)
                    
                    is_due = scheduled_date and current_time >= scheduled_date
                    days_since_decision = (current_time - outcome.decision_timestamp).days
                    
                    pending.append({
                        "decision_id": decision_id,
                        "user_id": outcome.user_id,
                        "decision_date": outcome.decision_timestamp.isoformat(),
                        "predicted_impact": outcome.predicted_impact,
                        "days_since_decision": days_since_decision,
                        "scheduled_measurement_date": scheduled_date.isoformat() if scheduled_date else None,
                        "is_due": is_due
                    })
            
            # Sort by due date
            pending.sort(key=lambda x: x["days_since_decision"], reverse=True)
            
            return pending
            
        except Exception as e:
            logger.error(f"Error getting pending measurements: {e}")
            raise
    
    async def get_impact_summary(self) -> Dict[str, Any]:
        """Get comprehensive impact monitoring summary"""
        try:
            total_decisions = len(self.decision_outcomes)
            measured_decisions = len([o for o in self.decision_outcomes.values() if o.is_measured])
            pending_decisions = len(self.pending_measurements)
            
            # Calculate totals for measured decisions
            measured_outcomes = [o for o in self.decision_outcomes.values() if o.is_measured]
            
            total_predicted = sum(o.predicted_impact for o in measured_outcomes)
            total_actual = sum(o.actual_impact for o in measured_outcomes)
            
            avg_accuracy = statistics.mean(self.accuracy_history) if self.accuracy_history else 0
            
            summary = {
                "timestamp": datetime.utcnow().isoformat(),
                "total_decisions_tracked": total_decisions,
                "measured_decisions": measured_decisions,
                "pending_measurements": pending_decisions,
                "measurement_rate": measured_decisions / total_decisions if total_decisions > 0 else 0,
                "total_predicted_impact": total_predicted,
                "total_actual_impact": total_actual,
                "prediction_accuracy": avg_accuracy,
                "impact_variance": total_actual - total_predicted,
                "recent_trends": {
                    "total_impact": self.impact_trends['total_impact'][-10:],
                    "revenue_impact": self.impact_trends['revenue_impact'][-10:],
                    "cost_impact": self.impact_trends['cost_impact'][-10:]
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating impact summary: {e}")
            raise

# Initialize the impact monitoring engine
impact_engine = ImpactMonitoringEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("Starting Impact Monitoring Service...")
    logger.info("Impact Monitoring Service started successfully")
    yield
    logger.info("Impact Monitoring Service shutting down...")

# Create FastAPI app

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
    title="LiftOS Impact Monitoring Service",
    description="Track decision outcomes and measure actual business impact",
    version="1.1.0",
    lifespan=lifespan
)

# Add CORS middleware
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
    return {
        "status": "healthy",
        "service": "impact-monitoring",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/decisions/track")
async def track_decision(data: Dict[str, Any]):
    """Track a decision for outcome monitoring"""
    try:
        outcome = await impact_engine.track_decision_outcome(data)
        return {"status": "success", "decision_id": outcome.decision_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error tracking decision: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/decisions/{decision_id}/measure")
async def measure_decision_impact(decision_id: str, data: Dict[str, Any]):
    """Measure actual impact of a decision"""
    try:
        measurement = await impact_engine.measure_impact(decision_id, data)
        return {"status": "success", "measurement_id": measurement.id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error measuring impact: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/roi")
async def get_roi_analytics(time_period_days: int = 30):
    """Get ROI impact analytics"""
    try:
        roi_impact = await impact_engine.calculate_roi_impact(time_period_days)
        return {"status": "success", "roi_impact": roi_impact}
    except Exception as e:
        logger.error(f"Error getting ROI analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/attribution")
async def get_attribution_analytics():
    """Get impact attribution analysis"""
    try:
        attribution = await impact_engine.get_impact_attribution_analysis()
        return {"status": "success", "attribution": attribution}
    except Exception as e:
        logger.error(f"Error getting attribution analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/accuracy")
async def get_accuracy_analytics():
    """Get prediction accuracy trends"""
    try:
        accuracy = await impact_engine.get_prediction_accuracy_trends()
        return {"status": "success", "accuracy": accuracy}
    except Exception as e:
        logger.error(f"Error getting accuracy analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pending")
async def get_pending_measurements():
    """Get decisions pending impact measurement"""
    try:
        pending = await impact_engine.get_pending_measurements()
        return {"status": "success", "pending": pending}
    except Exception as e:
        logger.error(f"Error getting pending measurements: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/summary")
async def get_impact_summary():
    """Get comprehensive impact monitoring summary"""
    try:
        summary = await impact_engine.get_impact_summary()
        return {"status": "success", "summary": summary}
    except Exception as e:
        logger.error(f"Error getting impact summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8014)