"""
LiftOS Business Metrics Service
Comprehensive business intelligence and KPI tracking service.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager

# Import business models
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# KSE-SDK Integration
from shared.kse_sdk.client import LiftKSEClient
from shared.kse_sdk.core import Entity, SearchQuery, SearchResult
from shared.kse_sdk.models import EntityType, Domain

from shared.models.business import (
    BusinessMetric, RevenueMetrics, CustomerMetrics, OperationalMetrics,
    ROIMetrics, BusinessKPI, BusinessGoal, BusinessImpactAssessment,
    CompetitiveIntelligence, MarketIntelligence, BusinessAlert,
    MetricType, MetricFrequency
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BusinessMetricsEngine:
    """Core business metrics calculation and tracking engine"""
    
    def __init__(self):
        self.metrics_store: Dict[str, BusinessMetric] = {}
        self.kpis_store: Dict[str, BusinessKPI] = {}
        self.goals_store: Dict[str, BusinessGoal] = {}
        self.alerts_store: Dict[str, BusinessAlert] = {}
        self.revenue_history: List[RevenueMetrics] = []
        self.customer_history: List[CustomerMetrics] = []
        self.operational_history: List[OperationalMetrics] = []
        
    async def calculate_revenue_metrics(self, data: Dict[str, Any]) -> RevenueMetrics:
        """Calculate comprehensive revenue metrics"""
        try:
            # Extract revenue data
            total_revenue = data.get('total_revenue', 0.0)
            previous_revenue = data.get('previous_revenue', 0.0)
            user_count = data.get('user_count', 1)
            decision_count = data.get('decision_count', 1)
            
            # Calculate growth rate
            revenue_growth_rate = 0.0
            if previous_revenue > 0:
                revenue_growth_rate = ((total_revenue - previous_revenue) / previous_revenue) * 100
            
            # Calculate per-unit metrics
            revenue_per_user = total_revenue / max(user_count, 1)
            revenue_per_decision = total_revenue / max(decision_count, 1)
            
            # LiftOS attribution (using ML-based attribution model)
            liftos_attributed_revenue = await self._calculate_liftos_attribution(data)
            attribution_confidence = await self._calculate_attribution_confidence(data)
            
            revenue_metrics = RevenueMetrics(
                total_revenue=total_revenue,
                revenue_growth_rate=revenue_growth_rate,
                revenue_per_user=revenue_per_user,
                revenue_per_decision=revenue_per_decision,
                recurring_revenue=data.get('recurring_revenue', 0.0),
                new_revenue=data.get('new_revenue', 0.0),
                lost_revenue=data.get('lost_revenue', 0.0),
                liftos_attributed_revenue=liftos_attributed_revenue,
                attribution_confidence=attribution_confidence,
                revenue_by_product=data.get('revenue_by_product', {}),
                revenue_by_channel=data.get('revenue_by_channel', {}),
                revenue_by_segment=data.get('revenue_by_segment', {})
            )
            
            self.revenue_history.append(revenue_metrics)
            logger.info(f"Calculated revenue metrics: ${total_revenue:,.2f} total, {revenue_growth_rate:.1f}% growth")
            return revenue_metrics
            
        except Exception as e:
            logger.error(f"Error calculating revenue metrics: {e}")
            raise
    
    async def calculate_customer_metrics(self, data: Dict[str, Any]) -> CustomerMetrics:
        """Calculate comprehensive customer metrics"""
        try:
            total_customers = data.get('total_customers', 0)
            new_customers = data.get('new_customers', 0)
            churned_customers = data.get('churned_customers', 0)
            
            # Calculate retention rate
            retention_rate = 1.0
            if total_customers > 0:
                retention_rate = max(0.0, 1.0 - (churned_customers / total_customers))
            
            # Calculate customer value metrics
            total_revenue = data.get('total_revenue', 0.0)
            acquisition_cost = data.get('acquisition_cost', 0.0)
            
            customer_lifetime_value = 0.0
            if total_customers > 0:
                customer_lifetime_value = total_revenue / total_customers
            
            customer_acquisition_cost = 0.0
            if new_customers > 0:
                customer_acquisition_cost = acquisition_cost / new_customers
            
            average_order_value = data.get('average_order_value', 0.0)
            
            customer_metrics = CustomerMetrics(
                total_customers=total_customers,
                new_customers=new_customers,
                churned_customers=churned_customers,
                retention_rate=retention_rate,
                customer_lifetime_value=customer_lifetime_value,
                customer_acquisition_cost=customer_acquisition_cost,
                average_order_value=average_order_value,
                net_promoter_score=data.get('net_promoter_score', 0.0),
                customer_satisfaction_score=data.get('customer_satisfaction_score', 5.0),
                support_ticket_volume=data.get('support_ticket_volume', 0),
                customers_by_segment=data.get('customers_by_segment', {}),
                value_by_segment=data.get('value_by_segment', {})
            )
            
            self.customer_history.append(customer_metrics)
            logger.info(f"Calculated customer metrics: {total_customers} customers, {retention_rate:.1%} retention")
            return customer_metrics
            
        except Exception as e:
            logger.error(f"Error calculating customer metrics: {e}")
            raise
    
    async def calculate_operational_metrics(self, data: Dict[str, Any]) -> OperationalMetrics:
        """Calculate operational efficiency metrics"""
        try:
            decision_volume = data.get('decision_volume', 0)
            correct_decisions = data.get('correct_decisions', 0)
            total_decision_time = data.get('total_decision_time', 0.0)
            automated_decisions = data.get('automated_decisions', 0)
            
            # Calculate accuracy and speed
            decision_accuracy = 0.0
            if decision_volume > 0:
                decision_accuracy = correct_decisions / decision_volume
            
            decision_speed = 0.0
            if decision_volume > 0:
                decision_speed = total_decision_time / decision_volume
            
            automation_rate = 0.0
            if decision_volume > 0:
                automation_rate = automated_decisions / decision_volume
            
            # Calculate efficiency metrics
            total_cost = data.get('total_cost', 0.0)
            cost_per_decision = 0.0
            if decision_volume > 0:
                cost_per_decision = total_cost / decision_volume
            
            operational_metrics = OperationalMetrics(
                decision_volume=decision_volume,
                decision_accuracy=decision_accuracy,
                decision_speed=decision_speed,
                automation_rate=automation_rate,
                cost_per_decision=cost_per_decision,
                time_savings=data.get('time_savings', 0.0),
                error_reduction=data.get('error_reduction', 0.0),
                process_efficiency=data.get('process_efficiency', 0.0),
                data_quality_score=data.get('data_quality_score', 1.0),
                system_uptime=data.get('system_uptime', 1.0),
                user_adoption_rate=data.get('user_adoption_rate', 0.0)
            )
            
            self.operational_history.append(operational_metrics)
            logger.info(f"Calculated operational metrics: {decision_volume} decisions, {decision_accuracy:.1%} accuracy")
            return operational_metrics
            
        except Exception as e:
            logger.error(f"Error calculating operational metrics: {e}")
            raise
    
    async def calculate_roi_metrics(self, data: Dict[str, Any]) -> ROIMetrics:
        """Calculate return on investment metrics"""
        try:
            total_investment = data.get('total_investment', 0.0)
            total_return = data.get('total_return', 0.0)
            
            # Calculate ROI percentage
            roi_percentage = 0.0
            if total_investment > 0:
                roi_percentage = ((total_return - total_investment) / total_investment) * 100
            
            # Calculate payback period (simplified)
            monthly_return = data.get('monthly_return', 0.0)
            payback_period = 0.0
            if monthly_return > 0:
                payback_period = total_investment / monthly_return
            
            roi_metrics = ROIMetrics(
                total_investment=total_investment,
                total_return=total_return,
                roi_percentage=roi_percentage,
                payback_period=payback_period,
                roi_by_feature=data.get('roi_by_feature', {}),
                roi_by_department=data.get('roi_by_department', {}),
                roi_by_use_case=data.get('roi_by_use_case', {}),
                short_term_roi=data.get('short_term_roi', 0.0),
                medium_term_roi=data.get('medium_term_roi', 0.0),
                long_term_roi=data.get('long_term_roi', 0.0)
            )
            
            logger.info(f"Calculated ROI metrics: {roi_percentage:.1f}% ROI, {payback_period:.1f} month payback")
            return roi_metrics
            
        except Exception as e:
            logger.error(f"Error calculating ROI metrics: {e}")
            raise
    
    async def _calculate_liftos_attribution(self, data: Dict[str, Any]) -> float:
        """Calculate revenue attribution to LiftOS using ML models"""
        try:
            # Simplified attribution model - in production, use ML attribution
            total_revenue = data.get('total_revenue', 0.0)
            liftos_decisions = data.get('liftos_decisions', 0)
            total_decisions = data.get('total_decisions', 1)
            
            # Attribution based on decision influence
            decision_influence = liftos_decisions / max(total_decisions, 1)
            
            # Apply attribution model (simplified)
            attribution_factor = min(0.8, decision_influence * 1.2)  # Cap at 80%
            
            return total_revenue * attribution_factor
            
        except Exception as e:
            logger.error(f"Error calculating LiftOS attribution: {e}")
            return 0.0
    
    async def _calculate_attribution_confidence(self, data: Dict[str, Any]) -> float:
        """Calculate confidence in attribution model"""
        try:
            # Factors affecting confidence
            data_quality = data.get('data_quality_score', 1.0)
            sample_size = data.get('sample_size', 0)
            time_period = data.get('time_period_days', 30)
            
            # Base confidence from data quality
            confidence = data_quality
            
            # Adjust for sample size
            if sample_size > 1000:
                confidence *= 1.0
            elif sample_size > 100:
                confidence *= 0.9
            else:
                confidence *= 0.7
            
            # Adjust for time period
            if time_period >= 30:
                confidence *= 1.0
            elif time_period >= 7:
                confidence *= 0.8
            else:
                confidence *= 0.6
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.error(f"Error calculating attribution confidence: {e}")
            return 0.5
    
    async def create_kpi(self, kpi_data: Dict[str, Any]) -> BusinessKPI:
        """Create a new business KPI"""
        try:
            # Calculate performance percentage
            current_value = kpi_data.get('current_value', 0.0)
            target_value = kpi_data.get('target_value', 1.0)
            
            performance_percentage = 0.0
            if target_value != 0:
                performance_percentage = (current_value / target_value) * 100
            
            # Determine trend (simplified)
            trend = "stable"
            if performance_percentage > 100:
                trend = "improving"
            elif performance_percentage < 80:
                trend = "declining"
            
            kpi = BusinessKPI(
                name=kpi_data['name'],
                description=kpi_data.get('description', ''),
                current_value=current_value,
                target_value=target_value,
                unit=kpi_data.get('unit', ''),
                performance_percentage=performance_percentage,
                trend=trend,
                critical_threshold=kpi_data.get('critical_threshold', target_value * 0.5),
                warning_threshold=kpi_data.get('warning_threshold', target_value * 0.8),
                excellent_threshold=kpi_data.get('excellent_threshold', target_value * 1.2),
                owner=kpi_data.get('owner', 'system'),
                review_frequency=MetricFrequency(kpi_data.get('review_frequency', 'daily'))
            )
            
            self.kpis_store[kpi.id] = kpi
            logger.info(f"Created KPI: {kpi.name} ({performance_percentage:.1f}% of target)")
            return kpi
            
        except Exception as e:
            logger.error(f"Error creating KPI: {e}")
            raise
    
    async def update_kpi(self, kpi_id: str, new_value: float) -> BusinessKPI:
        """Update KPI with new value"""
        try:
            if kpi_id not in self.kpis_store:
                raise ValueError(f"KPI {kpi_id} not found")
            
            kpi = self.kpis_store[kpi_id]
            old_performance = kpi.performance_percentage
            
            # Update value and recalculate performance
            kpi.current_value = new_value
            if kpi.target_value != 0:
                kpi.performance_percentage = (new_value / kpi.target_value) * 100
            
            # Update trend
            if kpi.performance_percentage > old_performance:
                kpi.trend = "improving"
            elif kpi.performance_percentage < old_performance:
                kpi.trend = "declining"
            else:
                kpi.trend = "stable"
            
            kpi.last_updated = datetime.utcnow()
            
            # Check for alerts
            await self._check_kpi_alerts(kpi)
            
            logger.info(f"Updated KPI {kpi.name}: {new_value} ({kpi.performance_percentage:.1f}% of target)")
            return kpi
            
        except Exception as e:
            logger.error(f"Error updating KPI: {e}")
            raise
    
    async def _check_kpi_alerts(self, kpi: BusinessKPI):
        """Check if KPI triggers any alerts"""
        try:
            # Check critical threshold
            if kpi.current_value <= kpi.critical_threshold:
                await self._trigger_alert(f"CRITICAL: {kpi.name} below critical threshold", "critical")
            elif kpi.current_value <= kpi.warning_threshold:
                await self._trigger_alert(f"WARNING: {kpi.name} below warning threshold", "medium")
            elif kpi.current_value >= kpi.excellent_threshold:
                await self._trigger_alert(f"EXCELLENT: {kpi.name} exceeds excellent threshold", "low")
                
        except Exception as e:
            logger.error(f"Error checking KPI alerts: {e}")
    
    async def _trigger_alert(self, message: str, severity: str):
        """Trigger a business alert"""
        try:
            logger.warning(f"Business Alert [{severity.upper()}]: {message}")
            # In production, send to notification system
            
        except Exception as e:
            logger.error(f"Error triggering alert: {e}")
    
    async def get_business_summary(self) -> Dict[str, Any]:
        """Get comprehensive business metrics summary"""
        try:
            # Get latest metrics
            latest_revenue = self.revenue_history[-1] if self.revenue_history else None
            latest_customer = self.customer_history[-1] if self.customer_history else None
            latest_operational = self.operational_history[-1] if self.operational_history else None
            
            # Calculate overall health score
            health_score = await self._calculate_business_health()
            
            summary = {
                "timestamp": datetime.utcnow().isoformat(),
                "business_health_score": health_score,
                "revenue_metrics": latest_revenue.dict() if latest_revenue else None,
                "customer_metrics": latest_customer.dict() if latest_customer else None,
                "operational_metrics": latest_operational.dict() if latest_operational else None,
                "active_kpis": len(self.kpis_store),
                "active_goals": len(self.goals_store),
                "active_alerts": len([a for a in self.alerts_store.values() if a.is_active])
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating business summary: {e}")
            raise
    
    async def _calculate_business_health(self) -> float:
        """Calculate overall business health score"""
        try:
            if not self.kpis_store:
                return 0.5  # Neutral score if no KPIs
            
            # Average KPI performance
            total_performance = sum(kpi.performance_percentage for kpi in self.kpis_store.values())
            avg_performance = total_performance / len(self.kpis_store)
            
            # Normalize to 0-1 scale
            health_score = min(1.0, avg_performance / 100.0)
            
            return health_score
            
        except Exception as e:
            logger.error(f"Error calculating business health: {e}")
            return 0.5

# Initialize the business metrics engine
business_engine = BusinessMetricsEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("Starting Business Metrics Service...")
    
    # Initialize default KPIs
    await business_engine.create_kpi({
        "name": "Revenue Growth Rate",
        "description": "Month-over-month revenue growth percentage",
        "current_value": 0.0,
        "target_value": 15.0,
        "unit": "percentage",
        "owner": "finance",
        "review_frequency": "monthly"
    })
    
    await business_engine.create_kpi({
        "name": "Customer Retention Rate",
        "description": "Percentage of customers retained",
        "current_value": 85.0,
        "target_value": 90.0,
        "unit": "percentage",
        "owner": "customer_success",
        "review_frequency": "monthly"
    })
    
    await business_engine.create_kpi({
        "name": "Decision Accuracy",
        "description": "Accuracy of LiftOS decisions",
        "current_value": 82.0,
        "target_value": 85.0,
        "unit": "percentage",
        "owner": "product",
        "review_frequency": "daily"
    })
    
    logger.info("Business Metrics Service started successfully")
    yield
    logger.info("Business Metrics Service shutting down...")

# Create FastAPI app
app = FastAPI(
    title="LiftOS Business Metrics Service",
    description="Comprehensive business intelligence and KPI tracking",
    version="1.0.0",
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
        "service": "business-metrics",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/metrics/revenue")
async def calculate_revenue(data: Dict[str, Any]):
    """Calculate revenue metrics"""
    try:
        metrics = await business_engine.calculate_revenue_metrics(data)
        return {"status": "success", "metrics": metrics.dict()}
    except Exception as e:
        logger.error(f"Error calculating revenue metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/metrics/customer")
async def calculate_customer(data: Dict[str, Any]):
    """Calculate customer metrics"""
    try:
        metrics = await business_engine.calculate_customer_metrics(data)
        return {"status": "success", "metrics": metrics.dict()}
    except Exception as e:
        logger.error(f"Error calculating customer metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/metrics/operational")
async def calculate_operational(data: Dict[str, Any]):
    """Calculate operational metrics"""
    try:
        metrics = await business_engine.calculate_operational_metrics(data)
        return {"status": "success", "metrics": metrics.dict()}
    except Exception as e:
        logger.error(f"Error calculating operational metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/metrics/roi")
async def calculate_roi(data: Dict[str, Any]):
    """Calculate ROI metrics"""
    try:
        metrics = await business_engine.calculate_roi_metrics(data)
        return {"status": "success", "metrics": metrics.dict()}
    except Exception as e:
        logger.error(f"Error calculating ROI metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/kpis")
async def create_kpi(kpi_data: Dict[str, Any]):
    """Create a new KPI"""
    try:
        kpi = await business_engine.create_kpi(kpi_data)
        return {"status": "success", "kpi": kpi.dict()}
    except Exception as e:
        logger.error(f"Error creating KPI: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/kpis/{kpi_id}")
async def update_kpi(kpi_id: str, data: Dict[str, Any]):
    """Update KPI value"""
    try:
        new_value = data.get('value')
        if new_value is None:
            raise HTTPException(status_code=400, detail="Value is required")
        
        kpi = await business_engine.update_kpi(kpi_id, new_value)
        return {"status": "success", "kpi": kpi.dict()}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating KPI: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/kpis")
async def get_kpis():
    """Get all KPIs"""
    try:
        kpis = [kpi.dict() for kpi in business_engine.kpis_store.values()]
        return {"status": "success", "kpis": kpis}
    except Exception as e:
        logger.error(f"Error getting KPIs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/summary")
async def get_business_summary():
    """Get comprehensive business summary"""
    try:
        summary = await business_engine.get_business_summary()
        return {"status": "success", "summary": summary}
    except Exception as e:
        logger.error(f"Error getting business summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8012)