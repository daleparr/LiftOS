"""
LiftOS Bayesian Analysis Service

Standalone service for Bayesian prior analysis, conflict detection, and SBC validation.
Supports both free audit tier for prospect onboarding and paid analysis tier.

Architecture: Hybrid approach leveraging shared Bayesian framework components.
"""

import sys
import os
# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# KSE-SDK Integration
from shared.kse_sdk.client import LiftKSEClient
from shared.kse_sdk.core import Entity, SearchQuery, SearchResult
from shared.kse_sdk.models import EntityType, Domain

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
import logging
from datetime import datetime
import uuid

# Import shared Bayesian framework components
from shared.utils.bayesian_diagnostics import ConflictAnalyzer, PriorUpdater
from shared.validation.simulation_based_calibration import SBCValidator, SBCDecisionFramework
from shared.models.bayesian_priors import (
    PriorElicitationSession, PriorSpecification, ConflictReport,
    EvidenceMetrics, PriorUpdateRecommendation
)
from shared.database.bayesian_models import (
    PriorElicitationSessionDB, PriorSpecificationDB, ConflictReportDB,
    SBCValidationResultDB, EvidenceMetricsDB
)

# Service-specific models
from models.audit_models import (
    AuditRequest, ConflictPreviewRequest, AuditReportRequest,
    LeadCaptureRequest
)
from models.analysis_models import (
    ComprehensiveSBCRequest, AdvancedPriorUpdateRequest,
    EvidenceAssessmentRequest
)

# Initialize FastAPI app

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
    title="LiftOS Bayesian Analysis Service",
    description="Bayesian prior analysis, conflict detection, and SBC validation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize shared components
conflict_analyzer = ConflictAnalyzer()
prior_updater = PriorUpdater()
sbc_validator = SBCValidator()
sbc_decision_framework = SBCDecisionFramework()

# Pricing tier middleware
async def verify_paid_subscription(request):
    """Verify user has valid paid subscription for analysis endpoints"""
    # Implementation depends on your authentication/billing system
    # For now, we'll simulate this
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(401, "Valid subscription required for analysis features")
    return True

# Health check endpoint
@app.get("/health")
async def health_check():
    """Service health check"""
    return {
        "status": "healthy",
        "service": "bayesian-analysis",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "conflict_analyzer": "ready",
            "prior_updater": "ready", 
            "sbc_validator": "ready",
            "sbc_decision_framework": "ready"
        }
    }

# =============================================================================
# FREE AUDIT TIER - Prospect Onboarding
# =============================================================================

@app.post("/api/v1/audit/elicit-priors")
async def elicit_client_priors(request: AuditRequest):
    """
    Free audit: Guided prior elicitation for prospects
    Limited to 5 parameters for free tier
    """
    try:
        # Validate free tier limits
        if len(request.parameters) > 5:
            raise HTTPException(
                400, 
                "Free audit limited to 5 parameters. Upgrade for unlimited analysis."
            )
        
        # Create elicitation session
        session = await conflict_analyzer.create_elicitation_session(
            client_id=request.prospect_id,
            parameters=request.parameters[:5],
            tier="free_audit",
            contact_info=request.contact_info
        )
        
        logger.info(f"Created free audit session {session.id} for prospect {request.prospect_id}")
        
        return {
            "session_id": session.id,
            "elicitation_questions": session.questions,
            "estimated_duration": "10-15 minutes",
            "parameters_included": len(request.parameters[:5]),
            "parameters_limit_reached": len(request.parameters) > 5,
            "next_step": "conflict_detection",
            "upgrade_message": "Upgrade for unlimited parameters and advanced analysis"
        }
        
    except Exception as e:
        logger.error(f"Error in prior elicitation: {str(e)}")
        raise HTTPException(500, f"Prior elicitation failed: {str(e)}")

@app.post("/api/v1/audit/detect-conflicts")
async def detect_prior_conflicts(request: ConflictPreviewRequest):
    """
    Free audit: Basic conflict detection with sample/demo data
    Shows potential conflicts without full analysis
    """
    try:
        # Use sample data for free audit to protect client data
        conflicts = await conflict_analyzer.detect_conflicts_preview(
            priors=request.client_priors,
            sample_data=True,  # Use demo data for free tier
            detail_level="basic",  # Limited detail
            max_conflicts=3  # Show top 3 conflicts only
        )
        
        # Calculate potential impact estimate
        potential_impact = await conflict_analyzer.estimate_impact(
            conflicts=conflicts,
            business_context=request.business_context
        )
        
        logger.info(f"Detected {len(conflicts)} conflicts for session {request.session_id}")
        
        return {
            "session_id": request.session_id,
            "conflicts_detected": len(conflicts),
            "severity_summary": conflicts.severity_distribution,
            "top_conflicts": conflicts.top_conflicts,
            "potential_impact": {
                "estimated_revenue_at_risk": potential_impact.revenue_impact,
                "confidence_improvement": potential_impact.confidence_gain,
                "decision_quality_score": potential_impact.decision_quality
            },
            "upgrade_benefits": {
                "full_data_analysis": "Analyze your actual data instead of sample data",
                "unlimited_parameters": "No limits on model complexity",
                "detailed_recommendations": "Specific actions to improve your priors",
                "sbc_validation": "Ensure your models are reliable"
            },
            "next_steps": [
                "Generate audit report",
                "Schedule consultation", 
                "Upgrade to full analysis"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error in conflict detection: {str(e)}")
        raise HTTPException(500, f"Conflict detection failed: {str(e)}")

@app.get("/api/v1/audit/generate-report/{session_id}")
async def generate_audit_report(session_id: str):
    """
    Free audit: Generate summary report for prospects
    Includes key findings and upgrade path
    """
    try:
        # Generate prospect-focused report
        report = await conflict_analyzer.generate_prospect_report(
            session_id=session_id,
            include_recommendations=True,
            include_upgrade_path=True,
            include_business_impact=True
        )
        
        logger.info(f"Generated audit report for session {session_id}")
        
        return {
            "session_id": session_id,
            "report_url": f"/reports/audit/{session_id}",
            "executive_summary": report.executive_summary,
            "key_findings": report.key_findings,
            "potential_improvements": report.potential_improvements,
            "business_impact": {
                "current_risk_level": report.risk_assessment,
                "improvement_opportunity": report.improvement_potential,
                "recommended_actions": report.priority_actions
            },
            "upgrade_value_proposition": {
                "additional_insights": report.upgrade_benefits,
                "roi_estimate": report.estimated_roi,
                "implementation_timeline": report.implementation_estimate
            },
            "next_steps": [
                {
                    "action": "Schedule consultation",
                    "description": "Discuss findings with Bayesian expert",
                    "timeline": "Within 1 week"
                },
                {
                    "action": "Upgrade to full analysis", 
                    "description": "Analyze your actual data with unlimited parameters",
                    "timeline": "Immediate"
                },
                {
                    "action": "Integrate with MMM platform",
                    "description": "Full attribution analysis with Bayesian validation",
                    "timeline": "2-4 weeks"
                }
            ]
        }
        
    except Exception as e:
        logger.error(f"Error generating audit report: {str(e)}")
        raise HTTPException(500, f"Report generation failed: {str(e)}")

@app.post("/api/v1/audit/capture-lead")
async def capture_prospect_lead(request: LeadCaptureRequest):
    """
    Capture prospect information during free audit
    Integrate with CRM and marketing automation
    """
    try:
        lead_data = {
            "email": request.email,
            "company": request.company,
            "name": request.name,
            "phone": request.phone,
            "audit_session_id": request.session_id,
            "conflicts_detected": request.conflicts_summary,
            "estimated_impact": request.potential_impact,
            "interest_level": request.interest_level,
            "source": "bayesian_audit",
            "status": "qualified_lead",
            "created_at": datetime.now().isoformat(),
            "follow_up_priority": "high" if request.potential_impact > 1000000 else "medium"
        }
        
        # TODO: Integrate with actual CRM/marketing automation
        # await send_to_crm(lead_data)
        # await trigger_follow_up_sequence(lead_data)
        
        logger.info(f"Captured lead for {request.email} from session {request.session_id}")
        
        return {
            "status": "captured",
            "lead_id": str(uuid.uuid4()),
            "next_steps": "Check email for detailed audit report",
            "consultation_link": "/schedule-consultation",
            "upgrade_offer": {
                "discount": "50% off first month",
                "valid_until": "7 days",
                "upgrade_link": "/upgrade-to-analysis"
            },
            "follow_up": {
                "email_sequence": "activated",
                "consultation_scheduling": "available",
                "priority": lead_data["follow_up_priority"]
            }
        }
        
    except Exception as e:
        logger.error(f"Error capturing lead: {str(e)}")
        raise HTTPException(500, f"Lead capture failed: {str(e)}")

# =============================================================================
# PAID ANALYSIS TIER - Advanced Bayesian Features
# =============================================================================

@app.post("/api/v1/analysis/comprehensive-sbc")
async def comprehensive_sbc_analysis(
    request: ComprehensiveSBCRequest,
    _: bool = Depends(verify_paid_subscription)
):
    """
    Paid feature: Comprehensive SBC validation with detailed reporting
    No limitations on model complexity or simulation count
    """
    try:
        # Full SBC analysis with no restrictions
        sbc_result = await sbc_validator.comprehensive_validation(
            model=request.model,
            num_simulations=min(request.num_simulations, 10000),  # Cap at 10k for performance
            validation_type="comprehensive",
            include_diagnostics=True,
            include_recommendations=True,
            include_plots=True
        )
        
        # Generate detailed recommendations
        recommendations = await sbc_decision_framework.generate_recommendations(
            sbc_result=sbc_result,
            business_context=request.business_context
        )
        
        logger.info(f"Completed comprehensive SBC analysis for model {request.model.id}")
        
        return {
            "validation_id": sbc_result.id,
            "model_id": request.model.id,
            "validation_status": sbc_result.status,
            "reliability_score": sbc_result.reliability_score,
            "rank_statistics": {
                "uniformity_test": sbc_result.rank_stats.uniformity_test,
                "coverage_probability": sbc_result.rank_stats.coverage_probability,
                "calibration_score": sbc_result.rank_stats.calibration_score
            },
            "diagnostic_analysis": {
                "rank_histogram": sbc_result.diagnostics.rank_histogram,
                "coverage_plots": sbc_result.diagnostics.coverage_plots,
                "shrinkage_analysis": sbc_result.diagnostics.shrinkage_analysis
            },
            "recommendations": {
                "model_improvements": recommendations.model_improvements,
                "prior_adjustments": recommendations.prior_adjustments,
                "validation_frequency": recommendations.validation_schedule,
                "risk_mitigation": recommendations.risk_mitigation
            },
            "business_impact": {
                "reliability_improvement": recommendations.reliability_gain,
                "decision_confidence": recommendations.confidence_improvement,
                "risk_reduction": recommendations.risk_reduction
            }
        }
        
    except Exception as e:
        logger.error(f"Error in comprehensive SBC analysis: {str(e)}")
        raise HTTPException(500, f"SBC analysis failed: {str(e)}")

@app.post("/api/v1/analysis/update-priors")
async def advanced_prior_updating(
    request: AdvancedPriorUpdateRequest,
    _: bool = Depends(verify_paid_subscription)
):
    """
    Paid feature: Sophisticated prior updating with evidence weighting
    Advanced algorithms and uncertainty quantification
    """
    try:
        # Advanced prior updating with full capabilities
        update_result = await prior_updater.advanced_update(
            current_priors=request.priors,
            evidence=request.evidence,
            update_strategy=request.strategy,
            confidence_threshold=request.confidence_threshold,
            include_uncertainty_quantification=True,
            include_sensitivity_analysis=True
        )
        
        # Generate integration recommendations
        integration_advice = await prior_updater.generate_integration_recommendations(
            update_result=update_result,
            current_workflow=request.current_workflow
        )
        
        logger.info(f"Completed advanced prior updating for {len(request.priors)} priors")
        
        return {
            "update_id": update_result.id,
            "updated_priors": update_result.new_priors,
            "evidence_analysis": {
                "strength_metrics": update_result.evidence_metrics,
                "reliability_assessment": update_result.evidence_reliability,
                "conflict_resolution": update_result.conflict_resolution
            },
            "uncertainty_quantification": {
                "posterior_uncertainty": update_result.uncertainty.posterior,
                "prediction_intervals": update_result.uncertainty.prediction_intervals,
                "sensitivity_analysis": update_result.uncertainty.sensitivity
            },
            "update_justification": {
                "statistical_reasoning": update_result.reasoning.statistical,
                "business_rationale": update_result.reasoning.business,
                "risk_assessment": update_result.reasoning.risk
            },
            "integration_recommendations": {
                "implementation_steps": integration_advice.implementation_steps,
                "validation_checkpoints": integration_advice.validation_points,
                "monitoring_strategy": integration_advice.monitoring_plan,
                "rollback_plan": integration_advice.rollback_strategy
            }
        }
        
    except Exception as e:
        logger.error(f"Error in advanced prior updating: {str(e)}")
        raise HTTPException(500, f"Prior updating failed: {str(e)}")

@app.post("/api/v1/analysis/evidence-assessment")
async def assess_evidence_strength(
    request: EvidenceAssessmentRequest,
    _: bool = Depends(verify_paid_subscription)
):
    """
    Paid feature: Comprehensive evidence strength assessment
    Advanced statistical analysis of evidence quality and reliability
    """
    try:
        # Comprehensive evidence assessment
        assessment = await conflict_analyzer.assess_evidence_strength(
            evidence=request.evidence,
            priors=request.priors,
            assessment_type="comprehensive",
            include_robustness_analysis=True
        )
        
        logger.info(f"Completed evidence assessment for {len(request.evidence)} evidence items")
        
        return {
            "assessment_id": assessment.id,
            "overall_strength": assessment.overall_strength,
            "evidence_breakdown": assessment.evidence_breakdown,
            "statistical_metrics": {
                "bayes_factors": assessment.bayes_factors,
                "kl_divergences": assessment.kl_divergences,
                "wasserstein_distances": assessment.wasserstein_distances
            },
            "robustness_analysis": {
                "sensitivity_to_outliers": assessment.robustness.outlier_sensitivity,
                "assumption_dependence": assessment.robustness.assumption_dependence,
                "data_quality_impact": assessment.robustness.data_quality_impact
            },
            "recommendations": {
                "evidence_improvements": assessment.recommendations.evidence_improvements,
                "data_collection": assessment.recommendations.data_collection,
                "analysis_refinements": assessment.recommendations.analysis_refinements
            }
        }
        
    except Exception as e:
        logger.error(f"Error in evidence assessment: {str(e)}")
        raise HTTPException(500, f"Evidence assessment failed: {str(e)}")

# =============================================================================
# INTEGRATION ENDPOINTS - Cross-service communication
# =============================================================================

@app.post("/api/v1/integration/validate-attribution")
async def validate_attribution_results(request: Dict[str, Any]):
    """
    Integration endpoint for causal service
    Validates attribution results against client priors
    """
    try:
        validation_result = await conflict_analyzer.validate_attribution(
            attribution_results=request["attribution_results"],
            client_priors=request["client_priors"],
            validation_level=request.get("validation_level", "standard")
        )
        
        return {
            "validation_status": validation_result.status,
            "conflicts_detected": validation_result.conflicts,
            "recommendations": validation_result.recommendations,
            "confidence_score": validation_result.confidence_score
        }
        
    except Exception as e:
        logger.error(f"Error in attribution validation: {str(e)}")
        raise HTTPException(500, f"Attribution validation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)