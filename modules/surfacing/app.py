"""
Lift OS Core - Surfacing Module
Product Analysis and Surfacing Capabilities
"""
import time
import uuid
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Header, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import asyncio

# Import shared modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.models.base import APIResponse, HealthCheck
from shared.utils.logging import setup_logging

# Import causal models and utilities
from shared.models.causal_marketing import (
    CausalMarketingData, CausalExperiment, CausalInsight,
    AttributionModel, ConfounderVariable, ExternalFactor,
    CausalOptimizationRequest, CausalOptimizationResponse,
    TreatmentRecommendationRequest, TreatmentRecommendationResponse,
    ExperimentDesignRequest, ExperimentDesignResponse
)
from shared.utils.causal_transforms import CausalDataTransformer

# Module configuration
MODULE_NAME = "surfacing"
MODULE_VERSION = "1.0.0"
MODULE_PORT = 9005

# Surfacing service configuration
SURFACING_SERVICE_URL = os.getenv("SURFACING_SERVICE_URL", "http://surfacing-service:3002")
CAUSAL_SERVICE_URL = os.getenv("CAUSAL_SERVICE_URL", "http://causal:8007")
MEMORY_SERVICE_URL = os.getenv("MEMORY_SERVICE_URL", "http://memory:8003")

# Setup logging
logger = setup_logging(MODULE_NAME)

# Initialize causal transformer
causal_transformer = None

# FastAPI app
app = FastAPI(
    title=f"Lift Module - {MODULE_NAME.title()}",
    description="Lift OS Core Module: Product Analysis and Surfacing",
    version=MODULE_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTTP client for surfacing service communication
http_client = httpx.AsyncClient(timeout=60.0)

# Memory service client
memory_service_url = MEMORY_SERVICE_URL



# Causal Optimization Classes

class CausalOptimizationEngine:
    """Engine for causal-based marketing optimization"""
    
    def __init__(self, causal_transformer=None):
        self.causal_transformer = causal_transformer
    
    async def optimize_based_on_causal_effects(self, causal_data: CausalMarketingData, optimization_goals: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize marketing parameters based on causal effects"""
        try:
            # Get causal analysis results
            causal_results = await self._get_causal_analysis(causal_data.experiment_id)
            
            if not causal_results:
                return {"error": "No causal analysis available for optimization"}
            
            treatment_effects = causal_results.get("treatment_effects", {})
            confounders = causal_results.get("confounders", [])
            
            # Generate optimization recommendations
            recommendations = []
            
            # Analyze treatment effects for optimization opportunities
            for treatment, effect_data in treatment_effects.items():
                effect_size = effect_data.get("effect_size", 0)
                confidence = effect_data.get("confidence", 0)
                
                if confidence > 0.8:  # High confidence threshold
                    if effect_size > 0:
                        recommendations.append({
                            "action": "increase",
                            "parameter": treatment,
                            "effect_size": effect_size,
                            "confidence": confidence,
                            "reasoning": f"Strong positive causal effect detected with {confidence:.1%} confidence"
                        })
                    elif effect_size < -0.1:  # Significant negative effect
                        recommendations.append({
                            "action": "decrease",
                            "parameter": treatment,
                            "effect_size": effect_size,
                            "confidence": confidence,
                            "reasoning": f"Negative causal effect detected with {confidence:.1%} confidence"
                        })
            
            # Account for confounders in recommendations
            confounder_adjustments = []
            for confounder in confounders:
                if confounder.get("impact_strength", 0) > 0.5:
                    confounder_adjustments.append({
                        "confounder": confounder.get("variable_name"),
                        "adjustment": "Monitor and control for this variable",
                        "impact": confounder.get("impact_strength")
                    })
            
            # Calculate expected ROI improvement
            expected_roi_improvement = self._calculate_expected_roi_improvement(recommendations)
            
            return {
                "optimization_recommendations": recommendations,
                "confounder_adjustments": confounder_adjustments,
                "expected_roi_improvement": expected_roi_improvement,
                "confidence_level": self._calculate_overall_confidence(recommendations),
                "optimization_strategy": self._generate_optimization_strategy(recommendations, optimization_goals)
            }
            
        except Exception as e:
            logger.error(f"Causal optimization failed: {e}")
            return {"error": f"Optimization failed: {str(e)}"}
    
    async def _get_causal_analysis(self, experiment_id: str) -> Dict[str, Any]:
        """Get causal analysis results from causal service"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{CAUSAL_SERVICE_URL}/api/v1/causal/experiments/{experiment_id}",
                    timeout=30.0
                )
                if response.status_code == 200:
                    return response.json().get("analysis_results", {})
                return {}
        except Exception as e:
            logger.error(f"Failed to get causal analysis: {e}")
            return {}
    
    def _calculate_expected_roi_improvement(self, recommendations: List[Dict[str, Any]]) -> float:
        """Calculate expected ROI improvement from recommendations"""
        total_improvement = 0.0
        for rec in recommendations:
            effect_size = abs(rec.get("effect_size", 0))
            confidence = rec.get("confidence", 0)
            # Weight by confidence and effect size
            total_improvement += effect_size * confidence
        
        return min(total_improvement * 100, 50.0)  # Cap at 50% improvement
    
    def _calculate_overall_confidence(self, recommendations: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence in optimization recommendations"""
        if not recommendations:
            return 0.0
        
        confidences = [rec.get("confidence", 0) for rec in recommendations]
        return sum(confidences) / len(confidences)
    
    def _generate_optimization_strategy(self, recommendations: List[Dict[str, Any]], goals: Dict[str, Any]) -> str:
        """Generate a strategic optimization plan"""
        if not recommendations:
            return "No clear optimization opportunities identified from causal analysis."
        
        strategy_parts = []
        
        # Prioritize by effect size and confidence
        high_impact = [r for r in recommendations if r.get("effect_size", 0) > 0.2 and r.get("confidence", 0) > 0.8]
        medium_impact = [r for r in recommendations if r.get("effect_size", 0) > 0.1 and r.get("confidence", 0) > 0.7]
        
        if high_impact:
            strategy_parts.append(f"Priority 1: Focus on {len(high_impact)} high-impact changes with strong causal evidence.")
        
        if medium_impact:
            strategy_parts.append(f"Priority 2: Implement {len(medium_impact)} medium-impact optimizations.")
        
        strategy_parts.append("Monitor results and iterate based on new causal evidence.")
        
        return " ".join(strategy_parts)

class TreatmentRecommendationEngine:
    """Engine for recommending marketing treatments based on causal analysis"""
    
    def __init__(self, causal_transformer=None):
        self.causal_transformer = causal_transformer
    
    async def recommend_treatments(self, current_performance: Dict[str, Any], goals: Dict[str, Any], constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """Recommend marketing treatments based on causal insights"""
        try:
            # Get historical causal data for similar scenarios
            similar_experiments = await self._find_similar_experiments(current_performance)
            
            if not similar_experiments:
                return self._generate_default_recommendations(current_performance, goals)
            
            # Analyze successful treatments from similar experiments
            successful_treatments = []
            for experiment in similar_experiments:
                treatments = experiment.get("successful_treatments", [])
                for treatment in treatments:
                    if treatment.get("effect_size", 0) > 0.1 and treatment.get("confidence", 0) > 0.7:
                        successful_treatments.append(treatment)
            
            # Generate recommendations based on causal evidence
            recommendations = []
            
            # Budget optimization recommendations
            budget_treatments = [t for t in successful_treatments if "budget" in t.get("treatment_type", "").lower()]
            if budget_treatments and goals.get("optimize_budget", False):
                best_budget_treatment = max(budget_treatments, key=lambda x: x.get("effect_size", 0))
                recommendations.append({
                    "treatment_type": "budget_optimization",
                    "recommendation": f"Adjust budget allocation based on causal evidence",
                    "expected_effect": best_budget_treatment.get("effect_size", 0),
                    "confidence": best_budget_treatment.get("confidence", 0),
                    "implementation": "Gradual budget reallocation over 2-week period"
                })
            
            # Targeting optimization recommendations
            targeting_treatments = [t for t in successful_treatments if "targeting" in t.get("treatment_type", "").lower()]
            if targeting_treatments and goals.get("optimize_targeting", False):
                best_targeting_treatment = max(targeting_treatments, key=lambda x: x.get("effect_size", 0))
                recommendations.append({
                    "treatment_type": "targeting_optimization",
                    "recommendation": f"Refine audience targeting based on causal insights",
                    "expected_effect": best_targeting_treatment.get("effect_size", 0),
                    "confidence": best_targeting_treatment.get("confidence", 0),
                    "implementation": "A/B test new targeting parameters"
                })
            
            # Creative optimization recommendations
            creative_treatments = [t for t in successful_treatments if "creative" in t.get("treatment_type", "").lower()]
            if creative_treatments and goals.get("optimize_creative", False):
                best_creative_treatment = max(creative_treatments, key=lambda x: x.get("effect_size", 0))
                recommendations.append({
                    "treatment_type": "creative_optimization",
                    "recommendation": f"Update creative elements based on causal analysis",
                    "expected_effect": best_creative_treatment.get("effect_size", 0),
                    "confidence": best_creative_treatment.get("confidence", 0),
                    "implementation": "Test new creative variations"
                })
            
            return {
                "recommendations": recommendations,
                "evidence_base": len(similar_experiments),
                "overall_confidence": self._calculate_recommendation_confidence(recommendations),
                "implementation_timeline": self._generate_implementation_timeline(recommendations),
                "risk_assessment": self._assess_recommendation_risks(recommendations, constraints)
            }
            
        except Exception as e:
            logger.error(f"Treatment recommendation failed: {e}")
            return {"error": f"Recommendation failed: {str(e)}"}
    
    async def _find_similar_experiments(self, current_performance: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find similar historical experiments for reference"""
        try:
            # Search memory for similar performance patterns
            search_query = f"causal experiment performance {current_performance.get('platform', '')} {current_performance.get('campaign_type', '')}"
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{MEMORY_SERVICE_URL}/api/v1/marketing/causal/search",
                    json={
                        "query": search_query,
                        "filters": {
                            "platform": current_performance.get("platform"),
                            "campaign_type": current_performance.get("campaign_type")
                        },
                        "limit": 10
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    return response.json().get("results", [])
                return []
                
        except Exception as e:
            logger.error(f"Failed to find similar experiments: {e}")
            return []
    
    def _generate_default_recommendations(self, current_performance: Dict[str, Any], goals: Dict[str, Any]) -> Dict[str, Any]:
        """Generate default recommendations when no causal evidence is available"""
        recommendations = [
            {
                "treatment_type": "baseline_testing",
                "recommendation": "Establish baseline measurements for future causal analysis",
                "expected_effect": 0.0,
                "confidence": 1.0,
                "implementation": "Set up proper measurement and control groups"
            }
        ]
        
        return {
            "recommendations": recommendations,
            "evidence_base": 0,
            "overall_confidence": 0.5,
            "implementation_timeline": "2-4 weeks for baseline establishment",
            "risk_assessment": "Low risk - establishing measurement foundation"
        }
    
    def _calculate_recommendation_confidence(self, recommendations: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence in recommendations"""
        if not recommendations:
            return 0.0
        
        confidences = [rec.get("confidence", 0) for rec in recommendations]
        return sum(confidences) / len(confidences)
    
    def _generate_implementation_timeline(self, recommendations: List[Dict[str, Any]]) -> str:
        """Generate implementation timeline for recommendations"""
        if not recommendations:
            return "No recommendations to implement"
        
        timeline_map = {
            "budget_optimization": "1-2 weeks",
            "targeting_optimization": "2-3 weeks", 
            "creative_optimization": "3-4 weeks",
            "baseline_testing": "2-4 weeks"
        }
        
        timelines = []
        for rec in recommendations:
            treatment_type = rec.get("treatment_type", "unknown")
            timeline = timeline_map.get(treatment_type, "2-3 weeks")
            timelines.append(f"{treatment_type}: {timeline}")
        
        return "; ".join(timelines)
    
    def _assess_recommendation_risks(self, recommendations: List[Dict[str, Any]], constraints: Dict[str, Any] = None) -> str:
        """Assess risks associated with recommendations"""
        if not recommendations:
            return "No risks - no recommendations"
        
        risk_factors = []
        
        for rec in recommendations:
            confidence = rec.get("confidence", 0)
            if confidence < 0.7:
                risk_factors.append(f"Low confidence in {rec.get('treatment_type', 'unknown')} recommendation")
        
        if constraints:
            budget_limit = constraints.get("budget_limit", float('inf'))
            if any("budget" in rec.get("treatment_type", "") for rec in recommendations):
                if budget_limit < 1000:  # Arbitrary threshold
                    risk_factors.append("Budget constraints may limit optimization effectiveness")
        
        if not risk_factors:
            return "Low risk - high confidence recommendations with adequate resources"
        
        return "; ".join(risk_factors)

class ExperimentDesigner:
    """Design causal experiments for marketing optimization"""
    
    def __init__(self, causal_transformer=None):
        self.causal_transformer = causal_transformer
    
    async def design_experiment(self, objective: str, current_setup: Dict[str, Any], constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """Design a causal experiment for marketing optimization"""
        try:
            # Determine experiment type based on objective
            experiment_type = self._determine_experiment_type(objective)
            
            # Design treatment and control groups
            treatment_design = self._design_treatment_groups(objective, current_setup)
            
            # Calculate required sample size
            sample_size = self._calculate_sample_size(objective, current_setup)
            
            # Identify potential confounders
            confounders = self._identify_potential_confounders(current_setup)
            
            # Generate randomization strategy
            randomization = self._design_randomization_strategy(treatment_design, constraints)
            
            # Create measurement plan
            measurement_plan = self._create_measurement_plan(objective)
            
            return {
                "experiment_design": {
                    "type": experiment_type,
                    "objective": objective,
                    "treatment_groups": treatment_design,
                    "control_group": self._design_control_group(current_setup),
                    "randomization_strategy": randomization,
                    "sample_size": sample_size,
                    "duration": self._calculate_experiment_duration(sample_size, current_setup),
                    "power_analysis": self._perform_power_analysis(sample_size, objective)
                },
                "confounders": confounders,
                "measurement_plan": measurement_plan,
                "success_criteria": self._define_success_criteria(objective),
                "implementation_checklist": self._create_implementation_checklist(experiment_type),
                "risk_mitigation": self._identify_risks_and_mitigation(experiment_type, constraints)
            }
            
        except Exception as e:
            logger.error(f"Experiment design failed: {e}")
            return {"error": f"Experiment design failed: {str(e)}"}
    
    def _determine_experiment_type(self, objective: str) -> str:
        """Determine the most appropriate experiment type"""
        objective_lower = objective.lower()
        
        if "budget" in objective_lower or "spend" in objective_lower:
            return "budget_allocation_experiment"
        elif "targeting" in objective_lower or "audience" in objective_lower:
            return "audience_targeting_experiment"
        elif "creative" in objective_lower or "ad" in objective_lower:
            return "creative_optimization_experiment"
        elif "bidding" in objective_lower or "bid" in objective_lower:
            return "bidding_strategy_experiment"
        else:
            return "general_optimization_experiment"
    
    def _design_treatment_groups(self, objective: str, current_setup: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Design treatment groups for the experiment"""
        experiment_type = self._determine_experiment_type(objective)
        
        if experiment_type == "budget_allocation_experiment":
            return [
                {"name": "increased_budget", "budget_multiplier": 1.2, "description": "20% budget increase"},
                {"name": "decreased_budget", "budget_multiplier": 0.8, "description": "20% budget decrease"},
                {"name": "reallocated_budget", "reallocation": "shift_to_top_performers", "description": "Reallocate to best performing segments"}
            ]
        elif experiment_type == "audience_targeting_experiment":
            return [
                {"name": "broader_targeting", "audience_expansion": 1.5, "description": "50% broader audience"},
                {"name": "narrower_targeting", "audience_expansion": 0.7, "description": "30% more focused audience"},
                {"name": "lookalike_targeting", "targeting_type": "lookalike", "description": "Lookalike audience based on converters"}
            ]
        elif experiment_type == "creative_optimization_experiment":
            return [
                {"name": "variant_a", "creative_type": "image_focused", "description": "Image-heavy creative"},
                {"name": "variant_b", "creative_type": "text_focused", "description": "Text-heavy creative"},
                {"name": "variant_c", "creative_type": "video_focused", "description": "Video creative"}
            ]
        else:
            return [
                {"name": "treatment_1", "modification": "optimization_1", "description": "Primary optimization approach"},
                {"name": "treatment_2", "modification": "optimization_2", "description": "Alternative optimization approach"}
            ]
    
    def _design_control_group(self, current_setup: Dict[str, Any]) -> Dict[str, Any]:
        """Design the control group"""
        return {
            "name": "control",
            "description": "Current setup with no modifications",
            "configuration": current_setup,
            "size_percentage": 0.2  # 20% of traffic
        }
    
    def _calculate_sample_size(self, objective: str, current_setup: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate required sample size for statistical power"""
        # Simplified sample size calculation
        baseline_conversion_rate = current_setup.get("conversion_rate", 0.02)
        minimum_detectable_effect = 0.1  # 10% relative improvement
        
        # Using simplified formula for sample size calculation
        # In practice, this would use more sophisticated statistical methods
        estimated_sample_size = int(16 * baseline_conversion_rate * (1 - baseline_conversion_rate) / (minimum_detectable_effect ** 2))
        
        return {
            "total_sample_size": estimated_sample_size,
            "per_group_size": estimated_sample_size // 4,  # Assuming 4 groups (3 treatments + 1 control)
            "minimum_detectable_effect": minimum_detectable_effect,
            "statistical_power": 0.8,
            "significance_level": 0.05
        }
    
    def _identify_potential_confounders(self, current_setup: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify potential confounding variables"""
        confounders = [
            {"variable": "time_of_day", "type": "temporal", "mitigation": "randomize across time periods"},
            {"variable": "day_of_week", "type": "temporal", "mitigation": "ensure equal distribution across days"},
            {"variable": "seasonality", "type": "temporal", "mitigation": "account for seasonal trends"},
            {"variable": "competitor_activity", "type": "external", "mitigation": "monitor and adjust for competitor changes"},
            {"variable": "economic_indicators", "type": "external", "mitigation": "track relevant economic metrics"}
        ]
        
        # Add platform-specific confounders
        platform = current_setup.get("platform", "").lower()
        if platform == "facebook" or platform == "meta":
            confounders.extend([
                {"variable": "ios_14_impact", "type": "platform", "mitigation": "separate analysis for iOS vs Android"},
                {"variable": "algorithm_changes", "type": "platform", "mitigation": "monitor platform announcements"}
            ])
        elif platform == "google":
            confounders.extend([
                {"variable": "quality_score_changes", "type": "platform", "mitigation": "monitor quality scores"},
                {"variable": "auction_dynamics", "type": "platform", "mitigation": "track competitive metrics"}
            ])
        
        return confounders
    
    def _design_randomization_strategy(self, treatment_design: List[Dict[str, Any]], constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """Design randomization strategy for the experiment"""
        return {
            "method": "stratified_randomization",
            "stratification_variables": ["geographic_region", "device_type", "user_segment"],
            "allocation_ratio": "equal",  # Equal allocation across groups
            "randomization_unit": "user_id",
            "blocking_variables": ["historical_performance_tier"],
            "implementation": "Use hash-based randomization for consistent assignment"
        }
    
    def _calculate_experiment_duration(self, sample_size: Dict[str, Any], current_setup: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal experiment duration"""
        daily_traffic = current_setup.get("daily_impressions", 10000)
        required_sample = sample_size.get("total_sample_size", 1000)
        
        estimated_days = max(7, int(required_sample / daily_traffic))  # Minimum 1 week
        
        return {
            "estimated_duration_days": estimated_days,
            "minimum_duration_days": 7,
            "recommended_duration_days": max(14, estimated_days),  # Minimum 2 weeks recommended
            "factors": [
                "Statistical power requirements",
                "Business cycle considerations", 
                "Seasonal effects",
                "Learning period for algorithms"
            ]
        }
    
    def _perform_power_analysis(self, sample_size: Dict[str, Any], objective: str) -> Dict[str, Any]:
        """Perform statistical power analysis"""
        return {
            "statistical_power": sample_size.get("statistical_power", 0.8),
            "significance_level": sample_size.get("significance_level", 0.05),
            "minimum_detectable_effect": sample_size.get("minimum_detectable_effect", 0.1),
            "type_i_error_rate": 0.05,
            "type_ii_error_rate": 0.2,
            "confidence_interval": "95%"
        }
    
    def _create_measurement_plan(self, objective: str) -> Dict[str, Any]:
        """Create comprehensive measurement plan"""
        return {
            "primary_metrics": self._define_primary_metrics(objective),
            "secondary_metrics": self._define_secondary_metrics(objective),
            "measurement_frequency": "daily",
            "data_collection_methods": [
                "Platform APIs",
                "Google Analytics",
                "Custom tracking pixels",
                "Server-side tracking"
            ],
            "quality_checks": [
                "Data completeness validation",
                "Outlier detection",
                "Randomization balance checks",
                "Sample ratio mismatch detection"
            ]
        }
    
    def _define_primary_metrics(self, objective: str) -> List[str]:
        """Define primary metrics based on objective"""
        objective_lower = objective.lower()
        
        if "conversion" in objective_lower:
            return ["conversion_rate", "cost_per_conversion"]
        elif "revenue" in objective_lower or "roas" in objective_lower:
            return ["revenue", "return_on_ad_spend"]
        elif "engagement" in objective_lower:
            return ["click_through_rate", "engagement_rate"]
        else:
            return ["conversion_rate", "cost_per_conversion", "return_on_ad_spend"]
    
    def _define_secondary_metrics(self, objective: str) -> List[str]:
        """Define secondary metrics for comprehensive analysis"""
        return [
            "impressions",
            "clicks", 
            "cost_per_click",
            "frequency",
            "reach",
            "brand_awareness_lift",
            "customer_lifetime_value"
        ]
    
    def _define_success_criteria(self, objective: str) -> Dict[str, Any]:
        """Define clear success criteria for the experiment"""
        return {
            "primary_success_threshold": "10% improvement in primary metric",
            "statistical_significance": "p-value < 0.05",
            "practical_significance": "Effect size > 0.2",
            "business_significance": "ROI improvement > 15%",
            "decision_framework": "Require both statistical and practical significance for implementation"
        }
    
    def _create_implementation_checklist(self, experiment_type: str) -> List[str]:
        """Create implementation checklist"""
        base_checklist = [
            "Set up tracking and measurement infrastructure",
            "Configure randomization system",
            "Implement treatment variations",
            "Set up monitoring and alerting",
            "Train team on experiment protocols",
            "Prepare analysis plan and scripts",
            "Set up regular review meetings",
            "Document experiment design and rationale"
        ]
        
        type_specific = {
            "budget_allocation_experiment": [
                "Configure budget allocation rules",
                "Set up automated budget adjustments",
                "Implement budget monitoring alerts"
            ],
            "audience_targeting_experiment": [
                "Create audience segments",
                "Set up audience exclusions",
                "Configure lookalike audiences"
            ],
            "creative_optimization_experiment": [
                "Prepare creative variations",
                "Set up creative rotation rules",
                "Implement creative performance tracking"
            ]
        }
        
        return base_checklist + type_specific.get(experiment_type, [])
    
    def _identify_risks_and_mitigation(self, experiment_type: str, constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """Identify risks and mitigation strategies"""
        risks = {
            "statistical_risks": [
                {"risk": "Insufficient sample size", "mitigation": "Monitor power analysis and extend duration if needed"},
                {"risk": "Multiple testing issues", "mitigation": "Apply Bonferroni correction or FDR control"},
                {"risk": "Selection bias", "mitigation": "Ensure proper randomization and monitor balance"}
            ],
            "business_risks": [
                {"risk": "Revenue impact during test", "mitigation": "Limit test to small percentage of traffic initially"},
                {"risk": "Customer experience degradation", "mitigation": "Monitor customer satisfaction metrics"},
                {"risk": "Competitive disadvantage", "mitigation": "Keep test duration minimal while maintaining statistical power"}
            ],
            "technical_risks": [
                {"risk": "Tracking failures", "mitigation": "Implement redundant tracking and monitoring"},
                {"risk": "Platform API changes", "mitigation": "Monitor platform announcements and have backup plans"},
                {"risk": "Data quality issues", "mitigation": "Implement automated data quality checks"}
            ]
        }
        
        return risks

# Initialize causal optimization components
causal_optimization_engine = None
treatment_recommendation_engine = None
experiment_designer = None

async def get_causal_service_data(endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get data from the causal service"""
    try:
        async with httpx.AsyncClient() as client:
            if params:
                response = await client.post(f"{CAUSAL_SERVICE_URL}{endpoint}", json=params, timeout=30.0)
            else:
                response = await client.get(f"{CAUSAL_SERVICE_URL}{endpoint}", timeout=30.0)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Failed to get causal service data: {e}")
        return {}

class ProductAnalysisRequest(BaseModel):
    url: Optional[str] = None
    product_data: Optional[Dict[str, Any]] = None
    analysis_type: str = "comprehensive"
    include_hybrid_analysis: bool = True
    include_knowledge_graph: bool = True
    optimization_level: str = "standard"


class BatchAnalysisRequest(BaseModel):
    products: List[Dict[str, Any]]
    analysis_id: Optional[str] = None
    optimization_level: str = "standard"


class OptimizeRequest(BaseModel):
    product_data: Dict[str, Any]
    analysis_id: Optional[str] = None
    optimization_level: str = "standard"


def get_user_context(
    x_user_id: Optional[str] = Header(None),
    x_org_id: Optional[str] = Header(None),
    x_memory_context: Optional[str] = Header(None),
    x_user_roles: Optional[str] = Header(None),
    x_module_id: Optional[str] = Header(None)
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
        "memory_context": x_memory_context,
        "roles": x_user_roles.split(",") if x_user_roles else [],
        "module_id": x_module_id
    }


async def search_memory(query: str, search_type: str = "hybrid", user_context: Dict[str, Any] = None):
    """Search memory using the memory service"""
    try:
        headers = {
            "X-User-ID": user_context["user_id"],
            "X-Org-ID": user_context["org_id"],
            "X-Memory-Context": user_context["memory_context"]
        }
        
        response = await http_client.post(
            f"{memory_service_url}/search",
            json={
                "query": query,
                "search_type": search_type,
                "limit": 10
            },
            headers=headers
        )
        
        if response.status_code == 200:
            return response.json().get("data", {}).get("results", [])
        else:
            return []
            
    except Exception as e:
        logger.error(f"Memory search failed: {str(e)}")
        return []


async def store_memory(content: str, memory_type: str = "general", metadata: Dict = None, user_context: Dict[str, Any] = None):
    """Store content in memory using the memory service"""
    try:
        headers = {
            "X-User-ID": user_context["user_id"],
            "X-Org-ID": user_context["org_id"],
            "X-Memory-Context": user_context["memory_context"]
        }
        
        response = await http_client.post(
            f"{memory_service_url}/store",
            json={
                "content": content,
                "memory_type": memory_type,
                "metadata": metadata or {}
            },
            headers=headers
        )
        
        if response.status_code == 200:
            return response.json().get("data", {}).get("memory_id")
        else:
            return None
            
    except Exception as e:
        logger.error(f"Memory storage failed: {str(e)}")
        return None


async def call_surfacing_service(endpoint: str, method: str = "POST", data: Dict = None):
    """Call the Node.js surfacing service"""
    try:
        url = f"{SURFACING_SERVICE_URL}{endpoint}"
        
        if method.upper() == "POST":
            response = await http_client.post(url, json=data)
        elif method.upper() == "GET":
            response = await http_client.get(url, params=data)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Surfacing service error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Surfacing service call failed: {str(e)}")
        return None


@app.get("/health")
async def health_check():
    """Module health check"""
    # Check surfacing service health
    surfacing_health = "unknown"
    try:
        response = await http_client.get(f"{SURFACING_SERVICE_URL}/health", timeout=5.0)
        surfacing_health = "healthy" if response.status_code == 200 else "unhealthy"
    except Exception:
        surfacing_health = "unreachable"
    
    return HealthCheck(
        status="healthy" if surfacing_health in ["healthy", "unknown"] else "degraded",
        dependencies={
            "surfacing_service": surfacing_health
        },
        timestamp=datetime.utcnow(),
        version=MODULE_VERSION,
        uptime=time.time() - getattr(app.state, "start_time", time.time())
    )


@app.get("/")
async def root():
    """Module root endpoint"""
    return APIResponse(
        message=f"Lift Module: {MODULE_NAME}",
        data={
            "name": MODULE_NAME,
            "version": MODULE_VERSION,
            "features": [
                "product_analysis",
                "batch_analysis", 
                "optimization",
                "memory_integration",
                "hybrid_analysis",
                "knowledge_graph"
            ],
            "docs": "/docs"
        }
    )


@app.get("/api/v1/info")
async def get_module_info(user_context: Dict[str, Any] = Depends(get_user_context)):
    """Get module information"""
    return APIResponse(
        message="Surfacing module information",
        data={
            "name": MODULE_NAME,
            "version": MODULE_VERSION,
            "user_context": user_context,
            "capabilities": [
                "product_analysis",
                "batch_analysis",
                "optimization",
                "memory_integration",
                "user_context_aware",
                "health_monitoring"
            ],
            "surfacing_service_url": SURFACING_SERVICE_URL
        }
    )


@app.post("/api/v1/analyze")
async def analyze_product(
    request: ProductAnalysisRequest,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Analyze a single product"""
    correlation_id = str(uuid.uuid4())
    
    try:
        # Search memory for similar products
        search_query = ""
        if request.url:
            search_query = f"product analysis {request.url}"
        elif request.product_data and request.product_data.get("title"):
            search_query = f"product analysis {request.product_data['title']}"
        
        memory_results = []
        if search_query:
            memory_results = await search_memory(
                query=search_query,
                search_type="hybrid",
                user_context=user_context
            )
        
        # Prepare request for surfacing service
        surfacing_request = {
            "url": request.url,
            "productData": request.product_data,
            "options": {
                "includeHybridAnalysis": request.include_hybrid_analysis,
                "includeKnowledgeGraph": request.include_knowledge_graph,
                "analysisDepth": request.analysis_type,
                "optimizationLevel": request.optimization_level
            }
        }
        
        # Call surfacing service
        surfacing_result = await call_surfacing_service("/api/analyze", "POST", surfacing_request)
        
        if not surfacing_result:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Surfacing service unavailable"
            )
        
        # Enhance result with memory context
        enhanced_result = {
            "analysis": surfacing_result,
            "memory_context": {
                "similar_products": len(memory_results),
                "related_analyses": memory_results[:3]
            },
            "correlation_id": correlation_id,
            "processed_at": datetime.utcnow().isoformat()
        }
        
        # Store analysis in memory
        memory_content = f"Product analysis completed for {request.url or 'product data'}"
        if surfacing_result.get("analysis", {}).get("summary"):
            memory_content += f": {surfacing_result['analysis']['summary']}"
        
        await store_memory(
            content=memory_content,
            memory_type="surfacing_analysis",
            metadata={
                "correlation_id": correlation_id,
                "analysis_type": request.analysis_type,
                "url": request.url,
                "has_hybrid_analysis": request.include_hybrid_analysis,
                "has_knowledge_graph": request.include_knowledge_graph
            },
            user_context=user_context
        )
        
        return APIResponse(
            message="Product analysis completed successfully",
            data=enhanced_result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Product analysis failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


@app.post("/api/v1/batch-analyze")
async def batch_analyze_products(
    request: BatchAnalysisRequest,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Analyze multiple products in batch"""
    correlation_id = request.analysis_id or str(uuid.uuid4())
    
    try:
        # Prepare request for surfacing service
        surfacing_request = {
            "products": request.products,
            "analysisId": correlation_id,
            "optimizationLevel": request.optimization_level
        }
        
        # Call surfacing service
        surfacing_result = await call_surfacing_service("/api/batch-analyze", "POST", surfacing_request)
        
        if not surfacing_result:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Surfacing service unavailable"
            )
        
        # Store batch analysis in memory
        await store_memory(
            content=f"Batch analysis completed for {len(request.products)} products",
            memory_type="surfacing_batch_analysis",
            metadata={
                "correlation_id": correlation_id,
                "product_count": len(request.products),
                "optimization_level": request.optimization_level
            },
            user_context=user_context
        )
        
        return APIResponse(
            message="Batch analysis completed successfully",
            data={
                "analysis": surfacing_result,
                "correlation_id": correlation_id,
                "processed_at": datetime.utcnow().isoformat(),
                "product_count": len(request.products)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch analysis failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch analysis failed: {str(e)}"
        )


@app.post("/api/v1/optimize")
async def optimize_product(
    request: OptimizeRequest,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Optimize product data"""
    correlation_id = request.analysis_id or str(uuid.uuid4())
    
    try:
        # Search memory for optimization patterns
        memory_results = await search_memory(
            query=f"product optimization {request.product_data.get('title', '')}",
            search_type="hybrid",
            user_context=user_context
        )
        
        # Prepare request for surfacing service
        surfacing_request = {
            "productData": request.product_data,
            "analysisId": correlation_id,
            "optimizationLevel": request.optimization_level
        }
        
        # Call surfacing service
        surfacing_result = await call_surfacing_service("/api/optimize", "POST", surfacing_request)
        
        if not surfacing_result:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Surfacing service unavailable"
            )
        
        # Store optimization in memory
        await store_memory(
            content=f"Product optimization completed for {request.product_data.get('title', 'product')}",
            memory_type="surfacing_optimization",
            metadata={
                "correlation_id": correlation_id,
                "optimization_level": request.optimization_level,
                "product_title": request.product_data.get("title")
            },
            user_context=user_context
        )
        
        return APIResponse(
            message="Product optimization completed successfully",
            data={
                "optimization": surfacing_result,
                "memory_context": {
                    "related_optimizations": len(memory_results)
                },
                "correlation_id": correlation_id,
                "processed_at": datetime.utcnow().isoformat()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Product optimization failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization failed: {str(e)}"
        )


@app.get("/api/v1/memory/search")
async def search_surfacing_memory(
    query: str,
    search_type: str = "hybrid",
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Search surfacing-related memory"""
    try:
        results = await search_memory(
            query=f"surfacing {query}",
            search_type=search_type,
            user_context=user_context
        )
        
        return APIResponse(
            message="Memory search completed",
            data={
                "query": query,
                "search_type": search_type,
                "results": results
            }
        )
        
    except Exception as e:
        logger.error(f"Memory search failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Memory search failed: {str(e)}"
        )


@app.on_event("startup")
async def startup_event():
    """Initialize module on startup"""
    global causal_transformer, causal_optimization_engine, treatment_recommendation_engine, experiment_designer
    
    app.state.start_time = time.time()
    logger.info(f"Surfacing module started on port {MODULE_PORT}")
    
    # Initialize causal transformer
    try:
        causal_transformer = CausalDataTransformer()
        logger.info("Causal data transformer initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize causal transformer: {e}")
        causal_transformer = None
    
    # Initialize causal optimization components
    try:
        causal_optimization_engine = CausalOptimizationEngine(causal_transformer)
        treatment_recommendation_engine = TreatmentRecommendationEngine(causal_transformer)
        experiment_designer = ExperimentDesigner(causal_transformer)
        logger.info("Causal optimization components initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize causal optimization components: {e}")
    
    # Test surfacing service connection
    try:
        response = await http_client.get(f"{SURFACING_SERVICE_URL}/health", timeout=5.0)
        if response.status_code == 200:
            logger.info("Successfully connected to surfacing service")
        else:
            logger.warning(f"Surfacing service health check failed: {response.status_code}")
    except Exception as e:
        logger.warning(f"Could not connect to surfacing service: {str(e)}")


# Causal Optimization API Endpoints

@app.post("/api/v1/surfacing/optimize/causal")
async def optimize_with_causal_analysis(
    request: CausalOptimizationRequest,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Optimize marketing campaigns based on causal analysis"""
    try:
        if not causal_optimization_engine:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Causal optimization engine not available"
            )
        
        # Get causal data from memory or causal service
        causal_data = None
        if request.experiment_id:
            causal_results = await get_causal_service_data(f"/api/v1/causal/experiments/{request.experiment_id}")
            if causal_results:
                causal_data = CausalMarketingData(**causal_results.get("causal_data", {}))
        
        if not causal_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Causal experiment data not found"
            )
        
        # Perform causal optimization
        optimization_results = await causal_optimization_engine.optimize_based_on_causal_effects(
            causal_data, 
            request.optimization_goals
        )
        
        if "error" in optimization_results:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=optimization_results["error"]
            )
        
        # Store optimization results in memory
        await store_memory(
            content=f"Causal optimization completed for experiment {request.experiment_id}",
            memory_type="causal_optimization",
            metadata={
                "experiment_id": request.experiment_id,
                "optimization_goals": request.optimization_goals,
                "recommendations_count": len(optimization_results.get("optimization_recommendations", [])),
                "expected_roi_improvement": optimization_results.get("expected_roi_improvement", 0)
            },
            user_context=user_context
        )
        
        return CausalOptimizationResponse(
            experiment_id=request.experiment_id,
            optimization_recommendations=optimization_results.get("optimization_recommendations", []),
            confounder_adjustments=optimization_results.get("confounder_adjustments", []),
            expected_roi_improvement=optimization_results.get("expected_roi_improvement", 0),
            confidence_level=optimization_results.get("confidence_level", 0),
            optimization_strategy=optimization_results.get("optimization_strategy", ""),
            metadata={
                "optimized_at": datetime.utcnow().isoformat(),
                "optimization_goals": request.optimization_goals,
                "causal_evidence_strength": "high" if optimization_results.get("confidence_level", 0) > 0.8 else "medium"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Causal optimization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Causal optimization failed: {str(e)}"
        )

@app.post("/api/v1/surfacing/recommendations/treatments")
async def recommend_marketing_treatments(
    request: TreatmentRecommendationRequest,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Recommend marketing treatments based on causal insights"""
    try:
        if not treatment_recommendation_engine:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Treatment recommendation engine not available"
            )
        
        # Generate treatment recommendations
        recommendations = await treatment_recommendation_engine.recommend_treatments(
            request.current_performance,
            request.goals,
            request.constraints
        )
        
        if "error" in recommendations:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=recommendations["error"]
            )
        
        # Store recommendations in memory
        await store_memory(
            content=f"Treatment recommendations generated for {request.current_performance.get('campaign_name', 'campaign')}",
            memory_type="treatment_recommendations",
            metadata={
                "campaign_name": request.current_performance.get("campaign_name"),
                "platform": request.current_performance.get("platform"),
                "recommendations_count": len(recommendations.get("recommendations", [])),
                "evidence_base": recommendations.get("evidence_base", 0),
                "overall_confidence": recommendations.get("overall_confidence", 0)
            },
            user_context=user_context
        )
        
        return TreatmentRecommendationResponse(
            recommendations=recommendations.get("recommendations", []),
            evidence_base=recommendations.get("evidence_base", 0),
            overall_confidence=recommendations.get("overall_confidence", 0),
            implementation_timeline=recommendations.get("implementation_timeline", ""),
            risk_assessment=recommendations.get("risk_assessment", ""),
            metadata={
                "generated_at": datetime.utcnow().isoformat(),
                "current_performance": request.current_performance,
                "goals": request.goals
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Treatment recommendation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Treatment recommendation failed: {str(e)}"
        )

@app.post("/api/v1/surfacing/experiments/design")
async def design_causal_experiment(
    request: ExperimentDesignRequest,
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Design a causal experiment for marketing optimization"""
    try:
        if not experiment_designer:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Experiment designer not available"
            )
        
        # Design the experiment
        experiment_design = await experiment_designer.design_experiment(
            request.objective,
            request.current_setup,
            request.constraints
        )
        
        if "error" in experiment_design:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=experiment_design["error"]
            )
        
        # Store experiment design in memory
        await store_memory(
            content=f"Experiment design created for objective: {request.objective}",
            memory_type="experiment_design",
            metadata={
                "objective": request.objective,
                "experiment_type": experiment_design.get("experiment_design", {}).get("type"),
                "estimated_duration": experiment_design.get("experiment_design", {}).get("duration", {}).get("estimated_duration_days"),
                "sample_size": experiment_design.get("experiment_design", {}).get("sample_size", {}).get("total_sample_size"),
                "treatment_groups": len(experiment_design.get("experiment_design", {}).get("treatment_groups", []))
            },
            user_context=user_context
        )
        
        return ExperimentDesignResponse(
            experiment_design=experiment_design.get("experiment_design", {}),
            confounders=experiment_design.get("confounders", []),
            measurement_plan=experiment_design.get("measurement_plan", {}),
            success_criteria=experiment_design.get("success_criteria", {}),
            implementation_checklist=experiment_design.get("implementation_checklist", []),
            risk_mitigation=experiment_design.get("risk_mitigation", {}),
            metadata={
                "designed_at": datetime.utcnow().isoformat(),
                "objective": request.objective,
                "designer_version": "1.0.0"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Experiment design failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Experiment design failed: {str(e)}"
        )

@app.get("/api/v1/surfacing/causal/capabilities")
async def get_causal_capabilities(
    user_context: Dict[str, Any] = Depends(get_user_context)
):
    """Get available causal optimization capabilities"""
    try:
        capabilities = {
            "optimization_engine": {
                "available": causal_optimization_engine is not None,
                "features": [
                    "causal_effect_analysis",
                    "confounder_adjustment",
                    "roi_improvement_estimation",
                    "optimization_strategy_generation"
                ]
            },
            "treatment_recommendation": {
                "available": treatment_recommendation_engine is not None,
                "features": [
                    "historical_evidence_analysis",
                    "treatment_effect_prediction",
                    "implementation_timeline_planning",
                    "risk_assessment"
                ]
            },
            "experiment_design": {
                "available": experiment_designer is not None,
                "features": [
                    "causal_experiment_design",
                    "sample_size_calculation",
                    "randomization_strategy",
                    "confounder_identification",
                    "power_analysis"
                ]
            },
            "supported_platforms": ["facebook", "google", "klaviyo", "tiktok", "linkedin"],
            "supported_objectives": [
                "conversion_optimization",
                "revenue_maximization",
                "cost_reduction",
                "audience_expansion",
                "creative_optimization",
                "budget_allocation"
            ]
        }
        
        return APIResponse(
            message="Causal optimization capabilities",
            data=capabilities
        )
        
    except Exception as e:
        logger.error(f"Failed to get causal capabilities: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get capabilities: {str(e)}"
        )


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await http_client.aclose()
    logger.info(f"Surfacing module stopped")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=MODULE_PORT,
        reload=True
    )