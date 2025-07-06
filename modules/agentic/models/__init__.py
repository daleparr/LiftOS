"""
Agentic Module Data Models

This package contains all data models for the Agentic microservice,
including agent configurations, evaluation results, and test cases.
"""

from .agent_models import (
    MarketingAgent,
    MarketingAgentType,
    AgentCapability,
    ModelConfig,
    MarketingContext
)

from .evaluation_models import (
    AgentEvaluationResult,
    CategoryAssessment,
    MetricScore,
    MarketingMetrics,
    DeploymentReadiness,
    EvaluationMatrix,
    EvaluationCategory
)

from .test_models import (
    MarketingTestCase,
    TestStep,
    TestResult,
    MarketingTestData,
    SuccessCriteria,
    TestScenario
)

__all__ = [
    # Agent Models
    "MarketingAgent",
    "MarketingAgentType", 
    "AgentCapability",
    "ModelConfig",
    "MarketingContext",
    
    # Evaluation Models
    "AgentEvaluationResult",
    "CategoryAssessment",
    "MetricScore",
    "MarketingMetrics",
    "DeploymentReadiness",
    "EvaluationMatrix",
    "EvaluationCategory",
    
    # Test Models
    "MarketingTestCase",
    "TestStep",
    "TestResult",
    "MarketingTestData",
    "SuccessCriteria",
    "TestScenario"
]