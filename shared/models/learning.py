"""
Learning Models for LiftOS Intelligence Enhancement
Core data structures for learning algorithms, pattern discovery, and knowledge accumulation
"""
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum
import numpy as np


class LearningType(str, Enum):
    """Types of learning algorithms"""
    PATTERN_DISCOVERY = "pattern_discovery"
    CAUSAL_LEARNING = "causal_learning"
    PERFORMANCE_LEARNING = "performance_learning"
    USER_BEHAVIOR_LEARNING = "user_behavior_learning"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    TRANSFER_LEARNING = "transfer_learning"


class PatternType(str, Enum):
    """Types of patterns that can be discovered"""
    SEASONAL = "seasonal"
    TREND = "trend"
    ANOMALY = "anomaly"
    CORRELATION = "correlation"
    CAUSAL = "causal"
    CYCLICAL = "cyclical"
    THRESHOLD = "threshold"


class LearningStatus(str, Enum):
    """Status of learning processes"""
    INITIALIZING = "initializing"
    TRAINING = "training"
    VALIDATING = "validating"
    DEPLOYED = "deployed"
    UPDATING = "updating"
    FAILED = "failed"
    DEPRECATED = "deprecated"


class Pattern(BaseModel):
    """Discovered pattern in data"""
    id: str
    pattern_type: PatternType
    name: str
    description: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    strength: float = Field(..., ge=0.0, le=1.0)
    variables: List[str]
    parameters: Dict[str, Any]
    time_range: Tuple[datetime, datetime]
    frequency: Optional[str] = None  # daily, weekly, monthly, etc.
    context: Dict[str, Any] = {}
    evidence: List[str] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_validated: Optional[datetime] = None
    validation_score: Optional[float] = None


class LearningModel(BaseModel):
    """Machine learning model metadata and configuration"""
    id: str
    name: str
    learning_type: LearningType
    algorithm: str  # random_forest, neural_network, etc.
    version: str
    status: LearningStatus
    performance_metrics: Dict[str, float] = {}
    hyperparameters: Dict[str, Any] = {}
    training_data_info: Dict[str, Any] = {}
    feature_importance: Dict[str, float] = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_trained: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    deployment_date: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class LearningExperiment(BaseModel):
    """Learning experiment tracking"""
    id: str
    name: str
    description: str
    learning_type: LearningType
    hypothesis: str
    models: List[str]  # Model IDs
    datasets: List[str]  # Dataset IDs
    metrics: Dict[str, float] = {}
    parameters: Dict[str, Any] = {}
    results: Dict[str, Any] = {}
    status: str = "running"
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    conclusions: List[str] = []


class KnowledgeItem(BaseModel):
    """Individual piece of learned knowledge"""
    id: str
    type: str  # insight, rule, pattern, relationship
    content: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    source: str  # learning_model_id or pattern_id
    domain: str  # marketing, causal, performance, etc.
    tags: List[str] = []
    dependencies: List[str] = []  # Other knowledge items this depends on
    applications: List[str] = []  # Where this knowledge can be applied
    evidence: List[str] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_validated: Optional[datetime] = None
    validation_count: int = 0
    success_rate: Optional[float] = None


class LearningOutcome(BaseModel):
    """Result of a learning process"""
    id: str
    learning_model_id: str
    experiment_id: Optional[str] = None
    patterns_discovered: List[str] = []  # Pattern IDs
    knowledge_items: List[str] = []  # KnowledgeItem IDs
    performance_improvement: Optional[float] = None
    insights: List[str] = []
    recommendations: List[str] = []
    confidence: float = Field(..., ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = {}


class CausalLearningResult(BaseModel):
    """Results from causal learning algorithms"""
    id: str
    causal_graph_updates: Dict[str, Any] = {}
    new_relationships: List[Dict[str, Any]] = []
    relationship_updates: List[Dict[str, Any]] = []
    confounders_discovered: List[str] = []
    treatment_effects: Dict[str, float] = {}
    confidence_intervals: Dict[str, Tuple[float, float]] = {}
    statistical_tests: Dict[str, Dict[str, Any]] = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PerformanceLearningResult(BaseModel):
    """Results from performance learning algorithms"""
    id: str
    performance_patterns: List[str] = []  # Pattern IDs
    optimization_opportunities: List[Dict[str, Any]] = []
    benchmark_comparisons: Dict[str, float] = {}
    efficiency_metrics: Dict[str, float] = {}
    resource_utilization: Dict[str, float] = {}
    bottleneck_analysis: List[Dict[str, Any]] = []
    recommendations: List[str] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)


class UserBehaviorLearningResult(BaseModel):
    """Results from user behavior learning"""
    id: str
    behavior_patterns: List[str] = []  # Pattern IDs
    user_segments: List[Dict[str, Any]] = []
    preference_models: Dict[str, Any] = {}
    workflow_optimizations: List[Dict[str, Any]] = []
    personalization_rules: List[Dict[str, Any]] = []
    engagement_insights: Dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)


class LearningRequest(BaseModel):
    """Request to initiate learning process"""
    learning_type: LearningType
    name: str
    description: Optional[str] = None
    data_sources: List[str]
    target_variables: List[str] = []
    parameters: Dict[str, Any] = {}
    experiment_config: Optional[Dict[str, Any]] = None
    user_id: str
    organization_id: str


class LearningResponse(BaseModel):
    """Response from learning process"""
    learning_id: str
    status: LearningStatus
    progress: float = Field(..., ge=0.0, le=1.0)
    estimated_completion: Optional[datetime] = None
    preliminary_results: Optional[Dict[str, Any]] = None
    message: str = "Learning process initiated"


class CompoundLearning(BaseModel):
    """Learning that builds on previous learning"""
    id: str
    name: str
    base_learning_ids: List[str]  # Previous learning this builds on
    synthesis_method: str
    compound_insights: List[str] = []
    emergent_patterns: List[str] = []
    meta_knowledge: Dict[str, Any] = {}
    complexity_level: int = Field(..., ge=1, le=10)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class LearningMetrics(BaseModel):
    """Metrics for evaluating learning effectiveness"""
    learning_id: str
    accuracy_metrics: Dict[str, float] = {}
    precision_metrics: Dict[str, float] = {}
    recall_metrics: Dict[str, float] = {}
    f1_scores: Dict[str, float] = {}
    auc_scores: Dict[str, float] = {}
    custom_metrics: Dict[str, float] = {}
    validation_results: Dict[str, Any] = {}
    cross_validation_scores: List[float] = []
    feature_importance_stability: Optional[float] = None
    model_interpretability_score: Optional[float] = None
    computational_efficiency: Dict[str, float] = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AdaptiveLearning(BaseModel):
    """Learning that adapts to changing conditions"""
    id: str
    base_model_id: str
    adaptation_triggers: List[str] = []
    adaptation_history: List[Dict[str, Any]] = []
    current_adaptation_level: float = Field(..., ge=0.0, le=1.0)
    adaptation_strategy: str
    performance_tracking: Dict[str, List[float]] = {}
    drift_detection: Dict[str, Any] = {}
    retraining_schedule: Optional[str] = None
    last_adaptation: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)