"""
Test Models for Agentic Module

Defines data structures for test cases, scenarios, and results
for marketing agent testing and validation.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator, ConfigDict
from datetime import datetime


class TestStatus(str, Enum):
    """Status of test execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TestPriority(str, Enum):
    """Priority levels for test cases."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MarketingScenarioType(str, Enum):
    """Types of marketing scenarios for testing."""
    ATTRIBUTION_ANALYSIS = "attribution_analysis"
    BUDGET_OPTIMIZATION = "budget_optimization"
    CAMPAIGN_PERFORMANCE = "campaign_performance"
    AUDIENCE_SEGMENTATION = "audience_segmentation"
    CREATIVE_TESTING = "creative_testing"
    CHANNEL_MIX = "channel_mix"
    FORECASTING = "forecasting"
    INCREMENTALITY = "incrementality"
    ROI_ANALYSIS = "roi_analysis"
    COMPETITIVE_ANALYSIS = "competitive_analysis"


class SuccessCriteria(BaseModel):
    """Success criteria for test validation."""
    metric_name: str = Field(..., description="Name of the metric to evaluate")
    operator: str = Field(..., description="Comparison operator (>=, <=, ==, !=)")
    threshold: float = Field(..., description="Threshold value for success")
    weight: float = Field(default=1.0, ge=0.0, description="Weight of this criteria")
    description: Optional[str] = Field(None, description="Description of the criteria")
    
    @validator('operator')
    def validate_operator(cls, v):
        """Validate operator is supported."""
        valid_operators = ['>=', '<=', '==', '!=', '>', '<']
        if v not in valid_operators:
            raise ValueError(f"Operator must be one of {valid_operators}")
        return v


class MarketingTestData(BaseModel):
    """Test data for marketing scenarios."""
    # Campaign data
    campaign_data: Optional[Dict[str, Any]] = Field(None, description="Campaign performance data")
    spend_data: Optional[Dict[str, Any]] = Field(None, description="Media spend data")
    conversion_data: Optional[Dict[str, Any]] = Field(None, description="Conversion data")
    
    # Channel data
    channel_performance: Optional[Dict[str, Any]] = Field(None, description="Channel performance metrics")
    attribution_data: Optional[Dict[str, Any]] = Field(None, description="Attribution model data")
    
    # Audience data
    audience_segments: Optional[Dict[str, Any]] = Field(None, description="Audience segmentation data")
    demographic_data: Optional[Dict[str, Any]] = Field(None, description="Demographic information")
    
    # Creative data
    creative_assets: Optional[Dict[str, Any]] = Field(None, description="Creative asset data")
    creative_performance: Optional[Dict[str, Any]] = Field(None, description="Creative performance metrics")
    
    # External data
    market_data: Optional[Dict[str, Any]] = Field(None, description="Market condition data")
    competitor_data: Optional[Dict[str, Any]] = Field(None, description="Competitor analysis data")
    
    # Metadata
    data_period: Optional[str] = Field(None, description="Time period for the data")
    data_source: Optional[str] = Field(None, description="Source of the test data")
    data_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Data quality assessment")


class TestStep(BaseModel):
    """Individual step in a test case."""
    step_id: str = Field(..., description="Unique step identifier")
    name: str = Field(..., description="Step name")
    description: str = Field(..., description="Step description")
    action: str = Field(..., description="Action to perform")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Step parameters")
    expected_output: Optional[Dict[str, Any]] = Field(None, description="Expected output")
    timeout_seconds: int = Field(default=60, gt=0, description="Step timeout")
    retry_count: int = Field(default=0, ge=0, description="Number of retries allowed")
    depends_on: List[str] = Field(default_factory=list, description="Dependencies on other steps")


class TestResult(BaseModel):
    """Result of test execution."""
    test_id: str = Field(..., description="Test case identifier")
    status: TestStatus = Field(..., description="Test execution status")
    
    # Execution details
    start_time: datetime = Field(..., description="Test start time")
    end_time: Optional[datetime] = Field(None, description="Test end time")
    duration_seconds: Optional[float] = Field(None, ge=0.0, description="Test duration")
    
    # Results
    success_criteria_met: List[bool] = Field(default_factory=list, description="Success criteria results")
    overall_success: bool = Field(default=False, description="Overall test success")
    score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Test score")
    
    # Step results
    step_results: List[Dict[str, Any]] = Field(default_factory=list, description="Individual step results")
    failed_steps: List[str] = Field(default_factory=list, description="Failed step IDs")
    
    # Output data
    agent_outputs: Dict[str, Any] = Field(default_factory=dict, description="Agent outputs")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Measured metrics")
    
    # Error handling
    errors: List[str] = Field(default_factory=list, description="Error messages")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    
    # Cost tracking
    execution_cost: Optional[float] = Field(None, ge=0.0, description="Execution cost in USD")
    token_usage: Optional[Dict[str, int]] = Field(None, description="Token usage breakdown")
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate from criteria results."""
        if not self.success_criteria_met:
            return 0.0
        return sum(self.success_criteria_met) / len(self.success_criteria_met)


class TestScenario(BaseModel):
    """Marketing test scenario configuration."""
    scenario_id: str = Field(..., description="Unique scenario identifier")
    name: str = Field(..., description="Scenario name")
    description: str = Field(..., description="Scenario description")
    scenario_type: MarketingScenarioType = Field(..., description="Type of marketing scenario")
    
    # Configuration
    complexity_level: str = Field(default="medium", description="Scenario complexity (low/medium/high)")
    estimated_duration: int = Field(default=300, gt=0, description="Estimated duration in seconds")
    
    # Test data
    test_data: MarketingTestData = Field(..., description="Test data for the scenario")
    
    # Validation
    success_criteria: List[SuccessCriteria] = Field(..., description="Success criteria")
    validation_rules: List[str] = Field(default_factory=list, description="Additional validation rules")
    
    # Metadata
    tags: List[str] = Field(default_factory=list, description="Scenario tags")
    industry: Optional[str] = Field(None, description="Target industry")
    business_model: Optional[str] = Field(None, description="Target business model")
    
    def calculate_complexity_score(self) -> float:
        """Calculate complexity score based on scenario characteristics."""
        base_score = {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.9
        }.get(self.complexity_level, 0.6)
        
        # Adjust based on number of success criteria
        criteria_factor = min(len(self.success_criteria) * 0.1, 0.3)
        
        # Adjust based on data complexity
        data_complexity = 0.0
        if self.test_data.campaign_data:
            data_complexity += 0.1
        if self.test_data.attribution_data:
            data_complexity += 0.1
        if self.test_data.audience_segments:
            data_complexity += 0.1
        
        return min(base_score + criteria_factor + data_complexity, 1.0)


class MarketingTestCase(BaseModel):
    """
    Complete test case for marketing agent evaluation.
    
    This model defines a comprehensive test case that can be used
    to evaluate marketing agents across various scenarios and metrics.
    """
    
    # Identity
    test_id: str = Field(..., description="Unique test case identifier")
    name: str = Field(..., description="Test case name")
    description: str = Field(..., description="Test case description")
    version: str = Field(default="1.0.0", description="Test case version")
    
    # Classification
    priority: TestPriority = Field(default=TestPriority.MEDIUM, description="Test priority")
    category: str = Field(..., description="Test category")
    tags: List[str] = Field(default_factory=list, description="Test tags")
    
    # Test configuration
    scenario: TestScenario = Field(..., description="Test scenario")
    steps: List[TestStep] = Field(..., description="Test steps")
    
    # Agent requirements
    required_capabilities: List[str] = Field(..., description="Required agent capabilities")
    agent_constraints: Dict[str, Any] = Field(default_factory=dict, description="Agent constraints")
    
    # Execution settings
    timeout_seconds: int = Field(default=600, gt=0, description="Overall test timeout")
    max_retries: int = Field(default=1, ge=0, description="Maximum retry attempts")
    parallel_execution: bool = Field(default=False, description="Allow parallel step execution")
    
    # Validation
    success_criteria: List[SuccessCriteria] = Field(..., description="Overall success criteria")
    expected_outputs: Dict[str, Any] = Field(default_factory=dict, description="Expected test outputs")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = Field(None, description="Test case author")
    
    # Execution history
    execution_count: int = Field(default=0, ge=0, description="Number of times executed")
    last_executed: Optional[datetime] = Field(None, description="Last execution time")
    average_duration: Optional[float] = Field(None, ge=0.0, description="Average execution duration")
    success_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Historical success rate")
    
    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True
    )
    
    def validate_agent_compatibility(self, agent_capabilities: List[str]) -> bool:
        """Check if an agent has required capabilities for this test."""
        return all(
            capability in agent_capabilities 
            for capability in self.required_capabilities
        )
    
    def estimate_execution_time(self) -> int:
        """Estimate total execution time based on steps and scenario."""
        step_time = sum(step.timeout_seconds for step in self.steps)
        scenario_time = self.scenario.estimated_duration
        return min(step_time + scenario_time, self.timeout_seconds)
    
    def get_critical_steps(self) -> List[TestStep]:
        """Get steps that are critical for test success."""
        # For now, consider steps with no retries as critical
        return [step for step in self.steps if step.retry_count == 0]
    
    def update_execution_stats(self, duration: float, success: bool) -> None:
        """Update execution statistics after test completion."""
        self.execution_count += 1
        self.last_executed = datetime.utcnow()
        
        # Update average duration
        if self.average_duration is None:
            self.average_duration = duration
        else:
            # Exponential moving average
            alpha = 0.1
            self.average_duration = (
                alpha * duration + 
                (1 - alpha) * self.average_duration
            )
        
        # Update success rate
        if self.success_rate is None:
            self.success_rate = 1.0 if success else 0.0
        else:
            # Exponential moving average
            alpha = 0.1
            current_success = 1.0 if success else 0.0
            self.success_rate = (
                alpha * current_success + 
                (1 - alpha) * self.success_rate
            )
        
        self.updated_at = datetime.utcnow()