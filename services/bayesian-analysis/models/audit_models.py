"""
Audit Models for Free Tier Bayesian Analysis

Pydantic models for free audit functionality including prior elicitation,
conflict detection, and lead capture for prospect onboarding.
"""

from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

class ContactInfo(BaseModel):
    """Contact information for prospects"""
    email: EmailStr
    name: Optional[str] = None
    company: Optional[str] = None
    phone: Optional[str] = None
    title: Optional[str] = None

class BusinessContext(BaseModel):
    """Business context for impact estimation"""
    industry: Optional[str] = None
    annual_revenue: Optional[float] = None
    marketing_budget: Optional[float] = None
    primary_channels: Optional[List[str]] = None
    decision_timeline: Optional[str] = None

class ParameterInfo(BaseModel):
    """Information about a parameter for prior elicitation"""
    name: str = Field(..., description="Parameter name (e.g., 'TV Attribution')")
    description: Optional[str] = Field(None, description="Parameter description")
    current_belief: Optional[float] = Field(None, description="Client's current belief about the value")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Confidence in current belief (0-1)")
    range_min: Optional[float] = Field(None, description="Minimum plausible value")
    range_max: Optional[float] = Field(None, description="Maximum plausible value")

class AuditRequest(BaseModel):
    """Request for free audit prior elicitation"""
    prospect_id: str = Field(..., description="Unique identifier for prospect")
    contact_info: ContactInfo
    business_context: Optional[BusinessContext] = None
    parameters: List[ParameterInfo] = Field(..., max_items=5, description="Parameters to analyze (max 5 for free tier)")
    source: Optional[str] = Field("website", description="Lead source")
    utm_params: Optional[Dict[str, str]] = Field(None, description="UTM tracking parameters")

class ClientPriorSimple(BaseModel):
    """Simplified client prior for free audit"""
    parameter_name: str
    mean: float
    std: float
    confidence: float = Field(ge=0, le=1)
    source: str = Field(default="client_belief")

class ConflictPreviewRequest(BaseModel):
    """Request for conflict detection preview (free tier)"""
    session_id: str
    client_priors: List[ClientPriorSimple]
    business_context: Optional[BusinessContext] = None
    use_sample_data: bool = Field(True, description="Use sample data for free tier")

class AuditReportRequest(BaseModel):
    """Request for audit report generation"""
    session_id: str
    include_upgrade_path: bool = Field(True)
    include_business_impact: bool = Field(True)
    format: str = Field("json", description="Report format: json, pdf, html")

class InterestLevel(str, Enum):
    """Prospect interest level"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNDECIDED = "undecided"

class LeadCaptureRequest(BaseModel):
    """Request to capture prospect lead information"""
    session_id: str
    email: EmailStr
    name: str
    company: str
    phone: Optional[str] = None
    title: Optional[str] = None
    conflicts_summary: Dict[str, Any]
    potential_impact: float = Field(description="Estimated potential impact in dollars")
    interest_level: InterestLevel
    preferred_contact_method: Optional[str] = Field("email")
    best_contact_time: Optional[str] = None
    specific_questions: Optional[str] = None
    marketing_consent: bool = Field(True, description="Consent to marketing communications")

class ConflictSummary(BaseModel):
    """Summary of detected conflicts for free audit"""
    total_conflicts: int
    severity_distribution: Dict[str, int]
    top_conflicts: List[Dict[str, Any]]
    estimated_impact: float

class UpgradeBenefits(BaseModel):
    """Benefits of upgrading to paid tier"""
    full_data_analysis: str
    unlimited_parameters: str
    detailed_recommendations: str
    sbc_validation: str
    integration_support: str

class AuditResponse(BaseModel):
    """Response from audit endpoints"""
    session_id: str
    status: str
    message: str
    data: Dict[str, Any]
    upgrade_message: Optional[str] = None
    next_steps: List[str]

class ElicitationQuestion(BaseModel):
    """Question for prior elicitation"""
    parameter: str
    question_type: str  # "point_estimate", "range", "confidence", "comparison"
    question_text: str
    options: Optional[List[str]] = None
    validation_rules: Optional[Dict[str, Any]] = None

class ElicitationSession(BaseModel):
    """Prior elicitation session for free audit"""
    session_id: str
    prospect_id: str
    contact_info: ContactInfo
    parameters: List[ParameterInfo]
    questions: List[ElicitationQuestion]
    tier: str = "free_audit"
    created_at: datetime
    estimated_duration: str
    status: str = "active"

class ConflictPreview(BaseModel):
    """Preview of potential conflicts (free tier)"""
    parameter: str
    conflict_type: str
    severity: str  # "weak", "moderate", "strong", "very_strong"
    description: str
    potential_impact: float
    sample_data_note: str = "Based on sample data. Upgrade for analysis with your actual data."

class ImpactEstimate(BaseModel):
    """Estimated business impact from resolving conflicts"""
    revenue_impact: float
    confidence_gain: float
    decision_quality: float
    risk_reduction: float
    implementation_effort: str

class AuditReport(BaseModel):
    """Comprehensive audit report for prospects"""
    session_id: str
    prospect_info: ContactInfo
    executive_summary: str
    key_findings: List[str]
    conflict_analysis: List[ConflictPreview]
    potential_improvements: List[str]
    business_impact: ImpactEstimate
    upgrade_value_proposition: Dict[str, Any]
    next_steps: List[Dict[str, str]]
    generated_at: datetime
    valid_until: datetime

class LeadData(BaseModel):
    """Captured lead data"""
    lead_id: str
    session_id: str
    contact_info: ContactInfo
    business_context: Optional[BusinessContext]
    audit_summary: ConflictSummary
    interest_level: InterestLevel
    source: str
    utm_params: Optional[Dict[str, str]]
    created_at: datetime
    follow_up_priority: str
    estimated_value: float

# Response models for API endpoints

class ElicitationResponse(BaseModel):
    """Response from prior elicitation endpoint"""
    session_id: str
    elicitation_questions: List[ElicitationQuestion]
    estimated_duration: str
    parameters_included: int
    parameters_limit_reached: bool
    next_step: str
    upgrade_message: str

class ConflictDetectionResponse(BaseModel):
    """Response from conflict detection endpoint"""
    session_id: str
    conflicts_detected: int
    severity_summary: Dict[str, int]
    top_conflicts: List[ConflictPreview]
    potential_impact: ImpactEstimate
    upgrade_benefits: UpgradeBenefits
    next_steps: List[str]

class ReportGenerationResponse(BaseModel):
    """Response from report generation endpoint"""
    session_id: str
    report_url: str
    executive_summary: str
    key_findings: List[str]
    potential_improvements: List[str]
    business_impact: ImpactEstimate
    upgrade_value_proposition: Dict[str, Any]
    next_steps: List[Dict[str, str]]

class LeadCaptureResponse(BaseModel):
    """Response from lead capture endpoint"""
    status: str
    lead_id: str
    next_steps: str
    consultation_link: str
    upgrade_offer: Dict[str, str]
    follow_up: Dict[str, str]

# Error models

class AuditError(BaseModel):
    """Error response for audit endpoints"""
    error_type: str
    message: str
    details: Optional[Dict[str, Any]] = None
    upgrade_suggestion: Optional[str] = None

class ValidationError(BaseModel):
    """Validation error for audit requests"""
    field: str
    message: str
    invalid_value: Any
    suggestion: Optional[str] = None