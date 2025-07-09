"""
SQLAlchemy Database Models for Bayesian Framework
Provides ORM models for prior elicitation, conflict analysis, and SBC validation
"""
from sqlalchemy import (
    Column, String, DateTime, Float, Integer, Boolean, Text, JSON,
    ForeignKey, Index, CheckConstraint, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid
from typing import Dict, List, Any, Optional

from shared.database.models import Base  # Assuming existing base model


class PriorElicitationSessionDB(Base):
    """Database model for prior elicitation sessions"""
    __tablename__ = 'prior_elicitation_sessions'
    
    session_id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    client_id = Column(String(50), nullable=False, index=True)
    organization_id = Column(String(50), nullable=False)
    session_date = Column(DateTime, default=datetime.utcnow, index=True)
    facilitator = Column(String(100))
    participants = Column(JSON, default=list)  # List of participant emails
    elicitation_method = Column(String(50), nullable=False)
    overall_confidence = Column(Float, CheckConstraint('overall_confidence >= 0 AND overall_confidence <= 1'))
    session_notes = Column(Text)
    validation_status = Column(String(20), default='pending', index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(100))
    
    # Relationships
    priors = relationship("PriorSpecificationDB", back_populates="session", cascade="all, delete-orphan")
    conflict_reports = relationship("ConflictReportDB", back_populates="session", cascade="all, delete-orphan")
    sbc_validations = relationship("SBCValidationResultDB", back_populates="session")
    update_sessions = relationship("BayesianUpdateSessionDB", back_populates="original_session")
    
    # Indexes
    __table_args__ = (
        Index('idx_sessions_client_date', 'client_id', 'session_date'),
        Index('idx_sessions_status_date', 'validation_status', 'session_date'),
        CheckConstraint('elicitation_method IN (\'expert_judgment\', \'historical_analogy\', \'industry_benchmark\', \'previous_study\', \'theoretical_constraint\', \'regulatory_requirement\')', name='chk_elicitation_method'),
        CheckConstraint('validation_status IN (\'pending\', \'validated\', \'rejected\', \'archived\')', name='chk_validation_status')
    )


class PriorSpecificationDB(Base):
    """Database model for individual prior specifications"""
    __tablename__ = 'prior_specifications'
    
    prior_id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(50), ForeignKey('prior_elicitation_sessions.session_id'), nullable=False, index=True)
    parameter_name = Column(String(100), nullable=False, index=True)
    parameter_category = Column(String(50), nullable=False, index=True)
    distribution_type = Column(String(30), nullable=False)
    hyperparameters = Column(JSON, nullable=False)  # Distribution parameters
    confidence_level = Column(Float, CheckConstraint('confidence_level >= 0 AND confidence_level <= 1'))
    domain_rationale = Column(Text)
    data_sources = Column(JSON, default=list)  # List of data sources
    elicitation_method = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String(100))
    is_active = Column(Boolean, default=True, index=True)
    
    # Relationships
    session = relationship("PriorElicitationSessionDB", back_populates="priors")
    evidence_metrics = relationship("EvidenceMetricsDB", back_populates="prior_spec")
    update_recommendations = relationship("PriorUpdateRecommendationDB", back_populates="original_prior")
    
    # Constraints and Indexes
    __table_args__ = (
        UniqueConstraint('session_id', 'parameter_name', name='uk_session_parameter'),
        Index('idx_priors_category_active', 'parameter_category', 'is_active'),
        CheckConstraint('distribution_type IN (\'normal\', \'lognormal\', \'beta\', \'gamma\', \'uniform\', \'exponential\', \'student_t\', \'half_normal\', \'half_cauchy\')', name='chk_distribution_type'),
        CheckConstraint('parameter_category IN (\'media_effect\', \'saturation\', \'adstock\', \'baseline\', \'seasonality\', \'trend\', \'external_factor\', \'interaction\')', name='chk_parameter_category')
    )


class ConflictReportDB(Base):
    """Database model for prior-data conflict reports"""
    __tablename__ = 'conflict_reports'
    
    report_id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String(50), ForeignKey('prior_elicitation_sessions.session_id'), nullable=False, index=True)
    analysis_date = Column(DateTime, default=datetime.utcnow, index=True)
    data_period_start = Column(DateTime, nullable=False)
    data_period_end = Column(DateTime, nullable=False)
    overall_conflict_score = Column(Float, CheckConstraint('overall_conflict_score >= 0 AND overall_conflict_score <= 1'), index=True)
    high_conflict_parameters = Column(JSON, default=list)  # List of parameter names
    recommendations = Column(JSON, default=list)  # List of recommendation strings
    status = Column(String(20), default='pending', index=True)
    processed_by = Column(String(100))
    processing_time_seconds = Column(Float)
    
    # Relationships
    session = relationship("PriorElicitationSessionDB", back_populates="conflict_reports")
    evidence_metrics = relationship("EvidenceMetricsDB", back_populates="report", cascade="all, delete-orphan")
    update_recommendations = relationship("PriorUpdateRecommendationDB", back_populates="report", cascade="all, delete-orphan")
    update_sessions = relationship("BayesianUpdateSessionDB", back_populates="conflict_report")
    
    # Constraints
    __table_args__ = (
        Index('idx_conflicts_score_date', 'overall_conflict_score', 'analysis_date'),
        CheckConstraint('status IN (\'pending\', \'completed\', \'failed\', \'archived\')', name='chk_conflict_status'),
        CheckConstraint('data_period_start <= data_period_end', name='chk_data_period_order')
    )


class EvidenceMetricsDB(Base):
    """Database model for evidence metrics"""
    __tablename__ = 'evidence_metrics'
    
    evidence_id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    report_id = Column(String(50), ForeignKey('conflict_reports.report_id'), nullable=False, index=True)
    prior_id = Column(String(50), ForeignKey('prior_specifications.prior_id'), nullable=True, index=True)
    parameter_name = Column(String(100), nullable=False, index=True)
    bayes_factor = Column(Float, nullable=False, index=True)
    kl_divergence = Column(Float, nullable=False)
    wasserstein_distance = Column(Float, nullable=False)
    overlap_coefficient = Column(Float, CheckConstraint('overlap_coefficient >= 0 AND overlap_coefficient <= 1'))
    effective_sample_size = Column(Float)
    prior_predictive_pvalue = Column(Float, CheckConstraint('prior_predictive_pvalue >= 0 AND prior_predictive_pvalue <= 1'))
    conflict_severity = Column(String(20), nullable=False, index=True)
    evidence_interpretation = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    report = relationship("ConflictReportDB", back_populates="evidence_metrics")
    prior_spec = relationship("PriorSpecificationDB", back_populates="evidence_metrics")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('report_id', 'parameter_name', name='uk_report_parameter'),
        Index('idx_evidence_severity_bf', 'conflict_severity', 'bayes_factor'),
        CheckConstraint('conflict_severity IN (\'none\', \'weak\', \'moderate\', \'strong\', \'decisive\')', name='chk_conflict_severity'),
        CheckConstraint('bayes_factor > 0', name='chk_positive_bayes_factor')
    )


class PriorUpdateRecommendationDB(Base):
    """Database model for prior update recommendations"""
    __tablename__ = 'prior_update_recommendations'
    
    recommendation_id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    report_id = Column(String(50), ForeignKey('conflict_reports.report_id'), nullable=False, index=True)
    parameter_name = Column(String(100), nullable=False, index=True)
    original_prior_id = Column(String(50), ForeignKey('prior_specifications.prior_id'), nullable=False)
    recommended_hyperparameters = Column(JSON, nullable=False)  # New hyperparameters
    update_strategy = Column(String(20), nullable=False)
    confidence_in_update = Column(Float, CheckConstraint('confidence_in_update >= 0 AND confidence_in_update <= 1'))
    business_impact = Column(Text)
    explanation = Column(Text)
    supporting_analysis = Column(JSON, default=dict)  # Additional analysis data
    implementation_priority = Column(Integer, CheckConstraint('implementation_priority >= 1 AND implementation_priority <= 5'), index=True)
    client_response = Column(String(20), index=True)
    client_response_date = Column(DateTime)
    client_response_notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    report = relationship("ConflictReportDB", back_populates="update_recommendations")
    original_prior = relationship("PriorSpecificationDB", back_populates="update_recommendations")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('report_id', 'parameter_name', name='uk_report_parameter_rec'),
        Index('idx_recommendations_priority_response', 'implementation_priority', 'client_response'),
        CheckConstraint('update_strategy IN (\'conservative\', \'moderate\', \'aggressive\', \'hierarchical\')', name='chk_update_strategy'),
        CheckConstraint('client_response IN (\'pending\', \'accepted\', \'rejected\', \'modified\')', name='chk_client_response')
    )


class SBCValidationResultDB(Base):
    """Database model for SBC validation results"""
    __tablename__ = 'sbc_validation_results'
    
    validation_id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_id = Column(String(50), index=True)
    client_id = Column(String(50), nullable=False, index=True)
    session_id = Column(String(50), ForeignKey('prior_elicitation_sessions.session_id'), nullable=True, index=True)
    validation_date = Column(DateTime, default=datetime.utcnow, index=True)
    model_configuration = Column(JSON, nullable=False)  # Model setup parameters
    n_simulations = Column(Integer, nullable=False)
    simulation_parameters = Column(JSON, nullable=False)  # Simulation settings
    calibration_quality = Column(Float, CheckConstraint('calibration_quality >= 0 AND calibration_quality <= 1'), index=True)
    coverage_probability = Column(Float, CheckConstraint('coverage_probability >= 0 AND coverage_probability <= 1'))
    rank_statistics = Column(JSON)  # Rank-based diagnostic statistics
    diagnostics = Column(JSON, nullable=False)  # Comprehensive diagnostic results
    passed = Column(Boolean, nullable=False, index=True)
    failure_reasons = Column(JSON, default=list)  # List of failure reasons
    recommendations = Column(JSON, default=list)  # List of recommendations
    execution_time_seconds = Column(Float)
    computational_resources = Column(JSON)  # Resource usage information
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String(100))
    
    # Relationships
    session = relationship("PriorElicitationSessionDB", back_populates="sbc_validations")
    
    # Constraints
    __table_args__ = (
        Index('idx_sbc_client_date_passed', 'client_id', 'validation_date', 'passed'),
        Index('idx_sbc_quality_date', 'calibration_quality', 'validation_date'),
        CheckConstraint('n_simulations > 0', name='chk_positive_simulations'),
        CheckConstraint('execution_time_seconds >= 0', name='chk_non_negative_time')
    )


class HierarchicalPriorStructureDB(Base):
    """Database model for hierarchical prior structures"""
    __tablename__ = 'hierarchical_prior_structures'
    
    structure_id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    industry_category = Column(String(100), nullable=False, index=True)
    structure_name = Column(String(200))
    hyperpriors = Column(JSON, nullable=False)  # Industry-level hyperpriors
    parameter_hierarchy = Column(JSON, nullable=False)  # Parameter grouping
    shrinkage_parameters = Column(JSON, default=dict)  # Shrinkage settings
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String(100))
    is_active = Column(Boolean, default=True, index=True)
    version = Column(String(20), index=True)
    
    # Constraints
    __table_args__ = (
        Index('idx_hierarchical_industry_active', 'industry_category', 'is_active'),
        Index('idx_hierarchical_version_active', 'version', 'is_active')
    )


class BayesianUpdateSessionDB(Base):
    """Database model for complete Bayesian update sessions"""
    __tablename__ = 'bayesian_update_sessions'
    
    update_session_id = Column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    original_session_id = Column(String(50), ForeignKey('prior_elicitation_sessions.session_id'), nullable=False, index=True)
    client_id = Column(String(50), nullable=False, index=True)
    update_date = Column(DateTime, default=datetime.utcnow, index=True)
    conflict_report_id = Column(String(50), ForeignKey('conflict_reports.report_id'), nullable=False)
    total_recommendations = Column(Integer, default=0)
    accepted_recommendations = Column(Integer, default=0)
    rejected_recommendations = Column(Integer, default=0)
    modified_recommendations = Column(Integer, default=0)
    update_summary = Column(JSON, default=dict)  # Summary statistics
    business_impact_assessment = Column(Text)
    client_satisfaction_score = Column(Float, CheckConstraint('client_satisfaction_score >= 1 AND client_satisfaction_score <= 5'), index=True)
    follow_up_scheduled = Column(Boolean, default=False)
    follow_up_date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    original_session = relationship("PriorElicitationSessionDB", back_populates="update_sessions")
    conflict_report = relationship("ConflictReportDB", back_populates="update_sessions")
    
    # Constraints
    __table_args__ = (
        Index('idx_update_sessions_client_date', 'client_id', 'update_date'),
        Index('idx_update_sessions_satisfaction', 'client_satisfaction_score'),
        CheckConstraint('total_recommendations >= 0', name='chk_non_negative_total'),
        CheckConstraint('accepted_recommendations >= 0', name='chk_non_negative_accepted'),
        CheckConstraint('rejected_recommendations >= 0', name='chk_non_negative_rejected'),
        CheckConstraint('modified_recommendations >= 0', name='chk_non_negative_modified'),
        CheckConstraint('accepted_recommendations + rejected_recommendations + modified_recommendations <= total_recommendations', name='chk_recommendation_totals')
    )


# Database utility functions
class BayesianDatabaseManager:
    """Utility class for managing Bayesian database operations"""
    
    def __init__(self, db_session):
        self.session = db_session
    
    def create_elicitation_session(
        self, 
        client_id: str, 
        facilitator: str, 
        participants: List[str],
        elicitation_method: str,
        **kwargs
    ) -> PriorElicitationSessionDB:
        """Create a new prior elicitation session"""
        session = PriorElicitationSessionDB(
            client_id=client_id,
            facilitator=facilitator,
            participants=participants,
            elicitation_method=elicitation_method,
            **kwargs
        )
        self.session.add(session)
        self.session.commit()
        return session
    
    def add_prior_specification(
        self,
        session_id: str,
        parameter_name: str,
        parameter_category: str,
        distribution_type: str,
        hyperparameters: Dict[str, float],
        confidence_level: float,
        domain_rationale: str,
        **kwargs
    ) -> PriorSpecificationDB:
        """Add a prior specification to a session"""
        prior = PriorSpecificationDB(
            session_id=session_id,
            parameter_name=parameter_name,
            parameter_category=parameter_category,
            distribution_type=distribution_type,
            hyperparameters=hyperparameters,
            confidence_level=confidence_level,
            domain_rationale=domain_rationale,
            **kwargs
        )
        self.session.add(prior)
        self.session.commit()
        return prior
    
    def create_conflict_report(
        self,
        session_id: str,
        data_period_start: datetime,
        data_period_end: datetime,
        overall_conflict_score: float,
        **kwargs
    ) -> ConflictReportDB:
        """Create a new conflict report"""
        report = ConflictReportDB(
            session_id=session_id,
            data_period_start=data_period_start,
            data_period_end=data_period_end,
            overall_conflict_score=overall_conflict_score,
            **kwargs
        )
        self.session.add(report)
        self.session.commit()
        return report
    
    def add_evidence_metrics(
        self,
        report_id: str,
        parameter_name: str,
        bayes_factor: float,
        kl_divergence: float,
        wasserstein_distance: float,
        conflict_severity: str,
        **kwargs
    ) -> EvidenceMetricsDB:
        """Add evidence metrics to a conflict report"""
        evidence = EvidenceMetricsDB(
            report_id=report_id,
            parameter_name=parameter_name,
            bayes_factor=bayes_factor,
            kl_divergence=kl_divergence,
            wasserstein_distance=wasserstein_distance,
            conflict_severity=conflict_severity,
            **kwargs
        )
        self.session.add(evidence)
        self.session.commit()
        return evidence
    
    def create_sbc_validation(
        self,
        client_id: str,
        model_configuration: Dict[str, Any],
        n_simulations: int,
        simulation_parameters: Dict[str, Any],
        calibration_quality: float,
        passed: bool,
        **kwargs
    ) -> SBCValidationResultDB:
        """Create a new SBC validation result"""
        validation = SBCValidationResultDB(
            client_id=client_id,
            model_configuration=model_configuration,
            n_simulations=n_simulations,
            simulation_parameters=simulation_parameters,
            calibration_quality=calibration_quality,
            passed=passed,
            **kwargs
        )
        self.session.add(validation)
        self.session.commit()
        return validation
    
    def get_client_sessions(self, client_id: str) -> List[PriorElicitationSessionDB]:
        """Get all elicitation sessions for a client"""
        return self.session.query(PriorElicitationSessionDB).filter(
            PriorElicitationSessionDB.client_id == client_id
        ).order_by(PriorElicitationSessionDB.session_date.desc()).all()
    
    def get_session_conflicts(self, session_id: str) -> List[ConflictReportDB]:
        """Get all conflict reports for a session"""
        return self.session.query(ConflictReportDB).filter(
            ConflictReportDB.session_id == session_id
        ).order_by(ConflictReportDB.analysis_date.desc()).all()
    
    def get_high_conflict_parameters(
        self, 
        client_id: str, 
        severity_threshold: str = 'moderate'
    ) -> List[EvidenceMetricsDB]:
        """Get parameters with high conflicts for a client"""
        severity_order = ['none', 'weak', 'moderate', 'strong', 'decisive']
        min_severity_index = severity_order.index(severity_threshold)
        target_severities = severity_order[min_severity_index:]
        
        return self.session.query(EvidenceMetricsDB).join(
            ConflictReportDB
        ).join(
            PriorElicitationSessionDB
        ).filter(
            PriorElicitationSessionDB.client_id == client_id,
            EvidenceMetricsDB.conflict_severity.in_(target_severities)
        ).order_by(EvidenceMetricsDB.bayes_factor.desc()).all()
    
    def get_sbc_validation_history(
        self, 
        client_id: str, 
        limit: int = 10
    ) -> List[SBCValidationResultDB]:
        """Get SBC validation history for a client"""
        return self.session.query(SBCValidationResultDB).filter(
            SBCValidationResultDB.client_id == client_id
        ).order_by(
            SBCValidationResultDB.validation_date.desc()
        ).limit(limit).all()
    
    def get_update_recommendations_pending(
        self, 
        client_id: str
    ) -> List[PriorUpdateRecommendationDB]:
        """Get pending update recommendations for a client"""
        return self.session.query(PriorUpdateRecommendationDB).join(
            ConflictReportDB
        ).join(
            PriorElicitationSessionDB
        ).filter(
            PriorElicitationSessionDB.client_id == client_id,
            PriorUpdateRecommendationDB.client_response == 'pending'
        ).order_by(
            PriorUpdateRecommendationDB.implementation_priority
        ).all()