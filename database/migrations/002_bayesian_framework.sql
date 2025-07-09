-- LiftOS Bayesian Framework Database Migration
-- Version: 2.1.0
-- Description: Add Bayesian prior analysis, conflict detection, and SBC validation tables

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. Bayesian Sessions Table
CREATE TABLE IF NOT EXISTS bayesian_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    session_name VARCHAR(255) NOT NULL,
    analysis_type VARCHAR(100) NOT NULL,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}',
    
    -- Indexes
    CONSTRAINT bayesian_sessions_user_id_idx UNIQUE (user_id, session_name)
);

CREATE INDEX IF NOT EXISTS idx_bayesian_sessions_user_id ON bayesian_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_bayesian_sessions_status ON bayesian_sessions(status);
CREATE INDEX IF NOT EXISTS idx_bayesian_sessions_created_at ON bayesian_sessions(created_at);

-- 2. Prior Beliefs Table
CREATE TABLE IF NOT EXISTS prior_beliefs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES bayesian_sessions(id) ON DELETE CASCADE,
    parameter_name VARCHAR(255) NOT NULL,
    parameter_category VARCHAR(100) NOT NULL,
    distribution_type VARCHAR(50) NOT NULL,
    distribution_params JSONB NOT NULL,
    confidence_level FLOAT NOT NULL DEFAULT 0.95,
    source VARCHAR(100) DEFAULT 'client_expertise',
    elicitation_method VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT prior_beliefs_confidence_check CHECK (confidence_level >= 0.0 AND confidence_level <= 1.0),
    CONSTRAINT prior_beliefs_unique_param UNIQUE (session_id, parameter_name)
);

CREATE INDEX IF NOT EXISTS idx_prior_beliefs_session_id ON prior_beliefs(session_id);
CREATE INDEX IF NOT EXISTS idx_prior_beliefs_parameter_category ON prior_beliefs(parameter_category);
CREATE INDEX IF NOT EXISTS idx_prior_beliefs_distribution_type ON prior_beliefs(distribution_type);

-- 3. Conflict Analysis Table
CREATE TABLE IF NOT EXISTS conflict_analysis (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES bayesian_sessions(id) ON DELETE CASCADE,
    analysis_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    conflict_detected BOOLEAN NOT NULL,
    conflict_severity VARCHAR(20) NOT NULL,
    evidence_strength VARCHAR(20) NOT NULL,
    bayes_factor FLOAT,
    kl_divergence FLOAT,
    wasserstein_distance FLOAT,
    p_value FLOAT,
    effect_size FLOAT,
    conflicting_parameters JSONB DEFAULT '[]',
    evidence_summary JSONB DEFAULT '{}',
    recommendations JSONB DEFAULT '[]',
    
    -- Constraints
    CONSTRAINT conflict_severity_check CHECK (conflict_severity IN ('low', 'moderate', 'high', 'severe')),
    CONSTRAINT evidence_strength_check CHECK (evidence_strength IN ('weak', 'moderate', 'strong', 'very_strong'))
);

CREATE INDEX IF NOT EXISTS idx_conflict_analysis_session_id ON conflict_analysis(session_id);
CREATE INDEX IF NOT EXISTS idx_conflict_analysis_timestamp ON conflict_analysis(analysis_timestamp);
CREATE INDEX IF NOT EXISTS idx_conflict_analysis_severity ON conflict_analysis(conflict_severity);

-- 4. Update Recommendations Table
CREATE TABLE IF NOT EXISTS update_recommendations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conflict_analysis_id UUID NOT NULL REFERENCES conflict_analysis(id) ON DELETE CASCADE,
    parameter_name VARCHAR(255) NOT NULL,
    current_prior JSONB NOT NULL,
    recommended_prior JSONB NOT NULL,
    update_strength FLOAT NOT NULL,
    evidence_weight FLOAT NOT NULL,
    rationale TEXT,
    implementation_priority VARCHAR(20) DEFAULT 'medium',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT update_strength_check CHECK (update_strength >= 0.0 AND update_strength <= 1.0),
    CONSTRAINT evidence_weight_check CHECK (evidence_weight >= 0.0 AND evidence_weight <= 1.0),
    CONSTRAINT priority_check CHECK (implementation_priority IN ('low', 'medium', 'high', 'critical'))
);

CREATE INDEX IF NOT EXISTS idx_update_recommendations_conflict_id ON update_recommendations(conflict_analysis_id);
CREATE INDEX IF NOT EXISTS idx_update_recommendations_parameter ON update_recommendations(parameter_name);
CREATE INDEX IF NOT EXISTS idx_update_recommendations_priority ON update_recommendations(implementation_priority);

-- 5. SBC Validation Table
CREATE TABLE IF NOT EXISTS sbc_validation (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES bayesian_sessions(id) ON DELETE CASCADE,
    validation_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    num_simulations INTEGER NOT NULL DEFAULT 1000,
    validation_passed BOOLEAN NOT NULL,
    rank_statistics JSONB NOT NULL,
    coverage_probability FLOAT,
    diagnostic_plots JSONB DEFAULT '[]',
    performance_metrics JSONB DEFAULT '{}',
    recommendations JSONB DEFAULT '[]',
    execution_time_seconds FLOAT,
    
    -- Constraints
    CONSTRAINT num_simulations_check CHECK (num_simulations > 0),
    CONSTRAINT coverage_probability_check CHECK (coverage_probability IS NULL OR (coverage_probability >= 0.0 AND coverage_probability <= 1.0))
);

CREATE INDEX IF NOT EXISTS idx_sbc_validation_session_id ON sbc_validation(session_id);
CREATE INDEX IF NOT EXISTS idx_sbc_validation_timestamp ON sbc_validation(validation_timestamp);
CREATE INDEX IF NOT EXISTS idx_sbc_validation_passed ON sbc_validation(validation_passed);

-- 6. SBC Decision Log Table
CREATE TABLE IF NOT EXISTS sbc_decision_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES bayesian_sessions(id) ON DELETE CASCADE,
    decision_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    sbc_required BOOLEAN NOT NULL,
    model_complexity INTEGER NOT NULL,
    conflict_severity VARCHAR(20) NOT NULL,
    business_impact FLOAT NOT NULL,
    client_confidence FLOAT NOT NULL,
    decision_rationale TEXT,
    automated_decision BOOLEAN DEFAULT TRUE,
    
    -- Constraints
    CONSTRAINT sbc_model_complexity_check CHECK (model_complexity > 0),
    CONSTRAINT sbc_business_impact_check CHECK (business_impact >= 0),
    CONSTRAINT sbc_client_confidence_check CHECK (client_confidence >= 0.0 AND client_confidence <= 1.0),
    CONSTRAINT sbc_conflict_severity_check CHECK (conflict_severity IN ('low', 'moderate', 'high', 'severe'))
);

CREATE INDEX IF NOT EXISTS idx_sbc_decision_session_id ON sbc_decision_log(session_id);
CREATE INDEX IF NOT EXISTS idx_sbc_decision_timestamp ON sbc_decision_log(decision_timestamp);
CREATE INDEX IF NOT EXISTS idx_sbc_decision_required ON sbc_decision_log(sbc_required);

-- 7. Prior Update History Table
CREATE TABLE IF NOT EXISTS prior_update_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES bayesian_sessions(id) ON DELETE CASCADE,
    parameter_name VARCHAR(255) NOT NULL,
    update_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    previous_prior JSONB NOT NULL,
    updated_prior JSONB NOT NULL,
    update_method VARCHAR(100) NOT NULL,
    evidence_data JSONB DEFAULT '{}',
    update_rationale TEXT,
    approved_by_user BOOLEAN DEFAULT FALSE,
    
    -- Constraints
    CONSTRAINT update_method_check CHECK (update_method IN ('bayes_rule', 'moment_matching', 'maximum_entropy', 'expert_adjustment'))
);

CREATE INDEX IF NOT EXISTS idx_prior_update_session_id ON prior_update_history(session_id);
CREATE INDEX IF NOT EXISTS idx_prior_update_parameter ON prior_update_history(parameter_name);
CREATE INDEX IF NOT EXISTS idx_prior_update_timestamp ON prior_update_history(update_timestamp);

-- 8. Evidence Data Table
CREATE TABLE IF NOT EXISTS evidence_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES bayesian_sessions(id) ON DELETE CASCADE,
    data_source VARCHAR(255) NOT NULL,
    data_type VARCHAR(100) NOT NULL,
    collection_date TIMESTAMP WITH TIME ZONE NOT NULL,
    data_content JSONB NOT NULL,
    quality_score FLOAT DEFAULT 1.0,
    relevance_score FLOAT DEFAULT 1.0,
    processing_notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT quality_score_check CHECK (quality_score >= 0.0 AND quality_score <= 1.0),
    CONSTRAINT relevance_score_check CHECK (relevance_score >= 0.0 AND relevance_score <= 1.0)
);

CREATE INDEX IF NOT EXISTS idx_evidence_data_session_id ON evidence_data(session_id);
CREATE INDEX IF NOT EXISTS idx_evidence_data_source ON evidence_data(data_source);
CREATE INDEX IF NOT EXISTS idx_evidence_data_type ON evidence_data(data_type);
CREATE INDEX IF NOT EXISTS idx_evidence_data_collection_date ON evidence_data(collection_date);

-- Create updated_at trigger function if it doesn't exist
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add updated_at triggers
CREATE TRIGGER update_bayesian_sessions_updated_at 
    BEFORE UPDATE ON bayesian_sessions 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_prior_beliefs_updated_at 
    BEFORE UPDATE ON prior_beliefs 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Add comments for documentation
COMMENT ON TABLE bayesian_sessions IS 'Bayesian analysis sessions for tracking client prior beliefs and evidence analysis';
COMMENT ON TABLE prior_beliefs IS 'Client prior beliefs and domain knowledge for marketing parameters';
COMMENT ON TABLE conflict_analysis IS 'Analysis results for conflicts between prior beliefs and observed data';
COMMENT ON TABLE update_recommendations IS 'Recommendations for updating prior beliefs based on evidence';
COMMENT ON TABLE sbc_validation IS 'Simulation Based Calibration validation results';
COMMENT ON TABLE sbc_decision_log IS 'Decision log for when SBC validation becomes necessary';
COMMENT ON TABLE prior_update_history IS 'History of prior belief updates and revisions';
COMMENT ON TABLE evidence_data IS 'Observed data and evidence used for Bayesian analysis';

-- Insert initial data for testing (optional)
INSERT INTO bayesian_sessions (user_id, session_name, analysis_type, metadata) VALUES 
('demo_user', 'Demo Marketing Attribution Analysis', 'marketing_attribution', '{"demo": true, "created_by": "migration"}')
ON CONFLICT (user_id, session_name) DO NOTHING;

-- Migration completion log
INSERT INTO public.migration_log (version, description, executed_at) VALUES 
('2.1.0', 'Bayesian Framework - Added 8 tables for prior analysis, conflict detection, and SBC validation', CURRENT_TIMESTAMP)
ON CONFLICT (version) DO UPDATE SET 
    executed_at = CURRENT_TIMESTAMP,
    description = EXCLUDED.description;

-- Grant permissions (adjust as needed for your environment)
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO liftos_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO liftos_app;