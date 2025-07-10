-- Migration 003: User Platform Connections
-- Adds tables for managing user-specific platform API connections

-- User Platform Connections Table
CREATE TABLE IF NOT EXISTS user_platform_connections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    org_id UUID NOT NULL,
    platform VARCHAR(50) NOT NULL,
    platform_display_name VARCHAR(100) NOT NULL,
    credential_id UUID REFERENCES encrypted_api_keys(id) ON DELETE CASCADE,
    connection_status VARCHAR(20) DEFAULT 'pending' CHECK (connection_status IN ('pending', 'active', 'error', 'disabled')),
    connection_config JSONB DEFAULT '{}',
    last_sync_at TIMESTAMP WITH TIME ZONE,
    last_test_at TIMESTAMP WITH TIME ZONE,
    sync_frequency INTERVAL DEFAULT '1 hour',
    auto_sync_enabled BOOLEAN DEFAULT true,
    error_count INTEGER DEFAULT 0,
    last_error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    UNIQUE(user_id, org_id, platform),
    CHECK (error_count >= 0)
);

-- Data Sync Logs Table
CREATE TABLE IF NOT EXISTS data_sync_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    connection_id UUID NOT NULL REFERENCES user_platform_connections(id) ON DELETE CASCADE,
    sync_type VARCHAR(20) NOT NULL CHECK (sync_type IN ('manual', 'scheduled', 'webhook', 'initial')),
    sync_status VARCHAR(20) NOT NULL CHECK (sync_status IN ('running', 'success', 'failed', 'partial', 'cancelled')),
    records_requested INTEGER,
    records_processed INTEGER DEFAULT 0,
    records_failed INTEGER DEFAULT 0,
    data_size_bytes BIGINT,
    error_message TEXT,
    error_details JSONB,
    sync_metadata JSONB DEFAULT '{}',
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    duration_seconds INTEGER,
    
    -- Constraints
    CHECK (records_processed >= 0),
    CHECK (records_failed >= 0),
    CHECK (data_size_bytes >= 0)
);

-- Platform OAuth States Table (for OAuth flow tracking)
CREATE TABLE IF NOT EXISTS platform_oauth_states (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    org_id UUID NOT NULL,
    platform VARCHAR(50) NOT NULL,
    state_token VARCHAR(255) NOT NULL UNIQUE,
    redirect_uri TEXT,
    scopes TEXT[],
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Cleanup expired states
    CHECK (expires_at > created_at)
);

-- User Data Preferences Table
CREATE TABLE IF NOT EXISTS user_data_preferences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    org_id UUID NOT NULL,
    prefer_live_data BOOLEAN DEFAULT true,
    fallback_to_mock BOOLEAN DEFAULT true,
    data_retention_days INTEGER DEFAULT 90,
    sync_preferences JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- One preference record per user/org
    UNIQUE(user_id, org_id),
    CHECK (data_retention_days > 0)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_user_platform_connections_user_org ON user_platform_connections(user_id, org_id);
CREATE INDEX IF NOT EXISTS idx_user_platform_connections_platform ON user_platform_connections(platform);
CREATE INDEX IF NOT EXISTS idx_user_platform_connections_status ON user_platform_connections(connection_status);
CREATE INDEX IF NOT EXISTS idx_user_platform_connections_sync ON user_platform_connections(last_sync_at) WHERE auto_sync_enabled = true;

CREATE INDEX IF NOT EXISTS idx_data_sync_logs_connection ON data_sync_logs(connection_id);
CREATE INDEX IF NOT EXISTS idx_data_sync_logs_status ON data_sync_logs(sync_status);
CREATE INDEX IF NOT EXISTS idx_data_sync_logs_started ON data_sync_logs(started_at);

CREATE INDEX IF NOT EXISTS idx_platform_oauth_states_user ON platform_oauth_states(user_id, org_id);
CREATE INDEX IF NOT EXISTS idx_platform_oauth_states_token ON platform_oauth_states(state_token);
CREATE INDEX IF NOT EXISTS idx_platform_oauth_states_expires ON platform_oauth_states(expires_at);

CREATE INDEX IF NOT EXISTS idx_user_data_preferences_user_org ON user_data_preferences(user_id, org_id);

-- Triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_user_platform_connections_updated_at 
    BEFORE UPDATE ON user_platform_connections 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_data_preferences_updated_at 
    BEFORE UPDATE ON user_data_preferences 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Cleanup function for expired OAuth states
CREATE OR REPLACE FUNCTION cleanup_expired_oauth_states()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM platform_oauth_states 
    WHERE expires_at < NOW();
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Comments for documentation
COMMENT ON TABLE user_platform_connections IS 'Stores user connections to marketing platforms with encrypted credentials';
COMMENT ON TABLE data_sync_logs IS 'Tracks all data synchronization attempts and their results';
COMMENT ON TABLE platform_oauth_states IS 'Temporary storage for OAuth flow state tokens';
COMMENT ON TABLE user_data_preferences IS 'User preferences for data handling and synchronization';

COMMENT ON COLUMN user_platform_connections.connection_config IS 'Platform-specific configuration like sync settings, field mappings, etc.';
COMMENT ON COLUMN data_sync_logs.sync_metadata IS 'Additional sync information like API rate limits, pagination tokens, etc.';
COMMENT ON COLUMN user_data_preferences.sync_preferences IS 'User preferences for sync frequency, data types, etc.';