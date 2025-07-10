"""
Database migration: Create security tables
Creates tables for enhanced security features including encrypted API keys,
audit logs, enhanced sessions, and security configuration.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import uuid

# revision identifiers
revision = '001_security_tables'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    """Create security tables"""
    
    # Create encrypted_api_keys table
    op.create_table(
        'encrypted_api_keys',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('org_id', sa.String(36), nullable=False),
        sa.Column('provider', sa.String(50), nullable=False),
        sa.Column('key_name', sa.String(100), nullable=False),
        sa.Column('encrypted_key', sa.Text, nullable=False),
        sa.Column('salt', sa.String(64), nullable=False),
        sa.Column('key_hash', sa.String(64), nullable=False),
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('expires_at', sa.DateTime(timezone=True)),
        sa.Column('last_used_at', sa.DateTime(timezone=True)),
        sa.Column('usage_count', sa.Integer, default=0),
        sa.Column('created_by', sa.String(36)),
        sa.Column('metadata', sa.JSON),
        sa.UniqueConstraint('org_id', 'provider', 'key_name', name='uq_org_provider_keyname'),
        sa.Index('idx_encrypted_api_keys_org_provider', 'org_id', 'provider'),
        sa.Index('idx_encrypted_api_keys_active', 'is_active'),
        sa.Index('idx_encrypted_api_keys_expires', 'expires_at')
    )
    
    # Create security_audit_logs table
    op.create_table(
        'security_audit_logs',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('event_type', sa.String(50), nullable=False),
        sa.Column('user_id', sa.String(36)),
        sa.Column('org_id', sa.String(36)),
        sa.Column('session_id', sa.String(36)),
        sa.Column('action', sa.String(100), nullable=False),
        sa.Column('resource_type', sa.String(50)),
        sa.Column('resource_id', sa.String(36)),
        sa.Column('ip_address', sa.String(45)),
        sa.Column('user_agent', sa.Text),
        sa.Column('success', sa.Boolean, nullable=False),
        sa.Column('error_message', sa.Text),
        sa.Column('risk_score', sa.Float, default=0.0),
        sa.Column('details', sa.JSON),
        sa.Column('compliance_flags', sa.JSON),
        sa.Index('idx_security_audit_timestamp', 'timestamp'),
        sa.Index('idx_security_audit_user', 'user_id'),
        sa.Index('idx_security_audit_org', 'org_id'),
        sa.Index('idx_security_audit_event_type', 'event_type'),
        sa.Index('idx_security_audit_success', 'success'),
        sa.Index('idx_security_audit_risk', 'risk_score')
    )
    
    # Create enhanced_user_sessions table
    op.create_table(
        'enhanced_user_sessions',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('user_id', sa.String(36), nullable=False),
        sa.Column('access_token_jti', sa.String(64), nullable=False),
        sa.Column('refresh_token_jti', sa.String(64), nullable=False),
        sa.Column('device_fingerprint', sa.String(64), nullable=False),
        sa.Column('ip_address', sa.String(45), nullable=False),
        sa.Column('user_agent', sa.Text),
        sa.Column('location', sa.JSON),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('last_activity', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('security_flags', sa.JSON),
        sa.UniqueConstraint('access_token_jti', name='uq_access_token_jti'),
        sa.UniqueConstraint('refresh_token_jti', name='uq_refresh_token_jti'),
        sa.Index('idx_enhanced_sessions_user', 'user_id'),
        sa.Index('idx_enhanced_sessions_active', 'is_active'),
        sa.Index('idx_enhanced_sessions_expires', 'expires_at'),
        sa.Index('idx_enhanced_sessions_device', 'device_fingerprint')
    )
    
    # Create revoked_tokens table
    op.create_table(
        'revoked_tokens',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('token_jti', sa.String(64), nullable=False),
        sa.Column('user_id', sa.String(36), nullable=False),
        sa.Column('token_type', sa.String(20), nullable=False),
        sa.Column('revoked_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('reason', sa.String(100)),
        sa.Column('revoked_by', sa.String(36)),
        sa.Column('original_expires_at', sa.DateTime(timezone=True)),
        sa.UniqueConstraint('token_jti', name='uq_token_jti'),
        sa.Index('idx_revoked_tokens_user', 'user_id'),
        sa.Index('idx_revoked_tokens_type', 'token_type'),
        sa.Index('idx_revoked_tokens_revoked_at', 'revoked_at')
    )
    
    # Create api_key_usage_analytics table
    op.create_table(
        'api_key_usage_analytics',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('api_key_id', sa.String(36), nullable=False),
        sa.Column('date', sa.Date, nullable=False),
        sa.Column('hour', sa.Integer, nullable=False),
        sa.Column('usage_count', sa.Integer, default=0),
        sa.Column('success_count', sa.Integer, default=0),
        sa.Column('error_count', sa.Integer, default=0),
        sa.Column('avg_response_time_ms', sa.Float),
        sa.Column('data_volume_bytes', sa.BigInteger, default=0),
        sa.Column('unique_endpoints', sa.JSON),
        sa.Column('error_types', sa.JSON),
        sa.UniqueConstraint('api_key_id', 'date', 'hour', name='uq_api_key_usage_hour'),
        sa.Index('idx_api_key_usage_key_date', 'api_key_id', 'date'),
        sa.Index('idx_api_key_usage_date', 'date')
    )
    
    # Create security_configurations table
    op.create_table(
        'security_configurations',
        sa.Column('id', sa.String(36), primary_key=True, default=lambda: str(uuid.uuid4())),
        sa.Column('org_id', sa.String(36), nullable=False),
        sa.Column('max_concurrent_sessions', sa.Integer, default=3),
        sa.Column('session_timeout_minutes', sa.Integer, default=480),
        sa.Column('require_mfa', sa.Boolean, default=False),
        sa.Column('allowed_ip_ranges', sa.JSON),
        sa.Column('api_rate_limits', sa.JSON),
        sa.Column('password_policy', sa.JSON),
        sa.Column('audit_retention_days', sa.Integer, default=365),
        sa.Column('compliance_settings', sa.JSON),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('updated_by', sa.String(36)),
        sa.UniqueConstraint('org_id', name='uq_security_config_org'),
        sa.Index('idx_security_config_org', 'org_id')
    )
    
    # Add foreign key constraints (if user/org tables exist)
    # Note: Uncomment these if you have user and organization tables
    # op.create_foreign_key(
    #     'fk_encrypted_api_keys_org',
    #     'encrypted_api_keys', 'organizations',
    #     ['org_id'], ['id'],
    #     ondelete='CASCADE'
    # )
    
    # op.create_foreign_key(
    #     'fk_security_audit_logs_user',
    #     'security_audit_logs', 'users',
    #     ['user_id'], ['id'],
    #     ondelete='SET NULL'
    # )
    
    # op.create_foreign_key(
    #     'fk_enhanced_user_sessions_user',
    #     'enhanced_user_sessions', 'users',
    #     ['user_id'], ['id'],
    #     ondelete='CASCADE'
    # )

def downgrade():
    """Drop security tables"""
    
    # Drop tables in reverse order to handle dependencies
    op.drop_table('security_configurations')
    op.drop_table('api_key_usage_analytics')
    op.drop_table('revoked_tokens')
    op.drop_table('enhanced_user_sessions')
    op.drop_table('security_audit_logs')
    op.drop_table('encrypted_api_keys')