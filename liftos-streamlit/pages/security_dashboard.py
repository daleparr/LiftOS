"""
LiftOS Security Dashboard
Enterprise security management interface for API keys, sessions, and audit logs
"""

import streamlit as st
import asyncio
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from shared.security.api_key_vault import get_api_key_vault
from shared.security.enhanced_jwt import get_enhanced_jwt_manager
from shared.security.audit_logger import SecurityAuditLogger
from shared.database.database import get_async_session
from services.data_ingestion.enhanced_credential_manager import get_enhanced_credential_manager
from shared.database.security_models import SecurityConfiguration

# Page configuration
st.set_page_config(
    page_title="Security Dashboard",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SecurityDashboard:
    """Security dashboard management class"""
    
    def __init__(self):
        self.vault = get_api_key_vault()
        self.jwt_manager = get_enhanced_jwt_manager()
        self.audit_logger = SecurityAuditLogger()
        self.credential_manager = get_enhanced_credential_manager()
        
        # Initialize session state
        if 'selected_org' not in st.session_state:
            st.session_state.selected_org = "default-org"
        if 'refresh_data' not in st.session_state:
            st.session_state.refresh_data = False

    async def get_security_overview(self, org_id: str) -> Dict[str, Any]:
        """Get security overview metrics"""
        try:
            async with get_async_session() as session:
                # Get API key count
                api_keys = await self.vault.list_api_keys(session, org_id)
                
                # Get usage analytics
                analytics = await self.credential_manager.get_credential_usage_analytics(
                    org_id=org_id,
                    days=30
                )
                
                # Get recent audit events (simplified query)
                from sqlalchemy import select, func
                from shared.database.security_models import SecurityAuditLog
                
                # Count recent events
                recent_events = await session.execute(
                    select(func.count(SecurityAuditLog.id))
                    .where(
                        SecurityAuditLog.org_id == org_id,
                        SecurityAuditLog.timestamp > datetime.now(timezone.utc) - timedelta(days=7)
                    )
                )
                recent_count = recent_events.scalar() or 0
                
                # Count failed events
                failed_events = await session.execute(
                    select(func.count(SecurityAuditLog.id))
                    .where(
                        SecurityAuditLog.org_id == org_id,
                        SecurityAuditLog.success == False,
                        SecurityAuditLog.timestamp > datetime.now(timezone.utc) - timedelta(days=7)
                    )
                )
                failed_count = failed_events.scalar() or 0
                
                # Count active sessions
                from shared.database.security_models import EnhancedUserSession
                active_sessions = await session.execute(
                    select(func.count(EnhancedUserSession.id))
                    .where(
                        EnhancedUserSession.user_id.like(f"%{org_id}%"),  # Simplified org filtering
                        EnhancedUserSession.is_active == True
                    )
                )
                session_count = active_sessions.scalar() or 0
                
                return {
                    "api_key_count": len(api_keys),
                    "active_sessions": session_count,
                    "recent_events": recent_count,
                    "failed_events": failed_count,
                    "success_rate": ((recent_count - failed_count) / max(recent_count, 1)) * 100,
                    "api_keys": api_keys,
                    "analytics": analytics
                }
                
        except Exception as e:
            st.error(f"Error loading security overview: {e}")
            return {
                "api_key_count": 0,
                "active_sessions": 0,
                "recent_events": 0,
                "failed_events": 0,
                "success_rate": 0,
                "api_keys": [],
                "analytics": {}
            }

    def render_overview_metrics(self, overview: Dict[str, Any]):
        """Render overview metrics cards"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🔑 API Keys",
                value=overview["api_key_count"],
                help="Total number of stored API keys"
            )
        
        with col2:
            st.metric(
                label="👥 Active Sessions",
                value=overview["active_sessions"],
                help="Currently active user sessions"
            )
        
        with col3:
            st.metric(
                label="📊 Security Events (7d)",
                value=overview["recent_events"],
                delta=f"-{overview['failed_events']} failed",
                help="Security events in the last 7 days"
            )
        
        with col4:
            st.metric(
                label="✅ Success Rate",
                value=f"{overview['success_rate']:.1f}%",
                help="Success rate of security operations"
            )

    def render_api_key_management(self, api_keys: List[Dict[str, Any]]):
        """Render API key management section"""
        st.subheader("🔐 API Key Management")
        
        if not api_keys:
            st.info("No API keys found. Add your first API key below.")
        else:
            # Create DataFrame for display
            df_data = []
            for key in api_keys:
                df_data.append({
                    "Provider": key.get("provider", "Unknown"),
                    "Key Name": key.get("key_name", "default"),
                    "Status": "🟢 Active" if key.get("is_active", False) else "🔴 Inactive",
                    "Created": key.get("created_at", "Unknown"),
                    "Last Used": key.get("last_used_at", "Never"),
                    "Usage Count": key.get("usage_count", 0)
                })
            
            if df_data:
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True)
                
                # Key management actions
                st.subheader("Key Actions")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🔄 Rotate Selected Key", help="Rotate the selected API key"):
                        st.info("Key rotation feature - implement with key selection")
                
                with col2:
                    if st.button("🗑️ Revoke Selected Key", help="Revoke the selected API key"):
                        st.warning("Key revocation feature - implement with key selection")
        
        # Add new API key section
        with st.expander("➕ Add New API Key"):
            self.render_add_api_key_form()

    def render_add_api_key_form(self):
        """Render form to add new API key"""
        with st.form("add_api_key_form"):
            st.subheader("Add New API Key")
            
            col1, col2 = st.columns(2)
            
            with col1:
                provider = st.selectbox(
                    "Provider",
                    ["meta", "google", "klaviyo", "shopify", "amazon", "salesforce", 
                     "stripe", "tiktok", "linkedin", "hubspot", "paypal"],
                    help="Select the API provider"
                )
                
                key_name = st.text_input(
                    "Key Name",
                    value="default",
                    help="Unique name for this API key"
                )
            
            with col2:
                # Dynamic credential fields based on provider
                credentials = {}
                
                if provider == "meta":
                    credentials["access_token"] = st.text_input("Access Token", type="password")
                    credentials["app_id"] = st.text_input("App ID")
                    credentials["app_secret"] = st.text_input("App Secret", type="password")
                
                elif provider == "google":
                    credentials["developer_token"] = st.text_input("Developer Token", type="password")
                    credentials["client_id"] = st.text_input("Client ID")
                    credentials["client_secret"] = st.text_input("Client Secret", type="password")
                    credentials["refresh_token"] = st.text_input("Refresh Token", type="password")
                
                elif provider == "klaviyo":
                    credentials["api_key"] = st.text_input("API Key", type="password")
                
                elif provider == "shopify":
                    credentials["shop_domain"] = st.text_input("Shop Domain")
                    credentials["access_token"] = st.text_input("Access Token", type="password")
                
                else:
                    # Generic key-value input
                    credentials["api_key"] = st.text_input("API Key", type="password")
                    credentials["secret"] = st.text_input("Secret", type="password")
            
            submitted = st.form_submit_button("🔐 Store API Key Securely")
            
            if submitted:
                if all(credentials.values()):
                    # Store the API key
                    success = asyncio.run(self._store_api_key(
                        st.session_state.selected_org,
                        provider,
                        key_name,
                        credentials
                    ))
                    
                    if success:
                        st.success(f"✅ {provider.title()} API key stored securely!")
                        st.session_state.refresh_data = True
                        st.rerun()
                    else:
                        st.error("❌ Failed to store API key")
                else:
                    st.error("Please fill in all required fields")

    async def _store_api_key(self, org_id: str, provider: str, key_name: str, credentials: Dict[str, str]) -> bool:
        """Store API key securely"""
        try:
            success = await self.credential_manager.store_credentials_in_vault(
                org_id=org_id,
                provider=provider,
                credentials=credentials,
                key_name=key_name,
                created_by="dashboard_user"
            )
            return success
        except Exception as e:
            st.error(f"Error storing API key: {e}")
            return False

    def render_security_analytics(self, analytics: Dict[str, Any]):
        """Render security analytics charts"""
        st.subheader("📊 Security Analytics")
        
        if not analytics:
            st.info("No analytics data available yet.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Usage by provider chart
            if "provider_usage" in analytics:
                provider_data = analytics["provider_usage"]
                if provider_data:
                    df_providers = pd.DataFrame(list(provider_data.items()), 
                                              columns=["Provider", "Usage Count"])
                    
                    fig_providers = px.pie(
                        df_providers, 
                        values="Usage Count", 
                        names="Provider",
                        title="API Usage by Provider"
                    )
                    st.plotly_chart(fig_providers, use_container_width=True)
        
        with col2:
            # Usage over time chart
            if "daily_usage" in analytics:
                daily_data = analytics["daily_usage"]
                if daily_data:
                    dates = list(daily_data.keys())
                    counts = list(daily_data.values())
                    
                    fig_timeline = go.Figure()
                    fig_timeline.add_trace(go.Scatter(
                        x=dates,
                        y=counts,
                        mode='lines+markers',
                        name='API Calls',
                        line=dict(color='#1f77b4', width=2)
                    ))
                    
                    fig_timeline.update_layout(
                        title="API Usage Over Time",
                        xaxis_title="Date",
                        yaxis_title="API Calls"
                    )
                    st.plotly_chart(fig_timeline, use_container_width=True)

    def render_audit_logs(self, org_id: str):
        """Render security audit logs"""
        st.subheader("🔍 Security Audit Logs")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            event_type = st.selectbox(
                "Event Type",
                ["All", "LOGIN_SUCCESS", "LOGIN_FAILED", "API_KEY_ACCESS", "SECURITY_VIOLATION"],
                help="Filter by event type"
            )
        
        with col2:
            days_back = st.selectbox(
                "Time Period",
                [1, 7, 30, 90],
                index=1,
                help="Number of days to look back"
            )
        
        with col3:
            if st.button("🔄 Refresh Logs"):
                st.session_state.refresh_data = True
                st.rerun()
        
        # Load and display audit logs
        logs = asyncio.run(self._get_audit_logs(org_id, event_type, days_back))
        
        if logs:
            df_logs = pd.DataFrame(logs)
            
            # Format the dataframe for display
            if not df_logs.empty:
                df_display = df_logs[["timestamp", "event_type", "action", "success", "ip_address", "user_agent"]].copy()
                df_display["timestamp"] = pd.to_datetime(df_display["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                df_display["success"] = df_display["success"].map({True: "✅", False: "❌"})
                
                st.dataframe(df_display, use_container_width=True)
                
                # Download logs
                csv = df_logs.to_csv(index=False)
                st.download_button(
                    label="📥 Download Audit Logs",
                    data=csv,
                    file_name=f"security_audit_logs_{org_id}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("No audit logs found for the selected criteria.")

    async def _get_audit_logs(self, org_id: str, event_type: str, days_back: int) -> List[Dict[str, Any]]:
        """Get audit logs from database"""
        try:
            async with get_async_session() as session:
                from sqlalchemy import select
                from shared.database.security_models import SecurityAuditLog
                
                query = select(SecurityAuditLog).where(
                    SecurityAuditLog.org_id == org_id,
                    SecurityAuditLog.timestamp > datetime.now(timezone.utc) - timedelta(days=days_back)
                )
                
                if event_type != "All":
                    query = query.where(SecurityAuditLog.event_type == event_type)
                
                query = query.order_by(SecurityAuditLog.timestamp.desc()).limit(100)
                
                result = await session.execute(query)
                logs = result.scalars().all()
                
                return [
                    {
                        "timestamp": log.timestamp,
                        "event_type": log.event_type,
                        "action": log.action,
                        "success": log.success,
                        "ip_address": log.ip_address,
                        "user_agent": log.user_agent,
                        "risk_score": log.risk_score,
                        "details": log.details
                    }
                    for log in logs
                ]
                
        except Exception as e:
            st.error(f"Error loading audit logs: {e}")
            return []

    def render_security_settings(self, org_id: str):
        """Render security settings configuration"""
        st.subheader("⚙️ Security Settings")
        
        # Load current configuration
        config = asyncio.run(self._get_security_config(org_id))
        
        with st.form("security_settings_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Session Management**")
                max_sessions = st.number_input(
                    "Max Concurrent Sessions",
                    min_value=1,
                    max_value=10,
                    value=config.get("max_concurrent_sessions", 3),
                    help="Maximum number of concurrent sessions per user"
                )
                
                session_timeout = st.number_input(
                    "Session Timeout (minutes)",
                    min_value=30,
                    max_value=1440,
                    value=config.get("session_timeout_minutes", 480),
                    help="Session timeout in minutes"
                )
                
                require_mfa = st.checkbox(
                    "Require Multi-Factor Authentication",
                    value=config.get("require_mfa", False),
                    help="Require MFA for all users"
                )
            
            with col2:
                st.write("**Rate Limiting**")
                auth_limit = st.number_input(
                    "Auth Attempts (per 5 min)",
                    min_value=1,
                    max_value=20,
                    value=config.get("api_rate_limits", {}).get("auth", {}).get("limit", 5),
                    help="Maximum authentication attempts per 5 minutes"
                )
                
                api_limit = st.number_input(
                    "API Calls (per hour)",
                    min_value=100,
                    max_value=10000,
                    value=config.get("api_rate_limits", {}).get("api", {}).get("limit", 1000),
                    help="Maximum API calls per hour"
                )
                
                audit_retention = st.number_input(
                    "Audit Log Retention (days)",
                    min_value=30,
                    max_value=2555,  # 7 years
                    value=config.get("audit_retention_days", 365),
                    help="How long to retain audit logs"
                )
            
            submitted = st.form_submit_button("💾 Save Security Settings")
            
            if submitted:
                new_config = {
                    "max_concurrent_sessions": max_sessions,
                    "session_timeout_minutes": session_timeout,
                    "require_mfa": require_mfa,
                    "api_rate_limits": {
                        "auth": {"limit": auth_limit, "window": 300},
                        "api": {"limit": api_limit, "window": 3600},
                        "sensitive": {"limit": 10, "window": 60}
                    },
                    "audit_retention_days": audit_retention
                }
                
                success = asyncio.run(self._update_security_config(org_id, new_config))
                
                if success:
                    st.success("✅ Security settings updated successfully!")
                else:
                    st.error("❌ Failed to update security settings")

    async def _get_security_config(self, org_id: str) -> Dict[str, Any]:
        """Get security configuration"""
        try:
            async with get_async_session() as session:
                from sqlalchemy import select
                
                result = await session.execute(
                    select(SecurityConfiguration).where(SecurityConfiguration.org_id == org_id)
                )
                config = result.scalar_one_or_none()
                
                if config:
                    return {
                        "max_concurrent_sessions": config.max_concurrent_sessions,
                        "session_timeout_minutes": config.session_timeout_minutes,
                        "require_mfa": config.require_mfa,
                        "api_rate_limits": config.api_rate_limits,
                        "audit_retention_days": config.audit_retention_days
                    }
                else:
                    return {}
                    
        except Exception as e:
            st.error(f"Error loading security configuration: {e}")
            return {}

    async def _update_security_config(self, org_id: str, new_config: Dict[str, Any]) -> bool:
        """Update security configuration"""
        try:
            async with get_async_session() as session:
                from sqlalchemy import select, update
                
                # Check if config exists
                result = await session.execute(
                    select(SecurityConfiguration).where(SecurityConfiguration.org_id == org_id)
                )
                existing_config = result.scalar_one_or_none()
                
                if existing_config:
                    # Update existing configuration
                    await session.execute(
                        update(SecurityConfiguration)
                        .where(SecurityConfiguration.org_id == org_id)
                        .values(**new_config, updated_by="dashboard_user")
                    )
                else:
                    # Create new configuration
                    config = SecurityConfiguration(
                        org_id=org_id,
                        updated_by="dashboard_user",
                        **new_config
                    )
                    session.add(config)
                
                await session.commit()
                return True
                
        except Exception as e:
            st.error(f"Error updating security configuration: {e}")
            return False

    def run(self):
        """Run the security dashboard"""
        st.title("🔐 LiftOS Security Dashboard")
        st.markdown("Enterprise security management and monitoring")
        
        # Sidebar for organization selection
        with st.sidebar:
            st.header("🏢 Organization")
            org_id = st.text_input(
                "Organization ID",
                value=st.session_state.selected_org,
                help="Enter your organization ID"
            )
            
            if org_id != st.session_state.selected_org:
                st.session_state.selected_org = org_id
                st.session_state.refresh_data = True
                st.rerun()
            
            st.header("🔄 Actions")
            if st.button("Refresh All Data"):
                st.session_state.refresh_data = True
                st.rerun()
            
            if st.button("🔒 Emergency Lockdown"):
                st.error("Emergency lockdown feature - implement with confirmation")
        
        # Main dashboard tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔑 API Keys", "🔍 Audit Logs", "⚙️ Settings"])
        
        # Load overview data
        overview = asyncio.run(self.get_security_overview(org_id))
        
        with tab1:
            st.header("Security Overview")
            self.render_overview_metrics(overview)
            
            st.markdown("---")
            self.render_security_analytics(overview.get("analytics", {}))
        
        with tab2:
            self.render_api_key_management(overview.get("api_keys", []))
        
        with tab3:
            self.render_audit_logs(org_id)
        
        with tab4:
            self.render_security_settings(org_id)
        
        # Reset refresh flag
        if st.session_state.refresh_data:
            st.session_state.refresh_data = False

# Run the dashboard
if __name__ == "__main__":
    dashboard = SecurityDashboard()
    dashboard.run()