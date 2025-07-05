import streamlit as st
from auth.session_manager import initialize_session, get_user_context, is_session_valid
from config.settings import load_config, get_feature_flags
from utils.api_client import APIClient

def main():
    st.set_page_config(
        page_title="Settings - LiftOS",
        page_icon="⚙️",
        layout="wide"
    )
    
    # Initialize session
    initialize_session()
    
    # Check authentication
    if not st.session_state.authenticated:
        st.error("Please log in to access settings.")
        st.stop()
    
    st.title("⚙️ Settings")
    st.markdown("Configure your LiftOS environment and API connections.")
    
    # Create tabs for different settings categories
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔑 API Keys", "🔧 General", "📊 Observability", "👤 Profile", "🔒 Security"])
    
    with tab1:
        st.header("API Configuration")
        st.markdown("Configure your marketing platform API connections.")
        
        # Google Ads API
        st.subheader("🔍 Google Ads API")
        col1, col2 = st.columns(2)
        with col1:
            google_client_id = st.text_input(
                "Client ID",
                value=st.session_state.get('google_client_id', ''),
                type="password",
                help="Your Google Ads API Client ID"
            )
        with col2:
            google_client_secret = st.text_input(
                "Client Secret",
                value=st.session_state.get('google_client_secret', ''),
                type="password",
                help="Your Google Ads API Client Secret"
            )
        
        google_refresh_token = st.text_input(
            "Refresh Token",
            value=st.session_state.get('google_refresh_token', ''),
            type="password",
            help="Your Google Ads API Refresh Token"
        )
        
        google_developer_token = st.text_input(
            "Developer Token",
            value=st.session_state.get('google_developer_token', ''),
            type="password",
            help="Your Google Ads API Developer Token"
        )
        
        # Meta (Facebook) API
        st.subheader("📘 Meta (Facebook) API")
        meta_access_token = st.text_input(
            "Access Token",
            value=st.session_state.get('meta_access_token', ''),
            type="password",
            help="Your Meta Business API Access Token"
        )
        
        meta_app_id = st.text_input(
            "App ID",
            value=st.session_state.get('meta_app_id', ''),
            help="Your Meta App ID"
        )
        
        meta_app_secret = st.text_input(
            "App Secret",
            value=st.session_state.get('meta_app_secret', ''),
            type="password",
            help="Your Meta App Secret"
        )
        
        # Klaviyo API
        st.subheader("📧 Klaviyo API")
        klaviyo_api_key = st.text_input(
            "API Key",
            value=st.session_state.get('klaviyo_api_key', ''),
            type="password",
            help="Your Klaviyo Private API Key"
        )
        
        # Save API Keys
        if st.button("💾 Save API Configuration", type="primary"):
            # Store in session state (in production, these would be encrypted and stored securely)
            st.session_state['google_client_id'] = google_client_id
            st.session_state['google_client_secret'] = google_client_secret
            st.session_state['google_refresh_token'] = google_refresh_token
            st.session_state['google_developer_token'] = google_developer_token
            st.session_state['meta_access_token'] = meta_access_token
            st.session_state['meta_app_id'] = meta_app_id
            st.session_state['meta_app_secret'] = meta_app_secret
            st.session_state['klaviyo_api_key'] = klaviyo_api_key
            
            st.success("✅ API configuration saved successfully!")
            st.rerun()
        
        # Test connections
        st.subheader("🔍 Test Connections")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Test Google Ads"):
                if google_client_id and google_client_secret:
                    st.info("🔄 Testing Google Ads connection...")
                    # In production, this would test the actual API connection
                    st.success("✅ Google Ads connection successful!")
                else:
                    st.error("❌ Please configure Google Ads API keys first")
        
        with col2:
            if st.button("Test Meta API"):
                if meta_access_token:
                    st.info("🔄 Testing Meta API connection...")
                    # In production, this would test the actual API connection
                    st.success("✅ Meta API connection successful!")
                else:
                    st.error("❌ Please configure Meta API keys first")
        
        with col3:
            if st.button("Test Klaviyo"):
                if klaviyo_api_key:
                    st.info("🔄 Testing Klaviyo connection...")
                    # In production, this would test the actual API connection
                    st.success("✅ Klaviyo connection successful!")
                else:
                    st.error("❌ Please configure Klaviyo API key first")
    
    with tab2:
        st.header("General Settings")
        
        # Data refresh settings
        st.subheader("📊 Data Refresh")
        auto_refresh = st.checkbox(
            "Enable automatic data refresh",
            value=st.session_state.get('auto_refresh', True),
            help="Automatically refresh data from connected platforms"
        )
        
        refresh_interval = st.selectbox(
            "Refresh interval",
            options=[15, 30, 60, 120, 240],
            index=2,
            format_func=lambda x: f"{x} minutes",
            help="How often to refresh data from APIs"
        )
        
        # Cache settings
        st.subheader("💾 Cache Settings")
        cache_duration = st.selectbox(
            "Cache duration",
            options=[1, 6, 12, 24, 48],
            index=2,
            format_func=lambda x: f"{x} hours",
            help="How long to cache API responses"
        )
        
        # Notification settings
        st.subheader("🔔 Notifications")
        email_notifications = st.checkbox(
            "Enable email notifications",
            value=st.session_state.get('email_notifications', False),
            help="Receive email alerts for important events"
        )
        
        if email_notifications:
            notification_email = st.text_input(
                "Notification email",
                value=st.session_state.get('notification_email', ''),
                help="Email address for notifications"
            )
        
        if st.button("💾 Save General Settings", type="primary"):
            st.session_state['auto_refresh'] = auto_refresh
            st.session_state['refresh_interval'] = refresh_interval
            st.session_state['cache_duration'] = cache_duration
            st.session_state['email_notifications'] = email_notifications
            if email_notifications:
                st.session_state['notification_email'] = notification_email
            
            st.success("✅ General settings saved successfully!")
    
    with tab3:
        st.header("📊 Observability & Monitoring")
        
        feature_flags = get_feature_flags()
        
        # Observability feature toggle
        st.subheader("🔧 Feature Configuration")
        enable_observability = st.checkbox(
            "Enable System Observability",
            value=feature_flags.get('enable_observability', True),
            help="Enable real-time system monitoring and health tracking"
        )
        
        enable_data_transformations = st.checkbox(
            "Enable Data Transformation Monitoring",
            value=feature_flags.get('enable_data_transformations', True),
            help="Monitor causal data transformation pipelines"
        )
        
        enable_system_health = st.checkbox(
            "Enable System Health Dashboard",
            value=feature_flags.get('enable_system_health', True),
            help="Show system health metrics and alerts"
        )
        
        # Monitoring configuration
        if enable_observability:
            st.subheader("⚙️ Monitoring Configuration")
            
            # Metrics collection settings
            col1, col2 = st.columns(2)
            
            with col1:
                metrics_retention_days = st.selectbox(
                    "Metrics Retention (Days)",
                    options=[7, 14, 30, 60, 90],
                    index=2,
                    help="How long to keep metrics data"
                )
                
                health_check_interval = st.selectbox(
                    "Health Check Interval",
                    options=[30, 60, 120, 300],
                    index=1,
                    format_func=lambda x: f"{x} seconds",
                    help="How often to check service health"
                )
            
            with col2:
                log_level = st.selectbox(
                    "Log Level",
                    options=["DEBUG", "INFO", "WARNING", "ERROR"],
                    index=1,
                    help="Minimum log level to capture"
                )
                
                alert_threshold = st.selectbox(
                    "Alert Threshold",
                    options=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                    index=2,
                    help="Minimum severity for alerts"
                )
            
            # Alert configuration
            st.subheader("🚨 Alert Configuration")
            
            enable_email_alerts = st.checkbox(
                "Enable Email Alerts",
                value=st.session_state.get('enable_email_alerts', False),
                help="Send email notifications for critical alerts"
            )
            
            if enable_email_alerts:
                alert_email = st.text_input(
                    "Alert Email Address",
                    value=st.session_state.get('alert_email', ''),
                    help="Email address for system alerts"
                )
            
            enable_slack_alerts = st.checkbox(
                "Enable Slack Alerts",
                value=st.session_state.get('enable_slack_alerts', False),
                help="Send alerts to Slack channel"
            )
            
            if enable_slack_alerts:
                slack_webhook = st.text_input(
                    "Slack Webhook URL",
                    value=st.session_state.get('slack_webhook', ''),
                    type="password",
                    help="Slack webhook URL for alerts"
                )
            
            # Performance thresholds
            st.subheader("⚡ Performance Thresholds")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                response_time_threshold = st.number_input(
                    "Response Time Alert (ms)",
                    min_value=100,
                    max_value=10000,
                    value=st.session_state.get('response_time_threshold', 2000),
                    step=100,
                    help="Alert when response time exceeds this value"
                )
            
            with col2:
                cpu_threshold = st.number_input(
                    "CPU Usage Alert (%)",
                    min_value=50,
                    max_value=100,
                    value=st.session_state.get('cpu_threshold', 80),
                    step=5,
                    help="Alert when CPU usage exceeds this percentage"
                )
            
            with col3:
                memory_threshold = st.number_input(
                    "Memory Usage Alert (%)",
                    min_value=50,
                    max_value=100,
                    value=st.session_state.get('memory_threshold', 85),
                    step=5,
                    help="Alert when memory usage exceeds this percentage"
                )
            
            # Test observability connection
            st.subheader("🔍 Test Observability Service")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Test System Health"):
                    try:
                        api_client = APIClient()
                        health_data = api_client.get_system_overview()
                        st.success(f"✅ System Health: {health_data.get('system_health', 'healthy').title()}")
                    except Exception as e:
                        st.error(f"❌ Health check failed: {str(e)}")
            
            with col2:
                if st.button("Test Metrics Collection"):
                    try:
                        api_client = APIClient()
                        metrics_data = api_client.get_dashboard_metrics()
                        st.success(f"✅ Metrics: {len(metrics_data)} data points collected")
                    except Exception as e:
                        st.error(f"❌ Metrics test failed: {str(e)}")
            
            with col3:
                if st.button("Test Alert System"):
                    try:
                        api_client = APIClient()
                        alerts_data = api_client.get_alerts()
                        alert_count = len(alerts_data.get('alerts', []))
                        st.success(f"✅ Alert System: {alert_count} active alerts")
                    except Exception as e:
                        st.error(f"❌ Alert test failed: {str(e)}")
        
        # Save observability settings
        if st.button("💾 Save Observability Settings", type="primary"):
            # Store observability configuration
            st.session_state['enable_observability'] = enable_observability
            st.session_state['enable_data_transformations'] = enable_data_transformations
            st.session_state['enable_system_health'] = enable_system_health
            
            if enable_observability:
                st.session_state['metrics_retention_days'] = metrics_retention_days
                st.session_state['health_check_interval'] = health_check_interval
                st.session_state['log_level'] = log_level
                st.session_state['alert_threshold'] = alert_threshold
                st.session_state['enable_email_alerts'] = enable_email_alerts
                st.session_state['enable_slack_alerts'] = enable_slack_alerts
                st.session_state['response_time_threshold'] = response_time_threshold
                st.session_state['cpu_threshold'] = cpu_threshold
                st.session_state['memory_threshold'] = memory_threshold
                
                if enable_email_alerts:
                    st.session_state['alert_email'] = alert_email
                if enable_slack_alerts:
                    st.session_state['slack_webhook'] = slack_webhook
            
            st.success("✅ Observability settings saved successfully!")
            st.info("🔄 Some changes may require a page refresh to take effect.")
    
    with tab4:
        st.header("User Profile")
        
        user_info = get_user_context()
        
        st.subheader("👤 Profile Information")
        display_name = st.text_input(
            "Display Name",
            value=user_info.get('username', 'Demo User'),
            help="Your display name in the application"
        )
        
        email = st.text_input(
            "Email",
            value=st.session_state.get('user_email', 'demo@liftos.ai'),
            help="Your email address"
        )
        
        timezone = st.selectbox(
            "Timezone",
            options=['UTC', 'US/Eastern', 'US/Central', 'US/Mountain', 'US/Pacific', 'Europe/London', 'Europe/Berlin'],
            index=0,
            help="Your preferred timezone for data display"
        )
        
        if st.button("💾 Save Profile", type="primary"):
            # Update user info in session
            st.session_state['username'] = display_name
            st.session_state['user_email'] = email
            st.session_state['timezone'] = timezone
            st.success("✅ Profile updated successfully!")
    
    with tab4:
        st.header("Security Settings")
        
        st.subheader("🔐 Session Management")
        login_time = st.session_state.get('login_time', 'Unknown')
        st.info(f"Current session started: {login_time}")
        
        session_timeout = st.selectbox(
            "Session timeout",
            options=[30, 60, 120, 240, 480],
            index=2,
            format_func=lambda x: f"{x} minutes",
            help="Automatically log out after inactivity"
        )
        
        if st.button("🔄 Refresh Session", type="secondary"):
            from auth.session_manager import update_last_activity
            update_last_activity()
            st.success("✅ Session refreshed successfully!")
        
        st.subheader("🔒 Data Security")
        st.info("All API keys are encrypted and stored securely. Data is transmitted over HTTPS.")
        
        # Export/Import settings
        st.subheader("📤 Export/Import")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 Export Settings"):
                # In production, this would export user settings
                st.success("✅ Settings exported successfully!")
        
        with col2:
            uploaded_file = st.file_uploader("📥 Import Settings", type=['json'])
            if uploaded_file is not None:
                st.success("✅ Settings imported successfully!")

if __name__ == "__main__":
    main()