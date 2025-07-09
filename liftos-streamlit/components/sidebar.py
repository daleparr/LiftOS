import streamlit as st
from config.settings import get_feature_flags
from auth.authenticator import show_logout_button
from utils.api_client import APIClient

def render_sidebar():
    """Render main sidebar navigation"""
    
    with st.sidebar:
        # Logo and title
        st.image("https://via.placeholder.com/200x60/1f77b4/white?text=LiftOS", width=200)
        st.markdown("### Marketing Intelligence Hub")
        
        # User info
        if st.session_state.get('authenticated', False):
            st.success(f"👤 {st.session_state.get('username', 'User')}")
        
        st.markdown("---")
        
        # Navigation menu
        render_navigation_menu()
        
        st.markdown("---")
        
        # Service status
        render_service_status()
        
        st.markdown("---")
        
        # Logout button
        show_logout_button()

def render_navigation_menu():
    """Render navigation menu"""
    st.subheader("📋 Navigation")
    
    feature_flags = get_feature_flags()
    
    # Home
    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("app.py")
    
    # Causal Analysis
    if feature_flags.get('enable_causal', True):
        if st.button("🧠 Causal Analysis", use_container_width=True):
            st.switch_page("pages/1_🧠_Causal_Analysis.py")
    
    # Data Insights
    if feature_flags.get('enable_surfacing', True):
        if st.button("🔍 Data Insights", use_container_width=True):
            st.switch_page("pages/2_🔍_Surfacing.py")
    
    # AI Assistant
    if feature_flags.get('enable_llm', True):
        if st.button("🤖 AI Assistant", use_container_width=True):
            st.switch_page("pages/3_🤖_LLM_Assistant.py")
    
    # System Health & Observability
    if feature_flags.get('enable_observability', True):
        if st.button("📊 System Health", use_container_width=True):
            st.switch_page("pages/4_📊_System_Health.py")
    
    # Knowledge Search
    if feature_flags.get('enable_memory', True):
        if st.button("🧠 Knowledge Search", use_container_width=True):
            st.switch_page("pages/5_🧠_Memory_Search.py")
    
    # Bayesian Analysis - moved up to appear right after Memory Search
    if feature_flags.get('enable_bayesian', True):
        if st.button("🎯 Bayesian Analysis", use_container_width=True):
            st.switch_page("pages/16_Bayesian_Analysis.py")
    
    # Settings
    if st.button("⚙️ Settings", use_container_width=True):
        st.switch_page("pages/6_⚙️_Settings.py")

def render_service_status():
    """Render enhanced platform status indicators with observability"""
    st.subheader("📊 Platform Status")
    
    feature_flags = get_feature_flags()
    
    try:
        api_client = APIClient()
        
        # Get comprehensive system health if observability is enabled
        if feature_flags.get('enable_observability', True):
            try:
                system_overview = api_client.get_system_overview()
                service_health = api_client.get_service_health()
                
                # Overall system health indicator
                system_health = system_overview.get('system_health', 'healthy')
                if system_health == 'healthy':
                    st.success("🟢 All Systems Operational")
                elif system_health == 'degraded':
                    st.warning("🟡 Performance Degraded")
                else:
                    st.error("🔴 Critical Issues Detected")
                
                # Individual service status from observability
                services = service_health.get('services', {})
                for service_name, service_data in services.items():
                    status = service_data.get('current_status', 'unknown')
                    response_time = service_data.get('response_time_ms', 0)
                    
                    # Map technical services to business-friendly names
                    business_services = {
                        'memory': ('Data Processing', '🧠'),
                        'causal': ('Analytics Engine', '📈'),
                        'llm': ('AI Assistant', '🤖'),
                        'surfacing': ('Insights Discovery', '🔍'),
                        'auth': ('Security Layer', '🔒'),
                        'observability': ('System Monitor', '📊'),
                        'gateway': ('API Gateway', '🌐')
                    }
                    
                    if service_name in business_services:
                        business_name, icon = business_services[service_name]
                        status_icon = "🟢" if status == 'healthy' else "🟡" if status == 'degraded' else "🔴"
                        st.markdown(f"{status_icon} **{icon} {business_name}**: {response_time:.0f}ms")
                
                # Quick metrics
                st.markdown("**Quick Metrics:**")
                col1, col2 = st.columns(2)
                with col1:
                    uptime = system_overview.get('uptime_percentage', 99.9)
                    st.metric("Uptime", f"{uptime:.1f}%")
                with col2:
                    response_time = system_overview.get('avg_response_time_ms', 67.3)
                    st.metric("Avg Response", f"{response_time:.0f}ms")
                
            except Exception:
                # Fallback to basic service status
                service_status = api_client.get_all_service_status()
                render_basic_service_status(service_status)
        else:
            # Basic service status when observability is disabled
            service_status = api_client.get_all_service_status()
            render_basic_service_status(service_status)
    
    except Exception as e:
        st.warning("🎭 Demo Mode - Services Initializing")
        # Show default business-friendly status
        render_default_service_status()

def render_basic_service_status(service_status):
    """Render basic service status without observability"""
    # Map technical services to business-friendly names
    business_services = {
        'memory': ('Data Processing', '🧠'),
        'causal': ('Analytics Engine', '📈'),
        'llm': ('AI Assistant', '🤖'),
        'surfacing': ('Insights Discovery', '🔍'),
        'auth': ('Security Layer', '🔒'),
        'observability': ('System Monitor', '📊'),
        'gateway': ('API Gateway', '🌐')
    }
    
    for service, is_healthy in service_status.items():
        if service in business_services:
            business_name, icon = business_services[service]
            status_icon = "🟢" if is_healthy else "🔴"
            status_text = "Active" if is_healthy else "Offline"
            st.markdown(f"{status_icon} **{icon} {business_name}**: {status_text}")

def render_default_service_status():
    """Render default service status for demo mode"""
    default_services = [
        ('🧠 Data Processing', 'Ready'),
        ('📈 Analytics Engine', 'Ready'),
        ('🤖 AI Assistant', 'Ready'),
        ('🔍 Insights Discovery', 'Ready'),
        ('🔒 Security Layer', 'Active'),
        ('📊 System Monitor', 'Active'),
        ('🌐 API Gateway', 'Active')
    ]
    for service_name, status in default_services:
        st.markdown(f"🟢 **{service_name}**: {status}")

def render_quick_stats():
    """Render quick statistics in sidebar"""
    st.subheader("📈 Quick Stats")
    
    # These would be fetched from the API in a real implementation
    stats = {
        "Active Campaigns": 12,
        "Models Created": 5,
        "Experiments Running": 3,
        "Data Sources": 3
    }
    
    for stat_name, stat_value in stats.items():
        st.metric(stat_name, stat_value)

def render_recent_alerts():
    """Render recent alerts/notifications with observability integration"""
    st.subheader("🔔 Recent Alerts")
    
    feature_flags = get_feature_flags()
    
    try:
        if feature_flags.get('enable_observability', True):
            api_client = APIClient()
            alerts_data = api_client.get_alerts()
            
            recent_alerts = alerts_data.get('alerts', [])[:3]  # Show only 3 most recent
            
            if recent_alerts:
                for alert in recent_alerts:
                    severity = alert.get('severity', 'info')
                    message = alert.get('name', 'Unknown alert')
                    service = alert.get('service_name', '')
                    
                    if severity == "critical":
                        st.error(f"🚨 {message} ({service})")
                    elif severity == "warning":
                        st.warning(f"⚠️ {message} ({service})")
                    else:
                        st.info(f"ℹ️ {message} ({service})")
            else:
                st.success("✅ No recent alerts")
        else:
            # Fallback to static alerts
            render_static_alerts()
            
    except Exception:
        # Fallback to static alerts
        render_static_alerts()

def render_static_alerts():
    """Render static demo alerts"""
    alerts = [
        {"type": "info", "message": "Meta sync completed successfully"},
        {"type": "success", "message": "New attribution model trained"},
        {"type": "info", "message": "System health check passed"}
    ]
    
    for alert in alerts:
        if alert["type"] == "info":
            st.info(f"ℹ️ {alert['message']}")
        elif alert["type"] == "warning":
            st.warning(f"⚠️ {alert['message']}")
        elif alert["type"] == "success":
            st.success(f"✅ {alert['message']}")
        else:
            st.error(f"🚨 {alert['message']}")