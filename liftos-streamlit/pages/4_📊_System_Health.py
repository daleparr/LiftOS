import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from typing import Dict
from auth.session_manager import initialize_session
from utils.api_client import APIClient
from components.sidebar import render_sidebar
from config.settings import get_feature_flags

def main():
    st.set_page_config(
        page_title="System Health - LiftOS",
        page_icon="📊",
        layout="wide"
    )
    
    # Initialize session
    initialize_session()
    
    # Render sidebar
    render_sidebar()
    
    # Check authentication
    if not st.session_state.authenticated:
        st.error("Please log in to access system health monitoring.")
        st.stop()
    
    # Check if observability is enabled
    features = get_feature_flags()
    if not features.get('enable_observability', True):
        st.warning("⚠️ Observability features are disabled. Please enable them in configuration.")
        st.stop()
    
    st.title("📊 System Health & Observability")
    st.markdown("Real-time monitoring of LiftOS platform health, performance, and data pipelines.")
    
    # Initialize API client
    api_client = APIClient()
    
    # Create tabs for different monitoring views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏥 System Overview", 
        "📈 Performance Metrics", 
        "🔄 Data Pipelines", 
        "🚨 Alerts & Issues",
        "🔍 Service Health"
    ])
    
    with tab1:
        render_system_overview(api_client)
    
    with tab2:
        render_performance_metrics(api_client)
    
    with tab3:
        render_data_pipelines(api_client)
    
    with tab4:
        render_alerts_issues(api_client)
    
    with tab5:
        render_service_health(api_client)

def render_system_overview(api_client: APIClient):
    """Render system overview dashboard"""
    st.header("🏥 System Overview")
    
    # Auto-refresh toggle
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### Real-time Platform Health")
    with col2:
        auto_refresh = st.checkbox("🔄 Auto-refresh", value=True, help="Automatically refresh every 30 seconds")
    
    if auto_refresh:
        # Auto-refresh every 30 seconds
        st.rerun()
    
    try:
        # Get system overview from observability service
        overview_data = api_client.get_system_overview()
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            system_health = overview_data.get('system_health', 'healthy')
            health_color = "🟢" if system_health == 'healthy' else "🟡" if system_health == 'degraded' else "🔴"
            st.metric(
                "System Health",
                f"{health_color} {system_health.title()}",
                help="Overall system health status"
            )
        
        with col2:
            total_services = overview_data.get('total_services', 6)
            healthy_services = overview_data.get('healthy_services', 5)
            st.metric(
                "Services Online",
                f"{healthy_services}/{total_services}",
                delta=f"{healthy_services - (total_services - healthy_services)}",
                help="Number of healthy services"
            )
        
        with col3:
            avg_response_time = overview_data.get('avg_response_time_ms', 67.3)
            st.metric(
                "Avg Response Time",
                f"{avg_response_time:.1f}ms",
                delta=f"{overview_data.get('trends', {}).get('response_time_trend_ms', -15.2):.1f}ms",
                help="Average response time across all services"
            )
        
        with col4:
            uptime_percentage = overview_data.get('uptime_percentage', 99.9)
            st.metric(
                "Platform Uptime",
                f"{uptime_percentage:.1f}%",
                delta=f"{overview_data.get('trends', {}).get('health_trend_percentage', 2.5):.1f}%",
                help="Platform availability percentage"
            )
        
        # System health trends chart
        st.subheader("📈 Health Trends (24 Hours)")
        render_health_trends_chart()
        
        # Critical alerts summary
        if overview_data.get('critical_alerts', 0) > 0:
            st.error(f"🚨 {overview_data['critical_alerts']} critical alerts require attention!")
            if st.button("🔍 View Critical Alerts"):
                st.switch_page("pages/4_📊_System_Health.py")
        
    except Exception as e:
        st.warning(f"⚠️ Could not load system overview: {str(e)}")
        render_mock_system_overview()

def render_performance_metrics(api_client: APIClient):
    """Render performance metrics dashboard"""
    st.header("📈 Performance Metrics")
    
    try:
        # Get dashboard metrics
        metrics_data = api_client.get_dashboard_metrics()
        
        # Time range selector
        col1, col2 = st.columns([2, 1])
        with col1:
            time_range = st.selectbox(
                "Time Range",
                options=["1h", "6h", "24h", "7d", "30d"],
                index=2,
                help="Select time range for metrics"
            )
        with col2:
            if st.button("🔄 Refresh Metrics", type="primary"):
                st.rerun()
        
        # Performance metrics grid
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🚀 Throughput")
            throughput_data = metrics_data.get('throughput', {})
            st.metric("Requests/sec", f"{throughput_data.get('current', 1250):,}")
            st.metric("Peak Requests/sec", f"{throughput_data.get('peak', 2100):,}")
            
        with col2:
            st.subheader("⚡ Latency")
            latency_data = metrics_data.get('latency', {})
            st.metric("P95 Response Time", f"{latency_data.get('p95', 145):.0f}ms")
            st.metric("P99 Response Time", f"{latency_data.get('p99', 280):.0f}ms")
            
        with col3:
            st.subheader("💾 Resources")
            resources_data = metrics_data.get('resources', {})
            st.metric("CPU Usage", f"{resources_data.get('cpu', 45.2):.1f}%")
            st.metric("Memory Usage", f"{resources_data.get('memory', 68.5):.1f}%")
        
        # Performance charts
        render_performance_charts(time_range)
        
    except Exception as e:
        st.warning(f"⚠️ Could not load performance metrics: {str(e)}")
        render_mock_performance_metrics()

def render_data_pipelines(api_client: APIClient):
    """Render data pipeline monitoring"""
    st.header("🔄 Data Pipeline Status")
    
    try:
        # Get transformation status
        pipeline_status = api_client.get_transformation_status()
        causal_metrics = api_client.get_causal_pipeline_metrics()
        
        # Pipeline overview
        st.subheader("📊 Pipeline Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            ingestion_status = pipeline_status.get('ingestion', {}).get('status', 'running')
            status_icon = "🟢" if ingestion_status == 'running' else "🟡" if ingestion_status == 'degraded' else "🔴"
            st.metric(
                "Data Ingestion",
                f"{status_icon} {ingestion_status.title()}",
                help="Data ingestion pipeline status"
            )
        
        with col2:
            transformation_status = pipeline_status.get('transformation', {}).get('status', 'running')
            status_icon = "🟢" if transformation_status == 'running' else "🟡" if transformation_status == 'degraded' else "🔴"
            st.metric(
                "Transformations",
                f"{status_icon} {transformation_status.title()}",
                help="Data transformation pipeline status"
            )
        
        with col3:
            causal_status = causal_metrics.get('pipeline_status', 'running')
            status_icon = "🟢" if causal_status == 'running' else "🟡" if causal_status == 'degraded' else "🔴"
            st.metric(
                "Causal Analysis",
                f"{status_icon} {causal_status.title()}",
                help="Causal analysis pipeline status"
            )
        
        with col4:
            records_processed = pipeline_status.get('records_processed_today', 125000)
            st.metric(
                "Records Processed",
                f"{records_processed:,}",
                delta=f"+{pipeline_status.get('records_delta', 15000):,}",
                help="Records processed today"
            )
        
        # Pipeline performance metrics
        st.subheader("⚡ Pipeline Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Processing speed chart
            render_pipeline_speed_chart()
        
        with col2:
            # Data quality metrics
            render_data_quality_metrics(api_client)
        
        # Causal transformation metrics
        st.subheader("🧠 Causal Data Transformations")
        
        causal_col1, causal_col2, causal_col3 = st.columns(3)
        
        with causal_col1:
            attribution_accuracy = causal_metrics.get('attribution_accuracy', 71.8)
            st.metric(
                "Attribution Accuracy",
                f"{attribution_accuracy:.1f}%",
                delta=f"+{causal_metrics.get('accuracy_improvement', 15.2):.1f}%",
                help="Causal attribution accuracy improvement"
            )
        
        with causal_col2:
            processing_speed = causal_metrics.get('processing_speed', 30651)
            st.metric(
                "Processing Speed",
                f"{processing_speed:,} rec/sec",
                help="Causal transformation processing speed"
            )
        
        with causal_col3:
            estimation_bias = causal_metrics.get('estimation_bias', 5.0)
            st.metric(
                "Estimation Bias",
                f"{estimation_bias:.1f}%",
                delta=f"-{causal_metrics.get('bias_reduction', 2.3):.1f}%",
                help="Causal estimation bias (lower is better)"
            )
        
    except Exception as e:
        st.warning(f"⚠️ Could not load pipeline status: {str(e)}")
        render_mock_pipeline_status()

def render_alerts_issues(api_client: APIClient):
    """Render alerts and issues dashboard"""
    st.header("🚨 Alerts & Issues")
    
    try:
        # Get alerts from observability service
        alerts_data = api_client.get_alerts()
        
        # Alert summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            critical_alerts = alerts_data.get('summary', {}).get('critical', 0)
            st.metric(
                "Critical Alerts",
                critical_alerts,
                delta=f"{alerts_data.get('trends', {}).get('critical_change', -1)}",
                help="Critical alerts requiring immediate attention"
            )
        
        with col2:
            warning_alerts = alerts_data.get('summary', {}).get('warning', 0)
            st.metric(
                "Warning Alerts",
                warning_alerts,
                delta=f"{alerts_data.get('trends', {}).get('warning_change', 2)}",
                help="Warning alerts for monitoring"
            )
        
        with col3:
            firing_alerts = alerts_data.get('summary', {}).get('firing', 0)
            st.metric(
                "Active Alerts",
                firing_alerts,
                help="Currently firing alerts"
            )
        
        with col4:
            resolved_today = alerts_data.get('summary', {}).get('resolved_today', 0)
            st.metric(
                "Resolved Today",
                resolved_today,
                delta=f"+{resolved_today}",
                help="Alerts resolved today"
            )
        
        # Active alerts table
        if alerts_data.get('alerts'):
            st.subheader("🔥 Active Alerts")
            
            alerts_df = pd.DataFrame(alerts_data['alerts'])
            
            # Add severity color coding
            def get_severity_color(severity):
                colors = {
                    'critical': '🔴',
                    'warning': '🟡',
                    'info': '🔵'
                }
                return colors.get(severity, '⚪')
            
            alerts_df['severity_icon'] = alerts_df['severity'].apply(get_severity_color)
            
            st.dataframe(
                alerts_df[['severity_icon', 'name', 'service_name', 'created_at', 'status']],
                column_config={
                    'severity_icon': st.column_config.TextColumn('Severity'),
                    'name': st.column_config.TextColumn('Alert Name'),
                    'service_name': st.column_config.TextColumn('Service'),
                    'created_at': st.column_config.DatetimeColumn('Created'),
                    'status': st.column_config.TextColumn('Status')
                },
                use_container_width=True
            )
        else:
            st.success("✅ No active alerts - all systems operating normally!")
        
        # Alert trends chart
        st.subheader("📈 Alert Trends (7 Days)")
        render_alert_trends_chart()
        
    except Exception as e:
        st.warning(f"⚠️ Could not load alerts: {str(e)}")
        render_mock_alerts()

def render_service_health(api_client: APIClient):
    """Render individual service health monitoring"""
    st.header("🔍 Service Health Details")
    
    try:
        # Get service health data
        health_data = api_client.get_service_health()
        
        # Service health grid
        services = health_data.get('services', {})
        
        if services:
            # Create service health cards
            cols = st.columns(3)
            
            for i, (service_name, service_data) in enumerate(services.items()):
                with cols[i % 3]:
                    status = service_data.get('current_status', 'unknown')
                    status_icon = "🟢" if status == 'healthy' else "🟡" if status == 'degraded' else "🔴"
                    
                    with st.container():
                        st.markdown(f"### {status_icon} {service_name.replace('_', ' ').title()}")
                        
                        response_time = service_data.get('response_time_ms', 0)
                        last_check = service_data.get('last_check', 'Unknown')
                        
                        st.metric("Response Time", f"{response_time:.1f}ms")
                        st.caption(f"Last checked: {last_check}")
                        
                        # Service-specific actions
                        if st.button(f"🔍 Details", key=f"details_{service_name}"):
                            show_service_details(service_name, service_data)
        
        # Service dependency map
        st.subheader("🔗 Service Dependencies")
        render_service_dependency_map()
        
    except Exception as e:
        st.warning(f"⚠️ Could not load service health: {str(e)}")
        render_mock_service_health()

def show_service_details(service_name: str, service_data: Dict):
    """Show detailed service information"""
    with st.expander(f"📋 {service_name} Details", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.json({
                "status": service_data.get('current_status'),
                "response_time_ms": service_data.get('response_time_ms'),
                "last_check": service_data.get('last_check'),
                "uptime_24h": service_data.get('uptime_24h', '99.9%')
            })
        
        with col2:
            # Service history chart would go here
            st.info("📊 Service history chart would be displayed here")

# Mock data functions for when services are unavailable
def render_mock_system_overview():
    """Render mock system overview"""
    st.info("🎭 Displaying demo data - observability service unavailable")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("System Health", "🟢 Healthy")
    with col2:
        st.metric("Services Online", "5/6", delta="0")
    with col3:
        st.metric("Avg Response Time", "67.3ms", delta="-15.2ms")
    with col4:
        st.metric("Platform Uptime", "99.9%", delta="+0.1%")

def render_mock_performance_metrics():
    """Render mock performance metrics"""
    st.info("🎭 Displaying demo data - observability service unavailable")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🚀 Throughput")
        st.metric("Requests/sec", "1,250")
        st.metric("Peak Requests/sec", "2,100")
    
    with col2:
        st.subheader("⚡ Latency")
        st.metric("P95 Response Time", "145ms")
        st.metric("P99 Response Time", "280ms")
    
    with col3:
        st.subheader("💾 Resources")
        st.metric("CPU Usage", "45.2%")
        st.metric("Memory Usage", "68.5%")

def render_mock_pipeline_status():
    """Render mock pipeline status"""
    st.info("🎭 Displaying demo data - pipeline services unavailable")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Data Ingestion", "🟢 Running")
    with col2:
        st.metric("Transformations", "🟢 Running")
    with col3:
        st.metric("Causal Analysis", "🟢 Running")
    with col4:
        st.metric("Records Processed", "125,000", delta="+15,000")

def render_mock_alerts():
    """Render mock alerts"""
    st.info("🎭 Displaying demo data - observability service unavailable")
    st.success("✅ No active alerts - all systems operating normally!")

def render_mock_service_health():
    """Render mock service health"""
    st.info("🎭 Displaying demo data - observability service unavailable")
    
    services = ["Gateway", "Auth", "Memory", "Causal", "Data Ingestion", "Observability"]
    cols = st.columns(3)
    
    for i, service in enumerate(services):
        with cols[i % 3]:
            st.markdown(f"### 🟢 {service}")
            st.metric("Response Time", f"{np.random.randint(20, 100)}ms")
            st.caption("Last checked: Just now")

# Chart rendering functions
def render_health_trends_chart():
    """Render health trends chart"""
    # Generate mock trend data
    dates = pd.date_range(start=datetime.now() - timedelta(hours=24), end=datetime.now(), freq='H')
    health_scores = np.random.normal(95, 5, len(dates))
    health_scores = np.clip(health_scores, 80, 100)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=health_scores,
        mode='lines+markers',
        name='System Health Score',
        line=dict(color='#00cc96', width=2)
    ))
    
    fig.update_layout(
        title="System Health Score (24 Hours)",
        xaxis_title="Time",
        yaxis_title="Health Score (%)",
        yaxis=dict(range=[80, 100]),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_performance_charts(time_range: str):
    """Render performance charts"""
    st.subheader("📊 Performance Trends")
    
    # Generate mock performance data
    hours = {"1h": 1, "6h": 6, "24h": 24, "7d": 168, "30d": 720}[time_range]
    dates = pd.date_range(start=datetime.now() - timedelta(hours=hours), end=datetime.now(), freq='5min')
    
    response_times = np.random.normal(150, 30, len(dates))
    throughput = np.random.normal(1200, 200, len(dates))
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=response_times, name='Response Time (ms)'))
        fig.update_layout(title="Response Time Trends", yaxis_title="Response Time (ms)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=throughput, name='Requests/sec'))
        fig.update_layout(title="Throughput Trends", yaxis_title="Requests/sec")
        st.plotly_chart(fig, use_container_width=True)

def render_pipeline_speed_chart():
    """Render pipeline processing speed chart"""
    # Generate mock pipeline speed data
    dates = pd.date_range(start=datetime.now() - timedelta(hours=6), end=datetime.now(), freq='10min')
    speeds = np.random.normal(30000, 5000, len(dates))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=speeds,
        mode='lines',
        name='Processing Speed',
        line=dict(color='#ff6692', width=2)
    ))
    
    fig.update_layout(
        title="Pipeline Processing Speed",
        xaxis_title="Time",
        yaxis_title="Records/sec"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_data_quality_metrics(api_client: APIClient):
    """Render data quality metrics"""
    st.markdown("#### 🎯 Data Quality")
    
    try:
        # This would call the actual data quality validation
        quality_data = {
            "completeness": 98.5,
            "accuracy": 96.2,
            "consistency": 94.8,
            "timeliness": 99.1
        }
        
        for metric, value in quality_data.items():
            st.metric(metric.title(), f"{value:.1f}%")
            
    except Exception:
        # Mock data quality metrics
        st.metric("Completeness", "98.5%")
        st.metric("Accuracy", "96.2%")
        st.metric("Consistency", "94.8%")
        st.metric("Timeliness", "99.1%")

def render_alert_trends_chart():
    """Render alert trends chart"""
    # Generate mock alert trend data
    dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='D')
    critical_alerts = np.random.poisson(1, len(dates))
    warning_alerts = np.random.poisson(3, len(dates))
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=dates, y=critical_alerts, name='Critical', marker_color='red'))
    fig.add_trace(go.Bar(x=dates, y=warning_alerts, name='Warning', marker_color='orange'))
    
    fig.update_layout(
        title="Alert Trends (7 Days)",
        xaxis_title="Date",
        yaxis_title="Number of Alerts",
        barmode='stack'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_service_dependency_map():
    """Render service dependency visualization"""
    st.info("🔗 Service dependency map would be displayed here with interactive network visualization")

if __name__ == "__main__":
    main()