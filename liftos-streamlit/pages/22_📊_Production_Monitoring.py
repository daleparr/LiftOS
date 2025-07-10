"""
Production Monitoring Dashboard

This page provides comprehensive production monitoring capabilities including
system health, alerts, metrics, and performance monitoring.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any, Optional

# Page configuration
st.set_page_config(
    page_title="Production Monitoring",
    page_icon="📊",
    layout="wide"
)

# API Configuration
API_BASE_URL = "http://localhost:8006/api/v1"

def get_auth_headers() -> Dict[str, str]:
    """Get authentication headers"""
    return {
        "Authorization": f"Bearer {st.session_state.get('auth_token', '')}",
        "Content-Type": "application/json"
    }

def get_health_status() -> Dict[str, Any]:
    """Get system health status"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/rollout-monitoring/health",
            headers=get_auth_headers()
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def get_dashboard_data() -> Dict[str, Any]:
    """Get comprehensive dashboard data"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/rollout-monitoring/dashboard",
            headers=get_auth_headers()
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def get_alerts(alert_type: str = None, severity: str = None, resolved: bool = None) -> Dict[str, Any]:
    """Get alerts with filtering"""
    try:
        params = {}
        if alert_type:
            params["alert_type"] = alert_type
        if severity:
            params["severity"] = severity
        if resolved is not None:
            params["resolved"] = resolved
        
        response = requests.get(
            f"{API_BASE_URL}/rollout-monitoring/alerts",
            headers=get_auth_headers(),
            params=params
        )
        return {"success": True, "alerts": response.json()}
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_metrics(metric_name: str = None, start_time: datetime = None, end_time: datetime = None) -> Dict[str, Any]:
    """Get metrics with filtering"""
    try:
        params = {}
        if metric_name:
            params["metric_name"] = metric_name
        if start_time:
            params["start_time"] = start_time.isoformat()
        if end_time:
            params["end_time"] = end_time.isoformat()
        
        response = requests.get(
            f"{API_BASE_URL}/rollout-monitoring/metrics",
            headers=get_auth_headers(),
            params=params
        )
        return {"success": True, "metrics": response.json()}
    except Exception as e:
        return {"success": False, "error": str(e)}

def resolve_alert(alert_id: str) -> Dict[str, Any]:
    """Resolve an alert"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/rollout-monitoring/alerts/{alert_id}/resolve",
            headers=get_auth_headers()
        )
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def create_alert(alert_type: str, severity: str, title: str, description: str, source: str) -> Dict[str, Any]:
    """Create a custom alert"""
    try:
        alert_data = {
            "alert_type": alert_type,
            "severity": severity,
            "title": title,
            "description": description,
            "source": source
        }
        response = requests.post(
            f"{API_BASE_URL}/rollout-monitoring/alerts",
            headers=get_auth_headers(),
            json=alert_data
        )
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def record_metric(name: str, value: float, metric_type: str = "gauge", tags: Dict[str, str] = None) -> Dict[str, Any]:
    """Record a custom metric"""
    try:
        metric_data = {
            "name": name,
            "value": value,
            "metric_type": metric_type,
            "tags": tags or {}
        }
        response = requests.post(
            f"{API_BASE_URL}/rollout-monitoring/metrics",
            headers=get_auth_headers(),
            json=metric_data
        )
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

# Main dashboard
def main():
    st.title("📊 Production Monitoring")
    st.markdown("Comprehensive monitoring and alerting for platform connections in production.")
    
    # Check authentication
    if 'auth_token' not in st.session_state:
        st.error("Please authenticate first using the main dashboard.")
        return
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=False)
    if auto_refresh:
        st.rerun()
    
    # Sidebar for navigation
    st.sidebar.title("Monitoring")
    page = st.sidebar.selectbox(
        "Select Page",
        ["System Health", "Alerts", "Metrics", "Performance", "Error Analysis", "Custom Monitoring"]
    )
    
    if page == "System Health":
        show_system_health()
    elif page == "Alerts":
        show_alerts()
    elif page == "Metrics":
        show_metrics()
    elif page == "Performance":
        show_performance()
    elif page == "Error Analysis":
        show_error_analysis()
    elif page == "Custom Monitoring":
        show_custom_monitoring()

def show_system_health():
    """Show system health overview"""
    st.header("System Health Overview")
    
    # Get health status
    health_data = get_health_status()
    
    if "error" not in health_data:
        # Overall health status
        overall_health = health_data.get("overall_health", "unknown")
        health_percentage = health_data.get("health_percentage", 0)
        
        # Health status indicator
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if overall_health == "healthy":
                st.success(f"🟢 System Healthy ({health_percentage:.1f}%)")
            elif overall_health == "degraded":
                st.warning(f"🟡 System Degraded ({health_percentage:.1f}%)")
            else:
                st.error(f"🔴 System Unhealthy ({health_percentage:.1f}%)")
        
        with col2:
            alerts = health_data.get("alerts", {})
            total_alerts = alerts.get("total_active", 0)
            st.metric("Active Alerts", total_alerts)
        
        with col3:
            st.metric("Health Score", f"{health_percentage:.1f}%")
        
        # Service health details
        st.subheader("Service Health")
        services = health_data.get("services", {})
        
        if services:
            service_data = []
            for service_name, service_info in services.items():
                service_data.append({
                    "Service": service_name.replace("_", " ").title(),
                    "Status": service_info.get("status", "unknown"),
                    "Response Time (ms)": service_info.get("response_time", 0),
                    "Last Check": service_info.get("last_check", "")
                })
            
            df = pd.DataFrame(service_data)
            
            # Color code status
            def color_status(val):
                if val == "healthy":
                    return "background-color: #d4edda"
                elif val == "degraded":
                    return "background-color: #fff3cd"
                else:
                    return "background-color: #f8d7da"
            
            styled_df = df.style.applymap(color_status, subset=["Status"])
            st.dataframe(styled_df, use_container_width=True)
        
        # Alerts summary
        st.subheader("Alerts Summary")
        alerts_by_severity = alerts.get("by_severity", {})
        
        if alerts_by_severity:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Critical", alerts_by_severity.get("critical", 0))
            with col2:
                st.metric("High", alerts_by_severity.get("high", 0))
            with col3:
                st.metric("Medium", alerts_by_severity.get("medium", 0))
            with col4:
                st.metric("Low", alerts_by_severity.get("low", 0))
        
        # Metrics summary
        st.subheader("Key Metrics")
        metrics = health_data.get("metrics", {})
        
        if metrics:
            metric_cols = st.columns(min(4, len(metrics)))
            for i, (metric_name, metric_data) in enumerate(metrics.items()):
                with metric_cols[i % 4]:
                    current_value = metric_data.get("current", 0)
                    if "error_rate" in metric_name:
                        st.metric(
                            metric_name.replace("_", " ").title(),
                            f"{current_value:.2%}"
                        )
                    elif "response_time" in metric_name:
                        st.metric(
                            metric_name.replace("_", " ").title(),
                            f"{current_value:.0f}ms"
                        )
                    else:
                        st.metric(
                            metric_name.replace("_", " ").title(),
                            f"{current_value:.0f}"
                        )
    
    else:
        st.error(f"Failed to load health status: {health_data.get('error', 'Unknown error')}")

def show_alerts():
    """Show alerts management"""
    st.header("Alerts Management")
    
    # Alert filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        alert_type_filter = st.selectbox(
            "Alert Type",
            ["All", "performance", "error_rate", "data_quality", "connection_failure", "security", "capacity", "anomaly"]
        )
    
    with col2:
        severity_filter = st.selectbox(
            "Severity",
            ["All", "critical", "high", "medium", "low"]
        )
    
    with col3:
        status_filter = st.selectbox(
            "Status",
            ["Active Only", "Resolved Only", "All"]
        )
    
    # Apply filters
    alert_type = None if alert_type_filter == "All" else alert_type_filter
    severity = None if severity_filter == "All" else severity_filter
    resolved = None if status_filter == "All" else (status_filter == "Resolved Only")
    
    # Get alerts
    alerts_result = get_alerts(alert_type, severity, resolved)
    
    if alerts_result.get("success"):
        alerts = alerts_result.get("alerts", [])
        
        if alerts:
            # Alert statistics
            st.subheader("Alert Statistics")
            
            df = pd.DataFrame(alerts)
            
            # Severity distribution
            severity_counts = df["severity"].value_counts()
            fig = px.pie(
                values=severity_counts.values,
                names=severity_counts.index,
                title="Alerts by Severity"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Alert list
            st.subheader("Alert Details")
            
            for alert in alerts:
                severity = alert.get("severity", "low")
                resolved = alert.get("resolved", False)
                
                # Color code by severity
                if severity == "critical":
                    alert_color = "🔴"
                elif severity == "high":
                    alert_color = "🟠"
                elif severity == "medium":
                    alert_color = "🟡"
                else:
                    alert_color = "🔵"
                
                status_icon = "✅" if resolved else "⚠️"
                
                with st.expander(f"{alert_color} {status_icon} {alert.get('title', 'Unknown Alert')}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Description:**", alert.get("description", ""))
                        st.write("**Source:**", alert.get("source", ""))
                        st.write("**Severity:**", severity.title())
                        st.write("**Type:**", alert.get("alert_type", "").replace("_", " ").title())
                    
                    with col2:
                        st.write("**Timestamp:**", alert.get("timestamp", ""))
                        if resolved:
                            st.write("**Resolved At:**", alert.get("resolved_at", ""))
                            st.write("**Resolved By:**", alert.get("resolved_by", ""))
                        else:
                            if st.button("Resolve Alert", key=f"resolve_{alert.get('alert_id')}"):
                                result = resolve_alert(alert.get("alert_id"))
                                if result.get("success"):
                                    st.success("Alert resolved!")
                                    st.rerun()
                                else:
                                    st.error(f"Failed to resolve alert: {result.get('error')}")
                    
                    # Metadata
                    metadata = alert.get("metadata", {})
                    if metadata:
                        st.write("**Additional Information:**")
                        st.json(metadata)
        else:
            st.info("No alerts found with the current filters.")
    
    else:
        st.error(f"Failed to load alerts: {alerts_result.get('error', 'Unknown error')}")

def show_metrics():
    """Show metrics dashboard"""
    st.header("Metrics Dashboard")
    
    # Time range selection
    col1, col2 = st.columns(2)
    
    with col1:
        start_time = st.datetime_input("Start Time", datetime.now() - timedelta(hours=24))
    
    with col2:
        end_time = st.datetime_input("End Time", datetime.now())
    
    # Metric selection
    metric_name = st.selectbox(
        "Select Metric",
        ["All", "connections.total", "connections.active", "operations.error_rate", "operations.total"]
    )
    
    metric_filter = None if metric_name == "All" else metric_name
    
    # Get metrics
    metrics_result = get_metrics(metric_filter, start_time, end_time)
    
    if metrics_result.get("success"):
        metrics = metrics_result.get("metrics", [])
        
        if metrics:
            df = pd.DataFrame(metrics)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Group by metric name
            metric_names = df['name'].unique()
            
            for metric in metric_names:
                metric_data = df[df['name'] == metric].copy()
                metric_data = metric_data.sort_values('timestamp')
                
                st.subheader(f"Metric: {metric}")
                
                # Time series chart
                fig = px.line(
                    metric_data,
                    x='timestamp',
                    y='value',
                    title=f"{metric} Over Time"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current", f"{metric_data['value'].iloc[-1]:.2f}")
                with col2:
                    st.metric("Average", f"{metric_data['value'].mean():.2f}")
                with col3:
                    st.metric("Min", f"{metric_data['value'].min():.2f}")
                with col4:
                    st.metric("Max", f"{metric_data['value'].max():.2f}")
        else:
            st.info("No metrics found for the selected time range.")
    
    else:
        st.error(f"Failed to load metrics: {metrics_result.get('error', 'Unknown error')}")

def show_performance():
    """Show performance monitoring"""
    st.header("Performance Monitoring")
    
    # Get dashboard data for performance metrics
    dashboard_result = get_dashboard_data()
    
    if "error" not in dashboard_result:
        performance_data = dashboard_result.get("performance_metrics", {})
        
        # Response times
        st.subheader("Response Times")
        response_times = performance_data.get("response_times", [])
        
        if response_times:
            df = pd.DataFrame(response_times)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            fig = px.bar(
                df,
                x='service',
                y='response_time',
                title="Response Times by Service",
                labels={"response_time": "Response Time (ms)"}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Error rate history
        st.subheader("Error Rate Trend")
        error_history = performance_data.get("error_rate_history", [])
        
        if error_history:
            df = pd.DataFrame(error_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            fig = px.line(
                df,
                x='timestamp',
                y='value',
                title="Error Rate Over Time",
                labels={"value": "Error Rate"}
            )
            fig.update_yaxis(tickformat='.2%')
            st.plotly_chart(fig, use_container_width=True)
        
        # Platform statistics
        st.subheader("Platform Performance")
        platform_stats = dashboard_result.get("platform_statistics", {})
        
        if platform_stats:
            for platform, stats in platform_stats.items():
                if stats:
                    st.write(f"**{platform.replace('_', ' ').title()}**")
                    latest_stat = stats[-1] if stats else {}
                    st.metric(f"{platform} Connections", latest_stat.get("count", 0))
    
    else:
        st.error(f"Failed to load performance data: {dashboard_result.get('error', 'Unknown error')}")

def show_error_analysis():
    """Show error analysis"""
    st.header("Error Analysis")
    
    # Get dashboard data for error analysis
    dashboard_result = get_dashboard_data()
    
    if "error" not in dashboard_result:
        error_data = dashboard_result.get("error_analysis", {})
        
        # Error summary
        st.subheader("Error Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Errors (24h)", error_data.get("total_errors", 0))
        
        with col2:
            error_types = error_data.get("error_types", {})
            most_common_error = max(error_types.items(), key=lambda x: x[1])[0] if error_types else "None"
            st.metric("Most Common Error", most_common_error)
        
        with col3:
            error_platforms = error_data.get("error_by_platform", {})
            most_error_platform = max(error_platforms.items(), key=lambda x: x[1])[0] if error_platforms else "None"
            st.metric("Most Errors Platform", most_error_platform)
        
        # Error types distribution
        if error_types:
            st.subheader("Error Types Distribution")
            fig = px.bar(
                x=list(error_types.keys()),
                y=list(error_types.values()),
                title="Errors by Type"
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Errors by platform
        if error_platforms:
            st.subheader("Errors by Platform")
            fig = px.pie(
                values=list(error_platforms.values()),
                names=list(error_platforms.keys()),
                title="Error Distribution by Platform"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent errors
        st.subheader("Recent Errors")
        recent_errors = error_data.get("recent_errors", [])
        
        if recent_errors:
            df = pd.DataFrame(recent_errors)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No recent errors found.")
    
    else:
        st.error(f"Failed to load error analysis: {dashboard_result.get('error', 'Unknown error')}")

def show_custom_monitoring():
    """Show custom monitoring tools"""
    st.header("Custom Monitoring")
    
    # Create custom alert
    st.subheader("Create Custom Alert")
    
    with st.form("create_alert_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            alert_type = st.selectbox(
                "Alert Type",
                ["performance", "error_rate", "data_quality", "connection_failure", "security", "capacity", "anomaly"]
            )
            severity = st.selectbox("Severity", ["low", "medium", "high", "critical"])
        
        with col2:
            title = st.text_input("Alert Title")
            source = st.text_input("Source", value="custom_dashboard")
        
        description = st.text_area("Description")
        
        if st.form_submit_button("Create Alert"):
            if title and description:
                result = create_alert(alert_type, severity, title, description, source)
                if result.get("success"):
                    st.success("Alert created successfully!")
                else:
                    st.error(f"Failed to create alert: {result.get('error')}")
            else:
                st.error("Title and description are required.")
    
    # Record custom metric
    st.subheader("Record Custom Metric")
    
    with st.form("record_metric_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            metric_name = st.text_input("Metric Name")
            metric_value = st.number_input("Metric Value", value=0.0)
        
        with col2:
            metric_type = st.selectbox("Metric Type", ["gauge", "counter", "histogram", "timer"])
            tags_input = st.text_input("Tags (key:value,key:value)")
        
        if st.form_submit_button("Record Metric"):
            if metric_name:
                tags = {}
                if tags_input:
                    try:
                        for tag_pair in tags_input.split(","):
                            key, value = tag_pair.split(":")
                            tags[key.strip()] = value.strip()
                    except:
                        st.error("Invalid tags format. Use key:value,key:value")
                        return
                
                result = record_metric(metric_name, metric_value, metric_type, tags)
                if result.get("success"):
                    st.success("Metric recorded successfully!")
                else:
                    st.error(f"Failed to record metric: {result.get('error')}")
            else:
                st.error("Metric name is required.")

if __name__ == "__main__":
    main()