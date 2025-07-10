"""
Data Quality Monitoring Dashboard
Monitor and validate data quality across platform connections
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import time
from typing import Dict, List, Any, Optional

# Page configuration
st.set_page_config(
    page_title="Data Quality - LiftOS",
    page_icon="📊",
    layout="wide"
)

# Import shared utilities
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.utils.config import get_service_config
from shared.utils.auth import get_user_context, require_auth

# Service configuration
config = get_service_config()
DATA_INGESTION_URL = config.get("data_ingestion_service_url", "http://localhost:8001")

def get_auth_headers() -> Dict[str, str]:
    """Get authentication headers for API requests"""
    user_context = get_user_context()
    return {
        "X-User-ID": user_context["user_id"],
        "X-Org-ID": user_context["org_id"],
        "X-Memory-Context": user_context.get("memory_context", ""),
        "X-User-Roles": ",".join(user_context.get("roles", []))
    }

def fetch_quality_summary() -> Optional[Dict[str, Any]]:
    """Fetch overall data quality summary"""
    try:
        response = requests.get(
            f"{DATA_INGESTION_URL}/api/v1/data-validation/quality-summary",
            headers=get_auth_headers(),
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch quality summary: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching quality summary: {str(e)}")
        return None

def fetch_validation_reports() -> Optional[List[Dict[str, Any]]]:
    """Fetch all validation reports"""
    try:
        response = requests.get(
            f"{DATA_INGESTION_URL}/api/v1/data-validation/connections/validate-all",
            headers=get_auth_headers(),
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch validation reports: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching validation reports: {str(e)}")
        return None

def fetch_connection_validation(connection_id: str) -> Optional[Dict[str, Any]]:
    """Fetch validation report for specific connection"""
    try:
        response = requests.get(
            f"{DATA_INGESTION_URL}/api/v1/data-validation/connections/{connection_id}/validate",
            headers=get_auth_headers(),
            timeout=20
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch connection validation: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching connection validation: {str(e)}")
        return None

def fetch_validation_rules(platform: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
    """Fetch available validation rules"""
    try:
        params = {"platform": platform} if platform else {}
        response = requests.get(
            f"{DATA_INGESTION_URL}/api/v1/data-validation/validation-rules",
            headers=get_auth_headers(),
            params=params,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch validation rules: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching validation rules: {str(e)}")
        return None

def get_quality_color(score: float) -> str:
    """Get color based on quality score"""
    if score >= 90:
        return "#28a745"  # Green
    elif score >= 75:
        return "#ffc107"  # Yellow
    elif score >= 60:
        return "#fd7e14"  # Orange
    else:
        return "#dc3545"  # Red

def get_quality_level_color(level: str) -> str:
    """Get color based on quality level"""
    colors = {
        "excellent": "#28a745",
        "good": "#20c997",
        "fair": "#ffc107",
        "poor": "#fd7e14",
        "critical": "#dc3545"
    }
    return colors.get(level.lower(), "#6c757d")

def render_quality_overview(summary: Dict[str, Any]):
    """Render quality overview section"""
    st.header("📊 Data Quality Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Connections",
            summary.get("total_connections", 0)
        )
    
    with col2:
        avg_score = summary.get("average_score", 0)
        st.metric(
            "Average Quality Score",
            f"{avg_score}%",
            delta=None
        )
    
    with col3:
        quality_dist = summary.get("quality_distribution", {})
        excellent_count = quality_dist.get("excellent", 0)
        st.metric(
            "Excellent Quality",
            excellent_count
        )
    
    with col4:
        critical_count = quality_dist.get("critical", 0)
        st.metric(
            "Critical Issues",
            critical_count,
            delta=f"-{critical_count}" if critical_count > 0 else None
        )

def render_quality_distribution(summary: Dict[str, Any]):
    """Render quality distribution chart"""
    quality_dist = summary.get("quality_distribution", {})
    
    if quality_dist:
        # Create pie chart
        labels = list(quality_dist.keys())
        values = list(quality_dist.values())
        colors = [get_quality_level_color(label) for label in labels]
        
        fig = go.Figure(data=[go.Pie(
            labels=[label.title() for label in labels],
            values=values,
            marker_colors=colors,
            hole=0.4
        )])
        
        fig.update_layout(
            title="Quality Level Distribution",
            showlegend=True,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_top_issues(summary: Dict[str, Any]):
    """Render top issues section"""
    st.subheader("🚨 Top Issues")
    
    top_issues = summary.get("top_issues", [])
    
    if top_issues:
        issues_df = pd.DataFrame(top_issues)
        issues_df.columns = ["Rule ID", "Affected Connections"]
        
        # Create bar chart
        fig = px.bar(
            issues_df,
            x="Affected Connections",
            y="Rule ID",
            orientation="h",
            title="Most Common Validation Issues"
        )
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No critical issues found! 🎉")

def render_recommendations(summary: Dict[str, Any]):
    """Render recommendations section"""
    st.subheader("💡 Recommendations")
    
    recommendations = summary.get("recommendations", [])
    
    if recommendations:
        for i, rec in enumerate(recommendations[:5], 1):
            st.write(f"{i}. {rec}")
    else:
        st.info("No specific recommendations at this time.")

def render_connection_details(reports: List[Dict[str, Any]]):
    """Render detailed connection validation results"""
    st.header("🔗 Connection Details")
    
    if not reports:
        st.warning("No validation reports available.")
        return
    
    # Create DataFrame for easier manipulation
    df_data = []
    for report in reports:
        df_data.append({
            "Connection ID": report["connection_id"],
            "Platform": report["platform"].title(),
            "Quality Score": report["overall_score"],
            "Quality Level": report["quality_level"].title(),
            "Issues": len([r for r in report["validation_results"] if r["status"] in ["failed", "warning"]]),
            "Last Updated": report["generated_at"]
        })
    
    df = pd.DataFrame(df_data)
    
    # Add filters
    col1, col2 = st.columns(2)
    
    with col1:
        platform_filter = st.selectbox(
            "Filter by Platform",
            ["All"] + sorted(df["Platform"].unique().tolist())
        )
    
    with col2:
        quality_filter = st.selectbox(
            "Filter by Quality Level",
            ["All"] + sorted(df["Quality Level"].unique().tolist())
        )
    
    # Apply filters
    filtered_df = df.copy()
    if platform_filter != "All":
        filtered_df = filtered_df[filtered_df["Platform"] == platform_filter]
    if quality_filter != "All":
        filtered_df = filtered_df[filtered_df["Quality Level"] == quality_filter]
    
    # Display table with styling
    st.dataframe(
        filtered_df.style.format({
            "Quality Score": "{:.1f}%"
        }).background_gradient(
            subset=["Quality Score"],
            cmap="RdYlGn",
            vmin=0,
            vmax=100
        ),
        use_container_width=True
    )
    
    # Connection selection for detailed view
    if not filtered_df.empty:
        st.subheader("📋 Detailed Validation Results")
        
        selected_connection = st.selectbox(
            "Select connection for detailed analysis",
            filtered_df["Connection ID"].tolist(),
            format_func=lambda x: f"{x} ({filtered_df[filtered_df['Connection ID'] == x]['Platform'].iloc[0]})"
        )
        
        if selected_connection:
            render_connection_validation_details(selected_connection, reports)

def render_connection_validation_details(connection_id: str, reports: List[Dict[str, Any]]):
    """Render detailed validation results for a specific connection"""
    # Find the report for this connection
    report = next((r for r in reports if r["connection_id"] == connection_id), None)
    
    if not report:
        st.error("Report not found for selected connection.")
        return
    
    # Basic info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Overall Score", f"{report['overall_score']:.1f}%")
    
    with col2:
        st.metric("Quality Level", report["quality_level"].title())
    
    with col3:
        issues_count = len([r for r in report["validation_results"] if r["status"] in ["failed", "warning"]])
        st.metric("Issues Found", issues_count)
    
    # Validation results
    st.subheader("Validation Results")
    
    validation_data = []
    for result in report["validation_results"]:
        validation_data.append({
            "Rule": result["rule_id"],
            "Status": result["status"].title(),
            "Score": f"{result['score']:.1f}%",
            "Message": result["message"]
        })
    
    validation_df = pd.DataFrame(validation_data)
    
    # Color code by status
    def color_status(val):
        if val == "Failed":
            return "background-color: #ffebee"
        elif val == "Warning":
            return "background-color: #fff3e0"
        elif val == "Passed":
            return "background-color: #e8f5e8"
        return ""
    
    st.dataframe(
        validation_df.style.applymap(color_status, subset=["Status"]),
        use_container_width=True
    )
    
    # Metrics breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Freshness")
        freshness = report.get("data_freshness", {})
        if freshness:
            st.write(f"**Most Recent:** {freshness.get('most_recent_date', 'N/A')}")
            st.write(f"**Days Since Update:** {freshness.get('days_since_recent', 'N/A')}")
            st.write(f"**Freshness Score:** {freshness.get('freshness_score', 0):.1f}%")
    
    with col2:
        st.subheader("Completeness")
        completeness = report.get("completeness_metrics", {})
        if completeness:
            st.write(f"**Total Records:** {completeness.get('total_records', 'N/A')}")
            st.write(f"**Overall Completeness:** {completeness.get('overall_completeness', 0):.1f}%")
    
    # Recommendations
    if report.get("recommendations"):
        st.subheader("Recommendations")
        for i, rec in enumerate(report["recommendations"], 1):
            st.write(f"{i}. {rec}")

def render_validation_rules():
    """Render validation rules information"""
    st.header("📋 Validation Rules")
    
    # Platform filter
    platform_filter = st.selectbox(
        "Filter rules by platform",
        ["All", "meta_business", "google_ads", "klaviyo", "shopify", "hubspot"]
    )
    
    rules = fetch_validation_rules(platform_filter if platform_filter != "All" else None)
    
    if rules:
        rules_data = []
        for rule in rules:
            rules_data.append({
                "Rule ID": rule["rule_id"],
                "Name": rule["name"],
                "Description": rule["description"],
                "Severity": rule["severity"].title(),
                "Platform Specific": "Yes" if rule["platform_specific"] else "No",
                "Applicable Platforms": ", ".join(rule.get("applicable_platforms", []) or ["All"])
            })
        
        rules_df = pd.DataFrame(rules_data)
        
        # Color code by severity
        def color_severity(val):
            if val == "Critical":
                return "background-color: #ffebee"
            elif val == "High":
                return "background-color: #fff3e0"
            elif val == "Medium":
                return "background-color: #e3f2fd"
            return ""
        
        st.dataframe(
            rules_df.style.applymap(color_severity, subset=["Severity"]),
            use_container_width=True
        )

def main():
    """Main application function"""
    # Authentication check
    if not require_auth():
        return
    
    st.title("📊 Data Quality Monitoring")
    st.markdown("Monitor and validate data quality across your platform connections")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select View",
        ["Overview", "Connection Details", "Validation Rules"]
    )
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)")
    
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    try:
        if page == "Overview":
            # Fetch data
            with st.spinner("Loading quality summary..."):
                summary = fetch_quality_summary()
            
            if summary:
                render_quality_overview(summary)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    render_quality_distribution(summary)
                
                with col2:
                    render_top_issues(summary)
                
                render_recommendations(summary)
            else:
                st.error("Failed to load quality summary.")
        
        elif page == "Connection Details":
            # Fetch validation reports
            with st.spinner("Loading validation reports..."):
                reports = fetch_validation_reports()
            
            if reports:
                render_connection_details(reports)
            else:
                st.error("Failed to load validation reports.")
        
        elif page == "Validation Rules":
            render_validation_rules()
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()