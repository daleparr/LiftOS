"""
Rollout Management Dashboard

This page provides comprehensive rollout management capabilities including
gradual rollout configuration, monitoring, and control.
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
    page_title="Rollout Management",
    page_icon="🚀",
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

def create_rollout(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new rollout"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/rollout-monitoring/rollouts",
            headers=get_auth_headers(),
            json=config
        )
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def start_rollout(rollout_id: str) -> Dict[str, Any]:
    """Start a rollout"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/rollout-monitoring/rollouts/{rollout_id}/start",
            headers=get_auth_headers()
        )
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def pause_rollout(rollout_id: str, reason: str = None) -> Dict[str, Any]:
    """Pause a rollout"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/rollout-monitoring/rollouts/{rollout_id}/pause",
            headers=get_auth_headers(),
            json=reason
        )
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def rollback_rollout(rollout_id: str, reason: str = None) -> Dict[str, Any]:
    """Rollback a rollout"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/rollout-monitoring/rollouts/{rollout_id}/rollback",
            headers=get_auth_headers(),
            json=reason
        )
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_rollout_status(rollout_id: str) -> Dict[str, Any]:
    """Get rollout status"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/rollout-monitoring/rollouts/{rollout_id}/status",
            headers=get_auth_headers()
        )
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def list_rollouts() -> Dict[str, Any]:
    """List all rollouts"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/rollout-monitoring/rollouts",
            headers=get_auth_headers()
        )
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_rollout_analytics() -> Dict[str, Any]:
    """Get rollout analytics"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/rollout-monitoring/rollouts/analytics",
            headers=get_auth_headers()
        )
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

# Main dashboard
def main():
    st.title("🚀 Rollout Management")
    st.markdown("Manage gradual rollouts of platform connections with monitoring and control capabilities.")
    
    # Check authentication
    if 'auth_token' not in st.session_state:
        st.error("Please authenticate first using the main dashboard.")
        return
    
    # Sidebar for navigation
    st.sidebar.title("Rollout Management")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Overview", "Create Rollout", "Active Rollouts", "Rollout Details", "Analytics"]
    )
    
    if page == "Overview":
        show_overview()
    elif page == "Create Rollout":
        show_create_rollout()
    elif page == "Active Rollouts":
        show_active_rollouts()
    elif page == "Rollout Details":
        show_rollout_details()
    elif page == "Analytics":
        show_analytics()

def show_overview():
    """Show rollout overview"""
    st.header("Rollout Overview")
    
    # Get rollout analytics
    analytics_result = get_rollout_analytics()
    
    if analytics_result.get("success"):
        analytics = analytics_result["analytics"]
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Rollouts",
                analytics.get("total_rollouts", 0)
            )
        
        with col2:
            st.metric(
                "Average Progress",
                f"{analytics.get('average_progress', 0):.1f}%"
            )
        
        with col3:
            rollout_types = analytics.get("rollout_types", {})
            most_common_type = max(rollout_types.items(), key=lambda x: x[1])[0] if rollout_types else "None"
            st.metric(
                "Most Common Type",
                most_common_type.replace("_", " ").title()
            )
        
        with col4:
            active_rollouts = len([r for r in analytics.get("rollouts", []) if r.get("progress", {}).get("percentage", 0) < 100])
            st.metric(
                "Active Rollouts",
                active_rollouts
            )
        
        # Rollout types distribution
        if rollout_types:
            st.subheader("Rollout Types Distribution")
            fig = px.pie(
                values=list(rollout_types.values()),
                names=[name.replace("_", " ").title() for name in rollout_types.keys()],
                title="Distribution of Rollout Types"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent rollouts
        st.subheader("Recent Rollouts")
        rollouts = analytics.get("rollouts", [])
        if rollouts:
            df = pd.DataFrame(rollouts)
            df["progress_percentage"] = df["progress"].apply(lambda x: x.get("percentage", 0) if isinstance(x, dict) else 0)
            df["rollout_type_display"] = df["rollout_type"].str.replace("_", " ").str.title()
            
            # Display as table
            display_df = df[["rollout_id", "name", "rollout_type_display", "progress_percentage"]].copy()
            display_df.columns = ["Rollout ID", "Name", "Type", "Progress (%)"]
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No rollouts found.")
    
    else:
        st.error(f"Failed to load rollout analytics: {analytics_result.get('error', 'Unknown error')}")

def show_create_rollout():
    """Show create rollout form"""
    st.header("Create New Rollout")
    
    with st.form("create_rollout_form"):
        # Basic information
        st.subheader("Basic Information")
        rollout_id = st.text_input("Rollout ID", help="Unique identifier for the rollout")
        name = st.text_input("Rollout Name", help="Descriptive name for the rollout")
        description = st.text_area("Description", help="Detailed description of the rollout")
        
        # Rollout type
        st.subheader("Rollout Configuration")
        rollout_type = st.selectbox(
            "Rollout Type",
            ["percentage", "user_based", "platform_based", "feature_flag", "a_b_test"],
            format_func=lambda x: x.replace("_", " ").title()
        )
        
        # Type-specific configuration
        target_percentage = None
        target_users = None
        target_platforms = None
        feature_flags = None
        
        if rollout_type == "percentage":
            target_percentage = st.slider("Target Percentage", 0.0, 100.0, 10.0, 0.1)
        
        elif rollout_type == "user_based":
            target_users_input = st.text_area("Target Users (one per line)")
            if target_users_input:
                target_users = [user.strip() for user in target_users_input.split("\n") if user.strip()]
        
        elif rollout_type == "platform_based":
            available_platforms = [
                "meta_business", "google_ads", "klaviyo", "shopify", "woocommerce",
                "amazon_seller_central", "hubspot", "salesforce", "stripe", "paypal",
                "tiktok", "snowflake", "databricks", "zoho_crm", "linkedin_ads", "x_ads"
            ]
            target_platforms = st.multiselect(
                "Target Platforms",
                available_platforms,
                format_func=lambda x: x.replace("_", " ").title()
            )
        
        elif rollout_type == "feature_flag":
            st.subheader("Feature Flags")
            flag_count = st.number_input("Number of Feature Flags", 1, 10, 1)
            feature_flags = {}
            for i in range(flag_count):
                col1, col2 = st.columns(2)
                with col1:
                    flag_name = st.text_input(f"Flag {i+1} Name", key=f"flag_name_{i}")
                with col2:
                    flag_value = st.checkbox(f"Flag {i+1} Enabled", key=f"flag_value_{i}")
                if flag_name:
                    feature_flags[flag_name] = flag_value
        
        # Scheduling
        st.subheader("Scheduling")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date (optional)")
            start_time = st.time_input("Start Time (optional)")
        with col2:
            end_date = st.date_input("End Date (optional)")
            end_time = st.time_input("End Time (optional)")
        
        # Criteria
        st.subheader("Success & Rollback Criteria")
        
        # Success criteria
        with st.expander("Success Criteria"):
            success_min_success_rate = st.slider("Minimum Success Rate", 0.0, 1.0, 0.95, 0.01)
            success_criteria = {"min_success_rate": success_min_success_rate}
        
        # Rollback criteria
        with st.expander("Rollback Criteria"):
            rollback_max_error_rate = st.slider("Maximum Error Rate", 0.0, 1.0, 0.05, 0.01)
            rollback_min_success_rate = st.slider("Minimum Success Rate (Rollback)", 0.0, 1.0, 0.80, 0.01)
            rollback_min_quality_score = st.slider("Minimum Quality Score", 0.0, 1.0, 0.70, 0.01)
            rollback_criteria = {
                "max_error_rate": rollback_max_error_rate,
                "min_success_rate": rollback_min_success_rate,
                "min_quality_score": rollback_min_quality_score
            }
        
        # Submit button
        submitted = st.form_submit_button("Create Rollout")
        
        if submitted:
            if not rollout_id or not name:
                st.error("Rollout ID and Name are required.")
                return
            
            # Prepare rollout configuration
            config = {
                "rollout_id": rollout_id,
                "name": name,
                "description": description,
                "rollout_type": rollout_type,
                "success_criteria": success_criteria,
                "rollback_criteria": rollback_criteria
            }
            
            # Add type-specific configuration
            if target_percentage is not None:
                config["target_percentage"] = target_percentage
            if target_users:
                config["target_users"] = target_users
            if target_platforms:
                config["target_platforms"] = target_platforms
            if feature_flags:
                config["feature_flags"] = feature_flags
            
            # Add scheduling
            if start_date:
                start_datetime = datetime.combine(start_date, start_time)
                config["start_date"] = start_datetime.isoformat()
            if end_date:
                end_datetime = datetime.combine(end_date, end_time)
                config["end_date"] = end_datetime.isoformat()
            
            # Create rollout
            result = create_rollout(config)
            
            if result.get("success"):
                st.success(f"Rollout '{name}' created successfully!")
                st.json(result)
            else:
                st.error(f"Failed to create rollout: {result.get('error', 'Unknown error')}")

def show_active_rollouts():
    """Show active rollouts"""
    st.header("Active Rollouts")
    
    # Get rollouts
    rollouts_result = list_rollouts()
    
    if rollouts_result.get("success"):
        rollouts = rollouts_result.get("rollouts", [])
        
        if rollouts:
            # Display rollouts
            for rollout in rollouts:
                with st.expander(f"🚀 {rollout['name']} ({rollout['rollout_id']})"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("**Type:**", rollout["rollout_type"].replace("_", " ").title())
                        progress = rollout.get("progress", {})
                        progress_pct = progress.get("percentage", 0)
                        st.progress(progress_pct / 100)
                        st.write(f"**Progress:** {progress_pct:.1f}%")
                    
                    with col2:
                        latest_metrics = rollout.get("latest_metrics")
                        if latest_metrics:
                            st.write("**Success Rate:**", f"{latest_metrics.get('success_rate', 0):.2%}")
                            st.write("**Error Rate:**", f"{latest_metrics.get('error_rate', 0):.2%}")
                            st.write("**Active Users:**", latest_metrics.get('active_users', 0))
                    
                    with col3:
                        # Control buttons
                        button_col1, button_col2, button_col3 = st.columns(3)
                        
                        with button_col1:
                            if st.button("▶️ Start", key=f"start_{rollout['rollout_id']}"):
                                result = start_rollout(rollout['rollout_id'])
                                if result.get("success"):
                                    st.success("Rollout started!")
                                    st.rerun()
                                else:
                                    st.error(f"Failed to start: {result.get('error')}")
                        
                        with button_col2:
                            if st.button("⏸️ Pause", key=f"pause_{rollout['rollout_id']}"):
                                result = pause_rollout(rollout['rollout_id'], "Manual pause")
                                if result.get("success"):
                                    st.success("Rollout paused!")
                                    st.rerun()
                                else:
                                    st.error(f"Failed to pause: {result.get('error')}")
                        
                        with button_col3:
                            if st.button("🔄 Rollback", key=f"rollback_{rollout['rollout_id']}"):
                                result = rollback_rollout(rollout['rollout_id'], "Manual rollback")
                                if result.get("success"):
                                    st.success("Rollout rolled back!")
                                    st.rerun()
                                else:
                                    st.error(f"Failed to rollback: {result.get('error')}")
        else:
            st.info("No active rollouts found.")
    
    else:
        st.error(f"Failed to load rollouts: {rollouts_result.get('error', 'Unknown error')}")

def show_rollout_details():
    """Show detailed rollout information"""
    st.header("Rollout Details")
    
    # Rollout selection
    rollouts_result = list_rollouts()
    
    if rollouts_result.get("success"):
        rollouts = rollouts_result.get("rollouts", [])
        
        if rollouts:
            rollout_options = {f"{r['name']} ({r['rollout_id']})": r['rollout_id'] for r in rollouts}
            selected_rollout_display = st.selectbox("Select Rollout", list(rollout_options.keys()))
            
            if selected_rollout_display:
                rollout_id = rollout_options[selected_rollout_display]
                
                # Get detailed status
                status_result = get_rollout_status(rollout_id)
                
                if status_result.get("success"):
                    status_data = status_result
                    config = status_data.get("config", {})
                    progress = status_data.get("progress", {})
                    latest_metrics = status_data.get("latest_metrics")
                    metrics_history = status_data.get("metrics_history", [])
                    recommendations = status_data.get("recommendations", [])
                    
                    # Configuration details
                    st.subheader("Configuration")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Rollout ID:**", config.get("rollout_id"))
                        st.write("**Name:**", config.get("name"))
                        st.write("**Type:**", config.get("rollout_type", "").replace("_", " ").title())
                        st.write("**Description:**", config.get("description"))
                    
                    with col2:
                        if config.get("target_percentage"):
                            st.write("**Target Percentage:**", f"{config['target_percentage']}%")
                        if config.get("target_users"):
                            st.write("**Target Users:**", len(config["target_users"]))
                        if config.get("target_platforms"):
                            st.write("**Target Platforms:**", ", ".join(config["target_platforms"]))
                    
                    # Progress and metrics
                    st.subheader("Progress & Metrics")
                    
                    if latest_metrics:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Success Rate", f"{latest_metrics.get('success_rate', 0):.2%}")
                        with col2:
                            st.metric("Error Rate", f"{latest_metrics.get('error_rate', 0):.2%}")
                        with col3:
                            st.metric("Active Users", latest_metrics.get('active_users', 0))
                        with col4:
                            st.metric("Total Users", latest_metrics.get('total_users', 0))
                        
                        # Progress bar
                        progress_pct = progress.get("percentage", 0)
                        st.progress(progress_pct / 100)
                        st.write(f"**Progress:** {progress_pct:.1f}%")
                    
                    # Metrics history chart
                    if metrics_history:
                        st.subheader("Metrics History")
                        
                        df = pd.DataFrame(metrics_history)
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        
                        # Success rate chart
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df['timestamp'],
                            y=df['success_rate'],
                            mode='lines+markers',
                            name='Success Rate',
                            line=dict(color='green')
                        ))
                        fig.add_trace(go.Scatter(
                            x=df['timestamp'],
                            y=df['error_rate'],
                            mode='lines+markers',
                            name='Error Rate',
                            line=dict(color='red')
                        ))
                        fig.update_layout(
                            title="Success and Error Rates Over Time",
                            xaxis_title="Time",
                            yaxis_title="Rate",
                            yaxis=dict(tickformat='.2%')
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommendations
                    if recommendations:
                        st.subheader("Recommendations")
                        for rec in recommendations:
                            rec_type = rec.get("type", "info")
                            if rec_type == "warning":
                                st.warning(f"**{rec.get('title')}**: {rec.get('description')}")
                            elif rec_type == "error":
                                st.error(f"**{rec.get('title')}**: {rec.get('description')}")
                            else:
                                st.info(f"**{rec.get('title')}**: {rec.get('description')}")
                            
                            if rec.get("action"):
                                st.write("*Recommended Action:*", rec["action"])
                    
                else:
                    st.error(f"Failed to load rollout status: {status_result.get('error', 'Unknown error')}")
        else:
            st.info("No rollouts found.")
    else:
        st.error(f"Failed to load rollouts: {rollouts_result.get('error', 'Unknown error')}")

def show_analytics():
    """Show rollout analytics"""
    st.header("Rollout Analytics")
    
    # Get analytics data
    analytics_result = get_rollout_analytics()
    
    if analytics_result.get("success"):
        analytics = analytics_result["analytics"]
        rollouts = analytics.get("rollouts", [])
        
        if rollouts:
            df = pd.DataFrame(rollouts)
            
            # Progress distribution
            st.subheader("Progress Distribution")
            df["progress_percentage"] = df["progress"].apply(lambda x: x.get("percentage", 0) if isinstance(x, dict) else 0)
            
            fig = px.histogram(
                df,
                x="progress_percentage",
                nbins=20,
                title="Distribution of Rollout Progress",
                labels={"progress_percentage": "Progress (%)", "count": "Number of Rollouts"}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Rollout types over time (if timestamp data available)
            st.subheader("Rollout Types Analysis")
            rollout_type_counts = df["rollout_type"].value_counts()
            
            fig = px.bar(
                x=rollout_type_counts.index,
                y=rollout_type_counts.values,
                title="Rollout Types Distribution",
                labels={"x": "Rollout Type", "y": "Count"}
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance metrics (if available)
            if any("latest_metrics" in rollout for rollout in rollouts):
                st.subheader("Performance Metrics")
                
                metrics_data = []
                for rollout in rollouts:
                    metrics = rollout.get("latest_metrics")
                    if metrics:
                        metrics_data.append({
                            "rollout_id": rollout["rollout_id"],
                            "name": rollout["name"],
                            "success_rate": metrics.get("success_rate", 0),
                            "error_rate": metrics.get("error_rate", 0),
                            "active_users": metrics.get("active_users", 0)
                        })
                
                if metrics_data:
                    metrics_df = pd.DataFrame(metrics_data)
                    
                    # Success vs Error rates
                    fig = px.scatter(
                        metrics_df,
                        x="success_rate",
                        y="error_rate",
                        size="active_users",
                        hover_data=["name"],
                        title="Success Rate vs Error Rate",
                        labels={"success_rate": "Success Rate", "error_rate": "Error Rate"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No rollout data available for analytics.")
    
    else:
        st.error(f"Failed to load analytics: {analytics_result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()