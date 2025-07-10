"""
Live Data Integration Testing Dashboard
Test and monitor live data integration across platform connections
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta, date
import time
import json
from typing import Dict, List, Any, Optional

# Page configuration
st.set_page_config(
    page_title="Integration Testing - LiftOS",
    page_icon="🧪",
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

def test_single_connection(connection_id: str) -> Optional[Dict[str, Any]]:
    """Test a single platform connection"""
    try:
        response = requests.post(
            f"{DATA_INGESTION_URL}/api/v1/live-integration/test-connection/{connection_id}",
            headers=get_auth_headers(),
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to test connection: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error testing connection: {str(e)}")
        return None

def test_all_connections() -> Optional[List[Dict[str, Any]]]:
    """Test all platform connections"""
    try:
        response = requests.post(
            f"{DATA_INGESTION_URL}/api/v1/live-integration/test-all-connections",
            headers=get_auth_headers(),
            timeout=60
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to test connections: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error testing connections: {str(e)}")
        return None

def get_integration_status() -> Optional[Dict[str, Any]]:
    """Get overall integration status"""
    try:
        response = requests.get(
            f"{DATA_INGESTION_URL}/api/v1/live-integration/integration-status",
            headers=get_auth_headers(),
            timeout=15
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to get integration status: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error getting integration status: {str(e)}")
        return None

def run_test_suite() -> Optional[Dict[str, Any]]:
    """Run comprehensive integration test suite"""
    try:
        response = requests.post(
            f"{DATA_INGESTION_URL}/api/v1/live-integration/test-suite",
            headers=get_auth_headers(),
            timeout=120
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to run test suite: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error running test suite: {str(e)}")
        return None

def get_status_color(status: str) -> str:
    """Get color based on status"""
    colors = {
        "available": "#28a745",
        "degraded": "#ffc107", 
        "unavailable": "#dc3545",
        "testing": "#17a2b8",
        "healthy": "#28a745",
        "critical": "#dc3545",
        "no_connections": "#6c757d"
    }
    return colors.get(status.lower(), "#6c757d")

def render_integration_overview(status: Dict[str, Any]):
    """Render integration status overview"""
    st.header("🧪 Integration Testing Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Data Sources",
            status.get("total_data_sources", 0)
        )
    
    with col2:
        health_pct = status.get("health_percentage", 0)
        st.metric(
            "Health Percentage", 
            f"{health_pct}%",
            delta=f"+{health_pct-80}%" if health_pct >= 80 else f"{health_pct-80}%"
        )
    
    with col3:
        st.metric(
            "Healthy Sources",
            status.get("healthy_sources", 0),
            delta=status.get("healthy_sources", 0) - status.get("degraded_sources", 0)
        )
    
    with col4:
        avg_quality = status.get("average_quality_score", 0)
        st.metric(
            "Avg Quality Score",
            f"{avg_quality}%"
        )
    
    # Overall status indicator
    overall_status = status.get("overall_status", "unknown")
    status_color = get_status_color(overall_status)
    
    st.markdown(f"""
    <div style="padding: 1rem; border-radius: 0.5rem; background-color: {status_color}20; border-left: 4px solid {status_color};">
        <h4 style="color: {status_color}; margin: 0;">Overall Status: {overall_status.replace('_', ' ').title()}</h4>
    </div>
    """, unsafe_allow_html=True)

def render_connection_tests(test_results: List[Dict[str, Any]]):
    """Render connection test results"""
    st.subheader("🔗 Connection Test Results")
    
    if not test_results:
        st.warning("No test results available.")
        return
    
    # Summary metrics
    total_tests = len(test_results)
    successful_tests = len([t for t in test_results if t["success"]])
    success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Tests", total_tests)
    
    with col2:
        st.metric("Successful", successful_tests)
    
    with col3:
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    # Test results table
    df_data = []
    for test in test_results:
        df_data.append({
            "Connection ID": test["connection_id"][:8] + "...",
            "Platform": test["platform"].title(),
            "Success": "✅" if test["success"] else "❌",
            "Data Retrieved": "✅" if test["data_retrieved"] else "❌",
            "Records": test["record_count"],
            "Quality Score": test["quality_score"],
            "Response Time (ms)": test["response_time_ms"],
            "Error": test.get("error_message", "None")[:30] + "..." if test.get("error_message") and len(test.get("error_message", "")) > 30 else test.get("error_message", "None")
        })
    
    df = pd.DataFrame(df_data)
    
    # Performance charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Response time chart
        fig = px.bar(
            df,
            x="Platform",
            y="Response Time (ms)",
            title="Response Time by Platform",
            color="Response Time (ms)",
            color_continuous_scale="RdYlGn_r"
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Quality score chart
        fig = px.bar(
            df,
            x="Platform", 
            y="Quality Score",
            title="Quality Score by Platform",
            color="Quality Score",
            color_continuous_scale="RdYlGn"
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed results table
    st.dataframe(df, use_container_width=True)

def main():
    """Main application function"""
    # Authentication check
    if not require_auth():
        return
    
    st.title("🧪 Live Data Integration Testing")
    st.markdown("Test and monitor live data integration across your platform connections")
    
    # Sidebar navigation
    st.sidebar.title("Testing Options")
    page = st.sidebar.radio(
        "Select Test Type",
        ["Overview", "Connection Tests", "Test Suite"]
    )
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    try:
        if page == "Overview":
            # Get integration status
            with st.spinner("Loading integration status..."):
                status = get_integration_status()
            
            if status:
                render_integration_overview(status)
                
                # Show recommendations
                if status.get("recommendations"):
                    st.subheader("💡 Recommendations")
                    for rec in status["recommendations"]:
                        st.write(f"• {rec}")
            else:
                st.error("Failed to load integration status.")
        
        elif page == "Connection Tests":
            st.header("🔗 Connection Testing")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Test All Connections", type="primary"):
                    with st.spinner("Testing all connections..."):
                        results = test_all_connections()
                    
                    if results:
                        render_connection_tests(results)
            
            with col2:
                connection_id = st.text_input("Test Single Connection (ID)")
                if st.button("Test Connection") and connection_id:
                    with st.spinner(f"Testing connection {connection_id}..."):
                        result = test_single_connection(connection_id)
                    
                    if result:
                        st.json(result)
        
        elif page == "Test Suite":
            st.header("🧪 Comprehensive Testing")
            
            st.info("Run a comprehensive test suite that includes connection tests, health checks, and hybrid data validation.")
            
            if st.button("Run Test Suite", type="primary"):
                with st.spinner("Running comprehensive test suite... This may take a few minutes."):
                    results = run_test_suite()
                
                if results:
                    st.success("✅ Test suite completed!")
                    
                    # Test summary
                    summary = results.get("test_summary", {})
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Connections Tested", summary.get("total_connections_tested", 0))
                    
                    with col2:
                        st.metric("Success Rate", f"{summary.get('success_rate', 0):.1f}%")
                    
                    with col3:
                        st.metric("Avg Quality", f"{summary.get('average_quality_score', 0):.1f}%")
                    
                    with col4:
                        st.metric("Test Duration", f"{summary.get('test_duration_seconds', 0):.1f}s")
                    
                    # Recommendations
                    if results.get("recommendations"):
                        st.subheader("📋 Recommendations")
                        for i, rec in enumerate(results["recommendations"], 1):
                            st.write(f"{i}. {rec}")
                    
                    # Show detailed results
                    if results.get("connection_tests"):
                        render_connection_tests(results["connection_tests"])
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()