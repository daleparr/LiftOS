"""
Advanced Analytics Dashboard
Provides comprehensive analytics, optimization insights, and intelligent recommendations
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

# Page configuration
st.set_page_config(
    page_title="Advanced Analytics - LiftOS",
    page_icon="📊",
    layout="wide"
)

# API Configuration
API_BASE_URL = "http://localhost:8006"

def get_auth_headers():
    """Get authentication headers"""
    if 'auth_token' not in st.session_state:
        st.error("Please log in to access this page")
        st.stop()
    
    return {
        "Authorization": f"Bearer {st.session_state.auth_token}",
        "Content-Type": "application/json"
    }

def fetch_analytics_overview(timeframe: str = "last_7_days") -> Dict[str, Any]:
    """Fetch analytics overview data"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/analytics/dashboard/overview",
            headers=get_auth_headers(),
            params={"timeframe": timeframe}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch analytics overview: {response.text}")
            return {}
    except Exception as e:
        st.error(f"Error fetching analytics overview: {str(e)}")
        return {}

def fetch_performance_analytics(platform: str, timeframe: str = "last_7_days") -> Dict[str, Any]:
    """Fetch performance analytics for a platform"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/analytics/performance/{platform}",
            headers=get_auth_headers(),
            params={"timeframe": timeframe}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch performance analytics: {response.text}")
            return {}
    except Exception as e:
        st.error(f"Error fetching performance analytics: {str(e)}")
        return {}

def fetch_cost_analytics(platform: str, timeframe: str = "last_30_days") -> Dict[str, Any]:
    """Fetch cost analytics for a platform"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/analytics/cost/{platform}",
            headers=get_auth_headers(),
            params={"timeframe": timeframe}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch cost analytics: {response.text}")
            return {}
    except Exception as e:
        st.error(f"Error fetching cost analytics: {str(e)}")
        return {}

def fetch_quality_trends(platform: str, timeframe: str = "last_30_days") -> Dict[str, Any]:
    """Fetch data quality trends for a platform"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/analytics/quality/{platform}",
            headers=get_auth_headers(),
            params={"timeframe": timeframe}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch quality trends: {response.text}")
            return {}
    except Exception as e:
        st.error(f"Error fetching quality trends: {str(e)}")
        return {}

def fetch_predictive_analytics(platform: str, horizon_days: int = 30) -> Dict[str, Any]:
    """Fetch predictive analytics for a platform"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/analytics/predictive/{platform}",
            headers=get_auth_headers(),
            params={"horizon_days": horizon_days}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch predictive analytics: {response.text}")
            return {}
    except Exception as e:
        st.error(f"Error fetching predictive analytics: {str(e)}")
        return {}

def fetch_optimization_recommendations(platform: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
    """Fetch optimization recommendations"""
    try:
        params = {"limit": limit}
        if platform:
            params["platform"] = platform
            
        response = requests.get(
            f"{API_BASE_URL}/analytics/recommendations",
            headers=get_auth_headers(),
            params=params
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch recommendations: {response.text}")
            return []
    except Exception as e:
        st.error(f"Error fetching recommendations: {str(e)}")
        return []

def fetch_intelligent_insights(limit: int = 5) -> List[Dict[str, Any]]:
    """Fetch intelligent insights"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/analytics/dashboard/insights",
            headers=get_auth_headers(),
            params={"limit": limit}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch insights: {response.text}")
            return []
    except Exception as e:
        st.error(f"Error fetching insights: {str(e)}")
        return []

def fetch_anomalies(platform: Optional[str] = None, timeframe: str = "last_24_hours") -> List[Dict[str, Any]]:
    """Fetch anomaly detection results"""
    try:
        params = {"timeframe": timeframe}
        if platform:
            params["platform"] = platform
            
        response = requests.get(
            f"{API_BASE_URL}/analytics/anomalies",
            headers=get_auth_headers(),
            params=params
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch anomalies: {response.text}")
            return []
    except Exception as e:
        st.error(f"Error fetching anomalies: {str(e)}")
        return []

def apply_recommendation(recommendation_id: str) -> bool:
    """Apply an optimization recommendation"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/analytics/recommendations/{recommendation_id}/apply",
            headers=get_auth_headers()
        )
        return response.status_code == 200
    except Exception as e:
        st.error(f"Error applying recommendation: {str(e)}")
        return False

def dismiss_recommendation(recommendation_id: str, reason: Optional[str] = None) -> bool:
    """Dismiss an optimization recommendation"""
    try:
        params = {}
        if reason:
            params["reason"] = reason
            
        response = requests.post(
            f"{API_BASE_URL}/analytics/recommendations/{recommendation_id}/dismiss",
            headers=get_auth_headers(),
            params=params
        )
        return response.status_code == 200
    except Exception as e:
        st.error(f"Error dismissing recommendation: {str(e)}")
        return False

def render_performance_metrics(data: Dict[str, Any]):
    """Render performance metrics visualization"""
    if not data or 'summary' not in data:
        st.warning("No performance data available")
        return
    
    summary = data['summary']
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Requests",
            f"{summary.get('total_requests', 0):,}",
            delta=summary.get('requests_change', 0)
        )
    
    with col2:
        st.metric(
            "Avg Response Time",
            f"{summary.get('avg_response_time', 0):.2f}ms",
            delta=f"{summary.get('response_time_change', 0):.2f}ms"
        )
    
    with col3:
        st.metric(
            "Success Rate",
            f"{summary.get('success_rate', 0):.1%}",
            delta=f"{summary.get('success_rate_change', 0):.1%}"
        )
    
    with col4:
        st.metric(
            "Error Rate",
            f"{summary.get('error_rate', 0):.1%}",
            delta=f"{summary.get('error_rate_change', 0):.1%}"
        )
    
    # Performance trends
    if 'trends' in data and data['trends']:
        st.subheader("Performance Trends")
        
        trends_df = pd.DataFrame(data['trends'])
        if not trends_df.empty:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Response Time', 'Request Volume', 'Success Rate', 'Error Rate'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Response time trend
            fig.add_trace(
                go.Scatter(x=trends_df['timestamp'], y=trends_df['response_time'],
                          name='Response Time', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Request volume trend
            fig.add_trace(
                go.Scatter(x=trends_df['timestamp'], y=trends_df['request_count'],
                          name='Requests', line=dict(color='green')),
                row=1, col=2
            )
            
            # Success rate trend
            fig.add_trace(
                go.Scatter(x=trends_df['timestamp'], y=trends_df['success_rate'],
                          name='Success Rate', line=dict(color='orange')),
                row=2, col=1
            )
            
            # Error rate trend
            fig.add_trace(
                go.Scatter(x=trends_df['timestamp'], y=trends_df['error_rate'],
                          name='Error Rate', line=dict(color='red')),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

def render_cost_analytics(data: Dict[str, Any]):
    """Render cost analytics visualization"""
    if not data:
        st.warning("No cost data available")
        return
    
    # Cost summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Cost",
            f"${data.get('total_cost', 0):,.2f}",
            delta=f"${data.get('cost_change', 0):,.2f}"
        )
    
    with col2:
        st.metric(
            "Cost per Request",
            f"${data.get('cost_per_request', 0):.4f}",
            delta=f"${data.get('cost_per_request_change', 0):.4f}"
        )
    
    with col3:
        st.metric(
            "Monthly Projection",
            f"${data.get('monthly_projection', 0):,.2f}",
            delta=f"${data.get('projection_change', 0):,.2f}"
        )
    
    # Cost trends
    if 'cost_trends' in data and data['cost_trends']:
        st.subheader("Cost Trends")
        
        trends_df = pd.DataFrame(data['cost_trends'])
        if not trends_df.empty:
            fig = px.line(trends_df, x='date', y='cost', title='Daily Cost Trends')
            st.plotly_chart(fig, use_container_width=True)
    
    # Cost optimization opportunities
    if 'optimization_opportunities' in data and data['optimization_opportunities']:
        st.subheader("Cost Optimization Opportunities")
        
        for opportunity in data['optimization_opportunities']:
            with st.expander(f"💰 {opportunity.get('title', 'Optimization Opportunity')}"):
                st.write(opportunity.get('description', ''))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Potential Savings", f"${opportunity.get('potential_savings', 0):,.2f}")
                with col2:
                    st.metric("Implementation Effort", opportunity.get('effort_level', 'Unknown'))

def render_quality_trends(data: Dict[str, Any]):
    """Render data quality trends visualization"""
    if not data:
        st.warning("No quality data available")
        return
    
    # Quality score
    quality_score = data.get('quality_score', 0)
    st.metric("Overall Quality Score", f"{quality_score:.1%}")
    
    # Quality trends
    if 'trends' in data and data['trends']:
        trends_df = pd.DataFrame(data['trends'])
        if not trends_df.empty:
            fig = px.line(trends_df, x='date', y='quality_score', 
                         title='Data Quality Trends Over Time')
            st.plotly_chart(fig, use_container_width=True)
    
    # Quality issues
    if 'issues' in data and data['issues']:
        st.subheader("Quality Issues")
        
        for issue in data['issues']:
            severity = issue.get('severity', 'medium')
            severity_color = {
                'high': '🔴',
                'medium': '🟡',
                'low': '🟢'
            }.get(severity, '🟡')
            
            with st.expander(f"{severity_color} {issue.get('title', 'Quality Issue')}"):
                st.write(issue.get('description', ''))
                st.write(f"**Affected Records:** {issue.get('affected_records', 0):,}")
                st.write(f"**Impact:** {issue.get('impact', 'Unknown')}")

def render_predictive_analytics(data: Dict[str, Any]):
    """Render predictive analytics visualization"""
    if not data or 'predictions' not in data:
        st.warning("No predictive data available")
        return
    
    predictions = data['predictions']
    confidence = data.get('confidence', 0)
    
    st.metric("Prediction Confidence", f"{confidence:.1%}")
    
    if predictions:
        # Predictions chart
        pred_df = pd.DataFrame(predictions)
        if not pred_df.empty:
            fig = go.Figure()
            
            # Historical data
            if 'historical_values' in pred_df.columns:
                fig.add_trace(go.Scatter(
                    x=pred_df['date'],
                    y=pred_df['historical_values'],
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue')
                ))
            
            # Predictions
            fig.add_trace(go.Scatter(
                x=pred_df['date'],
                y=pred_df['predicted_values'],
                mode='lines',
                name='Predicted',
                line=dict(color='red', dash='dash')
            ))
            
            # Confidence intervals
            if 'confidence_upper' in pred_df.columns and 'confidence_lower' in pred_df.columns:
                fig.add_trace(go.Scatter(
                    x=pred_df['date'],
                    y=pred_df['confidence_upper'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=pred_df['date'],
                    y=pred_df['confidence_lower'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.2)',
                    name='Confidence Interval'
                ))
            
            fig.update_layout(title='Predictive Analytics', xaxis_title='Date', yaxis_title='Value')
            st.plotly_chart(fig, use_container_width=True)
    
    # Potential issues
    if 'potential_issues' in data and data['potential_issues']:
        st.subheader("Potential Issues")
        
        for issue in data['potential_issues']:
            probability = issue.get('probability', 0)
            impact = issue.get('impact', 'medium')
            
            impact_color = {
                'high': '🔴',
                'medium': '🟡',
                'low': '🟢'
            }.get(impact, '🟡')
            
            with st.expander(f"{impact_color} {issue.get('title', 'Potential Issue')} ({probability:.1%} probability)"):
                st.write(issue.get('description', ''))
                st.write(f"**Expected Date:** {issue.get('expected_date', 'Unknown')}")
                st.write(f"**Recommended Action:** {issue.get('recommended_action', 'Monitor closely')}")

def render_optimization_recommendations(recommendations: List[Dict[str, Any]]):
    """Render optimization recommendations"""
    if not recommendations:
        st.info("No optimization recommendations available")
        return
    
    st.subheader(f"Optimization Recommendations ({len(recommendations)})")
    
    for rec in recommendations:
        priority = rec.get('priority', 0.5)
        priority_color = '🔴' if priority > 0.8 else '🟡' if priority > 0.5 else '🟢'
        
        with st.expander(f"{priority_color} {rec.get('title', 'Optimization Recommendation')} (Priority: {priority:.1%})"):
            st.write(rec.get('description', ''))
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Expected Impact", rec.get('expected_impact', 'Unknown'))
            
            with col2:
                st.metric("Implementation Cost", rec.get('implementation_cost', 'Unknown'))
            
            with col3:
                st.metric("Time to Implement", rec.get('time_to_implement', 'Unknown'))
            
            # Action buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button(f"Apply Recommendation", key=f"apply_{rec.get('id')}"):
                    if apply_recommendation(rec.get('id')):
                        st.success("Recommendation applied successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to apply recommendation")
            
            with col2:
                if st.button(f"Dismiss", key=f"dismiss_{rec.get('id')}"):
                    if dismiss_recommendation(rec.get('id'), "User dismissed"):
                        st.success("Recommendation dismissed")
                        st.rerun()
                    else:
                        st.error("Failed to dismiss recommendation")

def render_intelligent_insights(insights: List[Dict[str, Any]]):
    """Render intelligent insights"""
    if not insights:
        st.info("No intelligent insights available")
        return
    
    st.subheader("🧠 Intelligent Insights")
    
    for insight in insights:
        insight_type = insight.get('type', 'general')
        type_icon = {
            'performance': '⚡',
            'cost': '💰',
            'quality': '✅',
            'anomaly': '🚨',
            'trend': '📈'
        }.get(insight_type, '💡')
        
        with st.expander(f"{type_icon} {insight.get('title', 'Insight')}"):
            st.write(insight.get('description', ''))
            
            if 'confidence' in insight:
                st.metric("Confidence", f"{insight['confidence']:.1%}")
            
            if 'recommended_action' in insight:
                st.info(f"**Recommended Action:** {insight['recommended_action']}")

def render_anomaly_detection(anomalies: List[Dict[str, Any]]):
    """Render anomaly detection results"""
    if not anomalies:
        st.success("No anomalies detected")
        return
    
    st.subheader(f"🚨 Anomalies Detected ({len(anomalies)})")
    
    for anomaly in anomalies:
        severity = anomaly.get('severity', 0.5)
        severity_color = '🔴' if severity > 0.8 else '🟡' if severity > 0.5 else '🟢'
        
        with st.expander(f"{severity_color} {anomaly.get('title', 'Anomaly')} (Severity: {severity:.1%})"):
            st.write(anomaly.get('description', ''))
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Detected At", anomaly.get('detected_at', 'Unknown'))
            
            with col2:
                st.metric("Platform", anomaly.get('platform', 'Unknown'))
            
            if 'recommended_action' in anomaly:
                st.warning(f"**Recommended Action:** {anomaly['recommended_action']}")

# Main dashboard
def main():
    st.title("📊 Advanced Analytics Dashboard")
    st.markdown("Comprehensive analytics, optimization insights, and intelligent recommendations")
    
    # Sidebar controls
    st.sidebar.header("Analytics Controls")
    
    # Platform selection
    platforms = [
        "meta_business", "google_ads", "klaviyo", "shopify", "woocommerce",
        "amazon_seller_central", "hubspot", "salesforce", "stripe", "paypal",
        "tiktok", "snowflake", "databricks", "zoho_crm", "linkedin_ads", "x_ads"
    ]
    
    selected_platform = st.sidebar.selectbox(
        "Select Platform",
        ["All Platforms"] + platforms
    )
    
    # Timeframe selection
    timeframe = st.sidebar.selectbox(
        "Timeframe",
        ["last_24_hours", "last_7_days", "last_30_days", "last_90_days"]
    )
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", "⚡ Performance", "💰 Cost Analysis", 
        "✅ Quality", "🔮 Predictions", "🎯 Recommendations"
    ])
    
    with tab1:
        st.header("Analytics Overview")
        
        # Fetch overview data
        overview_data = fetch_analytics_overview(timeframe)
        
        if overview_data:
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Platforms", overview_data.get('total_platforms', 0))
            
            with col2:
                st.metric("Active Connections", overview_data.get('active_connections', 0))
            
            with col3:
                st.metric("Data Quality Score", f"{overview_data.get('avg_quality_score', 0):.1%}")
            
            with col4:
                st.metric("Total Cost", f"${overview_data.get('total_cost', 0):,.2f}")
        
        # Intelligent insights
        insights = fetch_intelligent_insights(limit=5)
        render_intelligent_insights(insights)
        
        # Anomaly detection
        anomalies = fetch_anomalies(
            platform=None if selected_platform == "All Platforms" else selected_platform,
            timeframe=timeframe
        )
        render_anomaly_detection(anomalies)
    
    with tab2:
        st.header("Performance Analytics")
        
        if selected_platform != "All Platforms":
            performance_data = fetch_performance_analytics(selected_platform, timeframe)
            render_performance_metrics(performance_data)
        else:
            st.info("Please select a specific platform to view performance analytics")
    
    with tab3:
        st.header("Cost Analysis")
        
        if selected_platform != "All Platforms":
            cost_data = fetch_cost_analytics(selected_platform, timeframe)
            render_cost_analytics(cost_data)
        else:
            st.info("Please select a specific platform to view cost analytics")
    
    with tab4:
        st.header("Data Quality Trends")
        
        if selected_platform != "All Platforms":
            quality_data = fetch_quality_trends(selected_platform, timeframe)
            render_quality_trends(quality_data)
        else:
            st.info("Please select a specific platform to view quality trends")
    
    with tab5:
        st.header("Predictive Analytics")
        
        if selected_platform != "All Platforms":
            horizon_days = st.slider("Prediction Horizon (days)", 7, 365, 30)
            predictive_data = fetch_predictive_analytics(selected_platform, horizon_days)
            render_predictive_analytics(predictive_data)
        else:
            st.info("Please select a specific platform to view predictive analytics")
    
    with tab6:
        st.header("Optimization Recommendations")
        
        # Filters
        col1, col2 = st.columns(2)
        
        with col1:
            optimization_type = st.selectbox(
                "Optimization Type",
                ["All", "Performance", "Cost", "Quality", "Security"]
            )
        
        with col2:
            priority_threshold = st.slider("Minimum Priority", 0.0, 1.0, 0.5)
        
        # Fetch recommendations
        recommendations = fetch_optimization_recommendations(
            platform=None if selected_platform == "All Platforms" else selected_platform,
            limit=20
        )
        
        # Filter by priority
        filtered_recommendations = [
            rec for rec in recommendations 
            if rec.get('priority', 0) >= priority_threshold
        ]
        
        render_optimization_recommendations(filtered_recommendations)

if __name__ == "__main__":
    main()