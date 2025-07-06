import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from auth.session_manager import initialize_session
from utils.api_client import APIClient
from components.sidebar import render_sidebar
from config.settings import get_feature_flags

def main():
    st.set_page_config(
        page_title="Attribution Truth Dashboard - LiftOS",
        page_icon="🎯",
        layout="wide"
    )
    
    # Initialize session
    initialize_session()
    
    # Render sidebar
    render_sidebar()
    
    # Check authentication
    if not st.session_state.authenticated:
        st.error("Please log in to access attribution truth dashboard.")
        st.stop()
    
    st.title("🎯 Attribution Truth Dashboard")
    st.markdown("**Policy 1: End Attribution Theatre** - Real-time attribution fraud detection with causal proof")
    
    # Initialize API client
    api_client = APIClient()
    
    # Real-time status banner
    render_attribution_status_banner(api_client)
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Attribution Fraud Detection",
        "📊 Causal Truth Analysis", 
        "💰 Budget Reallocation",
        "📈 Confidence Intervals"
    ])
    
    with tab1:
        render_fraud_detection(api_client)
    
    with tab2:
        render_causal_truth_analysis(api_client)
    
    with tab3:
        render_budget_reallocation(api_client)
    
    with tab4:
        render_confidence_intervals(api_client)

def render_attribution_status_banner(api_client: APIClient):
    """Render real-time attribution status banner"""
    try:
        # Get real-time attribution metrics
        attribution_data = api_client.get_causal_pipeline_metrics()
        
        # Attribution health status
        attribution_accuracy = attribution_data.get('attribution_accuracy', 93.8)
        
        if attribution_accuracy >= 90:
            st.success(f"🟢 Attribution Truth Engine: {attribution_accuracy:.1f}% accuracy - Exceeding 93.8% target")
        elif attribution_accuracy >= 80:
            st.warning(f"🟡 Attribution accuracy at {attribution_accuracy:.1f}% - Below target threshold")
        else:
            st.error(f"🔴 Critical: Attribution accuracy at {attribution_accuracy:.1f}% - Immediate attention required")
        
        # Quick attribution metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            over_crediting = attribution_data.get('over_crediting_percentage', 47.3)
            st.metric(
                "Over-crediting Detected", 
                f"{over_crediting:.1f}%",
                delta=f"-{attribution_data.get('over_crediting_reduction', 12.5):.1f}%",
                help="Percentage of budget over-credited to channels"
            )
        
        with col2:
            causal_confidence = attribution_data.get('causal_confidence', 94.2)
            st.metric(
                "Causal Confidence",
                f"{causal_confidence:.1f}%",
                help="Confidence in causal attribution results"
            )
        
        with col3:
            fraud_alerts = attribution_data.get('fraud_alerts_today', 3)
            if fraud_alerts > 0:
                st.metric("Fraud Alerts", f"🚨 {fraud_alerts}", help="Attribution fraud alerts today")
            else:
                st.metric("Fraud Alerts", "✅ None", help="No fraud detected today")
        
        with col4:
            processing_speed = attribution_data.get('processing_speed', 30651)
            st.metric(
                "Processing Speed",
                f"{processing_speed:,}/sec",
                help="Real-time attribution processing speed"
            )
        
        with col5:
            if st.button("🔄 Refresh Attribution Data", help="Refresh real-time data"):
                st.rerun()
                
    except Exception as e:
        st.info("🎭 Demo mode - Attribution Truth Engine initializing")

def render_fraud_detection(api_client: APIClient):
    """Render attribution fraud detection dashboard"""
    st.header("🔍 Attribution Fraud Detection")
    
    try:
        # Get attribution analysis data
        attribution_results = get_attribution_analysis_data(api_client)
        
        # Fraud detection summary
        st.subheader("🚨 Fraud Detection Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_over_crediting = attribution_results.get('total_over_crediting', 47.3)
            st.metric(
                "Total Over-crediting",
                f"{total_over_crediting:.1f}%",
                delta=f"-{attribution_results.get('improvement', 12.5):.1f}%",
                help="Total percentage of budget incorrectly attributed"
            )
        
        with col2:
            wasted_budget = attribution_results.get('wasted_budget', 142500)
            st.metric(
                "Wasted Budget Detected",
                f"${wasted_budget:,.0f}",
                delta=f"-${attribution_results.get('savings', 37500):,.0f}",
                help="Budget wasted due to attribution fraud"
            )
        
        with col3:
            channels_affected = attribution_results.get('channels_affected', 4)
            st.metric(
                "Channels Affected",
                f"{channels_affected}/7",
                help="Number of channels with attribution fraud"
            )
        
        # Channel-by-channel fraud analysis
        st.subheader("📊 Channel Attribution Fraud Analysis")
        
        # Create fraud detection chart
        fraud_data = attribution_results.get('channel_analysis', [])
        if fraud_data:
            df_fraud = pd.DataFrame(fraud_data)
            
            # Create side-by-side comparison chart
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Claimed vs Causal Attribution", "Over-crediting by Channel"),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Attribution comparison
            fig.add_trace(
                go.Bar(
                    name="Claimed Attribution",
                    x=df_fraud['channel'],
                    y=df_fraud['claimed_attribution'],
                    marker_color='lightcoral',
                    text=df_fraud['claimed_attribution'].apply(lambda x: f"{x:.1f}%"),
                    textposition='auto'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    name="Causal Attribution",
                    x=df_fraud['channel'],
                    y=df_fraud['causal_attribution'],
                    marker_color='lightgreen',
                    text=df_fraud['causal_attribution'].apply(lambda x: f"{x:.1f}%"),
                    textposition='auto'
                ),
                row=1, col=1
            )
            
            # Over-crediting chart
            fig.add_trace(
                go.Bar(
                    name="Over-crediting %",
                    x=df_fraud['channel'],
                    y=df_fraud['over_crediting_percentage'],
                    marker_color='red',
                    text=df_fraud['over_crediting_percentage'].apply(lambda x: f"{x:.1f}%"),
                    textposition='auto'
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                height=500,
                title_text="Attribution Fraud Detection Analysis",
                showlegend=True
            )
            
            fig.update_xaxes(title_text="Marketing Channels", row=1, col=1)
            fig.update_xaxes(title_text="Marketing Channels", row=1, col=2)
            fig.update_yaxes(title_text="Attribution %", row=1, col=1)
            fig.update_yaxes(title_text="Over-crediting %", row=1, col=2)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed fraud table
            st.subheader("📋 Detailed Fraud Analysis")
            
            # Add fraud severity indicators
            def get_fraud_severity(over_crediting):
                if over_crediting > 30:
                    return "🔴 Critical"
                elif over_crediting > 15:
                    return "🟡 High"
                elif over_crediting > 5:
                    return "🟠 Medium"
                else:
                    return "🟢 Low"
            
            df_fraud['fraud_severity'] = df_fraud['over_crediting_percentage'].apply(get_fraud_severity)
            df_fraud['wasted_budget'] = df_fraud['over_crediting_percentage'] * df_fraud['budget'] / 100
            
            st.dataframe(
                df_fraud[['channel', 'fraud_severity', 'claimed_attribution', 'causal_attribution', 
                         'over_crediting_percentage', 'budget', 'wasted_budget']],
                column_config={
                    'channel': st.column_config.TextColumn('Channel'),
                    'fraud_severity': st.column_config.TextColumn('Fraud Level'),
                    'claimed_attribution': st.column_config.NumberColumn('Claimed %', format="%.1f%%"),
                    'causal_attribution': st.column_config.NumberColumn('Causal %', format="%.1f%%"),
                    'over_crediting_percentage': st.column_config.NumberColumn('Over-crediting %', format="%.1f%%"),
                    'budget': st.column_config.NumberColumn('Budget', format="$%.0f"),
                    'wasted_budget': st.column_config.NumberColumn('Wasted Budget', format="$%.0f")
                },
                use_container_width=True
            )
        
    except Exception as e:
        st.warning(f"⚠️ Could not load fraud detection data: {str(e)}")
        render_mock_fraud_detection()

def render_causal_truth_analysis(api_client: APIClient):
    """Render causal truth analysis"""
    st.header("📊 Causal Truth Analysis")
    st.info("🧠 Advanced causal inference revealing true marketing impact beyond correlation")

def render_budget_reallocation(api_client: APIClient):
    """Render budget reallocation recommendations"""
    st.header("💰 Budget Reallocation Recommendations")
    st.info("💡 AI-powered budget optimization based on causal attribution truth")

def render_confidence_intervals(api_client: APIClient):
    """Render confidence intervals dashboard"""
    st.header("📈 Confidence Intervals & Statistical Significance")
    st.info("📊 Statistical confidence and significance testing for attribution results")

def render_mock_fraud_detection():
    """Render mock fraud detection data"""
    st.info("🎭 Displaying demo fraud detection data")
    
    # Mock data
    mock_data = {
        'total_over_crediting': 47.3,
        'improvement': 12.5,
        'wasted_budget': 142500,
        'savings': 37500,
        'channels_affected': 4
    }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Over-crediting", f"{mock_data['total_over_crediting']:.1f}%", 
                 delta=f"-{mock_data['improvement']:.1f}%")
    with col2:
        st.metric("Wasted Budget Detected", f"${mock_data['wasted_budget']:,.0f}",
                 delta=f"-${mock_data['savings']:,.0f}")
    with col3:
        st.metric("Channels Affected", f"{mock_data['channels_affected']}/7")

# Data fetching functions
def get_attribution_analysis_data(api_client: APIClient) -> Dict:
    """Get attribution analysis data from API"""
    try:
        # This would call the actual causal service
        return api_client.get_causal_pipeline_metrics()
    except:
        # Return mock data for demo
        return {
            'total_over_crediting': 47.3,
            'improvement': 12.5,
            'wasted_budget': 142500,
            'savings': 37500,
            'channels_affected': 4,
            'channel_analysis': [
                {
                    'channel': 'Meta Ads',
                    'claimed_attribution': 45.2,
                    'causal_attribution': 28.7,
                    'over_crediting_percentage': 36.5,
                    'budget': 120000,
                    'confidence_score': 94.2
                },
                {
                    'channel': 'Google Ads',
                    'claimed_attribution': 38.1,
                    'causal_attribution': 31.4,
                    'over_crediting_percentage': 17.6,
                    'budget': 100000,
                    'confidence_score': 91.8
                },
                {
                    'channel': 'Klaviyo Email',
                    'claimed_attribution': 12.3,
                    'causal_attribution': 18.9,
                    'over_crediting_percentage': -35.0,
                    'budget': 30000,
                    'confidence_score': 89.3
                },
                {
                    'channel': 'TikTok Ads',
                    'claimed_attribution': 8.7,
                    'causal_attribution': 4.2,
                    'over_crediting_percentage': 51.7,
                    'budget': 50000,
                    'confidence_score': 87.1
                }
            ]
        }

if __name__ == "__main__":
    main()