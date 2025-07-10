"""
Platform Connections Page
User interface for connecting and managing marketing platform APIs
"""

import streamlit as st
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.api_client import APIClient
from auth.session_manager import get_user_context
from config.settings import get_feature_flags

# Page configuration
st.set_page_config(
    page_title="Platform Connections - LiftOS",
    page_icon="🔗",
    layout="wide"
)

# Initialize API client
api_client = APIClient()

def get_platform_icon(platform: str) -> str:
    """Get emoji icon for platform"""
    icons = {
        'meta_business': '📘',
        'google_ads': '🔍',
        'klaviyo': '📧',
        'shopify': '🛍️',
        'woocommerce': '🛒',
        'amazon_seller_central': '📦',
        'hubspot': '🎯',
        'salesforce': '☁️',
        'stripe': '💳',
        'paypal': '💰',
        'tiktok': '🎵',
        'snowflake': '❄️',
        'databricks': '🧱',
        'zoho_crm': '📋',
        'linkedin_ads': '💼',
        'x_ads': '🐦'
    }
    return icons.get(platform, '🔌')

def get_platform_display_name(platform: str) -> str:
    """Get display name for platform"""
    names = {
        'meta_business': 'Meta Business',
        'google_ads': 'Google Ads',
        'klaviyo': 'Klaviyo',
        'shopify': 'Shopify',
        'woocommerce': 'WooCommerce',
        'amazon_seller_central': 'Amazon Seller Central',
        'hubspot': 'HubSpot',
        'salesforce': 'Salesforce',
        'stripe': 'Stripe',
        'paypal': 'PayPal',
        'tiktok': 'TikTok for Business',
        'snowflake': 'Snowflake',
        'databricks': 'Databricks',
        'zoho_crm': 'Zoho CRM',
        'linkedin_ads': 'LinkedIn Ads',
        'x_ads': 'X (Twitter) Ads'
    }
    return names.get(platform, platform.replace('_', ' ').title())

def get_connection_status_color(status: str) -> str:
    """Get color for connection status"""
    colors = {
        'active': 'green',
        'inactive': 'red',
        'pending': 'orange',
        'error': 'red',
        'expired': 'orange'
    }
    return colors.get(status, 'gray')

def render_connection_card(connection: Dict[str, Any]):
    """Render a connection card"""
    platform = connection['platform']
    status = connection['status']
    
    with st.container():
        col1, col2, col3, col4 = st.columns([1, 3, 2, 2])
        
        with col1:
            st.markdown(f"## {get_platform_icon(platform)}")
        
        with col2:
            st.markdown(f"**{get_platform_display_name(platform)}**")
            st.markdown(f"Connected: {connection['created_at'][:10]}")
            if connection.get('last_sync_at'):
                st.markdown(f"Last sync: {connection['last_sync_at'][:10]}")
        
        with col3:
            status_color = get_connection_status_color(status)
            st.markdown(f"<span style='color: {status_color}'>●</span> {status.title()}", 
                       unsafe_allow_html=True)
            
            if connection.get('sync_frequency'):
                st.markdown(f"Sync: {connection['sync_frequency']}")
        
        with col4:
            col4a, col4b, col4c = st.columns(3)
            
            with col4a:
                if st.button("Test", key=f"test_{connection['id']}"):
                    test_connection(connection['id'])
            
            with col4b:
                if st.button("Sync", key=f"sync_{connection['id']}"):
                    sync_connection(connection['id'])
            
            with col4c:
                if st.button("⚙️", key=f"settings_{connection['id']}"):
                    st.session_state.selected_connection = connection['id']
                    st.rerun()
        
        st.divider()

def render_platform_selection():
    """Render platform selection for new connections"""
    st.subheader("🔌 Connect New Platform")
    
    try:
        # Get supported platforms
        response = api_client.get("/api/v1/platform-connections/platforms")
        if response.status_code == 200:
            platforms = response.json()
            
            # Group platforms by tier
            tiers = {
                'Tier 0 - Core Marketing': ['meta_business', 'google_ads', 'klaviyo'],
                'Tier 1 - E-commerce': ['shopify', 'woocommerce', 'amazon_seller_central'],
                'Tier 2 - CRM & Payments': ['hubspot', 'salesforce', 'stripe', 'paypal'],
                'Tier 3 - Social & Analytics': ['tiktok', 'snowflake', 'databricks'],
                'Tier 4 - Extended Platforms': ['zoho_crm', 'linkedin_ads', 'x_ads']
            }
            
            for tier_name, tier_platforms in tiers.items():
                with st.expander(tier_name, expanded=(tier_name == 'Tier 0 - Core Marketing')):
                    cols = st.columns(3)
                    
                    for i, platform_id in enumerate(tier_platforms):
                        platform_config = next((p for p in platforms if p['platform'] == platform_id), None)
                        if platform_config:
                            with cols[i % 3]:
                                icon = get_platform_icon(platform_id)
                                name = get_platform_display_name(platform_id)
                                
                                if st.button(f"{icon} {name}", key=f"connect_{platform_id}"):
                                    initiate_connection(platform_id)
        else:
            st.error("Failed to load supported platforms")
            
    except Exception as e:
        st.error(f"Error loading platforms: {str(e)}")

def initiate_connection(platform: str):
    """Initiate OAuth flow for platform connection"""
    try:
        user_context = get_user_context()
        
        # Check if platform supports OAuth
        oauth_request = {
            "platform": platform,
            "redirect_uri": f"{st.secrets.get('app_url', 'http://localhost:8501')}/platform-connections/callback",
            "scopes": ["read_insights", "read_campaigns"]  # Default scopes
        }
        
        response = api_client.post("/api/v1/platform-connections/oauth/initiate", json=oauth_request)
        
        if response.status_code == 200:
            oauth_data = response.json()
            
            # Display OAuth URL for user to click
            st.success(f"Ready to connect {get_platform_display_name(platform)}!")
            st.markdown(f"[Click here to authorize LiftOS]({oauth_data['authorization_url']})")
            st.info("After authorization, you'll be redirected back to LiftOS.")
            
        else:
            # For non-OAuth platforms, show manual credential form
            show_manual_credential_form(platform)
            
    except Exception as e:
        st.error(f"Failed to initiate connection: {str(e)}")

def show_manual_credential_form(platform: str):
    """Show manual credential input form for platforms that don't support OAuth"""
    st.subheader(f"Connect {get_platform_display_name(platform)}")
    
    with st.form(f"credentials_{platform}"):
        st.markdown("Enter your API credentials:")
        
        credentials = {}
        
        # Platform-specific credential fields
        if platform == 'meta_business':
            credentials['access_token'] = st.text_input("Access Token", type="password")
            credentials['app_id'] = st.text_input("App ID")
            credentials['app_secret'] = st.text_input("App Secret", type="password")
            credentials['ad_account_id'] = st.text_input("Ad Account ID")
            
        elif platform == 'google_ads':
            credentials['client_id'] = st.text_input("Client ID")
            credentials['client_secret'] = st.text_input("Client Secret", type="password")
            credentials['refresh_token'] = st.text_input("Refresh Token", type="password")
            credentials['developer_token'] = st.text_input("Developer Token", type="password")
            credentials['customer_id'] = st.text_input("Customer ID")
            
        elif platform == 'klaviyo':
            credentials['api_key'] = st.text_input("API Key", type="password")
            
        elif platform == 'shopify':
            credentials['shop_domain'] = st.text_input("Shop Domain (e.g., mystore.myshopify.com)")
            credentials['access_token'] = st.text_input("Access Token", type="password")
            
        else:
            # Generic form for other platforms
            credentials['api_key'] = st.text_input("API Key", type="password")
            credentials['api_secret'] = st.text_input("API Secret", type="password")
        
        # Connection settings
        st.subheader("Connection Settings")
        connection_name = st.text_input("Connection Name", value=f"My {get_platform_display_name(platform)}")
        sync_frequency = st.selectbox("Sync Frequency", 
                                    ["daily", "hourly", "weekly", "manual"],
                                    index=0)
        
        submitted = st.form_submit_button("Create Connection")
        
        if submitted:
            create_manual_connection(platform, credentials, connection_name, sync_frequency)

def create_manual_connection(platform: str, credentials: Dict[str, str], name: str, sync_frequency: str):
    """Create a manual platform connection"""
    try:
        connection_request = {
            "platform": platform,
            "connection_name": name,
            "credentials": credentials,
            "sync_frequency": sync_frequency,
            "auto_sync_enabled": sync_frequency != "manual"
        }
        
        response = api_client.post("/api/v1/platform-connections/connections", json=connection_request)
        
        if response.status_code == 200:
            st.success(f"Successfully connected {get_platform_display_name(platform)}!")
            st.rerun()
        else:
            error_data = response.json()
            st.error(f"Failed to create connection: {error_data.get('detail', 'Unknown error')}")
            
    except Exception as e:
        st.error(f"Error creating connection: {str(e)}")

def test_connection(connection_id: str):
    """Test a platform connection"""
    try:
        response = api_client.post(f"/api/v1/platform-connections/connections/{connection_id}/test")
        
        if response.status_code == 200:
            test_result = response.json()
            
            if test_result['success']:
                st.success("✅ Connection test successful!")
                if test_result.get('data_preview'):
                    st.json(test_result['data_preview'])
            else:
                st.error(f"❌ Connection test failed: {test_result.get('error_message', 'Unknown error')}")
        else:
            st.error("Failed to test connection")
            
    except Exception as e:
        st.error(f"Error testing connection: {str(e)}")

def sync_connection(connection_id: str):
    """Manually sync a platform connection"""
    try:
        sync_request = {
            "sync_type": "incremental",
            "sync_config": {
                "date_range_days": 7
            }
        }
        
        response = api_client.post(f"/api/v1/platform-connections/connections/{connection_id}/sync", 
                                 json=sync_request)
        
        if response.status_code == 200:
            sync_result = response.json()
            st.success(f"✅ Sync initiated! Job ID: {sync_result['sync_job_id']}")
            st.info("Check the sync status in the dashboard below.")
        else:
            st.error("Failed to initiate sync")
            
    except Exception as e:
        st.error(f"Error syncing connection: {str(e)}")

def render_connection_dashboard():
    """Render the connections dashboard"""
    try:
        response = api_client.get("/api/v1/platform-connections/dashboard")
        
        if response.status_code == 200:
            dashboard_data = response.json()
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Connections", dashboard_data['total_connections'])
            
            with col2:
                st.metric("Active Connections", dashboard_data['active_connections'])
            
            with col3:
                st.metric("Data Sources", dashboard_data['unique_platforms'])
            
            with col4:
                last_sync = dashboard_data.get('last_successful_sync')
                if last_sync:
                    st.metric("Last Sync", last_sync[:10])
                else:
                    st.metric("Last Sync", "Never")
            
            # Connection status chart
            if dashboard_data.get('connection_status_summary'):
                st.subheader("📊 Connection Status Overview")
                
                status_data = dashboard_data['connection_status_summary']
                fig = px.pie(
                    values=list(status_data.values()),
                    names=list(status_data.keys()),
                    title="Connection Status Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Recent sync activity
            if dashboard_data.get('recent_sync_logs'):
                st.subheader("🔄 Recent Sync Activity")
                
                sync_logs = dashboard_data['recent_sync_logs']
                df = pd.DataFrame(sync_logs)
                
                if not df.empty:
                    # Format the dataframe for display
                    df['started_at'] = pd.to_datetime(df['started_at']).dt.strftime('%Y-%m-%d %H:%M')
                    df['platform'] = df['platform'].apply(get_platform_display_name)
                    
                    st.dataframe(
                        df[['platform', 'sync_type', 'status', 'records_synced', 'started_at']],
                        use_container_width=True
                    )
                else:
                    st.info("No recent sync activity")
            
        else:
            st.error("Failed to load dashboard data")
            
    except Exception as e:
        st.error(f"Error loading dashboard: {str(e)}")

def render_data_preferences():
    """Render data preferences settings"""
    st.subheader("⚙️ Data Preferences")
    
    try:
        response = api_client.get("/api/v1/platform-connections/preferences")
        
        if response.status_code == 200:
            preferences = response.json()
            
            with st.form("data_preferences"):
                col1, col2 = st.columns(2)
                
                with col1:
                    prefer_live_data = st.checkbox(
                        "Prefer Live Data",
                        value=preferences.get('prefer_live_data', True),
                        help="Use live data from connected platforms when available"
                    )
                    
                    fallback_to_mock = st.checkbox(
                        "Fallback to Mock Data",
                        value=preferences.get('fallback_to_mock', True),
                        help="Use mock data when live data is unavailable"
                    )
                
                with col2:
                    auto_sync_enabled = st.checkbox(
                        "Auto Sync Enabled",
                        value=preferences.get('auto_sync_enabled', True),
                        help="Automatically sync data based on connection settings"
                    )
                    
                    data_retention_days = st.number_input(
                        "Data Retention (Days)",
                        min_value=1,
                        max_value=365,
                        value=preferences.get('data_retention_days', 90),
                        help="How long to keep synced data"
                    )
                
                submitted = st.form_submit_button("Save Preferences")
                
                if submitted:
                    update_preferences = {
                        "prefer_live_data": prefer_live_data,
                        "fallback_to_mock": fallback_to_mock,
                        "auto_sync_enabled": auto_sync_enabled,
                        "data_retention_days": data_retention_days
                    }
                    
                    response = api_client.put("/api/v1/platform-connections/preferences", 
                                            json=update_preferences)
                    
                    if response.status_code == 200:
                        st.success("✅ Preferences updated successfully!")
                    else:
                        st.error("Failed to update preferences")
        else:
            st.error("Failed to load preferences")
            
    except Exception as e:
        st.error(f"Error loading preferences: {str(e)}")

def main():
    """Main page function"""
    st.title("🔗 Platform Connections")
    st.markdown("Connect and manage your marketing platform APIs for real-time data analysis")
    
    # Check authentication
    user_context = get_user_context()
    if not user_context:
        st.error("Please log in to access platform connections")
        st.stop()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 My Connections", "🔌 Connect New", "📊 Dashboard", "⚙️ Settings"])
    
    with tab1:
        st.subheader("📋 My Platform Connections")
        
        try:
            response = api_client.get("/api/v1/platform-connections/connections")
            
            if response.status_code == 200:
                connections = response.json()
                
                if connections:
                    for connection in connections:
                        render_connection_card(connection)
                else:
                    st.info("No platform connections found. Connect your first platform in the 'Connect New' tab.")
            else:
                st.error("Failed to load connections")
                
        except Exception as e:
            st.error(f"Error loading connections: {str(e)}")
    
    with tab2:
        render_platform_selection()
    
    with tab3:
        render_connection_dashboard()
    
    with tab4:
        render_data_preferences()
    
    # Handle OAuth callback if present
    query_params = st.experimental_get_query_params()
    if 'code' in query_params and 'state' in query_params:
        handle_oauth_callback(query_params)

def handle_oauth_callback(query_params: Dict[str, List[str]]):
    """Handle OAuth callback from platform"""
    try:
        callback_request = {
            "state": query_params['state'][0],
            "code": query_params['code'][0],
            "error": query_params.get('error', [None])[0],
            "error_description": query_params.get('error_description', [None])[0]
        }
        
        response = api_client.get("/api/v1/platform-connections/oauth/callback", params=callback_request)
        
        if response.status_code == 200:
            st.success("✅ Platform connected successfully!")
            st.rerun()
        else:
            st.error("❌ Failed to complete OAuth connection")
            
    except Exception as e:
        st.error(f"Error handling OAuth callback: {str(e)}")

if __name__ == "__main__":
    main()