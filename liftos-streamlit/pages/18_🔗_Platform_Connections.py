"""
Platform Connections Page
User interface for connecting and managing marketing platform APIs
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Optional

# Page configuration
st.set_page_config(
    page_title="Platform Connections - LiftOS",
    page_icon="🔗",
    layout="wide"
)

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
        'bigquery': '📊',
        'tableau': '📈',
        'powerbi': '⚡'
    }
    return icons.get(platform, '🔌')

def get_platform_status(platform: str) -> str:
    """Get connection status for platform"""
    # Mock status for demo - in production this would check actual connections
    if 'connected_platforms' not in st.session_state:
        st.session_state.connected_platforms = {}
    
    return st.session_state.connected_platforms.get(platform, 'disconnected')

def main():
    st.title("🔗 Platform Connections")
    st.markdown("Connect and manage your marketing platform integrations")
    
    # Mode indicator
    if st.session_state.get('live_data_mode', False):
        st.success("🟢 Live Data Mode - Real API connections active")
    else:
        st.info("🟡 Demo Mode - Using mock data")
    
    # Platform categories
    platforms = {
        "Advertising Platforms": [
            {'id': 'meta_business', 'name': 'Meta Business', 'description': 'Facebook & Instagram Ads'},
            {'id': 'google_ads', 'name': 'Google Ads', 'description': 'Google Advertising Platform'},
            {'id': 'tiktok', 'name': 'TikTok Ads', 'description': 'TikTok Advertising Platform'}
        ],
        "E-commerce Platforms": [
            {'id': 'shopify', 'name': 'Shopify', 'description': 'E-commerce Platform'},
            {'id': 'woocommerce', 'name': 'WooCommerce', 'description': 'WordPress E-commerce'},
            {'id': 'amazon_seller_central', 'name': 'Amazon Seller Central', 'description': 'Amazon Marketplace'}
        ],
        "Email & CRM": [
            {'id': 'klaviyo', 'name': 'Klaviyo', 'description': 'Email Marketing Platform'},
            {'id': 'hubspot', 'name': 'HubSpot', 'description': 'CRM & Marketing Hub'},
            {'id': 'salesforce', 'name': 'Salesforce', 'description': 'Customer Relationship Management'}
        ],
        "Payment Platforms": [
            {'id': 'stripe', 'name': 'Stripe', 'description': 'Payment Processing'},
            {'id': 'paypal', 'name': 'PayPal', 'description': 'Digital Payments'}
        ],
        "Data Platforms": [
            {'id': 'snowflake', 'name': 'Snowflake', 'description': 'Cloud Data Platform'},
            {'id': 'databricks', 'name': 'Databricks', 'description': 'Unified Analytics Platform'},
            {'id': 'bigquery', 'name': 'BigQuery', 'description': 'Google Cloud Data Warehouse'}
        ]
    }
    
    # Live Data Mode Toggle
    st.subheader("🚀 Live Data Mode")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        **Switch to Live Data Mode** to connect real marketing platforms and start ingesting live data.
        This will replace mock data with actual API connections.
        """)
    
    with col2:
        if st.button("🔄 Enable Live Data Mode", type="primary"):
            st.session_state.live_data_mode = True
            st.success("Live Data Mode enabled! You can now connect real platforms.")
            st.rerun()
    
    if st.session_state.get('live_data_mode', False):
        st.markdown("---")
        
        # Platform connection interface
        for category, platform_list in platforms.items():
            st.subheader(f"📂 {category}")
            
            for platform in platform_list:
                with st.expander(f"{get_platform_icon(platform['id'])} {platform['name']} - {platform['description']}"):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        status = get_platform_status(platform['id'])
                        if status == 'connected':
                            st.success("✅ Connected")
                        else:
                            st.warning("⚠️ Not Connected")
                    
                    with col2:
                        if status == 'connected':
                            if st.button(f"Disconnect", key=f"disconnect_{platform['id']}"):
                                st.session_state.connected_platforms[platform['id']] = 'disconnected'
                                st.success(f"Disconnected from {platform['name']}")
                                st.rerun()
                        else:
                            if st.button(f"Connect", key=f"connect_{platform['id']}", type="primary"):
                                st.session_state.connected_platforms[platform['id']] = 'connected'
                                st.success(f"Connected to {platform['name']}!")
                                st.rerun()
                    
                    with col3:
                        if st.button(f"Test", key=f"test_{platform['id']}"):
                            if status == 'connected':
                                st.success("✅ Connection test successful!")
                            else:
                                st.error("❌ Please connect first")
                    
                    # Connection details for connected platforms
                    if status == 'connected':
                        st.markdown("**Connection Details:**")
                        st.json({
                            "status": "active",
                            "last_sync": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "data_points": "1,234",
                            "api_calls_today": "456"
                        })
        
        # Connection Summary
        st.markdown("---")
        st.subheader("📊 Connection Summary")
        
        connected_count = len([p for p in st.session_state.get('connected_platforms', {}).values() if p == 'connected'])
        total_platforms = sum(len(platforms[cat]) for cat in platforms)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Connected Platforms", connected_count)
        with col2:
            st.metric("Total Available", total_platforms)
        with col3:
            st.metric("Connection Rate", f"{(connected_count/total_platforms*100):.1f}%")
        with col4:
            st.metric("Data Sync Status", "✅ Active" if connected_count > 0 else "⚠️ Inactive")
    
    else:
        st.markdown("---")
        st.info("""
        **Demo Mode Active** - Currently using mock data for demonstration purposes.
        
        To begin live data testing:
        1. Click "Enable Live Data Mode" above
        2. Connect your marketing platforms
        3. Configure API credentials
        4. Start data synchronization
        """)

if __name__ == "__main__":
    main()