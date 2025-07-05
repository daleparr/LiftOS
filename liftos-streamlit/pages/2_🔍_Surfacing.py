import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from auth.session_manager import initialize_session
from utils.api_client import APIClient
from components.sidebar import render_sidebar
from components.charts import create_surfacing_chart, create_keyword_performance_chart

def generate_mock_surfacing_data():
    """Generate mock data for surfacing analysis"""
    np.random.seed(42)
    
    # Generate keyword data
    keywords = [
        "marketing attribution", "customer journey", "conversion tracking",
        "ad spend optimization", "roi analysis", "marketing mix modeling",
        "incrementality testing", "brand awareness", "customer acquisition",
        "retention marketing", "email marketing", "social media ads",
        "google ads", "facebook ads", "programmatic advertising"
    ]
    
    data = []
    for keyword in keywords:
        for i in range(30):  # 30 days of data
            date = datetime.now() - timedelta(days=29-i)
            data.append({
                'date': date,
                'keyword': keyword,
                'impressions': np.random.randint(1000, 10000),
                'clicks': np.random.randint(50, 500),
                'conversions': np.random.randint(5, 50),
                'cost': np.random.uniform(100, 1000),
                'position': np.random.uniform(1, 10),
                'quality_score': np.random.randint(6, 10)
            })
    
    df = pd.DataFrame(data)
    df['ctr'] = df['clicks'] / df['impressions'] * 100
    df['conversion_rate'] = df['conversions'] / df['clicks'] * 100
    df['cpc'] = df['cost'] / df['clicks']
    df['cpa'] = df['cost'] / df['conversions']
    
    return df

def render_surfacing_overview():
    """Render the surfacing overview section"""
    st.header("🔍 Surfacing Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Keywords",
            "1,247",
            delta="23",
            help="Total number of tracked keywords"
        )
    
    with col2:
        st.metric(
            "Avg. Position",
            "3.2",
            delta="-0.4",
            help="Average search position across all keywords"
        )
    
    with col3:
        st.metric(
            "Click-Through Rate",
            "4.8%",
            delta="0.3%",
            help="Average CTR across all keywords"
        )
    
    with col4:
        st.metric(
            "Quality Score",
            "7.6",
            delta="0.2",
            help="Average quality score across all keywords"
        )

def render_keyword_analysis():
    """Render keyword analysis section"""
    st.header("📊 Keyword Performance Analysis")
    
    # Generate mock data
    df = generate_mock_surfacing_data()
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_keywords = st.multiselect(
            "Select Keywords",
            options=df['keyword'].unique(),
            default=df['keyword'].unique()[:5],
            help="Choose keywords to analyze"
        )
    
    with col2:
        date_range = st.date_input(
            "Date Range",
            value=[datetime.now() - timedelta(days=30), datetime.now()],
            help="Select date range for analysis"
        )
    
    with col3:
        metric = st.selectbox(
            "Primary Metric",
            options=['impressions', 'clicks', 'conversions', 'cost', 'ctr', 'conversion_rate'],
            index=1,
            help="Choose the primary metric to display"
        )
    
    # Filter data
    filtered_df = df[df['keyword'].isin(selected_keywords)]
    
    # Performance chart
    if not filtered_df.empty:
        fig = create_keyword_performance_chart(filtered_df, metric)
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance table
        st.subheader("📋 Keyword Performance Table")
        
        # Aggregate data by keyword
        agg_df = filtered_df.groupby('keyword').agg({
            'impressions': 'sum',
            'clicks': 'sum',
            'conversions': 'sum',
            'cost': 'sum',
            'position': 'mean',
            'quality_score': 'mean'
        }).round(2)
        
        # Calculate derived metrics
        agg_df['ctr'] = (agg_df['clicks'] / agg_df['impressions'] * 100).round(2)
        agg_df['conversion_rate'] = (agg_df['conversions'] / agg_df['clicks'] * 100).round(2)
        agg_df['cpc'] = (agg_df['cost'] / agg_df['clicks']).round(2)
        agg_df['cpa'] = (agg_df['cost'] / agg_df['conversions']).round(2)
        
        # Display table
        st.dataframe(
            agg_df,
            use_container_width=True,
            column_config={
                'impressions': st.column_config.NumberColumn('Impressions', format='%d'),
                'clicks': st.column_config.NumberColumn('Clicks', format='%d'),
                'conversions': st.column_config.NumberColumn('Conversions', format='%d'),
                'cost': st.column_config.NumberColumn('Cost', format='$%.2f'),
                'position': st.column_config.NumberColumn('Avg Position', format='%.1f'),
                'quality_score': st.column_config.NumberColumn('Quality Score', format='%.1f'),
                'ctr': st.column_config.NumberColumn('CTR', format='%.2f%%'),
                'conversion_rate': st.column_config.NumberColumn('Conv Rate', format='%.2f%%'),
                'cpc': st.column_config.NumberColumn('CPC', format='$%.2f'),
                'cpa': st.column_config.NumberColumn('CPA', format='$%.2f')
            }
        )

def render_search_trends():
    """Render search trends analysis"""
    st.header("📈 Search Trends Analysis")
    
    # Generate trend data
    np.random.seed(42)
    dates = pd.date_range(start=datetime.now() - timedelta(days=90), end=datetime.now(), freq='D')
    
    trend_data = {
        'date': dates,
        'search_volume': np.random.randint(10000, 50000, len(dates)) + 
                        np.sin(np.arange(len(dates)) * 2 * np.pi / 7) * 5000,  # Weekly pattern
        'competition': np.random.uniform(0.3, 0.9, len(dates)),
        'suggested_bid': np.random.uniform(1.5, 8.0, len(dates))
    }
    
    trend_df = pd.DataFrame(trend_data)
    
    # Trend visualization
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=trend_df['date'],
        y=trend_df['search_volume'],
        mode='lines',
        name='Search Volume',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.update_layout(
        title="Search Volume Trends (90 Days)",
        xaxis_title="Date",
        yaxis_title="Search Volume",
        hovermode='x unified',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Trend insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Trend Insights")
        st.info("📊 Search volume shows consistent weekly patterns with peaks on weekdays")
        st.info("📈 Overall trend is slightly increasing (+12% vs last month)")
        st.info("🎯 Competition levels remain stable around 0.6-0.7")
        
    with col2:
        st.subheader("💡 Recommendations")
        st.success("🚀 Increase bids during high-volume periods (Tue-Thu)")
        st.success("💰 Optimize for lower competition keywords")
        st.success("📅 Plan campaigns around seasonal trends")

def render_competitor_analysis():
    """Render competitor analysis section"""
    st.header("🏆 Competitor Analysis")
    
    # Mock competitor data
    competitors = ['Competitor A', 'Competitor B', 'Competitor C', 'Competitor D']
    
    competitor_data = []
    for comp in competitors:
        competitor_data.append({
            'competitor': comp,
            'share_of_voice': np.random.uniform(15, 35),
            'avg_position': np.random.uniform(2, 8),
            'estimated_traffic': np.random.randint(50000, 200000),
            'ad_spend_estimate': np.random.randint(10000, 50000)
        })
    
    comp_df = pd.DataFrame(competitor_data)
    
    # Competitor comparison chart
    fig = px.scatter(
        comp_df,
        x='share_of_voice',
        y='avg_position',
        size='estimated_traffic',
        color='ad_spend_estimate',
        hover_name='competitor',
        title="Competitor Positioning Map",
        labels={
            'share_of_voice': 'Share of Voice (%)',
            'avg_position': 'Average Position',
            'estimated_traffic': 'Estimated Traffic',
            'ad_spend_estimate': 'Estimated Ad Spend'
        }
    )
    
    fig.update_yaxes(autorange="reversed")  # Lower position numbers are better
    st.plotly_chart(fig, use_container_width=True)
    
    # Competitor table
    st.subheader("📊 Competitor Metrics")
    st.dataframe(
        comp_df,
        use_container_width=True,
        column_config={
            'competitor': 'Competitor',
            'share_of_voice': st.column_config.NumberColumn('Share of Voice', format='%.1f%%'),
            'avg_position': st.column_config.NumberColumn('Avg Position', format='%.1f'),
            'estimated_traffic': st.column_config.NumberColumn('Est. Traffic', format='%d'),
            'ad_spend_estimate': st.column_config.NumberColumn('Est. Ad Spend', format='$%d')
        }
    )

def main():
    st.set_page_config(
        page_title="Surfacing - LiftOS",
        page_icon="🔍",
        layout="wide"
    )
    
    # Initialize session
    initialize_session()
    
    # Render sidebar
    render_sidebar()
    
    # Check authentication
    if not st.session_state.authenticated:
        st.error("Please log in to access surfacing analysis.")
        st.stop()
    
    st.title("🔍 Surfacing Analysis")
    st.markdown("Analyze search visibility, keyword performance, and competitive positioning.")
    
    # Create tabs for different analysis views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔑 Keywords", "📈 Trends", "🏆 Competitors"])
    
    with tab1:
        render_surfacing_overview()
        
        # Quick actions
        st.header("⚡ Quick Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Refresh Data", type="primary"):
                st.success("✅ Data refreshed successfully!")
        
        with col2:
            if st.button("📊 Generate Report"):
                st.success("✅ Report generated and sent to your email!")
        
        with col3:
            if st.button("⚙️ Configure Alerts"):
                st.info("🔧 Redirecting to alert configuration...")
    
    with tab2:
        render_keyword_analysis()
    
    with tab3:
        render_search_trends()
    
    with tab4:
        render_competitor_analysis()
    
    # Footer
    st.markdown("---")
    st.markdown("💡 **Tip**: Use the Settings page to configure your search platform API keys for real-time data.")

if __name__ == "__main__":
    main()