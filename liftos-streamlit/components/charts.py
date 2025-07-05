import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
from typing import Dict, Any, List

def create_attribution_chart(attribution_data: Dict[str, Any]) -> go.Figure:
    """Create attribution analysis chart"""
    
    # Initialize default chart data structure
    chart_data = {
        'channel': [],
        'attribution_value': [],
        'model_type': []
    }
    
    # Process the data based on structure
    if attribution_data and isinstance(attribution_data, dict) and 'analysis' in attribution_data:
        # Handle new API response structure with 'analysis' field
        analysis_data = attribution_data.get('analysis', {})
        attribution_scores = analysis_data.get('attribution_scores', {})
        
        # Convert attribution scores to chart format
        for channel, score in attribution_scores.items():
            # Convert channel names to display format
            display_name = channel.replace('_', ' ').title()
            chart_data['channel'].append(display_name)
            # Convert score (0-1) to percentage
            chart_data['attribution_value'].append(score * 100)
            chart_data['model_type'].append(attribution_data.get('metadata', {}).get('model_type', 'time_decay'))
        
    elif attribution_data and isinstance(attribution_data, dict) and 'data' in attribution_data and isinstance(attribution_data['data'], list):
        # Legacy format with 'data' field containing list of records
        legacy_df = pd.DataFrame(attribution_data['data'])
        if 'channel' in legacy_df.columns and 'attribution_value' in legacy_df.columns:
            chart_data = legacy_df.to_dict('list')
        else:
            # Fallback to sample data if legacy format is unexpected
            chart_data = {
                'channel': ['Google Ads', 'Facebook Ads', 'Email Marketing', 'Organic Search', 'Direct Traffic'],
                'attribution_value': [35.0, 28.0, 15.0, 12.0, 10.0],
                'model_type': ['time_decay'] * 5
            }
    
    # If no valid data was processed, use sample data
    if not chart_data['channel']:
        chart_data = {
            'channel': ['Google Ads', 'Facebook Ads', 'Email Marketing', 'Organic Search', 'Direct Traffic'],
            'attribution_value': [35.0, 28.0, 15.0, 12.0, 10.0],
            'model_type': ['time_decay'] * 5
        }
    
    # Create DataFrame from processed chart data
    df = pd.DataFrame(chart_data)
    
    # Validate DataFrame before passing to plotly
    required_columns = ['channel', 'attribution_value']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame missing required columns: {missing_columns}")
    
    # Create the bar chart
    fig = px.bar(
        df,
        x='channel',
        y='attribution_value',
        color='channel',
        title="Attribution Analysis by Channel",
        labels={'attribution_value': 'Attribution Score (%)', 'channel': 'Marketing Channel'},
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        title_x=0.5,
        xaxis_title="Marketing Channel",
        yaxis_title="Attribution Score (%)",
        template="plotly_white"
    )
    
    # Add value labels on bars
    fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
    
    return fig

def create_lift_chart(lift_data: Dict[str, Any]) -> go.Figure:
    """Create lift measurement chart"""
    if not lift_data or 'data' not in lift_data:
        # Create sample data for demo
        dates = pd.date_range('2025-01-01', periods=30, freq='D')
        sample_data = {
            'date': dates,
            'control_group': [0.05 + 0.01 * (i % 7) for i in range(30)],
            'treatment_group': [0.07 + 0.015 * (i % 7) for i in range(30)]
        }
        df = pd.DataFrame(sample_data)
    else:
        df = pd.DataFrame(lift_data['data'])
    
    fig = go.Figure()
    
    # Add control group
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['control_group'],
        mode='lines+markers',
        name='Control Group',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=6)
    ))
    
    # Add treatment group
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['treatment_group'],
        mode='lines+markers',
        name='Treatment Group',
        line=dict(color='#ff7f0e', width=3),
        marker=dict(size=6)
    ))
    
    # Calculate and add lift percentage
    avg_control = df['control_group'].mean()
    avg_treatment = df['treatment_group'].mean()
    lift_percentage = ((avg_treatment - avg_control) / avg_control) * 100
    
    fig.update_layout(
        title=f"Lift Measurement: Treatment vs Control (Lift: {lift_percentage:.1f}%)",
        xaxis_title="Date",
        yaxis_title="Conversion Rate",
        height=500,
        template="plotly_white",
        title_x=0.5
    )
    
    return fig

def create_budget_optimization_chart(optimization_data: Dict[str, Any]) -> go.Figure:
    """Create budget optimization chart"""
    if not optimization_data or 'data' not in optimization_data:
        # Create sample data for demo
        sample_data = {
            'channel': ['Meta Ads', 'Google Ads', 'Klaviyo', 'TikTok'],
            'current_budget': [20000, 15000, 8000, 7000],
            'recommended_budget': [25000, 18000, 5000, 2000]
        }
        df = pd.DataFrame(sample_data)
    else:
        df = pd.DataFrame(optimization_data['data'])
    
    fig = go.Figure()
    
    # Add current budget bars
    fig.add_trace(go.Bar(
        name='Current Budget',
        x=df['channel'],
        y=df['current_budget'],
        marker_color='lightblue',
        text=[f'${x:,.0f}' for x in df['current_budget']],
        textposition='auto'
    ))
    
    # Add recommended budget bars
    fig.add_trace(go.Bar(
        name='Recommended Budget',
        x=df['channel'],
        y=df['recommended_budget'],
        marker_color='darkblue',
        text=[f'${x:,.0f}' for x in df['recommended_budget']],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Budget Optimization: Current vs Recommended",
        xaxis_title="Marketing Channel",
        yaxis_title="Budget ($)",
        barmode='group',
        height=500,
        template="plotly_white",
        title_x=0.5
    )
    
    return fig

def create_budget_allocation_pie(optimization_data: Dict[str, Any]) -> go.Figure:
    """Create budget allocation pie chart"""
    if not optimization_data or 'data' not in optimization_data:
        # Create sample data for demo
        sample_data = {
            'channel': ['Meta Ads', 'Google Ads', 'Klaviyo', 'TikTok'],
            'recommended_budget': [25000, 18000, 5000, 2000]
        }
        df = pd.DataFrame(sample_data)
    else:
        df = pd.DataFrame(optimization_data['data'])
    
    fig = px.pie(
        df,
        values='recommended_budget',
        names='channel',
        title="Recommended Budget Allocation",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_layout(
        height=500,
        title_x=0.5,
        template="plotly_white"
    )
    
    # Update traces to show percentage and value
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Budget: $%{value:,.0f}<br>Percentage: %{percent}<extra></extra>'
    )
    
    return fig

def create_time_series_chart(data: Dict[str, Any], title: str = "Time Series Analysis") -> go.Figure:
    """Create generic time series chart"""
    if not data or 'data' not in data:
        # Create sample data for demo
        dates = pd.date_range('2025-01-01', periods=30, freq='D')
        sample_data = {
            'date': dates,
            'value': [100 + 10 * (i % 7) + (i * 2) for i in range(30)]
        }
        df = pd.DataFrame(sample_data)
    else:
        df = pd.DataFrame(data['data'])
    
    fig = px.line(
        df,
        x='date',
        y='value',
        title=title,
        markers=True
    )
    
    fig.update_layout(
        height=400,
        template="plotly_white",
        title_x=0.5
    )
    
    return fig

def create_correlation_heatmap(correlation_data: Dict[str, Any]) -> go.Figure:
    """Create correlation heatmap"""
    if not correlation_data or 'data' not in correlation_data:
        # Create sample correlation matrix
        channels = ['Meta', 'Google', 'Klaviyo', 'Direct']
        correlation_matrix = [
            [1.0, 0.3, 0.2, 0.1],
            [0.3, 1.0, 0.4, 0.2],
            [0.2, 0.4, 1.0, 0.3],
            [0.1, 0.2, 0.3, 1.0]
        ]
        df = pd.DataFrame(correlation_matrix, index=channels, columns=channels)
    else:
        df = pd.DataFrame(correlation_data['data'])
    
    fig = px.imshow(
        df,
        title="Channel Correlation Matrix",
        color_continuous_scale="RdBu_r",
        aspect="auto"
    )
    
    fig.update_layout(
        height=500,
        title_x=0.5,
        template="plotly_white"
    )
    
    return fig

def create_surfacing_chart(data, metric='impressions'):
    """Create a surfacing performance chart"""
    import plotly.express as px
    
    fig = px.line(
        data,
        x='date',
        y=metric,
        color='keyword',
        title=f"Keyword {metric.title()} Over Time",
        labels={
            'date': 'Date',
            metric: metric.replace('_', ' ').title(),
            'keyword': 'Keyword'
        }
    )
    
    fig.update_layout(
        height=400,
        hovermode='x unified',
        template="plotly_white"
    )
    
    return fig

def create_keyword_performance_chart(data, metric='clicks'):
    """Create a keyword performance chart"""
    import plotly.express as px
    import pandas as pd
    
    # Aggregate data by keyword for the chart
    agg_data = data.groupby(['date', 'keyword'])[metric].sum().reset_index()
    
    fig = px.line(
        agg_data,
        x='date',
        y=metric,
        color='keyword',
        title=f"Keyword Performance: {metric.replace('_', ' ').title()}",
        labels={
            'date': 'Date',
            metric: metric.replace('_', ' ').title(),
            'keyword': 'Keyword'
        }
    )
    
    fig.update_layout(
        height=400,
        hovermode='x unified',
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_funnel_chart(funnel_data: Dict[str, Any]) -> go.Figure:
    """Create conversion funnel chart"""
    if not funnel_data or 'data' not in funnel_data:
        # Create sample funnel data
        sample_data = {
            'stage': ['Impressions', 'Clicks', 'Visits', 'Leads', 'Conversions'],
            'count': [100000, 5000, 4000, 800, 160]
        }
        df = pd.DataFrame(sample_data)
    else:
        df = pd.DataFrame(funnel_data['data'])
    
    fig = go.Figure(go.Funnel(
        y=df['stage'],
        x=df['count'],
        textinfo="value+percent initial",
        marker=dict(color=["deepskyblue", "lightsalmon", "tan", "teal", "silver"])
    ))
    
    fig.update_layout(
        title="Conversion Funnel",
        height=500,
        template="plotly_white",
        title_x=0.5
    )
    
    return fig

def display_chart_with_download(fig: go.Figure, filename: str = "chart"):
    """Display chart with download option"""
    st.plotly_chart(fig, use_container_width=True)
    
    # Add download button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button(f"📥 Download {filename}", use_container_width=True):
            # Convert to HTML and provide download
            html_str = fig.to_html()
            st.download_button(
                label="Download as HTML",
                data=html_str,
                file_name=f"{filename}.html",
                mime="text/html"
            )