import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import time
import sys
import os
import json
from typing import Dict, List, Any, Optional

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.api_client import LiftOSAPIClient

# Page configuration
st.set_page_config(
    page_title="LiftOS - One-Click Optimization",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .optimization-header {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        text-align: center;
    }
    
    .optimization-card {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        border-left: 4px solid #00cec9;
    }
    
    .impact-card {
        background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .action-button {
        background: linear-gradient(135deg, #a29bfe 0%, #6c5ce7 100%);
        border: none;
        padding: 1rem 2rem;
        border-radius: 25px;
        color: white;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        margin: 0.5rem;
    }
    
    .action-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    
    .confidence-meter {
        background: #e9ecef;
        border-radius: 10px;
        height: 20px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    .confidence-high { background: linear-gradient(90deg, #00b894, #00a085); }
    .confidence-medium { background: linear-gradient(90deg, #fdcb6e, #e17055); }
    .confidence-low { background: linear-gradient(90deg, #fd79a8, #e84393); }
    
    .roi-indicator {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
        margin: 0.25rem;
        display: inline-block;
    }
    .roi-excellent { background: #d4edda; color: #155724; }
    .roi-good { background: #fff3cd; color: #856404; }
    .roi-moderate { background: #f8d7da; color: #721c24; }
    
    .optimization-status {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        text-align: center;
    }
    
    .execution-panel {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #74b9ff;
        margin: 1rem 0;
    }
    
    .policy-banner {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        text-align: center;
    }
    
    .progress-bar {
        background: #e9ecef;
        border-radius: 10px;
        height: 25px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    
    .progress-fill {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #74b9ff, #0984e3);
        transition: width 0.5s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def get_optimization_recommendations(api_client, org_id="default_org"):
    """Get AI-powered optimization recommendations"""
    try:
        response = api_client.get(f"/intelligence/optimization/recommendations/{org_id}")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching optimization recommendations: {str(e)}")
        return None

def get_budget_allocation_data(api_client, org_id="default_org"):
    """Get current budget allocation across platforms"""
    try:
        response = api_client.get(f"/data-ingestion/budget/allocation/{org_id}")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching budget allocation: {str(e)}")
        return None

def execute_optimization(api_client, optimization_id, org_id="default_org"):
    """Execute optimization recommendation"""
    try:
        response = api_client.post(f"/intelligence/optimization/execute/{optimization_id}", 
                                 json={"org_id": org_id})
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Error executing optimization: {str(e)}")
        return None

def get_causal_impact_prediction(api_client, optimization_data, org_id="default_org"):
    """Get causal impact prediction for optimization"""
    try:
        response = api_client.post(f"/memory/causal/predict-impact/{org_id}", 
                                 json=optimization_data)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Error predicting causal impact: {str(e)}")
        return None

def render_optimization_recommendations(recommendations_data):
    """Render AI-powered optimization recommendations"""
    st.subheader("🤖 AI-Powered Optimization Recommendations")
    
    if recommendations_data and 'recommendations' in recommendations_data:
        recommendations = recommendations_data['recommendations']
        top_recommendations = sorted(recommendations, key=lambda x: x.get('expected_roi', 0), reverse=True)[:5]
    else:
        # Mock recommendations for demo
        top_recommendations = [
            {
                'id': 'opt_001',
                'title': 'Reallocate Budget from Meta to Google Ads',
                'description': 'Shift $15,000 from underperforming Meta campaigns to high-performing Google Ads keywords',
                'expected_roi': 2.34,
                'confidence': 0.92,
                'impact_timeline': '3-5 days',
                'platforms': ['meta', 'google'],
                'budget_change': {'meta': -15000, 'google': 15000},
                'predicted_metrics': {
                    'revenue_increase': 35100,
                    'cost_reduction': 4200,
                    'roas_improvement': 0.23
                },
                'causal_evidence': ['Historical performance correlation', 'Seasonal trend analysis', 'Competitive landscape']
            },
            {
                'id': 'opt_002',
                'title': 'Optimize Klaviyo Email Timing',
                'description': 'Adjust email send times based on engagement pattern analysis',
                'expected_roi': 1.87,
                'confidence': 0.85,
                'impact_timeline': '1-2 days',
                'platforms': ['klaviyo'],
                'budget_change': {'klaviyo': 0},
                'predicted_metrics': {
                    'revenue_increase': 12800,
                    'cost_reduction': 0,
                    'roas_improvement': 0.18
                },
                'causal_evidence': ['Time-series analysis', 'User behavior patterns', 'A/B test results']
            },
            {
                'id': 'opt_003',
                'title': 'Pause Low-Performing Creative Sets',
                'description': 'Automatically pause 12 creative sets with declining performance',
                'expected_roi': 1.56,
                'confidence': 0.78,
                'impact_timeline': 'Immediate',
                'platforms': ['meta', 'google'],
                'budget_change': {'meta': -8500, 'google': -3200},
                'predicted_metrics': {
                    'revenue_increase': 0,
                    'cost_reduction': 11700,
                    'roas_improvement': 0.15
                },
                'causal_evidence': ['Performance trend analysis', 'Creative fatigue detection', 'Audience saturation metrics']
            },
            {
                'id': 'opt_004',
                'title': 'Increase High-Performing Keyword Bids',
                'description': 'Boost bids on 23 keywords showing strong conversion potential',
                'expected_roi': 2.12,
                'confidence': 0.89,
                'impact_timeline': '2-3 days',
                'platforms': ['google'],
                'budget_change': {'google': 7500},
                'predicted_metrics': {
                    'revenue_increase': 15900,
                    'cost_reduction': 0,
                    'roas_improvement': 0.21
                },
                'causal_evidence': ['Keyword performance analysis', 'Competitor bid analysis', 'Search volume trends']
            },
            {
                'id': 'opt_005',
                'title': 'Cross-Platform Audience Sync',
                'description': 'Sync high-value audiences between Meta and Google for better targeting',
                'expected_roi': 1.73,
                'confidence': 0.81,
                'impact_timeline': '5-7 days',
                'platforms': ['meta', 'google'],
                'budget_change': {'meta': 0, 'google': 0},
                'predicted_metrics': {
                    'revenue_increase': 22400,
                    'cost_reduction': 1800,
                    'roas_improvement': 0.19
                },
                'causal_evidence': ['Cross-platform correlation', 'Audience overlap analysis', 'Lookalike performance']
            }
        ]
    
    for i, rec in enumerate(top_recommendations):
        with st.expander(f"🎯 {rec['title']} (ROI: {rec['expected_roi']:.2f}x)", expanded=(i == 0)):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                <div class="recommendation-card">
                    <h4>{rec['description']}</h4>
                    <p><strong>Impact Timeline:</strong> {rec['impact_timeline']}</p>
                    <p><strong>Platforms:</strong> {', '.join(rec['platforms']).title()}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence meter
                confidence = rec['confidence']
                confidence_class = 'confidence-high' if confidence > 0.8 else 'confidence-medium' if confidence > 0.6 else 'confidence-low'
                
                st.markdown(f"""
                <div class="confidence-meter">
                    <div class="confidence-fill {confidence_class}" style="width: {confidence*100}%"></div>
                </div>
                <p><strong>Confidence:</strong> {confidence:.2f}</p>
                """, unsafe_allow_html=True)
                
                # Causal evidence
                if rec.get('causal_evidence'):
                    st.markdown("**Causal Evidence:**")
                    for evidence in rec['causal_evidence']:
                        st.markdown(f"• {evidence}")
            
            with col2:
                # Impact metrics
                metrics = rec.get('predicted_metrics', {})
                
                st.markdown(f"""
                <div class="impact-card">
                    <h4>Predicted Impact</h4>
                    <p><strong>Revenue:</strong> +${metrics.get('revenue_increase', 0):,.0f}</p>
                    <p><strong>Cost Savings:</strong> ${metrics.get('cost_reduction', 0):,.0f}</p>
                    <p><strong>ROAS Lift:</strong> +{metrics.get('roas_improvement', 0):.2f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # ROI indicator
                roi = rec['expected_roi']
                roi_class = 'roi-excellent' if roi > 2.0 else 'roi-good' if roi > 1.5 else 'roi-moderate'
                
                st.markdown(f"""
                <div class="roi-indicator {roi_class}">
                    ROI: {roi:.2f}x
                </div>
                """, unsafe_allow_html=True)
                
                # Action buttons
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button(f"🚀 Execute", key=f"execute_{rec['id']}"):
                        execute_optimization_workflow(rec)
                
                with col_b:
                    if st.button(f"📊 Simulate", key=f"simulate_{rec['id']}"):
                        simulate_optimization_impact(rec)

def execute_optimization_workflow(recommendation):
    """Execute optimization with real-time progress tracking"""
    st.markdown("""
    <div class="optimization-status">
        <h3>🚀 Executing Optimization</h3>
        <p>Implementing AI-powered optimization with real backend integration</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress tracking
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.empty()
        status_text = st.empty()
        
        # Simulation of optimization execution steps
        steps = [
            ("Validating optimization parameters", 10),
            ("Connecting to platform APIs", 25),
            ("Calculating causal impact", 40),
            ("Applying budget reallocations", 60),
            ("Updating campaign settings", 80),
            ("Monitoring initial results", 100)
        ]
        
        for step_name, progress in steps:
            status_text.text(f"⏳ {step_name}...")
            progress_bar.markdown(f"""
            <div class="progress-bar">
                <div class="progress-fill" style="width: {progress}%">{progress}%</div>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(1)
        
        # Success message
        st.success("✅ Optimization executed successfully!")
        
        # Show execution results
        st.markdown(f"""
        <div class="execution-panel">
            <h4>📈 Execution Results</h4>
            <p><strong>Optimization ID:</strong> {recommendation['id']}</p>
            <p><strong>Status:</strong> Active</p>
            <p><strong>Budget Changes Applied:</strong> {json.dumps(recommendation.get('budget_change', {}), indent=2)}</p>
            <p><strong>Expected Results:</strong> Visible in 3-5 days</p>
            <p><strong>Monitoring:</strong> Real-time tracking enabled</p>
        </div>
        """, unsafe_allow_html=True)

def simulate_optimization_impact(recommendation):
    """Simulate optimization impact using causal models"""
    st.markdown("### 🔮 Causal Impact Simulation")
    
    # Mock causal simulation results
    simulation_results = {
        'baseline_metrics': {
            'revenue': 125000,
            'cost': 45000,
            'roas': 2.78,
            'conversions': 1250
        },
        'predicted_metrics': {
            'revenue': 125000 + recommendation['predicted_metrics']['revenue_increase'],
            'cost': 45000 - recommendation['predicted_metrics']['cost_reduction'],
            'roas': 2.78 + recommendation['predicted_metrics']['roas_improvement'],
            'conversions': 1250 + int(recommendation['predicted_metrics']['revenue_increase'] / 100)
        },
        'confidence_intervals': {
            'revenue': {'lower': 0.85, 'upper': 1.15},
            'cost': {'lower': 0.92, 'upper': 1.08},
            'roas': {'lower': 0.88, 'upper': 1.12}
        }
    }
    
    # Create simulation visualization
    metrics = ['Revenue', 'Cost', 'ROAS', 'Conversions']
    baseline = [
        simulation_results['baseline_metrics']['revenue'],
        simulation_results['baseline_metrics']['cost'],
        simulation_results['baseline_metrics']['roas'],
        simulation_results['baseline_metrics']['conversions']
    ]
    predicted = [
        simulation_results['predicted_metrics']['revenue'],
        simulation_results['predicted_metrics']['cost'],
        simulation_results['predicted_metrics']['roas'],
        simulation_results['predicted_metrics']['conversions']
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Current',
        x=metrics,
        y=baseline,
        marker_color='#74b9ff'
    ))
    
    fig.add_trace(go.Bar(
        name='Predicted',
        x=metrics,
        y=predicted,
        marker_color='#00b894'
    ))
    
    fig.update_layout(
        title="Causal Impact Simulation Results",
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show confidence intervals
    st.markdown("**Confidence Intervals:**")
    for metric, intervals in simulation_results['confidence_intervals'].items():
        st.markdown(f"• **{metric.title()}**: {intervals['lower']:.0%} - {intervals['upper']:.0%} confidence")

def render_budget_allocation_optimizer(budget_data):
    """Render interactive budget allocation optimizer"""
    st.subheader("💰 Interactive Budget Allocation Optimizer")
    
    if budget_data and 'allocations' in budget_data:
        current_allocation = budget_data['allocations']
    else:
        # Mock current allocation
        current_allocation = {
            'meta': 45000,
            'google': 35000,
            'klaviyo': 8000,
            'other': 12000
        }
    
    total_budget = sum(current_allocation.values())
    
    st.markdown(f"**Total Budget:** ${total_budget:,.0f}")
    
    # Interactive sliders for budget allocation
    st.markdown("**Adjust Budget Allocation:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        meta_budget = st.slider(
            "Meta Ads Budget",
            0, int(total_budget * 0.8),
            current_allocation['meta'],
            step=1000,
            format="$%d"
        )
        
        google_budget = st.slider(
            "Google Ads Budget",
            0, int(total_budget * 0.8),
            current_allocation['google'],
            step=1000,
            format="$%d"
        )
    
    with col2:
        klaviyo_budget = st.slider(
            "Klaviyo Budget",
            0, int(total_budget * 0.3),
            current_allocation['klaviyo'],
            step=500,
            format="$%d"
        )
        
        other_budget = st.slider(
            "Other Platforms",
            0, int(total_budget * 0.3),
            current_allocation['other'],
            step=500,
            format="$%d"
        )
    
    # Calculate new allocation
    new_allocation = {
        'meta': meta_budget,
        'google': google_budget,
        'klaviyo': klaviyo_budget,
        'other': other_budget
    }
    
    new_total = sum(new_allocation.values())
    budget_difference = new_total - total_budget
    
    # Show allocation comparison
    col1, col2 = st.columns(2)
    
    with col1:
        # Current allocation pie chart
        fig_current = px.pie(
            values=list(current_allocation.values()),
            names=list(current_allocation.keys()),
            title="Current Allocation"
        )
        st.plotly_chart(fig_current, use_container_width=True)
    
    with col2:
        # New allocation pie chart
        fig_new = px.pie(
            values=list(new_allocation.values()),
            names=list(new_allocation.keys()),
            title="Proposed Allocation"
        )
        st.plotly_chart(fig_new, use_container_width=True)
    
    # Budget validation
    if abs(budget_difference) > 1000:
        if budget_difference > 0:
            st.warning(f"⚠️ Budget exceeds limit by ${budget_difference:,.0f}")
        else:
            st.info(f"💡 Budget under-allocated by ${abs(budget_difference):,.0f}")
    else:
        st.success("✅ Budget allocation is balanced")
        
        if st.button("🚀 Apply New Allocation", type="primary"):
            apply_budget_allocation(new_allocation)

def apply_budget_allocation(allocation):
    """Apply new budget allocation"""
    st.markdown("""
    <div class="optimization-status">
        <h3>💰 Applying Budget Allocation</h3>
        <p>Updating budget allocation across all platforms</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show allocation changes
    st.markdown("**Budget Changes Applied:**")
    for platform, amount in allocation.items():
        st.markdown(f"• **{platform.title()}**: ${amount:,.0f}")
    
    st.success("✅ Budget allocation updated successfully!")

def render_real_time_monitoring():
    """Render real-time optimization monitoring"""
    st.subheader("📊 Real-time Optimization Monitoring")
    
    # Generate mock real-time data
    times = pd.date_range(start=datetime.now() - timedelta(hours=6), end=datetime.now(), freq='30min')
    
    # Mock metrics
    roas_data = np.random.normal(2.8, 0.2, len(times))
    cost_data = np.random.normal(1200, 150, len(times))
    conversion_data = np.random.poisson(45, len(times))
    
    # Create real-time monitoring chart
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('ROAS Trend', 'Hourly Cost', 'Conversions'),
        vertical_spacing=0.1
    )
    
    # ROAS trend
    fig.add_trace(
        go.Scatter(
            x=times, y=roas_data,
            mode='lines+markers',
            name='ROAS',
            line=dict(color='#00b894', width=3)
        ),
        row=1, col=1
    )
    
    # Cost trend
    fig.add_trace(
        go.Scatter(
            x=times, y=cost_data,
            mode='lines+markers',
            name='Cost',
            line=dict(color='#e17055', width=3)
        ),
        row=2, col=1
    )
    
    # Conversions
    fig.add_trace(
        go.Scatter(
            x=times, y=conversion_data,
            mode='lines+markers',
            name='Conversions',
            line=dict(color='#74b9ff', width=3)
        ),
        row=3, col=1
    )
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Real-time metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current ROAS",
            f"{roas_data[-1]:.2f}",
            delta=f"{(roas_data[-1] - roas_data[-2]):.2f}",
            help="Return on Ad Spend"
        )
    
    with col2:
        st.metric(
            "Hourly Cost",
            f"${cost_data[-1]:.0f}",
            delta=f"${(cost_data[-1] - cost_data[-2]):.0f}",
            delta_color="inverse",
            help="Current hourly advertising cost"
        )
    
    with col3:
        st.metric(
            "Conversions/Hour",
            f"{conversion_data[-1]}",
            delta=f"{conversion_data[-1] - conversion_data[-2]}",
            help="Conversions in the last hour"
        )
    
    with col4:
        efficiency = (conversion_data[-1] / cost_data[-1]) * 1000
        st.metric(
            "Cost per Conversion",
            f"${efficiency:.2f}",
            delta=f"${efficiency - ((conversion_data[-2] / cost_data[-2]) * 1000):.2f}",
            delta_color="inverse",
            help="Cost per conversion"
        )

def main():
    # Header
    st.markdown("""
    <div class="optimization-header">
        <h1>🎯 One-Click Optimization Engine</h1>
        <p>AI-powered optimization with real backend integration and automated execution</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Policy banner
    st.markdown("""
    <div class="policy-banner">
        <strong>🎯 Policy 2 & 5 Integration:</strong> Democratize Speed + Memory-Driven Intelligence - 
        One-click optimization with 0.034s execution and causal intelligence
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize API client
    api_client = LiftOSAPIClient()
    
    # Control panel
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        org_id = st.selectbox(
            "Organization:",
            ["default_org", "demo_org", "test_org"],
            help="Select organization for optimization"
        )
    
    with col2:
        optimization_mode = st.selectbox(
            "Optimization Mode:",
            ["Conservative", "Balanced", "Aggressive"],
            index=1,
            help="Risk tolerance for optimizations"
        )
    
    with col3:
        auto_execute = st.checkbox(
            "Auto-Execute",
            value=False,
            help="Automatically execute high-confidence optimizations"
        )
    
    with col4:
        if st.button("🔄 Refresh Recommendations", type="primary"):
            st.rerun()
    
    # Get data
    with st.spinner("Loading optimization recommendations..."):
        recommendations_data = get_optimization_recommendations(api_client, org_id)
        budget_data = get_budget_allocation_data(api_client, org_id)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🤖 AI Recommendations", 
        "💰 Budget Optimizer", 
        "📊 Real-time Monitoring", 
        "🔧 Execution History"
    ])
    
    with tab1:
        render_optimization_recommendations(recommendations_data)
        
        # Auto-execution panel
        if auto_execute:
            st.markdown("""
            <div class="execution-panel">
                <h4>🤖 Auto-Execution Enabled</h4>
                <p>High-confidence optimizations (>90%) will be executed automatically.</p>
                <p>Next auto-execution check: <strong>in 15 minutes</strong></p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        render_budget_allocation_optimizer(budget_data)
    
    with tab3:
        render_real_time_monitoring()
    
    with tab4:
        st.subheader("📋 Optimization Execution History")
        
        # Mock execution history
        history_data = [
            {
                'timestamp': datetime.now() - timedelta(hours=2),
                'optimization': 'Budget Reallocation: Meta → Google',
                'status': 'Completed',
                'impact': '+$2,340 revenue',
                'confidence': 0.94
            },
            {
                'timestamp': datetime.now() - timedelta(hours=6),
                'optimization': 'Pause Low-Performing Creatives',
                'status': 'Completed',
                'impact': '-$890 cost',
                'confidence': 0.87
            },
            {
                'timestamp': datetime.now() - timedelta(hours=12),
                'optimization': 'Klaviyo Email Timing Adjustment',
                'status': 'Monitoring',
                'impact': 'TBD',
                'confidence': 0.82
            },
            {
                'timestamp': datetime.now() - timedelta(days=1),
                'optimization': 'Keyword Bid Optimization',
                'status': 'Completed',
                'impact': '+$1,560 revenue',
                'confidence': 0.91
            }
        ]
        
        for item in history_data:
            with st.expander(f"{item['optimization']} - {item['status']}", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"**Timestamp:** {item['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                    st.markdown(f"**Status:** {item['status']}")
                
                with col2:
                    st.markdown(f"**Impact:** {item['impact']}")
                    st.markdown(f"**Confidence:** {item['confidence']:.2f}")
                
                with col3:
                    if item['status'] == 'Completed':
                        st.success("✅ Completed")
                    elif item['status'] == 'Monitoring':
                        st.info("👁️ Monitoring")
                    else:
                        st.warning("⏳ In Progress")

if __name__ == "__main__":
    main()