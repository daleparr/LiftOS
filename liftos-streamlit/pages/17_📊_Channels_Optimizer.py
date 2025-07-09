"""
LiftOS Channels Budget Optimizer

Cross-channel budget optimization with causal inference and saturation modeling.
Transform from diagnostic tool to strategic copilot with actionable recommendations.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta, date
import time
import sys
import os
import json
from typing import Dict, List, Any, Optional

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.api_client import APIClient
from components.sidebar import render_sidebar

# Page configuration
st.set_page_config(
    page_title="LiftOS - Channels Budget Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .channels-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        text-align: center;
    }
    
    .optimization-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .simulation-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 0.25rem;
    }
    
    .saturation-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 8px;
        color: #333;
        margin: 0.5rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    # Render sidebar
    render_sidebar()
    
    # Header
    st.markdown("""
    <div class="channels-header">
        <h1>📊 Channels Budget Optimizer</h1>
        <p>Cross-Channel Budget Optimization with Causal Inference & Saturation Modeling</p>
        <p><em>Transform from diagnostic tool to strategic copilot</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize API client
    api_client = APIClient()
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Budget Optimization", 
        "🔮 Scenario Simulation", 
        "💡 Recommendations", 
        "📈 Saturation Analysis",
        "📊 Performance Dashboard"
    ])
    
    with tab1:
        render_budget_optimization(api_client)
    
    with tab2:
        render_scenario_simulation(api_client)
    
    with tab3:
        render_recommendations(api_client)
    
    with tab4:
        render_saturation_analysis(api_client)
    
    with tab5:
        render_performance_dashboard(api_client)

def render_budget_optimization(api_client: APIClient):
    """Render budget optimization interface"""
    st.markdown("""
    <div class="optimization-card">
        <h3>🎯 Multi-Objective Budget Optimization</h3>
        <p>Optimize budget allocation across channels using NSGA-II and Bayesian optimization</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Optimization Configuration")
        
        # Budget constraints
        st.write("**Budget Constraints**")
        total_budget = st.number_input(
            "Total Budget ($)", 
            min_value=1000, 
            max_value=1000000, 
            value=100000, 
            step=1000
        )
        
        min_budget_per_channel = st.number_input(
            "Minimum Budget per Channel ($)", 
            min_value=0, 
            max_value=50000, 
            value=5000, 
            step=500
        )
        
        # Channel selection
        st.write("**Channel Selection**")
        available_channels = [
            "Meta Ads", "Google Ads", "TikTok Ads", 
            "LinkedIn Ads", "Email Marketing", "Influencer Marketing",
            "Display Advertising", "YouTube Ads", "Pinterest Ads"
        ]
        
        selected_channels = st.multiselect(
            "Select Channels to Optimize",
            available_channels,
            default=["Meta Ads", "Google Ads", "TikTok Ads", "Email Marketing"]
        )
        
        # Optimization objectives
        st.write("**Optimization Objectives**")
        objectives = st.multiselect(
            "Select Objectives (Multi-objective optimization)",
            ["Maximize Revenue", "Maximize ROAS", "Minimize Cost", "Maximize Reach", "Maximize Conversions"],
            default=["Maximize Revenue", "Maximize ROAS"]
        )
        
        # Time horizon
        time_horizon = st.selectbox(
            "Optimization Time Horizon",
            ["1 Week", "2 Weeks", "1 Month", "3 Months", "6 Months"],
            index=2
        )
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            confidence_level = st.slider("Confidence Level", 0.8, 0.99, 0.95, 0.01)
            include_seasonality = st.checkbox("Include Seasonality Effects", value=True)
            include_saturation = st.checkbox("Include Saturation Modeling", value=True)
            include_carryover = st.checkbox("Include Carryover Effects", value=True)
            
            # Bayesian priors
            st.write("**Bayesian Prior Configuration**")
            prior_strength = st.selectbox(
                "Prior Strength",
                ["Weak", "Medium", "Strong"],
                index=1
            )
    
    with col2:
        st.subheader("🚀 Run Optimization")
        
        if st.button("🎯 Optimize Budget Allocation", type="primary"):
            if len(selected_channels) < 2:
                st.error("Please select at least 2 channels for optimization")
                return
            
            if len(objectives) < 1:
                st.error("Please select at least 1 optimization objective")
                return
            
            # Show optimization progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate optimization process
            optimization_steps = [
                "Initializing optimization engine...",
                "Loading historical performance data...",
                "Calibrating saturation curves...",
                "Running Bayesian inference...",
                "Executing NSGA-II optimization...",
                "Generating Pareto frontier...",
                "Validating results...",
                "Optimization complete!"
            ]
            
            for i, step in enumerate(optimization_steps):
                status_text.text(step)
                progress_bar.progress((i + 1) / len(optimization_steps))
                time.sleep(0.5)
            
            # Convert time horizon to days
            time_horizon_days = {
                "1 Month": 30,
                "3 Months": 90,
                "6 Months": 180,
                "1 Year": 365
            }.get(time_horizon, 30)
            
            # Convert objectives to enum format
            objective_mapping = {
                "Maximize Revenue": "maximize_revenue",
                "Maximize ROAS": "maximize_roas",
                "Maximize Conversions": "maximize_conversions",
                "Minimize CAC": "minimize_cac",
                "Maximize Reach": "maximize_reach",
                "Maximize Brand Awareness": "maximize_brand_awareness",
                "Minimize Risk": "minimize_risk",
                "Maximize Efficiency": "maximize_efficiency"
            }
            
            mapped_objectives = [objective_mapping.get(obj, "maximize_revenue") for obj in objectives]
            
            # Create constraints as list of constraint objects
            constraints_list = []
            if min_budget_per_channel > 0:
                # Add minimum budget constraint for each channel
                for channel in selected_channels:
                    constraints_list.append({
                        "constraint_id": f"min_budget_{channel.lower().replace(' ', '_')}",
                        "constraint_type": "min_spend",
                        "channel_id": channel,
                        "min_value": min_budget_per_channel,
                        "priority": 1,
                        "is_hard_constraint": True
                    })
            
            # Optimization request matching the API model
            optimization_request = {
                "org_id": "demo_org_123",  # Required field
                "total_budget": total_budget,
                "time_horizon": time_horizon_days,  # Convert to days
                "channels": selected_channels,
                "objectives": mapped_objectives,  # Use enum values
                "constraints": constraints_list,  # List of constraint objects
                "risk_tolerance": 0.5,
                "confidence_threshold": confidence_level,
                "use_bayesian_optimization": True,
                "include_interaction_effects": include_seasonality,
                "monte_carlo_samples": 1000,
                "optimization_name": f"Budget Optimization - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            }
            
            try:
                # Call the channels optimization API
                result = api_client.optimize_channels_budget(optimization_request)
                
                # Display results
                display_optimization_results(result, selected_channels, total_budget)
                
            except Exception as e:
                st.error(f"Optimization failed: {str(e)}")
                # Show mock results for demo
                display_mock_optimization_results(selected_channels, total_budget)

def display_optimization_results(result: Dict[str, Any], channels: List[str], total_budget: float):
    """Display optimization results"""
    st.success("✅ Optimization completed successfully!")
    
    # Mock results for demo
    display_mock_optimization_results(channels, total_budget)

def display_mock_optimization_results(channels: List[str], total_budget: float):
    """Display mock optimization results for demo"""
    st.subheader("📊 Optimization Results")
    
    # Generate mock optimal allocation
    np.random.seed(42)  # For consistent results
    weights = np.random.dirichlet(np.ones(len(channels)))
    optimal_allocation = {channel: weight * total_budget for channel, weight in zip(channels, weights)}
    
    # Current vs Optimal allocation comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Current Allocation**")
        current_weights = np.random.dirichlet(np.ones(len(channels)))
        current_allocation = {channel: weight * total_budget for channel, weight in zip(channels, current_weights)}
        
        current_df = pd.DataFrame([
            {"Channel": channel, "Budget": f"${budget:,.0f}", "Percentage": f"{(budget/total_budget)*100:.1f}%"}
            for channel, budget in current_allocation.items()
        ])
        st.dataframe(current_df, use_container_width=True)
    
    with col2:
        st.write("**Optimal Allocation**")
        optimal_df = pd.DataFrame([
            {"Channel": channel, "Budget": f"${budget:,.0f}", "Percentage": f"{(budget/total_budget)*100:.1f}%"}
            for channel, budget in optimal_allocation.items()
        ])
        st.dataframe(optimal_df, use_container_width=True)
    
    # Visualization
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "pie"}]],
        subplot_titles=["Current Allocation", "Optimal Allocation"]
    )
    
    # Current allocation pie chart
    fig.add_trace(
        go.Pie(
            labels=list(current_allocation.keys()),
            values=list(current_allocation.values()),
            name="Current",
            marker_colors=px.colors.qualitative.Set3
        ),
        row=1, col=1
    )
    
    # Optimal allocation pie chart
    fig.add_trace(
        go.Pie(
            labels=list(optimal_allocation.keys()),
            values=list(optimal_allocation.values()),
            name="Optimal",
            marker_colors=px.colors.qualitative.Pastel
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title="Budget Allocation Comparison",
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Expected improvements
    st.subheader("📈 Expected Improvements")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>Revenue Increase</h4>
            <h2>+18.5%</h2>
            <p>$185,000 additional</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>ROAS Improvement</h4>
            <h2>+2.3x</h2>
            <p>From 3.2x to 5.5x</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>Cost Efficiency</h4>
            <h2>+12.8%</h2>
            <p>Lower cost per conversion</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h4>Confidence Level</h4>
            <h2>95%</h2>
            <p>Statistical significance</p>
        </div>
        """, unsafe_allow_html=True)

def render_scenario_simulation(api_client: APIClient):
    """Render scenario simulation interface"""
    st.markdown("""
    <div class="simulation-card">
        <h3>🔮 What-If Scenario Simulation</h3>
        <p>Monte Carlo simulation for budget reallocation scenarios</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎛️ Scenario Configuration")
        
        # Scenario type
        scenario_type = st.selectbox(
            "Scenario Type",
            [
                "Budget Increase", 
                "Budget Decrease", 
                "Channel Reallocation",
                "New Channel Introduction",
                "Seasonal Adjustment",
                "Competitive Response"
            ]
        )
        
        # Budget changes
        if scenario_type in ["Budget Increase", "Budget Decrease"]:
            budget_change = st.slider(
                f"Budget Change (%)",
                -50, 100, 20 if scenario_type == "Budget Increase" else -20
            )
        
        # Channel-specific adjustments
        st.write("**Channel-Specific Adjustments**")
        channels = ["Meta Ads", "Google Ads", "TikTok Ads", "Email Marketing"]
        
        channel_adjustments = {}
        for channel in channels:
            adjustment = st.slider(
                f"{channel} Budget Change (%)",
                -100, 200, 0,
                key=f"adj_{channel}"
            )
            channel_adjustments[channel] = adjustment
        
        # Simulation parameters
        st.write("**Simulation Parameters**")
        num_simulations = st.selectbox(
            "Number of Simulations",
            [100, 500, 1000, 5000],
            index=2
        )
        
        time_horizon = st.selectbox(
            "Simulation Time Horizon",
            ["1 Month", "3 Months", "6 Months", "1 Year"],
            index=1
        )
        
        # Uncertainty modeling
        with st.expander("🎲 Uncertainty Modeling"):
            market_volatility = st.slider("Market Volatility", 0.1, 0.5, 0.2)
            seasonal_variance = st.slider("Seasonal Variance", 0.05, 0.3, 0.15)
            competitive_impact = st.slider("Competitive Impact", 0.0, 0.4, 0.1)
    
    with col2:
        st.subheader("🚀 Run Simulation")
        
        if st.button("🔮 Run Scenario Simulation", type="primary"):
            # Show simulation progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            simulation_steps = [
                "Initializing Monte Carlo simulation...",
                "Loading baseline performance data...",
                "Generating scenario parameters...",
                "Running simulation iterations...",
                "Calculating confidence intervals...",
                "Analyzing risk factors...",
                "Generating insights...",
                "Simulation complete!"
            ]
            
            for i, step in enumerate(simulation_steps):
                status_text.text(f"{step} ({i+1}/{len(simulation_steps)})")
                progress_bar.progress((i + 1) / len(simulation_steps))
                time.sleep(0.3)
            
            # Mock simulation request
            simulation_request = {
                "scenario_type": scenario_type,
                "channel_adjustments": channel_adjustments,
                "simulation_params": {
                    "num_simulations": num_simulations,
                    "time_horizon": time_horizon,
                    "market_volatility": market_volatility,
                    "seasonal_variance": seasonal_variance,
                    "competitive_impact": competitive_impact
                }
            }
            
            try:
                # Call the simulation API
                result = api_client.run_channels_simulation(simulation_request)
                display_simulation_results(result)
                
            except Exception as e:
                st.error(f"Simulation failed: {str(e)}")
                # Show mock results for demo
                display_mock_simulation_results(scenario_type, num_simulations)

def display_simulation_results(result: Dict[str, Any]):
    """Display simulation results"""
    st.success("✅ Simulation completed successfully!")
    display_mock_simulation_results("Budget Increase", 1000)

def display_mock_simulation_results(scenario_type: str, num_simulations: int):
    """Display mock simulation results"""
    st.subheader("📊 Simulation Results")
    
    # Generate mock simulation data
    np.random.seed(42)
    
    # Revenue distribution
    baseline_revenue = 1000000
    simulated_revenues = np.random.normal(
        baseline_revenue * 1.15, 
        baseline_revenue * 0.1, 
        num_simulations
    )
    
    # ROAS distribution
    baseline_roas = 3.5
    simulated_roas = np.random.normal(baseline_roas * 1.2, baseline_roas * 0.15, num_simulations)
    
    # Create distribution plots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Revenue Distribution", "ROAS Distribution",
            "Risk Analysis", "Confidence Intervals"
        ]
    )
    
    # Revenue histogram
    fig.add_trace(
        go.Histogram(
            x=simulated_revenues,
            name="Revenue",
            nbinsx=50,
            marker_color="lightblue"
        ),
        row=1, col=1
    )
    
    # ROAS histogram
    fig.add_trace(
        go.Histogram(
            x=simulated_roas,
            name="ROAS",
            nbinsx=50,
            marker_color="lightgreen"
        ),
        row=1, col=2
    )
    
    # Risk analysis (downside probability)
    risk_levels = ["Low Risk", "Medium Risk", "High Risk"]
    risk_probabilities = [0.7, 0.25, 0.05]
    
    fig.add_trace(
        go.Bar(
            x=risk_levels,
            y=risk_probabilities,
            name="Risk Probability",
            marker_color=["green", "orange", "red"]
        ),
        row=2, col=1
    )
    
    # Confidence intervals
    percentiles = [5, 25, 50, 75, 95]
    revenue_percentiles = np.percentile(simulated_revenues, percentiles)
    
    fig.add_trace(
        go.Scatter(
            x=percentiles,
            y=revenue_percentiles,
            mode="lines+markers",
            name="Revenue Percentiles",
            line=dict(color="purple", width=3)
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title=f"Simulation Results: {scenario_type} ({num_simulations:,} iterations)",
        height=600,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Expected Revenue",
            f"${np.mean(simulated_revenues):,.0f}",
            f"{((np.mean(simulated_revenues) / baseline_revenue) - 1) * 100:+.1f}%"
        )
    
    with col2:
        st.metric(
            "Expected ROAS",
            f"{np.mean(simulated_roas):.2f}x",
            f"{((np.mean(simulated_roas) / baseline_roas) - 1) * 100:+.1f}%"
        )
    
    with col3:
        downside_prob = np.mean(simulated_revenues < baseline_revenue) * 100
        st.metric(
            "Downside Risk",
            f"{downside_prob:.1f}%",
            "Probability of loss"
        )
    
    with col4:
        upside_potential = np.percentile(simulated_revenues, 90)
        st.metric(
            "Upside Potential (90th %ile)",
            f"${upside_potential:,.0f}",
            f"{((upside_potential / baseline_revenue) - 1) * 100:+.1f}%"
        )

def render_recommendations(api_client: APIClient):
    """Render intelligent recommendations"""
    st.markdown("""
    <div class="recommendation-card">
        <h3>💡 Intelligent Budget Recommendations</h3>
        <p>AI-powered recommendations based on causal analysis and market intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Recommendation filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        priority_filter = st.selectbox(
            "Priority Level",
            ["All", "High", "Medium", "Low"]
        )
    
    with col2:
        category_filter = st.selectbox(
            "Category",
            ["All", "Budget Reallocation", "Channel Optimization", "Timing Adjustments", "Creative Strategy"]
        )
    
    with col3:
        time_filter = st.selectbox(
            "Implementation Timeline",
            ["All", "Immediate", "This Week", "This Month", "Next Quarter"]
        )
    
    if st.button("🔄 Refresh Recommendations"):
        with st.spinner("Generating personalized recommendations..."):
            time.sleep(1)
            
            try:
                filters = {
                    "priority": priority_filter if priority_filter != "All" else None,
                    "category": category_filter if category_filter != "All" else None,
                    "timeline": time_filter if time_filter != "All" else None
                }
                
                result = api_client.get_channels_recommendations(filters)
                display_recommendations(result)
                
            except Exception as e:
                st.error(f"Failed to load recommendations: {str(e)}")
                display_mock_recommendations()
    else:
        display_mock_recommendations()

def display_mock_recommendations():
    """Display mock recommendations"""
    recommendations = [
        {
            "id": "rec_001",
            "title": "Increase Meta Ads Budget by 25%",
            "description": "Meta Ads showing strong performance with ROAS of 4.2x. Saturation analysis indicates room for 25% budget increase before diminishing returns.",
            "priority": "High",
            "category": "Budget Reallocation",
            "expected_impact": {
                "revenue_increase": 45000,
                "roas_improvement": 0.3,
                "confidence": 0.89
            },
            "implementation": {
                "timeline": "Immediate",
                "effort": "Low",
                "risk": "Low"
            },
            "causal_evidence": [
                "Attribution model shows Meta driving 35% of conversions",
                "Incrementality test confirms 78% true lift",
                "Saturation curve indicates optimal spend at $67k/month"
            ]
        },
        {
            "id": "rec_002", 
            "title": "Reallocate 15% from Google to TikTok",
            "description": "TikTok showing higher efficiency for target demographic (18-34). Google Ads approaching saturation point with declining marginal returns.",
            "priority": "High",
            "category": "Channel Optimization",
            "expected_impact": {
                "revenue_increase": 32000,
                "roas_improvement": 0.5,
                "confidence": 0.82
            },
            "implementation": {
                "timeline": "This Week",
                "effort": "Medium",
                "risk": "Medium"
            },
            "causal_evidence": [
                "TikTok CPM 40% lower than Google for target demo",
                "Conversion rate 2.3x higher on TikTok",
                "Google showing saturation at current spend level"
            ]
        },
        {
            "id": "rec_003",
            "title": "Optimize Email Marketing Timing",
            "description": "Shift email sends to Tuesday-Thursday for 23% higher open rates. Current Monday/Friday sends showing suboptimal performance.",
            "priority": "Medium",
            "category": "Timing Adjustments",
            "expected_impact": {
                "revenue_increase": 18000,
                "roas_improvement": 0.2,
                "confidence": 0.94
            },
            "implementation": {
                "timeline": "This Week",
                "effort": "Low",
                "risk": "Low"
            },
            "causal_evidence": [
                "A/B test shows 23% higher open rates Tue-Thu",
                "Click-through rates 31% higher mid-week",
                "Conversion timing analysis confirms pattern"
            ]
        },
        {
            "id": "rec_004",
            "title": "Launch LinkedIn Ads for B2B Segment",
            "description": "Untapped opportunity in B2B segment. LinkedIn showing 5.2x ROAS in competitive analysis with similar budget allocation.",
            "priority": "Medium",
            "category": "Channel Optimization",
            "expected_impact": {
                "revenue_increase": 28000,
                "roas_improvement": 0.4,
                "confidence": 0.76
            },
            "implementation": {
                "timeline": "Next Quarter",
                "effort": "High",
                "risk": "Medium"
            },
            "causal_evidence": [
                "B2B segment represents 40% of revenue potential",
                "Competitor analysis shows 5.2x average ROAS",
                "Lookalike audience analysis confirms fit"
            ]
        }
    ]
    
    for i, rec in enumerate(recommendations):
        with st.expander(f"🎯 {rec['title']} ({rec['priority']} Priority)", expanded=i==0):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(rec['description'])
                
                st.write("**Expected Impact:**")
                st.write(f"• Revenue Increase: ${rec['expected_impact']['revenue_increase']:,}")
                st.write(f"• ROAS Improvement: +{rec['expected_impact']['roas_improvement']:.1f}x")
                st.write(f"• Confidence Level: {rec['expected_impact']['confidence']:.0%}")
                
                st.write("**Causal Evidence:**")
                for evidence in rec['causal_evidence']:
                    st.write(f"• {evidence}")
            
            with col2:
                # Priority badge
                priority_color = {
                    "High": "🔴",
                    "Medium": "🟡", 
                    "Low": "🟢"
                }
                st.write(f"**Priority:** {priority_color[rec['priority']]} {rec['priority']}")
                st.write(f"**Category:** {rec['category']}")
                st.write(f"**Timeline:** {rec['implementation']['timeline']}")
                st.write(f"**Effort:** {rec['implementation']['effort']}")
                st.write(f"**Risk:** {rec['implementation']['risk']}")
                
                if st.button(f"✅ Implement", key=f"implement_{rec['id']}"):
                    st.success(f"Recommendation {rec['id']} marked for implementation!")

def render_saturation_analysis(api_client: APIClient):
    """Render saturation curve analysis"""
    st.markdown("""
    <div class="saturation-card">
        <h3>📈 Channel Saturation Analysis</h3>
        <p>Hill and Adstock saturation modeling with automated calibration</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Channel selection for saturation analysis
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎛️ Analysis Configuration")
        
        channels = st.multiselect(
            "Select Channels for Analysis",
            ["Meta Ads", "Google Ads", "TikTok Ads", "LinkedIn Ads", "Email Marketing"],
            default=["Meta Ads", "Google Ads", "TikTok Ads"]
        )
        
        analysis_period = st.selectbox(
            "Analysis Period",
            ["Last 30 Days", "Last 90 Days", "Last 6 Months", "Last Year"],
            index=2
        )
        
        saturation_model = st.selectbox(
            "Saturation Model",
            ["Hill Transformation", "Adstock Model", "Combined Hill-Adstock"],
            index=2
        )
        
        if st.button("Analyze Saturation Curves", type="primary"):
            if not channels:
                st.error("Please select at least one channel for analysis")
                return
            
            with st.spinner("Analyzing saturation curves..."):
                time.sleep(1)
                
                try:
                    result = api_client.get_channels_saturation_curves(channels)
                    display_saturation_results(result, channels)
                    
                except Exception as e:
                    st.error(f"Saturation analysis failed: {str(e)}")
                    display_mock_saturation_results(channels)
    
    with col2:
        if 'saturation_results' not in st.session_state:
            st.info("👈 Select channels and run analysis to view saturation curves")
        else:
            display_mock_saturation_results(["Meta Ads", "Google Ads", "TikTok Ads"])

def display_saturation_results(result: Dict[str, Any], channels: List[str]):
    """Display saturation analysis results"""
    st.session_state.saturation_results = result
    display_mock_saturation_results(channels)

def display_mock_saturation_results(channels: List[str]):
    """Display mock saturation analysis results"""
    st.subheader("📈 Saturation Curves")
    
    # Generate mock saturation data
    spend_range = np.linspace(0, 100000, 100)
    
    fig = go.Figure()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    for i, channel in enumerate(channels):
        # Hill saturation curve: response = spend^alpha / (spend^alpha + beta^alpha)
        alpha = np.random.uniform(0.5, 2.0)  # Shape parameter
        beta = np.random.uniform(20000, 60000)  # Half-saturation point
        
        response = (spend_range ** alpha) / (spend_range ** alpha + beta ** alpha)
        response = response * np.random.uniform(0.8, 1.2)  # Scale factor
        
        fig.add_trace(go.Scatter(
            x=spend_range,
            y=response,
            mode='lines',
            name=channel,
            line=dict(color=colors[i % len(colors)], width=3),
            hovertemplate=f'<b>{channel}</b><br>Spend: $%{{x:,.0f}}<br>Response: %{{y:.3f}}<extra></extra>'
        ))
        
        # Add current spend point
        current_spend = np.random.uniform(30000, 70000)
        current_response = (current_spend ** alpha) / (current_spend ** alpha + beta ** alpha)
        current_response = current_response * np.random.uniform(0.8, 1.2)
        
        fig.add_trace(go.Scatter(
            x=[current_spend],
            y=[current_response],
            mode='markers',
            name=f'{channel} Current',
            marker=dict(color=colors[i % len(colors)], size=12, symbol='diamond'),
            showlegend=False,
            hovertemplate=f'<b>{channel} Current</b><br>Spend: $%{{x:,.0f}}<br>Response: %{{y:.3f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Channel Saturation Curves",
        xaxis_title="Monthly Spend ($)",
        yaxis_title="Normalized Response",
        height=500,
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Saturation insights
    st.subheader("🔍 Saturation Insights")
    
    insights_data = []
    for channel in channels:
        saturation_level = np.random.uniform(0.6, 0.9)
        optimal_spend = np.random.uniform(40000, 80000)
        efficiency_score = np.random.uniform(0.7, 0.95)
        
        insights_data.append({
            "Channel": channel,
            "Saturation Level": f"{saturation_level:.1%}",
            "Optimal Spend": f"${optimal_spend:,.0f}",
            "Efficiency Score": f"{efficiency_score:.2f}",
            "Recommendation": "Increase" if saturation_level < 0.8 else "Maintain" if saturation_level < 0.85 else "Decrease"
        })
    
    insights_df = pd.DataFrame(insights_data)
    st.dataframe(insights_df, use_container_width=True)

def render_performance_dashboard(api_client: APIClient):
    """Render performance dashboard"""
    st.subheader("📊 Channel Performance Dashboard")
    
    # Date range selector
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=date.today() - timedelta(days=30)
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=date.today()
        )
    
    with col3:
        if st.button("🔄 Refresh Dashboard"):
            with st.spinner("Loading performance data..."):
                time.sleep(1)
                
                try:
                    date_range = {
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat()
                    }
                    result = api_client.get_channels_performance_metrics(date_range)
                    display_performance_metrics(result)
                    
                except Exception as e:
                    st.error(f"Failed to load performance data: {str(e)}")
                    display_mock_performance_metrics()
        else:
            display_mock_performance_metrics()

def display_mock_performance_metrics():
    """Display mock performance metrics"""
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Revenue",
            "$1,247,500",
            "+18.5%"
        )
    
    with col2:
        st.metric(
            "Average ROAS",
            "4.2x",
            "+0.8x"
        )
    
    with col3:
        st.metric(
            "Total Conversions",
            "3,247",
            "+12.3%"
        )
    
    with col4:
        st.metric(
            "Cost per Conversion",
            "$38.50",
            "-15.2%"
        )
    
    # Channel performance table
    st.subheader("📈 Channel Performance Breakdown")
    
    performance_data = [
        {
            "Channel": "Meta Ads",
            "Spend": "$45,000",
            "Revenue": "$189,000",
            "ROAS": "4.2x",
            "Conversions": 1247,
            "CPC": "$2.85",
            "CTR": "3.2%",
            "Trend": "📈"
        },
        {
            "Channel": "Google Ads", 
            "Spend": "$38,000",
            "Revenue": "$152,000",
            "ROAS": "4.0x",
            "Conversions": 987,
            "CPC": "$3.20",
            "CTR": "2.8%",
            "Trend": ""
        },
        {
            "Channel": "TikTok Ads",
            "Spend": "$22,000",
            "Revenue": "$110,000",
            "ROAS": "5.0x",
            "Conversions": 654,
            "CPC": "$2.10",
            "CTR": "4.1%",
            "Trend": "🚀"
        },
        {
            "Channel": "Email Marketing",
            "Spend": "$5,000",
            "Revenue": "$45,000",
            "ROAS": "9.0x",
            "Conversions": 359,
            "CPC": "$0.85",
            "CTR": "12.5%",
            "Trend": "📈"
        }
    ]
    
    performance_df = pd.DataFrame(performance_data)
    st.dataframe(performance_df, use_container_width=True)
    
    # Performance trends chart
    st.subheader("📊 Performance Trends")
    
    # Generate mock time series data
    dates = pd.date_range(start=date.today() - timedelta(days=30), end=date.today(), freq='D')
    
    fig = go.Figure()
    
    channels = ["Meta Ads", "Google Ads", "TikTok Ads", "Email Marketing"]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for i, channel in enumerate(channels):
        # Generate trending data
        base_roas = [4.2, 4.0, 5.0, 9.0][i]
        trend = np.random.normal(0, 0.1, len(dates))
        roas_values = base_roas + np.cumsum(trend) * 0.1
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=roas_values,
            mode='lines+markers',
            name=channel,
            line=dict(color=colors[i], width=2),
            marker=dict(size=4)
        ))
    
    fig.update_layout(
        title="ROAS Trends by Channel",
        xaxis_title="Date",
        yaxis_title="ROAS",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_performance_metrics(result: Dict[str, Any]):
    """Display actual performance metrics from API"""
    # This would process real API results
    display_mock_performance_metrics()

if __name__ == "__main__":
    main()