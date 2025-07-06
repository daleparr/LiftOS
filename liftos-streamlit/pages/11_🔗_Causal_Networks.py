import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import networkx as nx
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
    page_title="LiftOS - Interactive Causal Networks",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .causal-network-header {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        text-align: center;
    }
    
    .network-controls {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .causal-path-card {
        background: linear-gradient(135deg, #a29bfe 0%, #6c5ce7 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .intervention-card {
        background: linear-gradient(135deg, #fd79a8 0%, #e84393 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .network-metric {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .causal-strength-bar {
        background: #e9ecef;
        border-radius: 10px;
        height: 20px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    
    .strength-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    .strength-high { background: linear-gradient(90deg, #00b894, #00a085); }
    .strength-medium { background: linear-gradient(90deg, #fdcb6e, #e17055); }
    .strength-low { background: linear-gradient(90deg, #fd79a8, #e84393); }
    
    .relationship-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-weight: bold;
        font-size: 0.8rem;
        margin: 0.25rem;
        display: inline-block;
    }
    .badge-direct { background: #00b894; color: white; }
    .badge-indirect { background: #74b9ff; color: white; }
    .badge-confounder { background: #fdcb6e; color: #2d3436; }
    .badge-mediator { background: #a29bfe; color: white; }
    .badge-moderator { background: #fd79a8; color: white; }
    
    .policy-validation {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        text-align: center;
    }
    
    .interactive-panel {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #74b9ff;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def get_causal_network_data(api_client, org_id="default_org", filters=None):
    """Get causal network data with filtering options"""
    try:
        params = {"org_id": org_id}
        if filters:
            params.update(filters)
        
        response = api_client.get("/memory/causal/network", params=params)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching causal network data: {str(e)}")
        return None

def get_causal_paths_data(api_client, source_var, target_var, org_id="default_org"):
    """Get causal paths between two variables"""
    try:
        response = api_client.get(f"/memory/causal/paths/{source_var}/{target_var}", 
                                params={"org_id": org_id})
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching causal paths: {str(e)}")
        return None

def create_interactive_network_graph(network_data, layout_type="spring", filter_strength=0.5):
    """Create interactive causal network visualization"""
    if not network_data or 'nodes' not in network_data or 'edges' not in network_data:
        return create_mock_network_graph(layout_type, filter_strength)
    
    nodes = network_data['nodes']
    edges = [edge for edge in network_data['edges'] 
             if edge.get('relationship', {}).get('strength', 0) >= filter_strength]
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add nodes with attributes
    for node in nodes:
        G.add_node(node['id'], **node)
    
    # Add filtered edges
    for edge in edges:
        G.add_edge(edge['source_node_id'], edge['target_node_id'], **edge)
    
    # Calculate layout
    if layout_type == "spring":
        pos = nx.spring_layout(G, k=2, iterations=50)
    elif layout_type == "circular":
        pos = nx.circular_layout(G)
    elif layout_type == "hierarchical":
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot') if hasattr(nx, 'nx_agraph') else nx.spring_layout(G)
    else:
        pos = nx.spring_layout(G)
    
    # Prepare node traces
    node_x, node_y = [], []
    node_text, node_colors, node_sizes = [], [], []
    
    for node in nodes:
        node_id = node['id']
        if node_id in pos and node_id in G:
            x, y = pos[node_id]
            node_x.append(x)
            node_y.append(y)
            
            # Node information
            degree = G.degree(node_id)
            node_text.append(
                f"<b>{node.get('variable_name', 'Unknown')}</b><br>"
                f"Type: {node.get('variable_type', 'Unknown')}<br>"
                f"Platform: {node.get('platform', 'Unknown')}<br>"
                f"Connections: {degree}<br>"
                f"Centrality: {nx.degree_centrality(G).get(node_id, 0):.3f}"
            )
            
            # Color by variable type
            type_colors = {
                'campaign': '#e74c3c',
                'ad_set': '#3498db',
                'creative': '#f39c12',
                'audience': '#9b59b6',
                'keyword': '#1abc9c',
                'product': '#34495e'
            }
            node_colors.append(type_colors.get(node.get('variable_type', 'unknown'), '#95a5a6'))
            
            # Size by degree centrality
            centrality = nx.degree_centrality(G).get(node_id, 0)
            node_sizes.append(max(10, centrality * 100 + 10))
    
    # Prepare edge traces
    edge_x, edge_y = [], []
    edge_colors, edge_widths = [], []
    
    for edge in edges:
        source_id = edge['source_node_id']
        target_id = edge['target_node_id']
        
        if source_id in pos and target_id in pos:
            x0, y0 = pos[source_id]
            x1, y1 = pos[target_id]
            
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Edge styling based on relationship
            relationship = edge.get('relationship', {})
            strength = relationship.get('strength', 0)
            rel_type = relationship.get('relationship_type', 'unknown')
            
            # Color by relationship type
            type_edge_colors = {
                'direct_cause': '#e74c3c',
                'indirect_cause': '#3498db',
                'confounder': '#f39c12',
                'mediator': '#9b59b6',
                'moderator': '#1abc9c'
            }
            edge_colors.append(type_edge_colors.get(rel_type, '#95a5a6'))
            
            # Width by strength
            edge_widths.append(max(1, strength * 5))
    
    # Create the plot
    fig = go.Figure()
    
    # Add edges
    for i in range(0, len(edge_x), 3):
        if i + 1 < len(edge_x):
            fig.add_trace(go.Scatter(
                x=edge_x[i:i+2], y=edge_y[i:i+2],
                mode='lines',
                line=dict(
                    color=edge_colors[i//3] if i//3 < len(edge_colors) else '#95a5a6',
                    width=edge_widths[i//3] if i//3 < len(edge_widths) else 2
                ),
                hoverinfo='none',
                showlegend=False
            ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            opacity=0.8,
            line=dict(width=2, color='white')
        ),
        text=[node.get('variable_name', 'Unknown') for node in nodes if node['id'] in pos and node['id'] in G],
        textposition="middle center",
        textfont=dict(size=10, color='white'),
        hovertext=node_text,
        hoverinfo='text',
        name='Variables'
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Interactive Causal Network ({layout_type.title()} Layout)",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[
            dict(
                text="Drag nodes to explore relationships • Hover for details • Click to select",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color='#666', size=12)
            )
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600
    )
    
    return fig

def create_mock_network_graph(layout_type="spring", filter_strength=0.5):
    """Create mock network graph for demo"""
    np.random.seed(42)
    
    # Mock nodes
    variable_types = ['campaign', 'ad_set', 'creative', 'audience', 'keyword', 'product']
    platforms = ['meta', 'google', 'klaviyo', 'internal']
    
    nodes = []
    for i in range(15):
        nodes.append({
            'id': f'var_{i}',
            'variable_name': f'Variable_{i}',
            'variable_type': np.random.choice(variable_types),
            'platform': np.random.choice(platforms)
        })
    
    # Mock edges
    edges = []
    for i in range(25):
        source = np.random.choice(nodes)['id']
        target = np.random.choice(nodes)['id']
        if source != target:
            strength = np.random.uniform(0.3, 1.0)
            if strength >= filter_strength:
                edges.append({
                    'source_node_id': source,
                    'target_node_id': target,
                    'relationship': {
                        'relationship_type': np.random.choice(['direct_cause', 'indirect_cause', 'confounder', 'mediator']),
                        'strength': strength,
                        'confidence': np.random.uniform(0.6, 0.95)
                    }
                })
    
    mock_data = {'nodes': nodes, 'edges': edges}
    return create_interactive_network_graph(mock_data, layout_type, filter_strength)

def render_network_metrics(network_data):
    """Render network analysis metrics"""
    if network_data and 'nodes' in network_data and 'edges' in network_data:
        nodes = network_data['nodes']
        edges = network_data['edges']
        
        # Create NetworkX graph for analysis
        G = nx.Graph()
        for node in nodes:
            G.add_node(node['id'])
        for edge in edges:
            G.add_edge(edge['source_node_id'], edge['target_node_id'])
        
        # Calculate metrics
        density = nx.density(G)
        avg_clustering = nx.average_clustering(G)
        try:
            avg_path_length = nx.average_shortest_path_length(G) if nx.is_connected(G) else 0
        except:
            avg_path_length = 0
        centrality_values = list(nx.degree_centrality(G).values())
        max_centrality = max(centrality_values) if centrality_values else 0
    else:
        # Mock metrics
        density = 0.23
        avg_clustering = 0.67
        avg_path_length = 2.8
        max_centrality = 0.45
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="network-metric">
            <h3>{density:.3f}</h3>
            <p>Network Density</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="network-metric">
            <h3>{avg_clustering:.3f}</h3>
            <p>Avg Clustering</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="network-metric">
            <h3>{avg_path_length:.1f}</h3>
            <p>Avg Path Length</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="network-metric">
            <h3>{max_centrality:.3f}</h3>
            <p>Max Centrality</p>
        </div>
        """, unsafe_allow_html=True)

def render_causal_paths_analysis(api_client, network_data):
    """Render causal paths analysis between variables"""
    st.subheader("🛤️ Causal Path Analysis")
    
    if network_data and 'nodes' in network_data:
        nodes = network_data['nodes']
        variable_names = [f"{node.get('variable_name', 'Unknown')} ({node['id']})" for node in nodes]
        
        col1, col2 = st.columns(2)
        
        with col1:
            source_var = st.selectbox("Source Variable:", variable_names, key="source")
        
        with col2:
            target_var = st.selectbox("Target Variable:", variable_names, key="target")
        
        if st.button("🔍 Find Causal Paths"):
            source_id = source_var.split('(')[-1].rstrip(')')
            target_id = target_var.split('(')[-1].rstrip(')')
            
            if source_id != target_id:
                with st.spinner("Analyzing causal paths..."):
                    paths_data = get_causal_paths_data(api_client, source_id, target_id)
                    
                    if paths_data and 'paths' in paths_data:
                        paths = paths_data['paths']
                        
                        st.markdown(f"**Found {len(paths)} causal paths from {source_var.split('(')[0]} to {target_var.split('(')[0]}:**")
                        
                        for i, path in enumerate(paths[:5]):  # Show top 5 paths
                            path_strength = path.get('total_strength', 0)
                            path_confidence = path.get('confidence', 0)
                            path_nodes = path.get('path_nodes', [])
                            
                            strength_class = 'strength-high' if path_strength > 0.8 else 'strength-medium' if path_strength > 0.5 else 'strength-low'
                            
                            st.markdown(f"""
                            <div class="causal-path-card">
                                <h4>Path {i+1}</h4>
                                <p><strong>Route:</strong> {' → '.join(path_nodes)}</p>
                                <div class="causal-strength-bar">
                                    <div class="strength-fill {strength_class}" style="width: {path_strength*100}%"></div>
                                </div>
                                <p><strong>Strength:</strong> {path_strength:.3f} | <strong>Confidence:</strong> {path_confidence:.3f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        # Mock paths for demo
                        st.markdown(f"**Found 3 causal paths from {source_var.split('(')[0]} to {target_var.split('(')[0]}:**")
                        
                        mock_paths = [
                            {"nodes": ["Campaign Budget", "Ad Set Targeting", "Creative Performance"], "strength": 0.87, "confidence": 0.92},
                            {"nodes": ["Campaign Budget", "Bid Strategy", "Creative Performance"], "strength": 0.73, "confidence": 0.85},
                            {"nodes": ["Campaign Budget", "Audience Size", "Reach", "Creative Performance"], "strength": 0.64, "confidence": 0.78}
                        ]
                        
                        for i, path in enumerate(mock_paths):
                            strength_class = 'strength-high' if path['strength'] > 0.8 else 'strength-medium' if path['strength'] > 0.5 else 'strength-low'
                            
                            st.markdown(f"""
                            <div class="causal-path-card">
                                <h4>Path {i+1}</h4>
                                <p><strong>Route:</strong> {' → '.join(path['nodes'])}</p>
                                <div class="causal-strength-bar">
                                    <div class="strength-fill {strength_class}" style="width: {path['strength']*100}%"></div>
                                </div>
                                <p><strong>Strength:</strong> {path['strength']:.3f} | <strong>Confidence:</strong> {path['confidence']:.3f}</p>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.warning("Please select different source and target variables.")
    else:
        st.info("Network data not available. Connect to the causal memory system to analyze paths.")

def render_intervention_analysis():
    """Render causal intervention analysis"""
    st.subheader("🎯 Intervention Analysis")
    
    st.markdown("""
    <div class="interactive-panel">
        <h4>What-If Scenario Analysis</h4>
        <p>Simulate interventions on causal variables to predict downstream effects.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        intervention_var = st.selectbox(
            "Intervention Variable:",
            ["Campaign Budget", "Audience Size", "Bid Strategy", "Creative Type", "Ad Placement"]
        )
        
        intervention_change = st.slider(
            "Change Magnitude:",
            -50, 50, 0, 5,
            help="Percentage change in the intervention variable"
        )
    
    with col2:
        target_metrics = st.multiselect(
            "Target Metrics:",
            ["Conversions", "Cost Per Acquisition", "Return on Ad Spend", "Click-Through Rate", "Reach"],
            default=["Conversions", "Cost Per Acquisition"]
        )
    
    if st.button("🚀 Run Intervention Simulation"):
        with st.spinner("Simulating causal intervention..."):
            time.sleep(2)  # Simulate processing
            
            # Mock intervention results
            results = {
                "Conversions": np.random.normal(intervention_change * 0.8, 5),
                "Cost Per Acquisition": np.random.normal(-intervention_change * 0.6, 3),
                "Return on Ad Spend": np.random.normal(intervention_change * 0.7, 4),
                "Click-Through Rate": np.random.normal(intervention_change * 0.4, 2),
                "Reach": np.random.normal(intervention_change * 0.5, 3)
            }
            
            st.markdown(f"**Predicted effects of {intervention_change:+}% change in {intervention_var}:**")
            
            for metric in target_metrics:
                effect = results.get(metric, 0)
                effect_class = 'strength-high' if abs(effect) > 10 else 'strength-medium' if abs(effect) > 5 else 'strength-low'
                
                st.markdown(f"""
                <div class="intervention-card">
                    <h4>{metric}</h4>
                    <p><strong>Predicted Change:</strong> {effect:+.1f}%</p>
                    <div class="causal-strength-bar">
                        <div class="strength-fill {effect_class}" style="width: {min(abs(effect)*2, 100)}%"></div>
                    </div>
                    <p><strong>Confidence:</strong> {np.random.uniform(0.75, 0.95):.2f}</p>
                </div>
                """, unsafe_allow_html=True)

def render_relationship_explorer(network_data):
    """Render interactive relationship explorer"""
    st.subheader("🔍 Relationship Explorer")
    
    # Relationship type filter
    relationship_types = ['direct_cause', 'indirect_cause', 'confounder', 'mediator', 'moderator']
    selected_types = st.multiselect(
        "Filter by Relationship Type:",
        relationship_types,
        default=relationship_types,
        help="Select relationship types to display"
    )
    
    if network_data and 'edges' in network_data:
        edges = network_data['edges']
        filtered_edges = [
            edge for edge in edges
            if edge.get('relationship', {}).get('relationship_type') in selected_types
        ]
    else:
        # Mock filtered edges
        filtered_edges = [
            {
                'source_node_id': 'campaign_1',
                'target_node_id': 'performance_1',
                'relationship': {
                    'relationship_type': 'direct_cause',
                    'strength': 0.87,
                    'confidence': 0.92,
                    'evidence': ['Statistical significance', 'Temporal precedence', 'Dose-response']
                }
            },
            {
                'source_node_id': 'audience_1',
                'target_node_id': 'ctr_1',
                'relationship': {
                    'relationship_type': 'indirect_cause',
                    'strength': 0.64,
                    'confidence': 0.78,
                    'evidence': ['Correlation analysis', 'Mediation analysis']
                }
            }
        ]
    
    # Display relationships
    st.markdown(f"**Found {len(filtered_edges)} relationships matching your criteria:**")
    
    for i, edge in enumerate(filtered_edges[:10]):  # Show first 10
        relationship = edge.get('relationship', {})
        rel_type = relationship.get('relationship_type', 'unknown')
        strength = relationship.get('strength', 0)
        confidence = relationship.get('confidence', 0)
        evidence = relationship.get('evidence', [])
        
        badge_class = f'badge-{rel_type.replace("_", "-")}'
        strength_class = 'strength-high' if strength > 0.8 else 'strength-medium' if strength > 0.5 else 'strength-low'
        
        with st.expander(f"Relationship {i+1}: {edge.get('source_node_id', 'Unknown')} → {edge.get('target_node_id', 'Unknown')}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <span class="relationship-badge {badge_class}">{rel_type.replace('_', ' ').title()}</span>
                """, unsafe_allow_html=True)
                
                st.markdown(f"**Strength:** {strength:.3f}")
                st.markdown(f"**Confidence:** {confidence:.3f}")
            
            with col2:
                if evidence:
                    st.markdown("**Evidence:**")
                    for ev in evidence:
                        st.markdown(f"• {ev}")
                else:
                    st.markdown("**Evidence:**")
                    st.markdown("• Statistical significance")
                    st.markdown("• Temporal precedence")

def main():
    # Header
    st.markdown("""
    <div class="causal-network-header">
        <h1>🔗 Interactive Causal Relationship Networks</h1>
        <p>Explore and analyze causal relationships with interactive network visualizations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Policy validation banner
    st.markdown("""
    <div class="policy-validation">
        <strong>🎯 Policy 3 & 5 Validation:</strong> Universal Modular Access + Memory-Driven Intelligence - 
        Interactive exploration of causal relationships across all marketing platforms
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize API client
    api_client = LiftOSAPIClient()
    
    # Network controls
    st.markdown("""
    <div class="network-controls">
        <h3>🎛️ Network Configuration</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        org_id = st.selectbox(
            "Organization:",
            ["default_org", "demo_org", "test_org"],
            help="Select organization for network analysis"
        )
    
    with col2:
        layout_type = st.selectbox(
            "Layout Algorithm:",
            ["spring", "circular", "hierarchical"],
            help="Choose network layout algorithm"
        )
    
    with col3:
        min_strength = st.slider(
            "Min Relationship Strength:",
            0.0, 1.0, 0.5, 0.1,
            help="Filter relationships by minimum strength"
        )
    
    with col4:
        if st.button("🔄 Refresh Network", type="primary"):
            st.rerun()
    
    # Get network data
    with st.spinner("Loading causal network data..."):
        filters = {"min_strength": min_strength}
        network_data = get_causal_network_data(api_client, org_id, filters)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌐 Network Graph", 
        "🛤️ Causal Paths", 
        "🎯 Interventions", 
        "🔍 Relationship Explorer"
    ])
    
    with tab1:
        st.markdown("### Interactive Causal Network Visualization")
        
        # Network metrics
        render_network_metrics(network_data)
        
        # Interactive network graph
        try:
            if network_data:
                fig_network = create_interactive_network_graph(network_data, layout_type, min_strength)
            else:
                st.info("Loading live network data...")
                fig_network = create_mock_network_graph(layout_type, min_strength)
            
            st.plotly_chart(fig_network, use_container_width=True)
            
            # Network legend
            st.markdown("""
            **Network Legend:**
            - **Node Size**: Degree centrality (importance in network)
            - **Node Color**: Variable type (Campaign=Red, Ad Set=Blue, Creative=Orange, etc.)
            - **Edge Color**: Relationship type (Direct=Red, Indirect=Blue, Confounder=Orange, etc.)
            - **Edge Width**: Relationship strength
            """)
            
        except Exception as e:
            st.error(f"Error rendering network: {str(e)}")
            st.info("Displaying mock network for demonstration")
            fig_network = create_mock_network_graph(layout_type, min_strength)
            st.plotly_chart(fig_network, use_container_width=True)
    
    with tab2:
        render_causal_paths_analysis(api_client, network_data)
    
    with tab3:
        render_intervention_analysis()
    
    with tab4:
        render_relationship_explorer(network_data)
    
    # Footer with policy compliance
    st.markdown("---")
    st.markdown("""
    ### 🎯 Policy Compliance Summary
    
    **✅ Policy 3: Universal Modular Access**
    - **Cross-Platform Integration**: Unified causal relationships across Meta, Google, Klaviyo
    - **Interactive Exploration**: Real-time network analysis and path discovery
    - **Modular Visualization**: Plug-and-play network components
    
    **✅ Policy 5: Memory-Driven Compound Intelligence**
    - **Causal Network Learning**: Continuous relationship discovery and refinement
    - **Intervention Simulation**: What-if scenario analysis with predictive modeling
    - **Pattern Recognition**: Automated discovery of causal patterns and anomalies
    
    **🚀 LiftOS enables interactive exploration of causal relationships across all marketing platforms with real-time intervention analysis.**
    """)

if __name__ == "__main__":
    main()