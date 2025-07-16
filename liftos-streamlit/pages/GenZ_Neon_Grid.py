import streamlit as st
import plotly.graph_objects as go
import numpy as np
import random
import time
from datetime import datetime

# LiftOS Cyberpunk Colors
COLORS = {
    "hot_pink": "#FF1493",
    "electric_blue": "#00D4FF",
    "neon_green": "#00FF41",
    "cyber_purple": "#8A2BE2",
    "warning_orange": "#FF8C00",
    "deep_black": "#0A0A0A",
    "dark_gray": "#1A1A1A",
    "white": "#FFFFFF"
}

st.set_page_config(page_title="LiftOS: Neon Grid", page_icon="🔮", layout="wide")

# Cyberpunk Grid CSS
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    
    .stApp {{
        background: 
            radial-gradient(circle at 20% 20%, {COLORS['hot_pink']}22 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, {COLORS['electric_blue']}22 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, {COLORS['cyber_purple']}22 0%, transparent 50%),
            linear-gradient(135deg, {COLORS['deep_black']} 0%, {COLORS['dark_gray']} 100%);
        color: {COLORS['white']};
        font-family: 'Orbitron', monospace;
    }}
    
    .neon-title {{
        text-align: center;
        font-size: 4rem;
        font-weight: 900;
        background: linear-gradient(45deg, {COLORS['hot_pink']}, {COLORS['electric_blue']}, {COLORS['neon_green']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 30px {COLORS['hot_pink']};
        margin-bottom: 2rem;
        animation: neon-flicker 3s infinite;
    }}
    
    @keyframes neon-flicker {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.8; }}
    }}
    
    .holo-card {{
        background: linear-gradient(135deg, 
            rgba(255,20,147,0.1) 0%, 
            rgba(0,212,255,0.1) 50%, 
            rgba(138,43,226,0.1) 100%);
        border: 1px solid {COLORS['hot_pink']};
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        box-shadow: 
            0 0 20px rgba(255,20,147,0.3),
            inset 0 0 20px rgba(255,20,147,0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }}
    
    .holo-card::before {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        transform: rotate(45deg);
        animation: hologram-sweep 4s infinite;
    }}
    
    @keyframes hologram-sweep {{
        0% {{ transform: translateX(-100%) translateY(-100%) rotate(45deg); }}
        100% {{ transform: translateX(100%) translateY(100%) rotate(45deg); }}
    }}
    
    .holo-card:hover {{
        transform: translateY(-10px) scale(1.02);
        border-color: {COLORS['electric_blue']};
        box-shadow: 
            0 10px 40px rgba(0,212,255,0.4),
            inset 0 0 30px rgba(0,212,255,0.2);
    }}
    
    .service-online {{
        color: {COLORS['neon_green']};
        text-shadow: 0 0 10px {COLORS['neon_green']};
    }}
    
    .service-dev {{
        color: {COLORS['warning_orange']};
        text-shadow: 0 0 10px {COLORS['warning_orange']};
    }}
    
    .metric-display {{
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(45deg, {COLORS['hot_pink']}, {COLORS['electric_blue']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
    }}
    
    .cyber-grid {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -1;
        opacity: 0.1;
        background-image: 
            linear-gradient({COLORS['hot_pink']} 1px, transparent 1px),
            linear-gradient(90deg, {COLORS['electric_blue']} 1px, transparent 1px);
        background-size: 50px 50px;
        animation: grid-move 20s linear infinite;
    }}
    
    @keyframes grid-move {{
        0% {{ transform: translate(0, 0); }}
        100% {{ transform: translate(50px, 50px); }}
    }}
    
    .pulse-button {{
        background: linear-gradient(45deg, {COLORS['hot_pink']}, {COLORS['electric_blue']});
        border: none;
        color: {COLORS['deep_black']};
        padding: 1rem 2rem;
        border-radius: 30px;
        font-weight: bold;
        font-size: 1.2rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 0 20px rgba(255,20,147,0.5);
        animation: pulse-glow 2s infinite;
    }}
    
    @keyframes pulse-glow {{
        0%, 100% {{ box-shadow: 0 0 20px rgba(255,20,147,0.5); }}
        50% {{ box-shadow: 0 0 30px rgba(255,20,147,0.8), 0 0 40px rgba(0,212,255,0.5); }}
    }}
    
    .status-bar {{
        background: linear-gradient(90deg, {COLORS['deep_black']}, {COLORS['dark_gray']});
        border: 1px solid {COLORS['hot_pink']};
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }}
</style>
""", unsafe_allow_html=True)

# Add animated grid background
st.markdown('<div class="cyber-grid"></div>', unsafe_allow_html=True)

# Header
st.markdown('<h1 class="neon-title">LIFT OS: NEON GRID</h1>', unsafe_allow_html=True)

# Create cyberpunk visualization
def create_neon_network():
    fig = go.Figure()
    
    # Create network nodes
    nodes = {
        "CORE": {"x": 0, "y": 0, "color": COLORS['white'], "size": 40},
        "MEMORY": {"x": 2, "y": 0, "color": COLORS['white'], "size": 30},
        "SURFACING": {"x": -2, "y": 2, "color": COLORS['hot_pink'], "size": 25},
        "CAUSAL": {"x": 2, "y": 2, "color": COLORS['electric_blue'], "size": 25},
        "AGENTIC": {"x": -2, "y": -2, "color": COLORS['warning_orange'], "size": 20},
        "LLM": {"x": 2, "y": -2, "color": COLORS['cyber_purple'], "size": 20},
        "SENTIMENT": {"x": 0, "y": 3, "color": COLORS['hot_pink'], "size": 20},
        "EVAL": {"x": 0, "y": -3, "color": COLORS['electric_blue'], "size": 20}
    }
    
    # Add connections
    connections = [
        ("CORE", "MEMORY"), ("CORE", "SURFACING"), ("CORE", "CAUSAL"),
        ("CORE", "AGENTIC"), ("CORE", "LLM"), ("MEMORY", "SURFACING"),
        ("MEMORY", "CAUSAL"), ("SURFACING", "SENTIMENT"), ("CAUSAL", "EVAL")
    ]
    
    for start, end in connections:
        fig.add_trace(go.Scatter(
            x=[nodes[start]["x"], nodes[end]["x"]],
            y=[nodes[start]["y"], nodes[end]["y"]],
            mode='lines',
            line=dict(color=COLORS['neon_green'], width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add nodes
    for name, props in nodes.items():
        fig.add_trace(go.Scatter(
            x=[props["x"]],
            y=[props["y"]],
            mode='markers+text',
            marker=dict(
                size=props["size"],
                color=props["color"],
                line=dict(width=3, color=COLORS['neon_green'])
            ),
            text=name,
            textposition="middle center",
            textfont=dict(color=COLORS['deep_black'], size=8, family="Orbitron"),
            showlegend=False,
            hovertemplate=f"<b>{name}</b><br>Network Node<extra></extra>"
        ))
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
        title=dict(
            text="NEURAL NETWORK TOPOLOGY",
            x=0.5,
            font=dict(color=COLORS['hot_pink'], size=18, family="Orbitron")
        )
    )
    
    return fig

# Main dashboard layout
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.markdown("### 🎮 CORE SYSTEMS")
    
    st.markdown('<div class="holo-card">', unsafe_allow_html=True)
    st.markdown("**⚪ CORE**")
    st.markdown('<span class="service-online">● ACTIVE</span>', unsafe_allow_html=True)
    st.markdown('<div class="metric-display">100%</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="holo-card">', unsafe_allow_html=True)
    st.markdown("**⚪ MEMORY**")
    st.markdown('<span class="service-online">● ACTIVE</span>', unsafe_allow_html=True)
    st.markdown('<div class="metric-display">2.4TB</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.plotly_chart(create_neon_network(), use_container_width=True)
    
    # Action buttons
    st.markdown("### 🚀 MISSION CONTROL")
    btn_cols = st.columns(4)
    
    with btn_cols[0]:
        if st.button("🔥 IGNITE", key="ignite"):
            st.success("Systems ignited!")
    
    with btn_cols[1]:
        if st.button("⚡ CHARGE", key="charge"):
            st.info("Charging capacitors...")
    
    with btn_cols[2]:
        if st.button("🌟 BOOST", key="boost"):
            st.warning("Boosting power!")
    
    with btn_cols[3]:
        if st.button("🎯 LOCK", key="lock"):
            st.error("Target locked!")

with col3:
    st.markdown("### 🔮 MODULES")
    
    modules = [
        {"name": "SURFACING", "color": COLORS['hot_pink'], "status": "ONLINE", "value": "847K"},
        {"name": "CAUSAL", "color": COLORS['electric_blue'], "status": "ONLINE", "value": "1.2M"},
        {"name": "AGENTIC", "color": COLORS['warning_orange'], "status": "DEV", "value": "23"},
        {"name": "LLM", "color": COLORS['cyber_purple'], "status": "DEV", "value": "GPT-4"},
        {"name": "SENTIMENT", "color": COLORS['hot_pink'], "status": "DEV", "value": "89%"},
        {"name": "EVAL", "color": COLORS['electric_blue'], "status": "DEV", "value": "156"}
    ]
    
    for module in modules:
        st.markdown('<div class="holo-card">', unsafe_allow_html=True)
        st.markdown(f"**{module['name']}**")
        status_class = "service-online" if module['status'] == 'ONLINE' else "service-dev"
        st.markdown(f'<span class="{status_class}">● {module["status"]}</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-display" style="font-size: 1.5rem;">{module["value"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Bottom status bar
st.markdown("---")
st.markdown("### 📊 REAL-TIME METRICS")

status_cols = st.columns(6)
metrics = [
    ("CPU", f"{random.randint(45, 85)}%"),
    ("MEMORY", f"{random.randint(60, 90)}%"),
    ("NETWORK", f"{random.randint(100, 999)} Mbps"),
    ("STORAGE", f"{random.randint(40, 80)}%"),
    ("TEMP", f"{random.randint(45, 75)}°C"),
    ("POWER", f"{random.randint(300, 500)}W")
]

for i, (label, value) in enumerate(metrics):
    with status_cols[i]:
        st.markdown('<div class="status-bar">', unsafe_allow_html=True)
        st.markdown(f"**{label}**")
        st.markdown(f'<div style="color: {COLORS["neon_green"]}; font-size: 1.2rem;">{value}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    f'<div style="text-align: center; font-size: 1.2rem; background: linear-gradient(45deg, {COLORS["hot_pink"]}, {COLORS["electric_blue"]}); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">'
    f'🔮 LIFT OS - NEURAL NETWORK FOR GROWTH INTELLIGENCE 🔮'
    f'</div>',
    unsafe_allow_html=True
) 