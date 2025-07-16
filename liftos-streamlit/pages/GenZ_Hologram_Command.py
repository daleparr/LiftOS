import streamlit as st
import plotly.graph_objects as go
import numpy as np
import random
import time
from datetime import datetime

# LiftOS Hologram Colors
COLORS = {
    "hot_pink": "#FF1493",
    "electric_blue": "#00D4FF",
    "neon_green": "#00FF41",
    "cyber_purple": "#8A2BE2",
    "warning_orange": "#FF8C00",
    "deep_black": "#0A0A0A",
    "dark_navy": "#0D1B2A",
    "white": "#FFFFFF"
}

st.set_page_config(page_title="LiftOS: Hologram Command", page_icon="🎮", layout="wide")

# Holographic Command Center CSS
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    
    .stApp {{
        background: 
            radial-gradient(ellipse at top, {COLORS['deep_black']} 0%, {COLORS['dark_navy']} 100%),
            radial-gradient(ellipse at bottom, {COLORS['cyber_purple']}22 0%, transparent 70%);
        color: {COLORS['white']};
        font-family: 'Orbitron', monospace;
        overflow-x: hidden;
    }}
    
    .command-title {{
        text-align: center;
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(45deg, {COLORS['hot_pink']}, {COLORS['electric_blue']}, {COLORS['neon_green']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 2rem;
        position: relative;
    }}
    
    .command-title::after {{
        content: '';
        position: absolute;
        bottom: -10px;
        left: 50%;
        transform: translateX(-50%);
        width: 200px;
        height: 2px;
        background: linear-gradient(90deg, transparent, {COLORS['hot_pink']}, transparent);
        animation: line-glow 2s infinite;
    }}
    
    @keyframes line-glow {{
        0%, 100% {{ box-shadow: 0 0 5px {COLORS['hot_pink']}; }}
        50% {{ box-shadow: 0 0 20px {COLORS['hot_pink']}, 0 0 30px {COLORS['electric_blue']}; }}
    }}
    
    .holo-panel {{
        background: linear-gradient(135deg, 
            rgba(255,20,147,0.15) 0%, 
            rgba(0,212,255,0.15) 50%, 
            rgba(0,255,65,0.15) 100%);
        border: 2px solid transparent;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        position: relative;
        backdrop-filter: blur(15px);
        box-shadow: 
            0 8px 32px rgba(0,0,0,0.4),
            inset 0 0 20px rgba(255,255,255,0.1);
        transition: all 0.4s ease;
        transform: perspective(1000px) rotateX(5deg);
    }}
    
    .holo-panel::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        border-radius: 20px;
        padding: 2px;
        background: linear-gradient(45deg, {COLORS['hot_pink']}, {COLORS['electric_blue']}, {COLORS['neon_green']});
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: exclude;
        mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        mask-composite: exclude;
    }}
    
    .holo-panel:hover {{
        transform: perspective(1000px) rotateX(0deg) translateY(-10px);
        box-shadow: 
            0 20px 40px rgba(0,0,0,0.6),
            inset 0 0 30px rgba(255,255,255,0.2);
    }}
    
    .floating-card {{
        background: linear-gradient(135deg, 
            rgba(255,20,147,0.2) 0%, 
            rgba(0,212,255,0.2) 100%);
        border: 1px solid {COLORS['hot_pink']};
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(255,20,147,0.3);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }}
    
    .floating-card::after {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        transform: rotate(45deg);
        animation: scan-line 3s infinite;
    }}
    
    @keyframes scan-line {{
        0% {{ transform: translateX(-100%) translateY(-100%) rotate(45deg); }}
        100% {{ transform: translateX(100%) translateY(100%) rotate(45deg); }}
    }}
    
    .floating-card:hover {{
        transform: translateY(-5px) scale(1.02);
        border-color: {COLORS['electric_blue']};
        box-shadow: 0 8px 30px rgba(0,212,255,0.4);
    }}
    
    .status-active {{
        color: {COLORS['neon_green']};
        text-shadow: 0 0 10px {COLORS['neon_green']};
        font-weight: bold;
    }}
    
    .status-dev {{
        color: {COLORS['warning_orange']};
        text-shadow: 0 0 10px {COLORS['warning_orange']};
        font-weight: bold;
    }}
    
    .metric-hologram {{
        font-size: 2.5rem;
        font-weight: 900;
        background: linear-gradient(45deg, {COLORS['hot_pink']}, {COLORS['electric_blue']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        text-shadow: 0 0 20px rgba(255,20,147,0.5);
    }}
    
    .command-button {{
        background: linear-gradient(45deg, {COLORS['hot_pink']}, {COLORS['electric_blue']});
        border: none;
        color: {COLORS['deep_black']};
        padding: 1rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(255,20,147,0.4);
        position: relative;
        overflow: hidden;
    }}
    
    .command-button::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }}
    
    .command-button:hover::before {{
        left: 100%;
    }}
    
    .command-button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0,212,255,0.6);
    }}
    
    .system-monitor {{
        background: linear-gradient(135deg, {COLORS['deep_black']}, {COLORS['dark_navy']});
        border: 1px solid {COLORS['cyber_purple']};
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
    }}
    
    .scan-lines {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 1000;
        background: linear-gradient(
            transparent 50%,
            rgba(0,255,65,0.03) 50%,
            rgba(0,255,65,0.03) 51%,
            transparent 51%
        );
        background-size: 100% 4px;
        animation: scan 0.1s linear infinite;
    }}
    
    @keyframes scan {{
        0% {{ background-position: 0 0; }}
        100% {{ background-position: 0 4px; }}
    }}
</style>
""", unsafe_allow_html=True)

# Add scan lines effect
st.markdown('<div class="scan-lines"></div>', unsafe_allow_html=True)

# Header
st.markdown('<h1 class="command-title">LIFT OS: HOLOGRAM COMMAND</h1>', unsafe_allow_html=True)

# Create 3D holographic visualization
def create_hologram_display():
    fig = go.Figure()
    
    # Create 3D-style service visualization
    services = [
        {"name": "CORE", "x": 0, "y": 0, "z": 0, "color": COLORS['white'], "size": 50},
        {"name": "MEMORY", "x": 1, "y": 0, "z": 1, "color": COLORS['white'], "size": 40},
        {"name": "SURFACING", "x": -1, "y": 1, "z": 0, "color": COLORS['hot_pink'], "size": 30},
        {"name": "CAUSAL", "x": 1, "y": 1, "z": 0, "color": COLORS['electric_blue'], "size": 30},
        {"name": "AGENTIC", "x": -1, "y": -1, "z": 1, "color": COLORS['warning_orange'], "size": 25},
        {"name": "LLM", "x": 1, "y": -1, "z": 1, "color": COLORS['cyber_purple'], "size": 25},
        {"name": "SENTIMENT", "x": 0, "y": 2, "z": 0, "color": COLORS['hot_pink'], "size": 25},
        {"name": "EVAL", "x": 0, "y": -2, "z": 0, "color": COLORS['electric_blue'], "size": 25}
    ]
    
    # Add 3D scatter plot
    for service in services:
        fig.add_trace(go.Scatter3d(
            x=[service["x"]],
            y=[service["y"]],
            z=[service["z"]],
            mode='markers+text',
            marker=dict(
                size=service["size"],
                color=service["color"],
                opacity=0.8,
                line=dict(width=2, color=COLORS['neon_green'])
            ),
            text=service["name"],
            textposition="middle center",
            textfont=dict(color=COLORS['deep_black'], size=10),
            showlegend=False,
            hovertemplate=f"<b>{service['name']}</b><br>Hologram Node<extra></extra>"
        ))
    
    # Add connecting lines
    connections = [
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3), (2, 6), (3, 7)
    ]
    
    for start, end in connections:
        fig.add_trace(go.Scatter3d(
            x=[services[start]["x"], services[end]["x"]],
            y=[services[start]["y"], services[end]["y"]],
            z=[services[start]["z"], services[end]["z"]],
            mode='lines',
            line=dict(color=COLORS['neon_green'], width=3),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            bgcolor='rgba(0,0,0,0)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=600,
        title=dict(
            text="HOLOGRAPHIC SYSTEM ARCHITECTURE",
            x=0.5,
            font=dict(color=COLORS['hot_pink'], size=18, family="Orbitron")
        )
    )
    
    return fig

# Main command center layout
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.markdown('<div class="holo-panel">', unsafe_allow_html=True)
    st.markdown("### 🎮 CORE MATRIX")
    
    st.markdown('<div class="floating-card">', unsafe_allow_html=True)
    st.markdown("**⚪ CORE SYSTEM**")
    st.markdown('<span class="status-active">● ACTIVE</span>', unsafe_allow_html=True)
    st.markdown('<div class="metric-hologram">100%</div>', unsafe_allow_html=True)
    st.markdown("System Health")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="floating-card">', unsafe_allow_html=True)
    st.markdown("**⚪ MEMORY CORE**")
    st.markdown('<span class="status-active">● ACTIVE</span>', unsafe_allow_html=True)
    st.markdown('<div class="metric-hologram">3.2TB</div>', unsafe_allow_html=True)
    st.markdown("Data Stored")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.plotly_chart(create_hologram_display(), use_container_width=True)
    
    # Command controls
    st.markdown("### 🚀 COMMAND INTERFACE")
    ctrl_cols = st.columns(4)
    
    with ctrl_cols[0]:
        if st.button("⚡ INITIALIZE", key="init"):
            st.success("System initialized!")
    
    with ctrl_cols[1]:
        if st.button("🔮 SCAN", key="scan"):
            st.info("Deep scan initiated...")
    
    with ctrl_cols[2]:
        if st.button("🎯 TARGET", key="target"):
            st.warning("Target acquired!")
    
    with ctrl_cols[3]:
        if st.button("🚀 EXECUTE", key="execute"):
            st.error("Command executed!")

with col3:
    st.markdown('<div class="holo-panel">', unsafe_allow_html=True)
    st.markdown("### 🔮 MODULE STATUS")
    
    modules = [
        {"name": "SURFACING", "status": "ACTIVE", "value": "1.8M", "unit": "Insights"},
        {"name": "CAUSAL", "status": "ACTIVE", "value": "923", "unit": "Models"},
        {"name": "AGENTIC", "status": "DEV", "value": "47", "unit": "Agents"},
        {"name": "LLM", "status": "DEV", "value": "GPT-4", "unit": "Engine"},
        {"name": "SENTIMENT", "status": "DEV", "value": "91%", "unit": "Accuracy"},
        {"name": "EVAL", "status": "DEV", "value": "203", "unit": "Tests"}
    ]
    
    for module in modules:
        st.markdown('<div class="floating-card">', unsafe_allow_html=True)
        st.markdown(f"**{module['name']}**")
        status_class = "status-active" if module['status'] == 'ACTIVE' else "status-dev"
        st.markdown(f'<span class="{status_class}">● {module["status"]}</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-hologram" style="font-size: 1.5rem;">{module["value"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size: 0.8rem; color: {COLORS["cyber_purple"]};">{module["unit"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# System monitoring panel
st.markdown("---")
st.markdown("### 📊 SYSTEM TELEMETRY")

monitor_cols = st.columns(4)
telemetry = [
    ("NEURAL LOAD", f"{random.randint(35, 75)}%", COLORS['neon_green']),
    ("QUANTUM FLUX", f"{random.randint(200, 800)} QPS", COLORS['electric_blue']),
    ("HOLOGRAM SYNC", f"{random.randint(85, 99)}%", COLORS['hot_pink']),
    ("POWER MATRIX", f"{random.randint(400, 650)}W", COLORS['warning_orange'])
]

for i, (label, value, color) in enumerate(telemetry):
    with monitor_cols[i]:
        st.markdown('<div class="system-monitor">', unsafe_allow_html=True)
        st.markdown(f"**{label}**")
        st.markdown(f'<div style="color: {color}; font-size: 1.5rem; font-weight: bold;">{value}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    f'<div style="text-align: center; font-size: 1.3rem; background: linear-gradient(45deg, {COLORS["hot_pink"]}, {COLORS["electric_blue"]}, {COLORS["neon_green"]}); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">'
    f'🎮 LIFT OS - HOLOGRAPHIC COMMAND CENTER 🎮'
    f'</div>',
    unsafe_allow_html=True
) 