import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import random
from datetime import datetime

# LiftOS Branding Colors (from the images)
COLORS = {
    "neon_green": "#00FF41",
    "electric_blue": "#00D4FF", 
    "hot_pink": "#FF1493",
    "cyber_purple": "#8A2BE2",
    "warning_orange": "#FF8C00",
    "deep_black": "#0A0A0A",
    "white": "#FFFFFF",
    "gray": "#808080"
}

st.set_page_config(page_title="LiftOS: Orbit of Truth", page_icon="🌌", layout="wide")

# Custom CSS for Orbit of Truth theme
st.markdown(f"""
<style>
    .stApp {{
        background: linear-gradient(135deg, #0A0A0A 0%, #1A1A2E 50%, #0A0A0A 100%);
        color: {COLORS['white']};
    }}
    
    .orbit-header {{
        text-align: center;
        font-family: 'Orbitron', monospace;
        font-size: 3rem;
        font-weight: 900;
        color: {COLORS['neon_green']};
        text-shadow: 0 0 20px {COLORS['neon_green']};
        margin-bottom: 2rem;
        animation: pulse 2s infinite;
    }}
    
    @keyframes pulse {{
        0% {{ text-shadow: 0 0 20px {COLORS['neon_green']}; }}
        50% {{ text-shadow: 0 0 30px {COLORS['neon_green']}, 0 0 40px {COLORS['neon_green']}; }}
        100% {{ text-shadow: 0 0 20px {COLORS['neon_green']}; }}
    }}
    
    .microservice-card {{
        background: linear-gradient(135deg, rgba(0,255,65,0.1) 0%, rgba(0,0,0,0.8) 100%);
        border: 2px solid {COLORS['neon_green']};
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 0 20px rgba(0,255,65,0.3);
        transition: all 0.3s ease;
    }}
    
    .microservice-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0,255,65,0.5);
        border-color: {COLORS['electric_blue']};
    }}
    
    .core-service {{
        background: linear-gradient(135deg, rgba(255,255,255,0.2) 0%, rgba(0,0,0,0.8) 100%);
        border-color: {COLORS['white']};
        box-shadow: 0 0 20px rgba(255,255,255,0.3);
    }}
    
    .metric-value {{
        font-size: 2.5rem;
        font-weight: bold;
        color: {COLORS['neon_green']};
        text-shadow: 0 0 10px {COLORS['neon_green']};
    }}
    
    .status-online {{
        color: {COLORS['neon_green']};
        font-weight: bold;
    }}
    
    .status-dev {{
        color: {COLORS['warning_orange']};
        font-weight: bold;
    }}
    
    .cyber-button {{
        background: linear-gradient(45deg, {COLORS['neon_green']} 0%, {COLORS['electric_blue']} 100%);
        border: none;
        color: {COLORS['deep_black']};
        padding: 0.7rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,255,65,0.4);
    }}
    
    .cyber-button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,255,65,0.6);
    }}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="orbit-header">LIFT OS: ORBIT OF TRUTH</h1>', unsafe_allow_html=True)

# Create the orbital visualization
def create_orbit_chart():
    fig = go.Figure()
    
    # Create concentric circles (orbits)
    theta = np.linspace(0, 2*np.pi, 100)
    orbits = [2, 4, 6, 8, 10]
    
    for i, radius in enumerate(orbits):
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            line=dict(color=COLORS['neon_green'], width=2, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Microservices positions
    services = [
        {"name": "CORE", "color": COLORS['white'], "radius": 0, "angle": 0},
        {"name": "MEMORY", "color": COLORS['white'], "radius": 2, "angle": 0},
        {"name": "SURFACING", "color": COLORS['hot_pink'], "radius": 4, "angle": 45},
        {"name": "CAUSAL", "color": COLORS['electric_blue'], "radius": 6, "angle": 90},
        {"name": "AGENTIC", "color": COLORS['warning_orange'], "radius": 8, "angle": 135},
        {"name": "LLM", "color": COLORS['gray'], "radius": 6, "angle": 180},
        {"name": "SENTIMENT", "color": COLORS['cyber_purple'], "radius": 10, "angle": 225},
        {"name": "EVAL", "color": COLORS['electric_blue'], "radius": 10, "angle": 270},
    ]
    
    for service in services:
        if service['radius'] == 0:  # CORE at center
            x, y = 0, 0
        else:
            angle_rad = np.radians(service['angle'])
            x = service['radius'] * np.cos(angle_rad)
            y = service['radius'] * np.sin(angle_rad)
        
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(
                size=30 if service['name'] == 'CORE' else 20,
                color=service['color'],
                line=dict(width=3, color=COLORS['neon_green'])
            ),
            text=service['name'],
            textposition="middle center",
            textfont=dict(color=COLORS['deep_black'], size=10, family="Orbitron"),
            showlegend=False,
            hovertemplate=f"<b>{service['name']}</b><br>Status: Active<extra></extra>"
        ))
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
        title=dict(
            text="MICROSERVICES CONSTELLATION",
            x=0.5,
            font=dict(color=COLORS['neon_green'], size=20, family="Orbitron")
        )
    )
    
    return fig

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.plotly_chart(create_orbit_chart(), use_container_width=True)

with col2:
    st.markdown("### 🎯 MISSION CONTROL")
    
    # Core Services (Always On)
    st.markdown('<div class="microservice-card core-service">', unsafe_allow_html=True)
    st.markdown("**⚪ CORE + MEMORY**")
    st.markdown('<span class="status-online">● ONLINE</span>', unsafe_allow_html=True)
    st.markdown('<div class="metric-value">99.9%</div>', unsafe_allow_html=True)
    st.markdown("Uptime")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Modular Services
    services_data = [
        {"name": "🔴 SURFACING", "status": "ONLINE", "metric": "1.2M", "label": "Insights"},
        {"name": "🔵 CAUSAL", "status": "ONLINE", "metric": "847", "label": "Models"},
        {"name": "🟡 AGENTIC", "status": "DEV", "metric": "23", "label": "Agents"},
        {"name": "⚫ LLM", "status": "DEV", "metric": "GPT-4", "label": "Engine"},
        {"name": "🟣 SENTIMENT", "status": "DEV", "metric": "89%", "label": "Accuracy"},
        {"name": "🔵 EVAL", "status": "DEV", "metric": "156", "label": "Tests"}
    ]
    
    for service in services_data:
        st.markdown('<div class="microservice-card">', unsafe_allow_html=True)
        st.markdown(f"**{service['name']}**")
        status_class = "status-online" if service['status'] == 'ONLINE' else "status-dev"
        st.markdown(f'<span class="{status_class}">● {service["status"]}</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{service["metric"]}</div>', unsafe_allow_html=True)
        st.markdown(service['label'])
        st.markdown('</div>', unsafe_allow_html=True)

# Bottom action bar
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🚀 LAUNCH MISSION", key="launch"):
        st.success("Mission Control Activated!")

with col2:
    if st.button("🔍 DEEP SCAN", key="scan"):
        st.info("Scanning orbit for anomalies...")

with col3:
    if st.button("⚡ BOOST POWER", key="boost"):
        st.warning("Power levels increasing...")

with col4:
    if st.button("🛸 WARP SPEED", key="warp"):
        st.error("Engaging hyperdrive!")

# Real-time stats ticker
st.markdown("### 📊 REAL-TIME INTELLIGENCE")
cols = st.columns(5)
metrics = [
    ("Active Campaigns", random.randint(12, 18)),
    ("Data Points", f"{random.randint(800, 1200)}K"),
    ("Models Running", random.randint(5, 12)),
    ("Predictions/sec", random.randint(450, 650)),
    ("Accuracy", f"{random.randint(92, 98)}%")
]

for i, (label, value) in enumerate(metrics):
    with cols[i]:
        st.metric(label, value, delta=f"+{random.randint(1, 10)}")

# Footer
st.markdown("---")
st.markdown(
    f'<div style="text-align: center; color: {COLORS["neon_green"]}; font-family: Orbitron;">'
    f'🌌 LIFT OS - The Platform for Causal Growth Intelligence 🌌'
    f'</div>',
    unsafe_allow_html=True
) 