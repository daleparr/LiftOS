import streamlit as st
import plotly.graph_objects as go
import numpy as np
import random
import time
from datetime import datetime

# LiftOS Gradient Wave Colors
COLORS = {
    "hot_pink": "#FF1493",
    "electric_blue": "#00D4FF",
    "neon_green": "#00FF41",
    "cyber_purple": "#8A2BE2",
    "warning_orange": "#FF8C00",
    "deep_black": "#0A0A0A",
    "dark_purple": "#2D1B69",
    "white": "#FFFFFF"
}

st.set_page_config(page_title="LiftOS: Gradient Wave", page_icon="🌊", layout="wide")

# Gradient Wave CSS
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    
    .stApp {{
        background: 
            linear-gradient(45deg, {COLORS['deep_black']} 0%, {COLORS['dark_purple']} 50%, {COLORS['deep_black']} 100%),
            radial-gradient(circle at 30% 30%, {COLORS['hot_pink']}33 0%, transparent 50%),
            radial-gradient(circle at 70% 70%, {COLORS['electric_blue']}33 0%, transparent 50%);
        color: {COLORS['white']};
        font-family: 'Orbitron', monospace;
        overflow-x: hidden;
        position: relative;
    }}
    
    .wave-title {{
        text-align: center;
        font-size: 4rem;
        font-weight: 900;
        background: linear-gradient(45deg, {COLORS['hot_pink']}, {COLORS['electric_blue']}, {COLORS['neon_green']}, {COLORS['cyber_purple']});
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradient-wave 4s ease infinite;
        margin-bottom: 2rem;
    }}
    
    @keyframes gradient-wave {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    
    .wave-background {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -1;
        background: 
            linear-gradient(45deg, transparent 40%, {COLORS['hot_pink']}11 50%, transparent 60%),
            linear-gradient(-45deg, transparent 40%, {COLORS['electric_blue']}11 50%, transparent 60%);
        background-size: 100px 100px;
        animation: wave-move 10s linear infinite;
    }}
    
    @keyframes wave-move {{
        0% {{ transform: translateX(0) translateY(0); }}
        100% {{ transform: translateX(100px) translateY(100px); }}
    }}
    
    .fluid-card {{
        background: linear-gradient(135deg, 
            rgba(255,20,147,0.2) 0%, 
            rgba(0,212,255,0.2) 25%, 
            rgba(0,255,65,0.2) 50%, 
            rgba(138,43,226,0.2) 75%, 
            rgba(255,140,0,0.2) 100%);
        background-size: 300% 300%;
        border: 2px solid transparent;
        border-radius: 25px;
        padding: 2rem;
        margin: 1rem 0;
        backdrop-filter: blur(20px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        transition: all 0.5s ease;
        position: relative;
        overflow: hidden;
        animation: card-wave 6s ease-in-out infinite;
    }}
    
    @keyframes card-wave {{
        0%, 100% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
    }}
    
    .fluid-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        border-radius: 25px;
        padding: 2px;
        background: linear-gradient(45deg, 
            {COLORS['hot_pink']}, 
            {COLORS['electric_blue']}, 
            {COLORS['neon_green']}, 
            {COLORS['cyber_purple']}, 
            {COLORS['warning_orange']});
        background-size: 400% 400%;
        animation: border-wave 3s ease infinite;
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: exclude;
        mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        mask-composite: exclude;
    }}
    
    @keyframes border-wave {{
        0%, 100% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
    }}
    
    .fluid-card:hover {{
        transform: translateY(-15px) scale(1.03);
        box-shadow: 0 15px 50px rgba(255,20,147,0.4);
    }}
    
    .wave-metric {{
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(45deg, {COLORS['hot_pink']}, {COLORS['electric_blue']}, {COLORS['neon_green']});
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: metric-wave 2s ease infinite;
        text-align: center;
    }}
    
    @keyframes metric-wave {{
        0%, 100% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
    }}
    
    .status-flowing {{
        background: linear-gradient(45deg, {COLORS['neon_green']}, {COLORS['electric_blue']});
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: status-flow 3s ease infinite;
        font-weight: bold;
    }}
    
    .status-dev-flow {{
        background: linear-gradient(45deg, {COLORS['warning_orange']}, {COLORS['hot_pink']});
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: status-flow 3s ease infinite;
        font-weight: bold;
    }}
    
    @keyframes status-flow {{
        0%, 100% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
    }}
    
    .wave-button {{
        background: linear-gradient(45deg, 
            {COLORS['hot_pink']}, 
            {COLORS['electric_blue']}, 
            {COLORS['neon_green']}, 
            {COLORS['cyber_purple']});
        background-size: 300% 300%;
        border: none;
        color: {COLORS['deep_black']};
        padding: 1rem 2rem;
        border-radius: 30px;
        font-weight: bold;
        font-size: 1.2rem;
        cursor: pointer;
        transition: all 0.3s ease;
        animation: button-wave 4s ease infinite;
        box-shadow: 0 4px 20px rgba(255,20,147,0.3);
    }}
    
    @keyframes button-wave {{
        0%, 100% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
    }}
    
    .wave-button:hover {{
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 8px 30px rgba(0,212,255,0.5);
    }}
    
    .flow-monitor {{
        background: linear-gradient(135deg, 
            rgba(255,20,147,0.1) 0%, 
            rgba(0,212,255,0.1) 100%);
        border: 1px solid {COLORS['electric_blue']};
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }}
    
    .flow-monitor:hover {{
        border-color: {COLORS['hot_pink']};
        box-shadow: 0 4px 20px rgba(255,20,147,0.3);
    }}
    
    .flowing-particles {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 1;
    }}
    
    .particle {{
        position: absolute;
        width: 4px;
        height: 4px;
        background: {COLORS['neon_green']};
        border-radius: 50%;
        animation: float 8s linear infinite;
    }}
    
    @keyframes float {{
        0% {{ transform: translateY(100vh) translateX(0); opacity: 0; }}
        10% {{ opacity: 1; }}
        90% {{ opacity: 1; }}
        100% {{ transform: translateY(-100vh) translateX(100px); opacity: 0; }}
    }}
</style>
""", unsafe_allow_html=True)

# Add wave background and particles
st.markdown('<div class="wave-background"></div>', unsafe_allow_html=True)
st.markdown('''
<div class="flowing-particles">
    <div class="particle" style="left: 10%; animation-delay: 0s;"></div>
    <div class="particle" style="left: 20%; animation-delay: 1s;"></div>
    <div class="particle" style="left: 30%; animation-delay: 2s;"></div>
    <div class="particle" style="left: 40%; animation-delay: 3s;"></div>
    <div class="particle" style="left: 50%; animation-delay: 4s;"></div>
    <div class="particle" style="left: 60%; animation-delay: 5s;"></div>
    <div class="particle" style="left: 70%; animation-delay: 6s;"></div>
    <div class="particle" style="left: 80%; animation-delay: 7s;"></div>
    <div class="particle" style="left: 90%; animation-delay: 8s;"></div>
</div>
''', unsafe_allow_html=True)

# Header
st.markdown('<h1 class="wave-title">LIFT OS: GRADIENT WAVE</h1>', unsafe_allow_html=True)

# Create flowing wave visualization
def create_wave_chart():
    fig = go.Figure()
    
    # Create wave surface
    x = np.linspace(0, 4*np.pi, 100)
    y = np.linspace(0, 2*np.pi, 50)
    X, Y = np.meshgrid(x, y)
    
    # Multiple wave layers
    Z1 = np.sin(X) * np.cos(Y) * 0.5
    Z2 = np.sin(X*2) * np.cos(Y*2) * 0.3
    Z3 = np.sin(X*0.5) * np.cos(Y*0.5) * 0.2
    Z = Z1 + Z2 + Z3
    
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale=[[0, COLORS['deep_black']], 
                   [0.2, COLORS['hot_pink']], 
                   [0.4, COLORS['electric_blue']], 
                   [0.6, COLORS['neon_green']], 
                   [0.8, COLORS['cyber_purple']], 
                   [1, COLORS['warning_orange']]],
        opacity=0.8,
        showscale=False
    ))
    
    # Add service nodes floating on the wave
    services = [
        {"name": "CORE", "x": 2, "y": 1, "z": 1, "color": COLORS['white']},
        {"name": "MEMORY", "x": 3, "y": 1, "z": 1, "color": COLORS['white']},
        {"name": "SURFACING", "x": 1, "y": 2, "z": 1.5, "color": COLORS['hot_pink']},
        {"name": "CAUSAL", "x": 4, "y": 2, "z": 1.5, "color": COLORS['electric_blue']},
        {"name": "AGENTIC", "x": 1, "y": 0.5, "z": 1.2, "color": COLORS['warning_orange']},
        {"name": "LLM", "x": 4, "y": 0.5, "z": 1.2, "color": COLORS['cyber_purple']}
    ]
    
    for service in services:
        fig.add_trace(go.Scatter3d(
            x=[service["x"]],
            y=[service["y"]],
            z=[service["z"]],
            mode='markers+text',
            marker=dict(
                size=15,
                color=service["color"],
                line=dict(width=2, color=COLORS['neon_green'])
            ),
            text=service["name"],
            textposition="middle center",
            textfont=dict(color=COLORS['deep_black'], size=8),
            showlegend=False
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
        height=500,
        title=dict(
            text="FLOWING DATA WAVES",
            x=0.5,
            font=dict(color=COLORS['hot_pink'], size=18, family="Orbitron")
        )
    )
    
    return fig

# Main wave dashboard layout
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.markdown('<div class="fluid-card">', unsafe_allow_html=True)
    st.markdown("### 🌊 CORE FLOW")
    
    st.markdown("**⚪ CORE SYSTEM**")
    st.markdown('<span class="status-flowing">● FLOWING</span>', unsafe_allow_html=True)
    st.markdown('<div class="wave-metric">100%</div>', unsafe_allow_html=True)
    st.markdown("Flow Rate")
    
    st.markdown("**⚪ MEMORY STREAM**")
    st.markdown('<span class="status-flowing">● FLOWING</span>', unsafe_allow_html=True)
    st.markdown('<div class="wave-metric">4.1TB</div>', unsafe_allow_html=True)
    st.markdown("Data Stream")
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.plotly_chart(create_wave_chart(), use_container_width=True)
    
    # Wave control buttons
    st.markdown("### 🌊 WAVE CONTROLS")
    wave_cols = st.columns(4)
    
    with wave_cols[0]:
        if st.button("🌊 FLOW", key="flow"):
            st.success("Wave flow activated!")
    
    with wave_cols[1]:
        if st.button("⚡ SURGE", key="surge"):
            st.info("Power surge initiated!")
    
    with wave_cols[2]:
        if st.button("🔄 CYCLE", key="cycle"):
            st.warning("Cycle mode engaged!")
    
    with wave_cols[3]:
        if st.button("🎯 FOCUS", key="focus"):
            st.error("Focus beam locked!")

with col3:
    st.markdown('<div class="fluid-card">', unsafe_allow_html=True)
    st.markdown("### 🔮 WAVE MODULES")
    
    modules = [
        {"name": "SURFACING", "status": "FLOWING", "value": "2.1M", "unit": "Waves"},
        {"name": "CAUSAL", "status": "FLOWING", "value": "1.4K", "unit": "Ripples"},
        {"name": "AGENTIC", "status": "DEV", "value": "89", "unit": "Agents"},
        {"name": "LLM", "status": "DEV", "value": "GPT-4", "unit": "Neural"},
        {"name": "SENTIMENT", "status": "DEV", "value": "93%", "unit": "Harmony"},
        {"name": "EVAL", "status": "DEV", "value": "267", "unit": "Pulses"}
    ]
    
    for module in modules:
        st.markdown("---")
        st.markdown(f"**{module['name']}**")
        status_class = "status-flowing" if module['status'] == 'FLOWING' else "status-dev-flow"
        st.markdown(f'<span class="{status_class}">● {module["status"]}</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="wave-metric" style="font-size: 1.8rem;">{module["value"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size: 0.9rem; color: {COLORS["cyber_purple"]};">{module["unit"]}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Flow monitoring section
st.markdown("---")
st.markdown("### 📊 FLOW DYNAMICS")

flow_cols = st.columns(5)
dynamics = [
    ("AMPLITUDE", f"{random.randint(60, 95)}%"),
    ("FREQUENCY", f"{random.randint(120, 180)} Hz"),
    ("RESONANCE", f"{random.randint(85, 99)}%"),
    ("PHASE", f"{random.randint(0, 360)}°"),
    ("HARMONY", f"{random.randint(90, 100)}%")
]

for i, (label, value) in enumerate(dynamics):
    with flow_cols[i]:
        st.markdown('<div class="flow-monitor">', unsafe_allow_html=True)
        st.markdown(f"**{label}**")
        st.markdown(f'<div class="wave-metric" style="font-size: 1.3rem;">{value}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    f'<div style="text-align: center; font-size: 1.4rem;" class="wave-title">'
    f'🌊 LIFT OS - FLOWING THROUGH DATA DIMENSIONS 🌊'
    f'</div>',
    unsafe_allow_html=True
) 