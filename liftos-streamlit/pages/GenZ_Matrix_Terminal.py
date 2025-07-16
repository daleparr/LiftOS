import streamlit as st
import plotly.graph_objects as go
import numpy as np
import random
import time
from datetime import datetime

# LiftOS Matrix Colors
COLORS = {
    "hot_pink": "#FF1493",
    "electric_blue": "#00D4FF",
    "neon_green": "#00FF41",
    "cyber_purple": "#8A2BE2",
    "warning_orange": "#FF8C00",
    "deep_black": "#0A0A0A",
    "matrix_green": "#00FF00",
    "white": "#FFFFFF"
}

st.set_page_config(page_title="LiftOS: Matrix Terminal", page_icon="💻", layout="wide")

# Matrix Terminal CSS
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Fira+Code:wght@400;500;700&display=swap');
    
    .stApp {{
        background: 
            linear-gradient(135deg, {COLORS['deep_black']} 0%, #001100 100%),
            radial-gradient(circle at 20% 20%, {COLORS['neon_green']}11 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, {COLORS['hot_pink']}11 0%, transparent 50%);
        color: {COLORS['neon_green']};
        font-family: 'Fira Code', monospace;
        overflow-x: hidden;
    }}
    
    .matrix-title {{
        text-align: center;
        font-size: 3.5rem;
        font-weight: 900;
        font-family: 'Orbitron', monospace;
        background: linear-gradient(45deg, {COLORS['neon_green']}, {COLORS['electric_blue']}, {COLORS['hot_pink']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 20px {COLORS['neon_green']};
        margin-bottom: 2rem;
        animation: matrix-glow 3s ease-in-out infinite;
    }}
    
    @keyframes matrix-glow {{
        0%, 100% {{ text-shadow: 0 0 20px {COLORS['neon_green']}; }}
        50% {{ text-shadow: 0 0 30px {COLORS['neon_green']}, 0 0 40px {COLORS['electric_blue']}; }}
    }}
    
    .digital-rain {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 1;
        overflow: hidden;
    }}
    
    .rain-column {{
        position: absolute;
        top: -100%;
        font-family: 'Fira Code', monospace;
        font-size: 14px;
        color: {COLORS['neon_green']};
        opacity: 0.7;
        animation: rain-fall 3s linear infinite;
    }}
    
    @keyframes rain-fall {{
        0% {{ top: -100%; opacity: 0; }}
        10% {{ opacity: 1; }}
        90% {{ opacity: 1; }}
        100% {{ top: 100%; opacity: 0; }}
    }}
    
    .terminal-panel {{
        background: linear-gradient(135deg, 
            rgba(0,255,65,0.1) 0%, 
            rgba(0,0,0,0.9) 100%);
        border: 2px solid {COLORS['neon_green']};
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        font-family: 'Fira Code', monospace;
        position: relative;
        box-shadow: 
            0 0 20px rgba(0,255,65,0.3),
            inset 0 0 20px rgba(0,255,65,0.1);
        transition: all 0.3s ease;
    }}
    
    .terminal-panel::before {{
        content: '> LIFT_OS_TERMINAL';
        position: absolute;
        top: -10px;
        left: 10px;
        background: {COLORS['deep_black']};
        color: {COLORS['neon_green']};
        padding: 0 10px;
        font-size: 0.8rem;
        font-weight: bold;
    }}
    
    .terminal-panel:hover {{
        border-color: {COLORS['electric_blue']};
        box-shadow: 
            0 0 30px rgba(0,212,255,0.4),
            inset 0 0 30px rgba(0,212,255,0.1);
    }}
    
    .code-block {{
        background: rgba(0,0,0,0.8);
        border: 1px solid {COLORS['neon_green']};
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-family: 'Fira Code', monospace;
        font-size: 0.9rem;
        color: {COLORS['neon_green']};
        overflow-x: auto;
        white-space: pre;
        position: relative;
    }}
    
    .code-block::before {{
        content: '$ ';
        color: {COLORS['hot_pink']};
        font-weight: bold;
    }}
    
    .status-online {{
        color: {COLORS['neon_green']};
        font-weight: bold;
        text-shadow: 0 0 10px {COLORS['neon_green']};
    }}
    
    .status-dev {{
        color: {COLORS['warning_orange']};
        font-weight: bold;
        text-shadow: 0 0 10px {COLORS['warning_orange']};
    }}
    
    .matrix-metric {{
        font-size: 2.5rem;
        font-weight: 900;
        font-family: 'Orbitron', monospace;
        color: {COLORS['neon_green']};
        text-align: center;
        text-shadow: 0 0 15px {COLORS['neon_green']};
        animation: metric-flicker 2s infinite;
    }}
    
    @keyframes metric-flicker {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.8; }}
    }}
    
    .terminal-button {{
        background: linear-gradient(45deg, {COLORS['neon_green']}, {COLORS['electric_blue']});
        border: 2px solid {COLORS['neon_green']};
        color: {COLORS['deep_black']};
        padding: 0.8rem 1.5rem;
        border-radius: 5px;
        font-family: 'Fira Code', monospace;
        font-weight: bold;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 0 15px rgba(0,255,65,0.4);
        text-transform: uppercase;
    }}
    
    .terminal-button:hover {{
        background: linear-gradient(45deg, {COLORS['electric_blue']}, {COLORS['hot_pink']});
        border-color: {COLORS['electric_blue']};
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(0,212,255,0.5);
    }}
    
    .system-log {{
        background: rgba(0,0,0,0.9);
        border: 1px solid {COLORS['cyber_purple']};
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-family: 'Fira Code', monospace;
        font-size: 0.8rem;
        color: {COLORS['cyber_purple']};
        max-height: 200px;
        overflow-y: auto;
    }}
    
    .typing-effect {{
        border-right: 2px solid {COLORS['neon_green']};
        animation: typing 3s steps(40, end), blink 1s step-end infinite;
        white-space: nowrap;
        overflow: hidden;
    }}
    
    @keyframes typing {{
        from {{ width: 0; }}
        to {{ width: 100%; }}
    }}
    
    @keyframes blink {{
        0%, 50% {{ border-color: {COLORS['neon_green']}; }}
        51%, 100% {{ border-color: transparent; }}
    }}
    
    .glitch {{
        position: relative;
        color: {COLORS['neon_green']};
        animation: glitch 2s infinite;
    }}
    
    @keyframes glitch {{
        0%, 100% {{ transform: translate(0); }}
        20% {{ transform: translate(-2px, 2px); }}
        40% {{ transform: translate(-2px, -2px); }}
        60% {{ transform: translate(2px, 2px); }}
        80% {{ transform: translate(2px, -2px); }}
    }}
</style>
""", unsafe_allow_html=True)

# Add digital rain effect
rain_columns = []
for i in range(20):
    left_pos = random.randint(0, 100)
    delay = random.uniform(0, 3)
    rain_columns.append(f'<div class="rain-column" style="left: {left_pos}%; animation-delay: {delay}s;">01001010<br/>11010101<br/>00110011<br/>10101010<br/>01110001</div>')

st.markdown(f'<div class="digital-rain">{"".join(rain_columns)}</div>', unsafe_allow_html=True)

# Header
st.markdown('<h1 class="matrix-title glitch">LIFT OS: MATRIX TERMINAL</h1>', unsafe_allow_html=True)

# Create matrix-style network visualization
def create_matrix_network():
    fig = go.Figure()
    
    # Create network graph
    nodes = [
        {"name": "CORE", "x": 0, "y": 0, "color": COLORS['white']},
        {"name": "MEMORY", "x": 2, "y": 0, "color": COLORS['white']},
        {"name": "SURFACING", "x": -2, "y": 2, "color": COLORS['hot_pink']},
        {"name": "CAUSAL", "x": 2, "y": 2, "color": COLORS['electric_blue']},
        {"name": "AGENTIC", "x": -2, "y": -2, "color": COLORS['warning_orange']},
        {"name": "LLM", "x": 2, "y": -2, "color": COLORS['cyber_purple']},
        {"name": "SENTIMENT", "x": 0, "y": 3, "color": COLORS['hot_pink']},
        {"name": "EVAL", "x": 0, "y": -3, "color": COLORS['electric_blue']}
    ]
    
    # Add connections with matrix-style lines
    connections = [
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3), (2, 6), (3, 7)
    ]
    
    for start, end in connections:
        fig.add_trace(go.Scatter(
            x=[nodes[start]["x"], nodes[end]["x"]],
            y=[nodes[start]["y"], nodes[end]["y"]],
            mode='lines',
            line=dict(color=COLORS['neon_green'], width=3, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add nodes
    for node in nodes:
        fig.add_trace(go.Scatter(
            x=[node["x"]],
            y=[node["y"]],
            mode='markers+text',
            marker=dict(
                size=25,
                color=node["color"],
                line=dict(width=3, color=COLORS['neon_green']),
                symbol='square'
            ),
            text=node["name"],
            textposition="middle center",
            textfont=dict(color=COLORS['deep_black'], size=8, family="Fira Code"),
            showlegend=False,
            hovertemplate=f"<b>{node['name']}</b><br>Matrix Node<extra></extra>"
        ))
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
        title=dict(
            text="NEURAL NETWORK MATRIX",
            x=0.5,
            font=dict(color=COLORS['neon_green'], size=18, family="Orbitron")
        )
    )
    
    return fig

# Main terminal layout
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.markdown('<div class="terminal-panel">', unsafe_allow_html=True)
    st.markdown("### 💻 CORE SYSTEMS")
    
    st.markdown('<div class="code-block">system.core.status()</div>', unsafe_allow_html=True)
    st.markdown("**CORE**")
    st.markdown('<span class="status-online">● ONLINE</span>', unsafe_allow_html=True)
    st.markdown('<div class="matrix-metric">100%</div>', unsafe_allow_html=True)
    st.markdown("Operational")
    
    st.markdown('<div class="code-block">memory.core.status()</div>', unsafe_allow_html=True)
    st.markdown("**MEMORY**")
    st.markdown('<span class="status-online">● ONLINE</span>', unsafe_allow_html=True)
    st.markdown('<div class="matrix-metric">5.7TB</div>', unsafe_allow_html=True)
    st.markdown("Allocated")
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.plotly_chart(create_matrix_network(), use_container_width=True)
    
    # Terminal commands
    st.markdown("### 💻 COMMAND INTERFACE")
    cmd_cols = st.columns(4)
    
    with cmd_cols[0]:
        if st.button("EXECUTE", key="exec"):
            st.success("Command executed successfully!")
    
    with cmd_cols[1]:
        if st.button("SCAN", key="scan"):
            st.info("System scan initiated...")
    
    with cmd_cols[2]:
        if st.button("DEPLOY", key="deploy"):
            st.warning("Deployment in progress...")
    
    with cmd_cols[3]:
        if st.button("TERMINATE", key="term"):
            st.error("Process terminated!")

with col3:
    st.markdown('<div class="terminal-panel">', unsafe_allow_html=True)
    st.markdown("### 🔮 MODULE STATUS")
    
    modules = [
        {"name": "SURFACING", "status": "ONLINE", "value": "3.2M", "cmd": "surfacing.run()"},
        {"name": "CAUSAL", "status": "ONLINE", "value": "1.8K", "cmd": "causal.analyze()"},
        {"name": "AGENTIC", "status": "DEV", "value": "127", "cmd": "agentic.init()"},
        {"name": "LLM", "status": "DEV", "value": "GPT-4", "cmd": "llm.load()"},
        {"name": "SENTIMENT", "status": "DEV", "value": "94%", "cmd": "sentiment.test()"},
        {"name": "EVAL", "status": "DEV", "value": "384", "cmd": "eval.start()"}
    ]
    
    for module in modules:
        st.markdown(f'<div class="code-block">{module["cmd"]}</div>', unsafe_allow_html=True)
        st.markdown(f"**{module['name']}**")
        status_class = "status-online" if module['status'] == 'ONLINE' else "status-dev"
        st.markdown(f'<span class="{status_class}">● {module["status"]}</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="matrix-metric" style="font-size: 1.5rem;">{module["value"]}</div>', unsafe_allow_html=True)
        st.markdown("---")
    
    st.markdown('</div>', unsafe_allow_html=True)

# System logs section
st.markdown("---")
st.markdown("### 📊 SYSTEM LOGS")

log_cols = st.columns(2)

with log_cols[0]:
    st.markdown('<div class="system-log">', unsafe_allow_html=True)
    st.markdown("**REAL-TIME LOGS**")
    logs = [
        "[2025-01-16 14:30:15] CORE: System initialized",
        "[2025-01-16 14:30:16] MEMORY: Cache warmed up",
        "[2025-01-16 14:30:17] SURFACING: 1.2M insights processed",
        "[2025-01-16 14:30:18] CAUSAL: Model training complete",
        "[2025-01-16 14:30:19] AGENTIC: 47 agents deployed",
        "[2025-01-16 14:30:20] LLM: GPT-4 model loaded",
        "[2025-01-16 14:30:21] SENTIMENT: Analysis at 94% accuracy",
        "[2025-01-16 14:30:22] EVAL: 384 tests passed"
    ]
    for log in logs:
        st.markdown(f'<div style="color: {COLORS["neon_green"]}; font-size: 0.8rem;">{log}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with log_cols[1]:
    st.markdown('<div class="system-log">', unsafe_allow_html=True)
    st.markdown("**PERFORMANCE METRICS**")
    metrics = [
        f"CPU Usage: {random.randint(45, 75)}%",
        f"Memory Usage: {random.randint(60, 85)}%",
        f"Network I/O: {random.randint(200, 800)} MB/s",
        f"Disk Usage: {random.randint(40, 70)}%",
        f"Active Connections: {random.randint(150, 300)}",
        f"Requests/sec: {random.randint(500, 1200)}",
        f"Response Time: {random.randint(10, 50)}ms",
        f"Error Rate: {random.uniform(0.1, 2.0):.1f}%"
    ]
    for metric in metrics:
        st.markdown(f'<div style="color: {COLORS["electric_blue"]}; font-size: 0.8rem;">{metric}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    f'<div style="text-align: center; font-size: 1.2rem; color: {COLORS["neon_green"]}; font-family: Orbitron;">'
    f'💻 LIFT OS - MATRIX TERMINAL INTERFACE 💻'
    f'</div>',
    unsafe_allow_html=True
) 