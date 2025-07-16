import streamlit as st
import random
import time

# Neon Pulse Color Palette
COLORS = {
    "electric_blue": "#3A86FF",
    "neon_pink": "#FF006E", 
    "lime_green": "#B9FF66",
    "deep_purple": "#8338EC",
    "midnight_black": "#1A1A2E",
    "soft_white": "#F8F7FF",
    "gradient": "linear-gradient(90deg, #3A86FF 0%, #FF006E 100%)"
}

st.set_page_config(page_title="Gen-Z Dashboard (Simple)", page_icon="🚀", layout="wide")

# Custom CSS for modern styling
st.markdown(f"""
<style>
    .main-header {{
        background: {COLORS['gradient']};
        padding: 2rem 1rem;
        border-radius: 1rem;
        text-align: center;
        margin-bottom: 2rem;
        color: {COLORS['soft_white']};
    }}
    
    .metric-card {{
        background: {COLORS['deep_purple']};
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        color: {COLORS['soft_white']};
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }}
    
    .service-card {{
        background: {COLORS['soft_white']};
        padding: 1.2rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid {COLORS['electric_blue']};
    }}
    
    .achievement-card {{
        background: {COLORS['electric_blue']};
        padding: 1rem;
        border-radius: 1rem;
        color: {COLORS['soft_white']};
        margin-bottom: 1rem;
    }}
    
    .activity-card {{
        background: {COLORS['neon_pink']};
        padding: 1rem;
        border-radius: 1rem;
        color: {COLORS['soft_white']};
        margin-bottom: 1rem;
    }}
    
    .pulse {{
        animation: pulse 2s infinite;
    }}
    
    @keyframes pulse {{
        0% {{ opacity: 1; }}
        50% {{ opacity: 0.7; }}
        100% {{ opacity: 1; }}
    }}
</style>
""", unsafe_allow_html=True)

# Welcome Header
st.markdown(f"""
<div class="main-header">
    <h1>🚀 Welcome to LiftOS</h1>
    <p style="font-size: 1.2rem; margin-bottom: 1rem;">The next-gen marketing intelligence platform for creators and innovators</p>
    <div style="font-size: 2rem;">✨ Let's Go! ✨</div>
</div>
""", unsafe_allow_html=True)

# Core Analytics Row
st.markdown("## 🔥 Core Analytics")
col1, col2 = st.columns(2)

with col1:
    campaigns = random.randint(12, 25)
    st.markdown(f"""
    <div class="metric-card">
        <h3>🧠 Core Analytics</h3>
        <p>Always-on insights, attribution, and campaign stats</p>
        <div style="font-size: 2.5rem; font-weight: bold; color: {COLORS['lime_green']};">{campaigns}</div>
        <div style="font-size: 1.2rem;">Active Campaigns</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    results = random.randint(120, 300)
    st.markdown(f"""
    <div class="metric-card">
        <h3>🔍 Memory Search</h3>
        <p>Semantic, hybrid, and keyword search across all your data</p>
        <div style="font-size: 2.5rem; font-weight: bold; color: {COLORS['neon_pink']};">{results}</div>
        <div style="font-size: 1.2rem;">Results Indexed</div>
    </div>
    """, unsafe_allow_html=True)

# Modular Services
st.markdown("## 🎯 Modular Services")
services = [
    {"name": "Surfacing", "icon": "🌊", "status": "Active"},
    {"name": "Causal", "icon": "🧩", "status": "Active"},
    {"name": "LLM Assistant", "icon": "🤖", "status": "Beta"},
    {"name": "Agentic", "icon": "🦾", "status": "Coming Soon"}
]

service_cols = st.columns(4)
for i, service in enumerate(services):
    with service_cols[i]:
        enabled = st.toggle(f"{service['icon']} {service['name']}", value=True, key=f"service_{i}")
        status_color = COLORS['lime_green'] if enabled else COLORS['midnight_black']
        st.markdown(f"""
        <div class="service-card">
            <h4 style="color: {COLORS['deep_purple']};">{service['icon']} {service['name']}</h4>
            <span style="background: {status_color}; color: white; padding: 0.3rem 0.8rem; border-radius: 1rem; font-size: 0.9rem;">
                {'Active' if enabled else 'Inactive'}
            </span>
            <p style="color: {COLORS['midnight_black']}; margin-top: 0.5rem;">{service['status']}</p>
        </div>
        """, unsafe_allow_html=True)

# Progress and Stats
st.markdown("## 📊 Your Progress")
progress_col1, progress_col2 = st.columns(2)

with progress_col1:
    st.markdown(f"""
    <div class="achievement-card">
        <h4>🏅 Achievements</h4>
        <ul style="text-align: left; margin-top: 1rem;">
            <li>🎉 First Attribution Model Created</li>
            <li>🔥 7-Day Data Sync Streak</li>
            <li>🚀 Campaign ROAS > 3.0</li>
            <li>💡 AI Insights Unlocked</li>
        </ul>
        <div style="margin-top: 1rem; font-weight: bold;">Keep going! More badges await 🏆</div>
    </div>
    """, unsafe_allow_html=True)

with progress_col2:
    st.markdown(f"""
    <div class="activity-card">
        <h4>🕒 Recent Activity</h4>
        <ul style="text-align: left; margin-top: 1rem;">
            <li>🧠 Memory search: "Meta ads performance last month"</li>
            <li>🧩 Causal analysis: Attribution run completed</li>
            <li>🤖 LLM: "Suggest a meme for campaign"</li>
            <li>🌊 Surfacing: Product analysis finished</li>
        </ul>
        <div style="margin-top: 1rem;">Share your wins with your team! 🎊</div>
    </div>
    """, unsafe_allow_html=True)

# Interactive Elements
st.markdown("## 💫 How are you feeling about LiftOS today?")
feedback_cols = st.columns(3)

with feedback_cols[0]:
    if st.button("😎 You rock!", use_container_width=True):
        st.success("Thanks! You're amazing too! 🌟")
        st.balloons()

with feedback_cols[1]:
    if st.button("🔥 On fire!", use_container_width=True):
        st.success("That's the energy we love! 🚀")
        st.balloons()

with feedback_cols[2]:
    if st.button("💡 Inspired!", use_container_width=True):
        st.success("Innovation is our middle name! ✨")
        st.balloons()

# Footer
st.markdown(f"""
<div style="text-align: center; margin-top: 3rem; padding: 2rem; background: {COLORS['deep_purple']}; border-radius: 1rem; color: {COLORS['soft_white']};">
    <div style="font-size: 1.2rem; margin-bottom: 1rem;">Made with ❤️ for Gen-Z Marketers</div>
    <div><a href="https://liftos.ai" style="color: {COLORS['lime_green']}; text-decoration: none;">liftos.ai</a> | v2.0 | 🚀 The Future is Now</div>
</div>
""", unsafe_allow_html=True) 