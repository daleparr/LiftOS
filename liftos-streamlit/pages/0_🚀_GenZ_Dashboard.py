import streamlit as st
import random

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

st.set_page_config(page_title="Gen-Z Dashboard", page_icon="🚀", layout="wide")

# --- Welcome Banner ---
st.markdown(
    f"""
    <div style="background: {COLORS['gradient']}; padding: 2.5rem 1rem 2rem 1rem; border-radius: 1.5rem; text-align: center; margin-bottom: 2rem;">
        <h1 style="color: {COLORS['soft_white']}; font-size: 2.7rem; letter-spacing: 1px;">Welcome to LiftOS <span style='font-size:2.2rem;'>🚀</span></h1>
        <p style="color: {COLORS['soft_white']}; font-size: 1.25rem; margin-bottom: 1.5rem;">The next-gen marketing intelligence platform for creators and innovators.</p>
        <button id="letsgo-btn" style="background: {COLORS['lime_green']}; color: {COLORS['midnight_black']}; border: none; padding: 1rem 2.5rem; border-radius: 2rem; font-size: 1.1rem; font-weight: bold; cursor: pointer; box-shadow: 0 4px 16px {COLORS['deep_purple']}33;">Let’s Go!</button>
    </div>
    <script>
    const btn = window.parent.document.getElementById('letsgo-btn');
    if (btn) {{
        btn.onclick = function() {{
            window.parent.postMessage({{type: 'confetti'}}, '*');
        }}
    }}
    </script>
    """,
    unsafe_allow_html=True
)

# --- Core Cards Row ---
col1, col2 = st.columns(2, gap="large")
with col1:
    st.markdown(f"""
    <div style="background: {COLORS['deep_purple']}; border-radius: 1.2rem; padding: 1.5rem; box-shadow: 0 2px 12px {COLORS['electric_blue']}33;">
        <h3 style="color: {COLORS['lime_green']}; font-size: 1.4rem;">🧠 Core Analytics</h3>
        <p style="color: {COLORS['soft_white']}; font-size: 1.1rem;">Always-on insights, attribution, and campaign stats.</p>
        <div style="color: {COLORS['electric_blue']}; font-size: 2.2rem; font-weight: bold;">{random.randint(12, 25)} Active Campaigns</div>
        <button style="background: {COLORS['electric_blue']}; color: {COLORS['soft_white']}; border: none; padding: 0.7rem 1.5rem; border-radius: 1.2rem; font-size: 1rem; font-weight: bold; margin-top: 1rem; cursor: pointer;">View Analytics</button>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div style="background: {COLORS['deep_purple']}; border-radius: 1.2rem; padding: 1.5rem; box-shadow: 0 2px 12px {COLORS['neon_pink']}33;">
        <h3 style="color: {COLORS['electric_blue']}; font-size: 1.4rem;">🔍 Memory Search</h3>
        <p style="color: {COLORS['soft_white']}; font-size: 1.1rem;">Semantic, hybrid, and keyword search across all your data.</p>
        <div style="color: {COLORS['neon_pink']}; font-size: 2.2rem; font-weight: bold;">{random.randint(120, 300)} Results Indexed</div>
        <button style="background: {COLORS['neon_pink']}; color: {COLORS['soft_white']}; border: none; padding: 0.7rem 1.5rem; border-radius: 1.2rem; font-size: 1rem; font-weight: bold; margin-top: 1rem; cursor: pointer;">Search Now</button>
    </div>
    """, unsafe_allow_html=True)

# --- Modular Services Row ---
modular_services = [
    {"name": "Surfacing", "icon": "🌊", "color": COLORS['electric_blue']},
    {"name": "Causal", "icon": "🧩", "color": COLORS['neon_pink']},
    {"name": "LLM Assistant", "icon": "🤖", "color": COLORS['lime_green']},
    {"name": "Agentic", "icon": "🦾", "color": COLORS['deep_purple']},
]

st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)

mod_cols = st.columns(4, gap="large")
for i, svc in enumerate(modular_services):
    with mod_cols[i]:
        enabled = st.toggle(f"{svc['icon']} {svc['name']}", value=True, key=f"toggle_{svc['name']}")
        status = "Active" if enabled else "Inactive"
        badge_color = svc['color'] if enabled else COLORS['midnight_black']
        st.markdown(f"""
        <div style="background: {COLORS['soft_white']}; border-radius: 1.2rem; padding: 1.2rem; box-shadow: 0 2px 12px {svc['color']}33; margin-top: 0.5rem;">
            <h4 style="color: {svc['color']}; font-size: 1.2rem; margin-bottom: 0.5rem;">{svc['icon']} {svc['name']}</h4>
            <span style="background: {badge_color}; color: {COLORS['soft_white'] if enabled else COLORS['soft_white']}; padding: 0.3rem 1rem; border-radius: 1rem; font-size: 0.95rem; font-weight: bold;">{status}</span>
            <p style="color: {COLORS['midnight_black']}; font-size: 1rem; margin-top: 0.7rem;">{random.choice(['Ready to use!', 'Try the new features!', 'Coming soon!', 'Beta'])}</p>
            <button style="background: {svc['color']}; color: {COLORS['soft_white']}; border: none; padding: 0.6rem 1.2rem; border-radius: 1rem; font-size: 0.95rem; font-weight: bold; margin-top: 0.5rem; cursor: pointer;">Open</button>
        </div>
        """, unsafe_allow_html=True)

# --- Gamification/Stats Row ---
st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)
gam_col1, gam_col2 = st.columns(2, gap="large")
with gam_col1:
    st.markdown(f"""
    <div style="background: {COLORS['electric_blue']}; border-radius: 1.2rem; padding: 1.2rem; box-shadow: 0 2px 12px {COLORS['electric_blue']}33;">
        <h4 style="color: {COLORS['soft_white']}; font-size: 1.2rem;">🏅 Achievements</h4>
        <ul style="color: {COLORS['soft_white']}; font-size: 1rem;">
            <li>🎉 First Attribution Model Created</li>
            <li>🔥 7-Day Data Sync Streak</li>
            <li>🚀 Campaign ROAS > 3.0</li>
        </ul>
        <div style="margin-top: 1rem; color: {COLORS['lime_green']}; font-weight: bold; font-size: 1.1rem;">Keep going! More badges await 🏆</div>
    </div>
    """, unsafe_allow_html=True)
with gam_col2:
    st.markdown(f"""
    <div style="background: {COLORS['neon_pink']}; border-radius: 1.2rem; padding: 1.2rem; box-shadow: 0 2px 12px {COLORS['neon_pink']}33;">
        <h4 style="color: {COLORS['soft_white']}; font-size: 1.2rem;">🕒 Recent Activity</h4>
        <ul style="color: {COLORS['soft_white']}; font-size: 1rem;">
            <li>🧠 Memory search: "Meta ads performance last month"</li>
            <li>🧩 Causal analysis: Attribution run completed</li>
            <li>🤖 LLM: "Suggest a meme for campaign"</li>
        </ul>
        <div style="margin-top: 1rem; color: {COLORS['soft_white']}; font-size: 1rem;">Share your wins with your team! 🎊</div>
    </div>
    """, unsafe_allow_html=True)

# --- Footer ---
st.markdown(f"""
<div style="text-align: center; margin-top: 2.5rem; color: {COLORS['deep_purple']}; font-size: 1rem;">
    <span>Made with <span style='color: {COLORS['neon_pink']};'>&#10084;&#65039;</span> for Gen-Z Marketers | <a href='https://liftos.ai' style='color: {COLORS['electric_blue']}; text-decoration: none;'>liftos.ai</a> | v2.0</span>
</div>
""", unsafe_allow_html=True) 