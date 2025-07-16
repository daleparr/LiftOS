import streamlit as st
import requests
import json
import time
import random

# Neon Pulse Color Palette
COLORS = {
    "electric_blue": "#3A86FF",
    "neon_pink": "#FF006E", 
    "lime_green": "#B9FF66",
    "deep_purple": "#8338EC",
    "midnight_black": "#1A1A2E",
    "soft_white": "#F8F7FF",
    "gradient": "linear-gradient(135deg, #3A86FF 0%, #FF006E 50%, #B9FF66 100%)"
}

st.set_page_config(page_title="Gen-Z Layout 1 (Fixed)", page_icon="🚀", layout="wide")

# Custom CSS for modern styling
st.markdown(f"""
<style>
    /* Hide default Streamlit elements */
    .stDeployButton {{display: none;}}
    .stDecoration {{display: none;}}
    
    /* Hero Section */
    .hero-section {{
        background: {COLORS['gradient']};
        padding: 3rem 2rem;
        border-radius: 1.5rem;
        text-align: center;
        margin-bottom: 2rem;
        animation: pulse 2s infinite;
        box-shadow: 0 10px 30px rgba(255, 0, 110, 0.3);
    }}
    
    @keyframes pulse {{
        0% {{ transform: scale(1); }}
        50% {{ transform: scale(1.02); }}
        100% {{ transform: scale(1); }}
    }}
    
    .hero-title {{
        font-size: 3rem;
        font-weight: 900;
        color: white;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }}
    
    .hero-subtitle {{
        font-size: 1.2rem;
        color: white;
        opacity: 0.9;
        margin-bottom: 2rem;
    }}
    
    /* Animated Stats */
    .stat-card {{
        background: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        border-left: 4px solid {COLORS['neon_pink']};
    }}
    
    .stat-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(255, 0, 110, 0.2);
    }}
    
    .stat-number {{
        font-size: 2.5rem;
        font-weight: 900;
        color: {COLORS['electric_blue']};
        margin-bottom: 0.5rem;
    }}
    
    .stat-label {{
        font-size: 1rem;
        color: #666;
        font-weight: 600;
    }}
    
    /* Module Cards */
    .module-card {{
        background: white;
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border-top: 3px solid {COLORS['lime_green']};
    }}
    
    .module-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(185, 255, 102, 0.2);
    }}
    
    .module-title {{
        font-size: 1.5rem;
        font-weight: 700;
        color: {COLORS['deep_purple']};
        margin-bottom: 1rem;
    }}
    
    .module-status {{
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }}
    
    .status-active {{
        background: {COLORS['lime_green']};
        color: {COLORS['midnight_black']};
    }}
    
    .status-ready {{
        background: {COLORS['electric_blue']};
        color: white;
    }}
    
    /* Buttons */
    .neon-button {{
        background: {COLORS['gradient']};
        color: white;
        padding: 1rem 2rem;
        border: none;
        border-radius: 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
        display: inline-block;
        margin: 0.5rem;
    }}
    
    .neon-button:hover {{
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(255, 0, 110, 0.4);
    }}
    
    /* Animations */
    @keyframes slideIn {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    .slide-in {{
        animation: slideIn 0.6s ease-out;
    }}
</style>
""", unsafe_allow_html=True)

# Hero Section with Lottie Animation
try:
    from streamlit_lottie import st_lottie
    
    def load_lottie_url(url):
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return r.json()
        except:
            pass
        return None
    
    # Hero with Lottie
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        lottie_rocket = load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_V9t630.json")
        if lottie_rocket:
            st_lottie(lottie_rocket, height=200, key="hero_rocket")
        else:
            st.markdown("🚀", unsafe_allow_html=True)
            
except ImportError:
    st.markdown("🚀", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-section slide-in">
    <div class="hero-title">Welcome to LiftOS 🚀</div>
    <div class="hero-subtitle">Next-gen marketing intelligence for the bold</div>
</div>
""", unsafe_allow_html=True)

# Animated Stats Row
st.markdown("### 📊 Live Stats")
col1, col2, col3, col4 = st.columns(4)

stats = [
    {"number": "14", "label": "Active Campaigns", "color": COLORS['electric_blue']},
    {"number": "223", "label": "Results Indexed", "color": COLORS['neon_pink']},
    {"number": "98%", "label": "Uptime", "color": COLORS['lime_green']},
    {"number": "3.2x", "label": "ROAS Boost", "color": COLORS['deep_purple']}
]

for i, (col, stat) in enumerate(zip([col1, col2, col3, col4], stats)):
    with col:
        # Simulate animated counting
        if f"stat_{i}" not in st.session_state:
            st.session_state[f"stat_{i}"] = 0
        
        st.markdown(f"""
        <div class="stat-card slide-in" style="animation-delay: {i*0.1}s;">
            <div class="stat-number">{stat['number']}</div>
            <div class="stat-label">{stat['label']}</div>
        </div>
        """, unsafe_allow_html=True)

# Module Cards Section
st.markdown("### 🧩 Your Modules")

# Core modules (always visible)
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="module-card slide-in">
        <div class="module-title">🔥 Core Analytics</div>
        <div class="module-status status-active">ACTIVE</div>
        <p>Real-time campaign performance, attribution modeling, and ROI tracking across all channels.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Launch Core Analytics", key="core_btn"):
        st.success("🎉 Core Analytics launched!")

with col2:
    st.markdown("""
    <div class="module-card slide-in">
        <div class="module-title">🧠 Memory Search</div>
        <div class="module-status status-active">ACTIVE</div>
        <p>Semantic search across all your marketing data, insights, and historical performance.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔍 Open Memory Search", key="memory_btn"):
        st.success("🎉 Memory Search opened!")

# Modular components (toggleable)
st.markdown("### ⚡ Power-ups")

# Toggle states
if "surfacing_enabled" not in st.session_state:
    st.session_state.surfacing_enabled = True
if "causal_enabled" not in st.session_state:
    st.session_state.causal_enabled = True
if "llm_enabled" not in st.session_state:
    st.session_state.llm_enabled = False
if "agentic_enabled" not in st.session_state:
    st.session_state.agentic_enabled = False

# Module toggles
col1, col2, col3, col4 = st.columns(4)

modules = [
    {"name": "Surfacing", "icon": "🌊", "key": "surfacing_enabled", "desc": "Data surfacing & insights"},
    {"name": "Causal", "icon": "🔗", "key": "causal_enabled", "desc": "Causal inference & modeling"},
    {"name": "LLM", "icon": "🤖", "key": "llm_enabled", "desc": "AI-powered analysis"},
    {"name": "Agentic", "icon": "🎯", "key": "agentic_enabled", "desc": "Autonomous agents"}
]

for col, module in zip([col1, col2, col3, col4], modules):
    with col:
        enabled = st.session_state[module["key"]]
        status = "ACTIVE" if enabled else "READY"
        status_class = "status-active" if enabled else "status-ready"
        
        st.markdown(f"""
        <div class="module-card slide-in">
            <div class="module-title">{module['icon']} {module['name']}</div>
            <div class="module-status {status_class}">{status}</div>
            <p>{module['desc']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button(f"{'🔥 Launch' if enabled else '⚡ Enable'} {module['name']}", key=f"btn_{module['key']}"):
            st.session_state[module["key"]] = not st.session_state[module["key"]]
            st.rerun()

# Gamification Section
st.markdown("### 🏆 Your Progress")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="module-card slide-in">
        <div class="module-title">🎮 Achievements</div>
        <div style="display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 1rem;">
            <span style="background: #FFD700; color: #000; padding: 0.5rem 1rem; border-radius: 1rem; font-size: 0.9rem;">🥇 First Campaign</span>
            <span style="background: #FF6B6B; color: #fff; padding: 0.5rem 1rem; border-radius: 1rem; font-size: 0.9rem;">🔥 7-Day Streak</span>
            <span style="background: #4ECDC4; color: #fff; padding: 0.5rem 1rem; border-radius: 1rem; font-size: 0.9rem;">📈 ROAS Master</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="module-card slide-in">
        <div class="module-title">⚡ Recent Activity</div>
        <div style="margin-top: 1rem;">
            <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.5rem;">
                <span style="color: #3A86FF;">🔍</span>
                <span>Memory search: "Meta ads performance"</span>
                <span style="color: #999; font-size: 0.8rem;">2m ago</span>
            </div>
            <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.5rem;">
                <span style="color: #FF006E;">📊</span>
                <span>Updated attribution model</span>
                <span style="color: #999; font-size: 0.8rem;">15m ago</span>
            </div>
            <div style="display: flex; align-items: center; gap: 1rem;">
                <span style="color: #B9FF66;">🚀</span>
                <span>Launched new campaign</span>
                <span style="color: #999; font-size: 0.8rem;">1h ago</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# CTA Section
st.markdown("### 🎯 Ready to Launch?")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("🚀 Let's Go!", key="main_cta"):
        st.balloons()
        st.success("🎉 Welcome to the future of marketing intelligence!")
        
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; color: #666;">
        <p>🔥 Join the revolution • 🚀 Built for Gen-Z marketers • ⚡ Powered by AI</p>
    </div>
    """, unsafe_allow_html=True) 