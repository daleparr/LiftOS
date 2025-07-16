import streamlit as st
import random
import time
from datetime import datetime, timedelta

# Vibrant Gen-Z Color Palette
COLORS = {
    "primary": "#FF6B6B",
    "secondary": "#4ECDC4", 
    "accent": "#45B7D1",
    "warning": "#FFA07A",
    "success": "#98D8C8",
    "dark": "#2C3E50",
    "light": "#F8F9FA",
    "gradient": "linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%)"
}

st.set_page_config(page_title="Gen-Z Layout 2 (Fixed)", page_icon="🔥", layout="wide")

# Custom CSS for social media feel
st.markdown(f"""
<style>
    .stDeployButton {{display: none;}}
    .stDecoration {{display: none;}}
    
    /* Profile Card */
    .profile-card {{
        background: {COLORS['gradient']};
        padding: 2rem;
        border-radius: 1.5rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(255, 107, 107, 0.3);
    }}
    
    .profile-avatar {{
        width: 100px;
        height: 100px;
        border-radius: 50%;
        margin: 0 auto 1rem;
        border: 4px solid white;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }}
    
    .profile-name {{
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }}
    
    .profile-title {{
        font-size: 1rem;
        opacity: 0.9;
        margin-bottom: 1rem;
    }}
    
    .profile-stats {{
        display: flex;
        justify-content: space-around;
        margin-top: 1rem;
    }}
    
    .stat-item {{
        text-align: center;
    }}
    
    .stat-number {{
        font-size: 1.5rem;
        font-weight: 700;
        display: block;
    }}
    
    .stat-label {{
        font-size: 0.8rem;
        opacity: 0.8;
    }}
    
    /* Activity Feed */
    .activity-feed {{
        background: white;
        border-radius: 1rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }}
    
    .activity-item {{
        display: flex;
        align-items: center;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        transition: background 0.3s ease;
    }}
    
    .activity-item:hover {{
        background: #f8f9fa;
    }}
    
    .activity-icon {{
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 1rem;
        font-size: 1.2rem;
    }}
    
    .activity-content {{
        flex: 1;
    }}
    
    .activity-title {{
        font-weight: 600;
        margin-bottom: 0.2rem;
    }}
    
    .activity-time {{
        color: #666;
        font-size: 0.8rem;
    }}
    
    /* Achievement Badges */
    .badge {{
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.2rem;
        text-align: center;
        animation: bounce 2s infinite;
    }}
    
    @keyframes bounce {{
        0%, 20%, 50%, 80%, 100% {{ transform: translateY(0); }}
        40% {{ transform: translateY(-5px); }}
        60% {{ transform: translateY(-3px); }}
    }}
    
    .badge-gold {{
        background: linear-gradient(45deg, #FFD700, #FFA500);
        color: #000;
    }}
    
    .badge-silver {{
        background: linear-gradient(45deg, #C0C0C0, #808080);
        color: #000;
    }}
    
    .badge-bronze {{
        background: linear-gradient(45deg, #CD7F32, #A0522D);
        color: #fff;
    }}
    
    /* Widget Cards */
    .widget-card {{
        background: white;
        border-radius: 1rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        border-left: 4px solid {COLORS['primary']};
    }}
    
    .widget-card:hover {{
        transform: translateY(-5px);
    }}
    
    .widget-title {{
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: {COLORS['dark']};
    }}
    
    /* Progress Bars */
    .progress-bar {{
        background: #e9ecef;
        border-radius: 1rem;
        height: 8px;
        overflow: hidden;
        margin: 0.5rem 0;
    }}
    
    .progress-fill {{
        height: 100%;
        background: {COLORS['gradient']};
        border-radius: 1rem;
        transition: width 0.3s ease;
    }}
    
    /* Notification Badge */
    .notification-badge {{
        background: {COLORS['primary']};
        color: white;
        border-radius: 50%;
        width: 20px;
        height: 20px;
        font-size: 0.7rem;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-left: auto;
    }}
</style>
""", unsafe_allow_html=True)

# Navigation Menu
try:
    from streamlit_option_menu import option_menu
    
    selected = option_menu(
        menu_title=None,
        options=["Dashboard", "Analytics", "Memory", "Modules", "Settings"],
        icons=["house", "graph-up", "brain", "puzzle", "gear"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": COLORS['primary'], "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "center",
                "margin": "0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": COLORS['primary']},
        }
    )
except ImportError:
    selected = st.selectbox("Navigate", ["Dashboard", "Analytics", "Memory", "Modules", "Settings"])

st.markdown("<br>", unsafe_allow_html=True)

# Profile Section
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown(f"""
    <div class="profile-card">
        <div class="profile-avatar">
            <img src="https://randomuser.me/api/portraits/men/32.jpg" style="width: 100%; height: 100%; border-radius: 50%;">
        </div>
        <div class="profile-name">Alex Chen</div>
        <div class="profile-title">Growth Marketing Lead</div>
        <div class="profile-stats">
            <div class="stat-item">
                <span class="stat-number">🔥 7</span>
                <span class="stat-label">Day Streak</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">⭐ 2.4K</span>
                <span class="stat-label">Points</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">🏆 Pro</span>
                <span class="stat-label">Level</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Main Content Based on Selection
if selected == "Dashboard":
    # Achievement Badges
    st.markdown("### 🏅 Your Achievements")
    
    achievements = [
        {"name": "First Attribution Model", "type": "gold", "icon": "🥇"},
        {"name": "7-Day Data Sync Streak", "type": "silver", "icon": "🔥"},
        {"name": "Campaign ROAS > 3.0", "type": "bronze", "icon": "📈"},
        {"name": "Memory Search Master", "type": "gold", "icon": "🧠"},
        {"name": "Automation Expert", "type": "silver", "icon": "🤖"},
    ]
    
    badge_html = ""
    for achievement in achievements:
        badge_html += f'<span class="badge badge-{achievement["type"]}">{achievement["icon"]} {achievement["name"]}</span>'
    
    st.markdown(f'<div style="text-align: center; margin-bottom: 2rem;">{badge_html}</div>', unsafe_allow_html=True)
    
    # Activity Feed
    st.markdown("### 🕒 Recent Activity")
    
    activities = [
        {"title": "Memory Search", "desc": "Searched for 'Meta ads performance last month'", "time": "2 minutes ago", "icon": "🔍", "color": COLORS['accent']},
        {"title": "Attribution Model", "desc": "Updated multi-touch attribution settings", "time": "15 minutes ago", "icon": "📊", "color": COLORS['primary']},
        {"title": "Campaign Launch", "desc": "Launched 'Q4 Holiday Campaign' with $50K budget", "time": "1 hour ago", "icon": "🚀", "color": COLORS['success']},
        {"title": "Data Sync", "desc": "Synced Google Ads performance data", "time": "3 hours ago", "icon": "🔄", "color": COLORS['warning']},
        {"title": "Insight Generated", "desc": "AI found 23% improvement opportunity in Facebook ads", "time": "5 hours ago", "icon": "💡", "color": COLORS['secondary']},
    ]
    
    activity_html = ""
    for activity in activities:
        activity_html += f"""
        <div class="activity-item">
            <div class="activity-icon" style="background: {activity['color']}20; color: {activity['color']};">
                {activity['icon']}
            </div>
            <div class="activity-content">
                <div class="activity-title">{activity['title']}</div>
                <div>{activity['desc']}</div>
                <div class="activity-time">{activity['time']}</div>
            </div>
        </div>
        """
    
    st.markdown(f'<div class="activity-feed">{activity_html}</div>', unsafe_allow_html=True)

elif selected == "Analytics":
    # Quick Stats
    col1, col2, col3, col4 = st.columns(4)
    
    stats = [
        {"value": "14", "label": "Active Campaigns", "change": "+3 this week"},
        {"value": "223", "label": "Results Indexed", "change": "+45 today"},
        {"value": "98.2%", "label": "System Uptime", "change": "All green"},
        {"value": "3.2x", "label": "Avg ROAS", "change": "+0.4 vs last month"}
    ]
    
    for i, (col, stat) in enumerate(zip([col1, col2, col3, col4], stats)):
        with col:
            st.markdown(f"""
            <div class="widget-card">
                <div class="widget-title">{stat['value']}</div>
                <div style="color: #666; margin-bottom: 0.5rem;">{stat['label']}</div>
                <div style="color: {COLORS['success']}; font-size: 0.8rem;">{stat['change']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Progress Tracking
    st.markdown("### 📈 Campaign Progress")
    
    campaigns = [
        {"name": "Q4 Holiday Campaign", "progress": 75, "budget": "$50K", "spent": "$37.5K"},
        {"name": "Brand Awareness Push", "progress": 45, "budget": "$25K", "spent": "$11.2K"},
        {"name": "Retargeting Optimization", "progress": 90, "budget": "$15K", "spent": "$13.5K"},
    ]
    
    for campaign in campaigns:
        st.markdown(f"""
        <div class="widget-card">
            <div class="widget-title">{campaign['name']}</div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span>Budget: {campaign['budget']}</span>
                <span>Spent: {campaign['spent']}</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {campaign['progress']}%;"></div>
            </div>
            <div style="text-align: right; margin-top: 0.5rem; color: #666; font-size: 0.9rem;">
                {campaign['progress']}% complete
            </div>
        </div>
        """, unsafe_allow_html=True)

elif selected == "Memory":
    st.markdown("### 🧠 Memory Search Hub")
    
    # Search Interface
    search_query = st.text_input("🔍 Search your marketing intelligence...", placeholder="e.g., 'Facebook ads performance Q3 2024'")
    
    if search_query:
        st.markdown(f"""
        <div class="widget-card">
            <div class="widget-title">🎯 Search Results for: "{search_query}"</div>
            <div style="margin-top: 1rem;">
                <div style="padding: 1rem; background: #f8f9fa; border-radius: 0.5rem; margin-bottom: 1rem;">
                    <strong>📊 Campaign Performance Report</strong><br>
                    <span style="color: #666;">Facebook Ads Q3 2024 showed 23% improvement in CTR with video creative formats...</span><br>
                    <span style="color: {COLORS['primary']}; font-size: 0.8rem;">Relevance: 95% • 2 days ago</span>
                </div>
                <div style="padding: 1rem; background: #f8f9fa; border-radius: 0.5rem; margin-bottom: 1rem;">
                    <strong>🎯 Attribution Analysis</strong><br>
                    <span style="color: #666;">Multi-touch attribution data for Q3 reveals Facebook's role in customer journey...</span><br>
                    <span style="color: {COLORS['primary']}; font-size: 0.8rem;">Relevance: 87% • 1 week ago</span>
                </div>
                <div style="padding: 1rem; background: #f8f9fa; border-radius: 0.5rem;">
                    <strong>💡 Optimization Insights</strong><br>
                    <span style="color: #666;">AI-generated recommendations for Facebook ad optimization based on historical data...</span><br>
                    <span style="color: {COLORS['primary']}; font-size: 0.8rem;">Relevance: 82% • 3 weeks ago</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Saved Searches
    st.markdown("### 💾 Saved Searches")
    
    saved_searches = [
        {"query": "Google Ads conversion tracking", "results": 15, "last_run": "Yesterday"},
        {"query": "Attribution model comparison", "results": 8, "last_run": "3 days ago"},
        {"query": "TikTok campaign performance", "results": 23, "last_run": "1 week ago"},
    ]
    
    for search in saved_searches:
        st.markdown(f"""
        <div class="widget-card">
            <div style="display: flex; justify-content: between; align-items: center;">
                <div>
                    <div class="widget-title">{search['query']}</div>
                    <div style="color: #666; font-size: 0.9rem;">{search['results']} results • Last run: {search['last_run']}</div>
                </div>
                <div class="notification-badge">{search['results']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

elif selected == "Modules":
    st.markdown("### 🧩 Module Management")
    
    # Module toggles with enhanced UI
    modules = [
        {"name": "Surfacing", "desc": "Data surfacing & insights discovery", "enabled": True, "icon": "🌊"},
        {"name": "Causal", "desc": "Causal inference & modeling", "enabled": True, "icon": "🔗"},
        {"name": "LLM", "desc": "AI-powered analysis & recommendations", "enabled": False, "icon": "🤖"},
        {"name": "Agentic", "desc": "Autonomous marketing agents", "enabled": False, "icon": "🎯"},
    ]
    
    col1, col2 = st.columns(2)
    
    for i, module in enumerate(modules):
        col = col1 if i % 2 == 0 else col2
        with col:
            status = "ACTIVE" if module["enabled"] else "AVAILABLE"
            status_color = COLORS['success'] if module["enabled"] else COLORS['warning']
            
            st.markdown(f"""
            <div class="widget-card">
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                    <span style="font-size: 2rem; margin-right: 1rem;">{module['icon']}</span>
                    <div>
                        <div class="widget-title">{module['name']}</div>
                        <div style="color: {status_color}; font-size: 0.9rem; font-weight: 600;">{status}</div>
                    </div>
                </div>
                <div style="color: #666; margin-bottom: 1rem;">{module['desc']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"{'🔥 Launch' if module['enabled'] else '⚡ Enable'} {module['name']}", key=f"module_{i}"):
                st.success(f"🎉 {module['name']} {'launched' if module['enabled'] else 'enabled'}!")

else:  # Settings
    st.markdown("### ⚙️ Settings")
    
    # Settings sections
    st.markdown("#### 🎨 Appearance")
    theme = st.selectbox("Theme", ["Gen-Z Neon", "Dark Mode", "Classic"])
    
    st.markdown("#### 🔔 Notifications")
    email_notifications = st.checkbox("Email notifications", value=True)
    push_notifications = st.checkbox("Push notifications", value=True)
    
    st.markdown("#### 🔒 Privacy")
    data_sharing = st.checkbox("Allow anonymous usage analytics", value=False)
    
    if st.button("💾 Save Settings"):
        st.success("🎉 Settings saved successfully!")

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🔥 Built for the next generation of marketers • 🚀 Powered by LiftOS</p>
</div>
""", unsafe_allow_html=True) 