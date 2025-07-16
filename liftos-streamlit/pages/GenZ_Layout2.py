import streamlit as st
from streamlit_timeline import timeline
from streamlit_extras.badges import badge

st.set_page_config(page_title="Gen-Z Layout 2", page_icon="🔥", layout="wide")
st.markdown("## 👋 Welcome, demo_user!")
st.image("https://randomuser.me/api/portraits/men/32.jpg", width=80)
st.markdown("**Streak:** 🔥 7 days\n**Level:** 🏆 Pro Marketer")

st.markdown("### 🏅 Achievements")
badge(type="success", text="First Attribution Model")
badge(type="warning", text="7-Day Data Sync Streak")
badge(type="info", text="Campaign ROAS > 3.0")

st.markdown("### 🕒 Recent Activity")
timeline([
    {"title": "Memory Search", "content": "Meta ads performance last month", "date": "2024-07-01"},
    {"title": "Causal Analysis", "content": "Attribution run completed", "date": "2024-07-01"},
    {"title": "LLM", "content": "Suggest a meme for campaign", "date": "2024-06-30"},
]) 