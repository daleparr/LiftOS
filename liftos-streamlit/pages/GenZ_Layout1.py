import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_extras.animated_number import animated_number
from streamlit_card import card
import requests

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

st.set_page_config(page_title="Gen-Z Layout 1", page_icon="🚀", layout="wide")
st_lottie(load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_9cyyl8i4.json"), height=180, key="hero")
st.markdown("<h1 style='text-align:center; color:#FF006E;'>Welcome to LiftOS 🚀</h1>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("### Active Campaigns")
    animated_number("active_campaigns", value=14, format="{:d}")
with col2:
    st.markdown("### Results Indexed")
    animated_number("results_indexed", value=223, format="{:d}")

st.markdown("### Modules")
mod_names = ["Surfacing", "Causal", "LLM Assistant", "Agentic"]
for i, mod in enumerate(mod_names):
    card(title=mod, text="Ready to use!", url="#", key=f"mod_{i}")

st.success("🎉 First Attribution Model Created\n🔥 7-Day Data Sync Streak\n🚀 Campaign ROAS > 3.0") 