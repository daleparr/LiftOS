import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

with st.sidebar:
    selected = option_menu(
        "Menu", ["Dashboard", "Surfacing", "Causal", "LLM", "Agentic"],
        icons=["house", "water", "puzzle", "robot", "activity"],
        menu_icon="cast", default_index=0,
        styles={"container": {"background": "linear-gradient(90deg, #3A86FF 0%, #FF006E 100%)"}}
    )
st_lottie(load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_9cyyl8i4.json"), height=100, key="sidebar_anim")
st.markdown(f"## {selected} Module")
st.info("Animated microinteractions on every action!") 