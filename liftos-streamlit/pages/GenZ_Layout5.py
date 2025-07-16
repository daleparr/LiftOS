import streamlit as st
from streamlit_carousel import carousel
from streamlit_extras.animated_number import animated_number

st.set_page_config(page_title="Gen-Z Layout 5", page_icon="🎡", layout="wide")
carousel(items=[
    {"title": "Welcome to LiftOS", "img": "https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=400&q=80", "caption": "The next-gen platform"},
    {"title": "14 Active Campaigns", "img": "https://images.unsplash.com/photo-1461749280684-dccba630e2f6?auto=format&fit=crop&w=400&q=80", "caption": "Core Analytics"},
    {"title": "223 Results Indexed", "img": "https://images.unsplash.com/photo-1519125323398-675f0ddb6308?auto=format&fit=crop&w=400&q=80", "caption": "Memory Search"},
])
st.progress(0.7, text="Onboarding Progress: 70%")
st.markdown("### How do you feel about LiftOS today?")
if st.button("😎"): st.success("You rock!")
if st.button("🔥"): st.balloons()
if st.button("💡"): st.info("Thanks for your feedback!") 