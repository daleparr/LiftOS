import streamlit as st
from streamlit_sortable import sortable

st.set_page_config(page_title="Gen-Z Layout 3", page_icon="🧩", layout="wide")
widgets = [
    {"title": "Core Analytics", "content": "14 Active Campaigns"},
    {"title": "Memory Search", "content": "223 Results Indexed"},
    {"title": "Surfacing", "content": "Ready to use!"},
    {"title": "Causal", "content": "Try the new features!"},
]
order = sortable("dashboard", [w["title"] for w in widgets])
for i in order:
    w = next(x for x in widgets if x["title"] == i)
    st.markdown(f"### {w['title']}\n{w['content']}")

if st.button("➕ Add Widget"):
    st.info("Widget picker coming soon!") 