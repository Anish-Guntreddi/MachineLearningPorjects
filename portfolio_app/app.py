"""
ML Portfolio - Streamlit App
Entry point: page config, CSS loading, sidebar navigation.
"""
import streamlit as st
from pathlib import Path

# Page config must be first Streamlit command
st.set_page_config(
    page_title="ML Portfolio",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load CSS
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.markdown("### ML Portfolio")
    st.markdown("12 Projects | Deep Learning")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["Home", "Project Detail", "Compare"],
        label_visibility="collapsed",
    )

# Route to pages
if page == "Home":
    from pages import home
    home.render()
elif page == "Project Detail":
    from pages import project
    project.render()
elif page == "Compare":
    from pages import compare
    compare.render()
