"""
Main application entry point for the Agent Reasoning Beta platform.
"""

import streamlit as st

from pages.playground import render_playground
from pages.analytics import render_analytics
from pages.settings import render_settings

# Page configuration
st.set_page_config(
    page_title="Agent Reasoning Beta",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Navigation
pages = {
    "Playground": render_playground,
    "Analytics": render_analytics,
    "Settings": render_settings
}

# Sidebar navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(pages.keys()))

# Render selected page
pages[selection]()
