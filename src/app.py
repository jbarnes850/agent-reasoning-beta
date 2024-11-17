"""Main entry point for the Agent Reasoning Beta platform."""

import streamlit as st
from dotenv import load_dotenv
import os

from config.system import SystemConfig
from config.agent import AgentConfig
from config.model import ModelConfig
from config.visualization import VisualizationConfig
from visualization.layouts.main_layout import MainLayout
from visualization.layouts.dashboard_layout import DashboardLayout
from ui.state.session import SessionState

# Load environment variables
load_dotenv()

# Configure page settings
st.set_page_config(
    page_title="Agent Reasoning Beta",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_configurations():
    """Load all configuration objects."""
    return {
        "system": SystemConfig(),
        "agent": AgentConfig(),
        "model": ModelConfig(),
        "visualization": VisualizationConfig()
    }

def main():
    """Main application entry point."""
    try:
        # Load configurations
        configs = load_configurations()
        
        # Initialize session state
        session = SessionState()
        
        # Initialize layouts
        main_layout = MainLayout()
        dashboard_layout = DashboardLayout(session)
        
        # Render current page
        if session.current_page == "playground":
            main_layout.render(
                mcts_data=None,
                verification_data=None,
                consensus_data=None
            )
        elif session.current_page == "analytics":
            dashboard_layout.render()
        elif session.current_page == "settings":
            render_settings(configs, session)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        if os.getenv("DEBUG", "false").lower() == "true":
            st.exception(e)

def render_settings(configs: dict, session: SessionState):
    """Render settings page.
    
    Args:
        configs: Configuration objects
        session: Session state manager
    """
    st.title("Settings")
    
    # Model settings
    st.subheader("Model Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        provider = st.selectbox(
            "Model Provider",
            options=["groq", "openai", "anthropic"],
            index=["groq", "openai", "anthropic"].index(session.model_state.provider)
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=session.model_state.temperature,
            step=0.1
        )
    
    with col2:
        max_tokens = st.number_input(
            "Max Tokens",
            min_value=100,
            max_value=10000,
            value=session.model_state.max_tokens,
            step=100
        )
    
    # Update model state if changed
    if (provider != session.model_state.provider or
        temperature != session.model_state.temperature or
        max_tokens != session.model_state.max_tokens):
        session.model_state.provider = provider
        session.model_state.temperature = temperature
        session.model_state.max_tokens = max_tokens
    
    # Visualization settings
    st.markdown("---")
    st.subheader("Visualization Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        st.checkbox(
            "Dark Mode",
            value=session.dark_mode,
            key="dark_mode"
        )
        
        st.checkbox(
            "Auto Refresh",
            value=session.auto_refresh,
            key="auto_refresh"
        )
    
    with col2:
        if session.auto_refresh:
            st.slider(
                "Refresh Rate (s)",
                min_value=1,
                max_value=60,
                value=session.refresh_rate,
                key="refresh_rate"
            )
    
    # System settings
    st.markdown("---")
    st.subheader("System Settings")
    
    debug_mode = st.checkbox(
        "Debug Mode",
        value=os.getenv("DEBUG", "false").lower() == "true"
    )
    
    if debug_mode != (os.getenv("DEBUG", "false").lower() == "true"):
        os.environ["DEBUG"] = str(debug_mode).lower()

if __name__ == "__main__":
    main()
