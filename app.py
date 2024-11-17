"""Main entry point for the Agent Reasoning Beta platform."""

import streamlit as st
from dotenv import load_dotenv
import os

# Configuration imports
from src.config.system import SystemConfig
from src.config.agent import AgentConfig
from src.config.model import ModelConfig
from src.config.visualization import VisualizationConfig

# Core functionality imports
from src.core.agents import AgentManager
from src.core.reasoning import ReasoningEngine
from src.core.models import ModelProvider
from src.core.system import SystemManager
from src.core.tools import ToolRegistry

# Visualization imports
from src.visualization.layouts.main_layout import MainLayout
from src.visualization.layouts.dashboard_layout import DashboardLayout
from src.visualization.components.views.shared.graphs import GraphVisualizer
from src.visualization.components.views.shared.metrics import MetricsVisualizer

# UI and state management
from src.ui.state.session import SessionState

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

def initialize_components():
    """Initialize core application components."""
    if "core_components" not in st.session_state:
        st.session_state.core_components = {
            "model_provider": ModelProvider(),
            "agent_manager": AgentManager(),
            "reasoning_engine": ReasoningEngine(),
            "system_manager": SystemManager(),
            "tool_registry": ToolRegistry(),
            "graph_viz": GraphVisualizer(),
            "metrics_viz": MetricsVisualizer()
        }
    return st.session_state.core_components

def main():
    """Main application entry point."""
    try:
        # Load configurations
        if "configs" not in st.session_state:
            st.session_state.configs = load_configurations()
        
        # Initialize core components
        components = initialize_components()
        
        # Initialize session state if needed
        if "session" not in st.session_state:
            st.session_state.session = SessionState()
        
        # Store components in session state for pages to access
        st.session_state.update(components)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        if os.getenv("DEBUG", "false").lower() == "true":
            st.exception(e)

if __name__ == "__main__":
    main()
