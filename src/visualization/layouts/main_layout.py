"""Main layout component for the Agent Reasoning Beta platform."""

import streamlit as st
from typing import Dict, List, Optional
from datetime import datetime

from core.types import ReasoningType, AgentRole
from core.reasoning import MCTSNode, ReasoningPath, VerificationResult, ConsensusResult
from visualization.components.exploration_view import ExplorationView
from visualization.components.verification_view import VerificationView
from visualization.components.consensus_view import ConsensusView
from ui.state.session import SessionState


class MainLayout:
    """Main layout manager for the visualization platform."""

    def __init__(self):
        """Initialize main layout component."""
        # Initialize session state if not exists
        self.session = SessionState()
        
        # Initialize view components
        self.exploration_view = ExplorationView()
        self.verification_view = VerificationView()
        self.consensus_view = ConsensusView()
    
    def render_sidebar(self):
        """Render the application sidebar with navigation and settings."""
        with st.sidebar:
            st.title(" AgentViz")
            st.markdown("---")
            
            # Navigation
            st.subheader("Navigation")
            page = st.radio(
                "Select Page",
                options=["playground", "analytics", "settings"],
                format_func=lambda x: x.title(),
                key="current_page"
            )
            
            st.markdown("---")
            
            # Model selection
            st.subheader("Model Settings")
            provider = st.selectbox(
                "Provider",
                options=["groq", "openai", "anthropic"],
                index=["groq", "openai", "anthropic"].index(self.session.model_state.provider),
                key="model_provider"
            )
            
            # Show provider-specific models
            models = {
                "groq": ["llama-3.1-70b-versatile"],
                "openai": ["gpt-4o", "gpt-4o-mini"],
                "anthropic": ["claude-3-5-sonnet-latest", "claude-3-5-haiku-latest"]
            }
            
            model = st.selectbox(
                "Model",
                options=models[provider],
                key="model_name"
            )
            
            if provider != self.session.model_state.provider or model != self.session.model_state.model:
                self.session.model_state.provider = provider
                self.session.model_state.model = model
            
            st.markdown("---")
            
            # Settings
            st.subheader("Settings")
            
            # Theme toggle
            dark_mode = st.toggle(
                "Dark Mode",
                value=self.session.dark_mode,
                key="dark_mode",
                help="Toggle dark/light theme"
            )
            
            # Auto-refresh toggle
            auto_refresh = st.toggle(
                "Auto Refresh",
                value=self.session.auto_refresh,
                key="auto_refresh",
                help="Automatically refresh visualizations"
            )
            
            if auto_refresh:
                refresh_rate = st.slider(
                    "Refresh Rate (s)",
                    min_value=1,
                    max_value=60,
                    value=5,
                    help="How often to refresh the visualizations",
                    key="refresh_rate"
                )
            
            st.markdown("---")
            
            # System status
            st.subheader("System Status")
            status_container = st.container()
            
            # Update status periodically if auto-refresh is enabled
            if auto_refresh:
                with status_container:
                    self._render_system_status()
    
    def _render_system_status(self):
        """Render system status indicators."""
        # Cost tracking
        total_cost = sum(self.session.model_state.cost_tracking.values())
        st.metric("Total API Cost", f"${total_cost:.4f}")
        
        # Performance metrics
        if self.session.metrics["response_times"]:
            times = [t["value"] for t in self.session.metrics["response_times"]]
            avg_time = sum(times) / len(times)
            st.metric("Avg Response Time", f"{avg_time:.2f}s")
        
        if self.session.metrics["success_rate"]:
            rates = [r["value"] for r in self.session.metrics["success_rate"]]
            avg_rate = sum(rates) / len(rates)
            st.metric("Success Rate", f"{avg_rate:.1%}")
    
    def _apply_theme(self):
        """Apply the selected theme using custom CSS."""
        if self.session.dark_mode:
            st.markdown("""
                <style>
                    .stApp {
                        background-color: #1E1E1E;
                        color: #FFFFFF;
                    }
                    .stMarkdown {
                        color: #FFFFFF;
                    }
                    .stMetric {
                        background-color: #2D2D2D;
                    }
                    .stButton > button {
                        background-color: #4A4A4A;
                        color: #FFFFFF;
                    }
                    .stSelectbox > div > div {
                        background-color: #2D2D2D;
                        color: #FFFFFF;
                    }
                </style>
                """, unsafe_allow_html=True)
    
    def render_header(self):
        """Render the main header with global controls."""
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.title("Agent Reasoning Playground")
            st.markdown("""
                Explore and analyze multi-agent reasoning processes through
                interactive visualizations and real-time metrics.
            """)
        
        with col2:
            # Global controls
            if st.button("Reset View"):
                self.session.reset_visualization()
                st.experimental_rerun()
        
        with col3:
            # Export controls
            st.download_button(
                "Export Data",
                data=str(self.session.export_state()),
                file_name=f"agent_reasoning_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                help="Download all visualization data as JSON"
            )
    
    def render_visualization_controls(self):
        """Render visualization control panel."""
        st.sidebar.markdown("---")
        st.sidebar.subheader("Visualization Controls")
        
        # View selection
        view = st.sidebar.radio(
            "Select View",
            options=["exploration", "verification", "consensus"],
            format_func=lambda x: x.title()
        )
        
        # Zoom controls
        zoom = st.sidebar.slider(
            "Zoom Level",
            min_value=0.1,
            max_value=2.0,
            value=self.session.viz_state.zoom_level,
            step=0.1
        )
        
        if zoom != self.session.viz_state.zoom_level:
            self.session.viz_state.zoom_level = zoom
        
        # Pan controls
        col1, col2 = st.sidebar.columns(2)
        with col1:
            pan_x = st.number_input(
                "Pan X",
                value=self.session.viz_state.pan_position[0],
                step=10.0
            )
        with col2:
            pan_y = st.number_input(
                "Pan Y",
                value=self.session.viz_state.pan_position[1],
                step=10.0
            )
        
        if (pan_x, pan_y) != self.session.viz_state.pan_position:
            self.session.viz_state.pan_position = (pan_x, pan_y)
    
    def render(self, 
               mcts_data: Optional[MCTSNode] = None,
               verification_data: Optional[List[VerificationResult]] = None,
               consensus_data: Optional[List[ConsensusResult]] = None):
        """Render the complete application layout.
        
        Args:
            mcts_data: Optional MCTS data for exploration view
            verification_data: Optional verification results for verification view
            consensus_data: Optional consensus results for consensus view
        """
        # Apply theme
        self._apply_theme()
        
        # Render sidebar
        self.render_sidebar()
        
        # Render header
        self.render_header()
        
        # Render visualization controls
        self.render_visualization_controls()
        
        # Main content area
        main_container = st.container()
        with main_container:
            # Render selected view
            if st.session_state.current_page == "playground":
                if mcts_data:
                    self.exploration_view.render(
                        mcts_data,
                        zoom_level=self.session.viz_state.zoom_level,
                        pan_position=self.session.viz_state.pan_position
                    )
                elif verification_data:
                    self.verification_view.render(
                        verification_data,
                        zoom_level=self.session.viz_state.zoom_level,
                        pan_position=self.session.viz_state.pan_position
                    )
                elif consensus_data:
                    self.consensus_view.render(
                        consensus_data,
                        zoom_level=self.session.viz_state.zoom_level,
                        pan_position=self.session.viz_state.pan_position
                    )
                else:
                    st.info("No data available for the selected view")
            elif st.session_state.current_page == "analytics":
                st.info("Analytics page is under development")
            elif st.session_state.current_page == "settings":
                st.info("Settings page is under development")
