"""Session state management for the Agent Reasoning Beta platform."""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime

import streamlit as st

from core.types import ReasoningType, AgentRole
from core.reasoning import MCTSNode, ReasoningPath, VerificationResult, ConsensusResult


@dataclass
class VisualizationState:
    """State for visualization settings."""
    zoom_level: float = 1.0
    pan_position: tuple[float, float] = (0.0, 0.0)
    selected_node: Optional[str] = None
    highlight_path: Optional[List[str]] = None
    color_scheme: Dict[str, str] = None

@dataclass
class ModelState:
    """State for model settings."""
    provider: str = "groq"
    model: str = "llama-3.1-70b-versatile"
    temperature: float = 0.7
    max_tokens: int = 1000
    cost_tracking: Dict[str, float] = None

class SessionState:
    """Manages global session state for the application."""
    
    def __init__(self):
        """Initialize session state with default values."""
        if "initialized" not in st.session_state:
            self._initialize_state()
    
    def _initialize_state(self):
        """Initialize all session state variables."""
        # Core state
        st.session_state.initialized = True
        st.session_state.current_page = "playground"
        st.session_state.dark_mode = False
        st.session_state.auto_refresh = True
        st.session_state.refresh_rate = 5
        
        # Visualization state
        st.session_state.viz_state = VisualizationState(
            color_scheme={
                "mcts": {
                    "node": "#1f77b4",
                    "edge": "#7fcdbb",
                    "text": "#2c3e50"
                },
                "verification": {
                    "node": "#2ca02c",
                    "edge": "#98df8a",
                    "text": "#2c3e50"
                },
                "consensus": {
                    "node": "#ff7f0e",
                    "edge": "#ffbb78",
                    "text": "#2c3e50"
                }
            }
        )
        
        # Model state
        st.session_state.model_state = ModelState(
            cost_tracking={
                "groq": 0.0,
                "openai": 0.0,
                "anthropic": 0.0
            }
        )
        
        # Analytics state
        st.session_state.metrics = {
            "success_rate": [],
            "confidence_scores": [],
            "response_times": [],
            "error_counts": {}
        }
    
    @property
    def current_page(self) -> str:
        """Get current page."""
        return st.session_state.current_page
    
    @current_page.setter
    def current_page(self, value: str):
        """Set current page."""
        st.session_state.current_page = value
    
    @property
    def viz_state(self) -> VisualizationState:
        """Get visualization state."""
        return st.session_state.viz_state
    
    @property
    def model_state(self) -> ModelState:
        """Get model state."""
        return st.session_state.model_state
    
    def update_cost(self, provider: str, cost: float):
        """Update cost tracking for a provider."""
        self.model_state.cost_tracking[provider] += cost
    
    def log_metric(self, metric_type: str, value: float):
        """Log a metric value."""
        if metric_type in st.session_state.metrics:
            st.session_state.metrics[metric_type].append({
                "timestamp": datetime.now().isoformat(),
                "value": value
            })
    
    def reset_visualization(self):
        """Reset visualization state to defaults."""
        self.viz_state.zoom_level = 1.0
        self.viz_state.pan_position = (0.0, 0.0)
        self.viz_state.selected_node = None
        self.viz_state.highlight_path = None
    
    def export_state(self) -> dict:
        """Export current session state as dictionary."""
        return {
            "current_page": self.current_page,
            "dark_mode": st.session_state.dark_mode,
            "auto_refresh": st.session_state.auto_refresh,
            "refresh_rate": st.session_state.refresh_rate,
            "visualization": {
                "zoom_level": self.viz_state.zoom_level,
                "pan_position": self.viz_state.pan_position,
                "color_scheme": self.viz_state.color_scheme
            },
            "model": {
                "provider": self.model_state.provider,
                "model": self.model_state.model,
                "temperature": self.model_state.temperature,
                "max_tokens": self.model_state.max_tokens,
                "cost_tracking": self.model_state.cost_tracking
            },
            "metrics": st.session_state.metrics
        }
