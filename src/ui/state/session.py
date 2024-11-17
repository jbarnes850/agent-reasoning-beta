"""Session state management for the Agent Reasoning Beta platform."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import streamlit as st

from core.reasoning import ConsensusResult, MCTSNode, ReasoningPath, VerificationResult
from core.types import AgentRole, ReasoningType


@dataclass
class VisualizationState:
    """State for visualization settings."""

    zoom_level: float = 1.0
    pan_position: tuple[float, float] = (0.0, 0.0)
    selected_node: Optional[str] = None
    highlight_path: Optional[List[str]] = None
    color_scheme: Dict[str, str] = field(default_factory=dict)
    last_update: datetime = field(default_factory=datetime.now)


@dataclass
class ModelState:
    """State for model settings."""

    provider: str = "groq"
    model: str = "llama-3.1-70b-versatile"
    temperature: float = 0.7
    max_tokens: int = 1000
    cost_tracking: Dict[str, float] = field(default_factory=dict)
    last_call: datetime = field(default_factory=datetime.now)


class SessionState:
    """Manages global session state with cleanup and optimization."""

    def __init__(self):
        """Initialize session state with default values."""
        if "initialized" not in st.session_state:
            self._initialize_state()

        # Set up periodic cleanup
        if "last_cleanup" not in st.session_state:
            st.session_state.last_cleanup = datetime.now()

        self._maybe_cleanup_state()

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
                "mcts": {"node": "#1f77b4", "edge": "#7fcdbb"},
                "verification": {"node": "#2ecc71", "edge": "#a1d99b"},
                "consensus": {"node": "#9b59b6", "edge": "#bcbddc"},
            }
        )

        # Model state
        st.session_state.model_state = ModelState()

        # Cache state
        st.session_state.cache = {}
        st.session_state.cache_timestamps = {}

    def _maybe_cleanup_state(self):
        """Periodically clean up expired state."""
        now = datetime.now()
        if (now - st.session_state.last_cleanup) > timedelta(minutes=5):
            self._cleanup_state()
            st.session_state.last_cleanup = now

    def _cleanup_state(self):
        """Clean up expired or invalid state."""
        now = datetime.now()

        # Clean up visualization data older than 1 hour
        for key in list(st.session_state.keys()):
            if key.startswith("viz_"):
                state = st.session_state[key]
                if isinstance(state, VisualizationState):
                    if (now - state.last_update) > timedelta(hours=1):
                        del st.session_state[key]

        # Clean up cache entries older than 30 minutes
        expired_keys = [
            key
            for key, timestamp in st.session_state.cache_timestamps.items()
            if (now - timestamp) > timedelta(minutes=30)
        ]
        for key in expired_keys:
            if key in st.session_state.cache:
                del st.session_state.cache[key]
            del st.session_state.cache_timestamps[key]

        # Reset model state if no activity for 15 minutes
        if (now - st.session_state.model_state.last_call) > timedelta(minutes=15):
            st.session_state.model_state = ModelState()

    def cache_result(self, key: str, value: any) -> None:
        """Cache a result with timestamp."""
        st.session_state.cache[key] = value
        st.session_state.cache_timestamps[key] = datetime.now()

    def get_cached_result(self, key: str) -> Optional[any]:
        """Get a cached result if not expired."""
        if key in st.session_state.cache:
            timestamp = st.session_state.cache_timestamps.get(key)
            if timestamp and (datetime.now() - timestamp) <= timedelta(minutes=30):
                return st.session_state.cache[key]
        return None

    def update_visualization(self, key: str, data: any) -> None:
        """Update visualization state with timestamp."""
        if not isinstance(st.session_state.get(key), VisualizationState):
            st.session_state[key] = VisualizationState()

        state = st.session_state[key]
        for attr, value in data.items():
            if hasattr(state, attr):
                setattr(state, attr, value)
        state.last_update = datetime.now()

    def track_model_usage(self, cost: float) -> None:
        """Track model usage and costs."""
        state = st.session_state.model_state
        provider = state.provider

        if provider not in state.cost_tracking:
            state.cost_tracking[provider] = 0
        state.cost_tracking[provider] += cost
        state.last_call = datetime.now()

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
            st.session_state.metrics[metric_type].append(
                {"timestamp": datetime.now().isoformat(), "value": value}
            )

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
                "color_scheme": self.viz_state.color_scheme,
            },
            "model": {
                "provider": self.model_state.provider,
                "model": self.model_state.model,
                "temperature": self.model_state.temperature,
                "max_tokens": self.model_state.max_tokens,
                "cost_tracking": self.model_state.cost_tracking,
            },
            "metrics": st.session_state.metrics,
        }
