"""Tests for UI components."""

from unittest.mock import Mock, patch

import pytest
import streamlit as st
from agent_reasoning_beta.core.types import AgentResponse, Confidence
from agent_reasoning_beta.ui.pages.analytics import Analytics
from agent_reasoning_beta.ui.pages.playground import Playground
from agent_reasoning_beta.ui.pages.settings import Settings


@pytest.fixture
def mock_st():
    """Mock Streamlit components."""
    with (
        patch("streamlit.title") as mock_title,
        patch("streamlit.sidebar") as mock_sidebar,
        patch("streamlit.columns") as mock_columns,
    ):
        yield {"title": mock_title, "sidebar": mock_sidebar, "columns": mock_columns}


@pytest.fixture
def mock_session_state():
    """Mock session state."""
    return SessionState()


def test_playground_initialization(mock_st, mock_session_state):
    """Test playground page initialization."""
    with patch(
        "agent_reasoning_beta.ui.pages.playground.st.session_state", mock_session_state
    ):
        playground = Playground()
        assert playground is not None
        assert hasattr(playground, "agent_selector")
        assert hasattr(playground, "query_input")
        assert hasattr(playground, "response_display")
        mock_st["title"].assert_called_once()


@pytest.mark.asyncio
async def test_playground_interaction(mock_st, mock_session_state):
    """Test playground interaction flow."""
    with patch(
        "agent_reasoning_beta.ui.pages.playground.st.session_state", mock_session_state
    ):
        playground = Playground()

        # Mock agent response
        mock_response = AgentResponse(
            content="Test response",
            confidence=Confidence.HIGH,
            metadata={"test": "data"},
        )

        with patch(
            "agent_reasoning_beta.core.agents.Agent.process_query",
            return_value=mock_response,
        ):
            response = await playground.process_user_query("test query")
            assert response.content == "Test response"
            assert response.confidence == Confidence.HIGH


def test_analytics_page(mock_st, mock_session_state):
    """Test analytics page functionality."""
    with patch(
        "agent_reasoning_beta.ui.pages.analytics.st.session_state", mock_session_state
    ):
        analytics = Analytics()
        assert analytics is not None
        assert hasattr(analytics, "metrics_display")
        assert hasattr(analytics, "visualization_options")

        # Test visualization components
        assert analytics.has_visualization("time_series")
        assert analytics.has_visualization("heatmap")
        assert analytics.has_visualization("scatter")


def test_settings_page(mock_st, mock_session_state):
    """Test settings page functionality."""
    with patch(
        "agent_reasoning_beta.ui.pages.settings.st.session_state", mock_session_state
    ):
        settings = Settings()
        assert settings is not None
        assert hasattr(settings, "model_settings")
        assert hasattr(settings, "api_settings")

        # Test settings form
        form = settings.get_settings_form()
        assert "model" in form
        assert "api_keys" in form

        # Test API key validation
        assert settings.validate_api_key("test-key", "groq")
        assert not settings.validate_api_key("invalid-key", "groq")


def test_responsive_layout(mock_st, mock_session_state):
    """Test UI responsive layout."""
    with patch(
        "agent_reasoning_beta.ui.pages.playground.st.session_state", mock_session_state
    ):
        playground = Playground()
        analytics = Analytics()
        settings = Settings()

        # Test layout configurations
        assert playground.is_responsive()
        assert analytics.is_responsive()
        assert settings.is_responsive()

        # Test mobile layout
        assert playground.get_mobile_layout() is not None
        assert analytics.get_mobile_layout() is not None
        assert settings.get_mobile_layout() is not None


def test_theme_customization(mock_st, mock_session_state):
    """Test UI theme customization."""
    with patch(
        "agent_reasoning_beta.ui.pages.playground.st.session_state", mock_session_state
    ):
        playground = Playground()

        # Test theme switching
        playground.set_theme("dark")
        assert playground.get_current_theme() == "dark"

        playground.set_theme("light")
        assert playground.get_current_theme() == "light"

        # Test custom theme
        custom_theme = {
            "primary_color": "#FF0000",
            "secondary_color": "#00FF00",
            "font_family": "Arial",
        }
        playground.set_custom_theme(custom_theme)
        assert playground.get_current_theme() == "custom"


def test_playground_page():
    """Test playground page functionality."""
    playground = Playground()

    # Mock streamlit container
    container = Mock()

    # Test page rendering
    playground.render(container)
    container.title.assert_called_once()

    # Test input handling
    with patch("streamlit.text_area") as mock_input:
        mock_input.return_value = "Test input"
        assert playground.get_user_input() == "Test input"


def test_analytics_page():
    """Test analytics page functionality."""
    analytics = Analytics()

    # Mock data
    metrics = {
        "response_time": [1.2, 1.0, 1.5],
        "success_rate": [1, 1, 0],
        "confidence": [0.8, 0.9, 0.7],
    }

    # Test metrics display
    container = Mock()
    analytics.display_metrics(metrics, container)
    container.plotly_chart.assert_called()


def test_settings_page():
    """Test settings page functionality."""
    settings = Settings()

    # Test default settings
    assert settings.get_default_model() == "gpt-4"
    assert settings.get_temperature() == 0.7

    # Test settings update
    settings.update_model("claude-3")
    assert settings.get_default_model() == "claude-3"

    settings.update_temperature(0.9)
    assert settings.get_temperature() == 0.9
