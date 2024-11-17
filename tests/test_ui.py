"""Tests for UI components."""

import pytest
from unittest.mock import Mock, patch
import streamlit as st
from src.ui.pages import Playground, Analytics, Settings
from src.ui.state.session import SessionState
from src.core.types import AgentRole, ModelProvider

@pytest.fixture
def mock_st():
    """Mock Streamlit components."""
    with patch("streamlit.title") as mock_title, \
         patch("streamlit.sidebar") as mock_sidebar, \
         patch("streamlit.columns") as mock_columns:
        yield {
            "title": mock_title,
            "sidebar": mock_sidebar,
            "columns": mock_columns
        }

@pytest.fixture
def mock_session_state():
    """Mock session state."""
    return SessionState()

def test_playground_initialization(mock_st, mock_session_state):
    """Test playground page initialization."""
    with patch("src.ui.pages.1_Playground.st.session_state", mock_session_state):
        playground = Playground()
        assert playground is not None
        mock_st["title"].assert_called_once()

def test_analytics_metrics(mock_st, mock_session_state, mock_config):
    """Test analytics metrics calculation and display."""
    with patch("src.ui.pages.2_Analytics.st.session_state", mock_session_state):
        analytics = Analytics()
        
        # Test metric calculation
        metrics = analytics.calculate_metrics()
        assert "total_experiments" in metrics
        assert "avg_confidence" in metrics
        assert "success_rate" in metrics

def test_settings_form(mock_st, mock_session_state, mock_config):
    """Test settings form validation and saving."""
    with patch("src.ui.pages.3_Settings.st.session_state", mock_session_state):
        settings = Settings()
        
        # Test API key validation
        valid_key = "test-valid-key"
        assert settings.validate_api_key(valid_key, ModelProvider.GROQ)
        
        # Test model configuration
        model_config = settings.get_model_config(ModelProvider.GROQ)
        assert model_config.provider == ModelProvider.GROQ
        assert model_config.temperature <= 1.0

@pytest.mark.asyncio
async def test_experiment_creation(mock_st, mock_session_state, mock_config):
    """Test experiment creation and configuration."""
    with patch("src.ui.pages.1_Playground.st.session_state", mock_session_state):
        playground = Playground()
        
        # Test experiment configuration
        config = {
            "name": "Test Experiment",
            "description": "Test description",
            "agent_roles": [AgentRole.RESEARCHER, AgentRole.CRITIC],
            "model_provider": ModelProvider.GROQ
        }
        
        experiment = await playground.create_experiment(config)
        assert experiment.name == config["name"]
        assert len(experiment.agents) == len(config["agent_roles"])

@pytest.mark.asyncio
async def test_visualization_update(mock_st, mock_session_state, mock_config):
    """Test visualization component updates."""
    with patch("src.ui.pages.1_Playground.st.session_state", mock_session_state):
        playground = Playground()
        
        # Test visualization update
        mock_data = {
            "nodes": [{"id": 1, "label": "Test"}],
            "edges": [{"from": 1, "to": 2}]
        }
        
        updated = playground.update_visualization(mock_data)
        assert updated
        # Verify visualization state
        assert mock_session_state.visualization_data == mock_data
