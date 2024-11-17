"""Test configuration and fixtures for the agent-reasoning-beta package."""

import pytest
from pathlib import Path
import os

@pytest.fixture(scope="session")
def test_data_dir():
    """Get the test data directory."""
    return Path(__file__).parent / "data"

@pytest.fixture(scope="session")
def mock_env(monkeypatch):
    """Mock environment variables for testing."""
    env_vars = {
        "GROQ_API_KEY": "test-groq-key",
        "OPENAI_API_KEY": "test-openai-key",
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "TAVILY_API_KEY": "test-tavily-key",
        "ENVIRONMENT": "test",
        "LOG_LEVEL": "DEBUG",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars

@pytest.fixture(scope="session")
def mock_config():
    """Mock configuration for testing."""
    return {
        "model": {
            "provider": "groq",
            "name": "llama-3.1-70b-versatile",
            "temperature": 0.7,
            "max_tokens": 1000
        },
        "visualization": {
            "max_tree_depth": 5,
            "max_nodes_display": 100,
            "confidence_threshold": 0.7
        }
    }
