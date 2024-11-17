"""Test configuration and fixtures for the agent-reasoning-beta package."""

import pytest
from pathlib import Path
import os
import numpy as np
from datetime import datetime, timedelta
import sys

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.types import AgentRole, ReasoningType, ThoughtNode, ConsensusMetrics

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

@pytest.fixture
def mock_agents():
    """Mock agent data for testing."""
    return [
        {"id": "agent1", "role": AgentRole.EXPLORER, "confidence": 0.9},
        {"id": "agent2", "role": AgentRole.VERIFIER, "confidence": 0.8},
        {"id": "agent3", "role": AgentRole.COORDINATOR, "confidence": 0.85}
    ]

@pytest.fixture
def mock_interactions():
    """Mock agent interactions for testing."""
    return [
        {"source": "agent1", "target": "agent2", "type": ReasoningType.EXPLORATION},
        {"source": "agent2", "target": "agent3", "type": ReasoningType.VERIFICATION},
        {"source": "agent3", "target": "agent1", "type": ReasoningType.CONSENSUS}
    ]

@pytest.fixture
def mock_metrics():
    """Mock performance metrics for testing."""
    return {
        "agent1": {
            "success_rate": [0.8, 0.85, 0.9],
            "response_time": [1.2, 1.0, 0.9],
            "confidence": [0.7, 0.8, 0.9]
        },
        "agent2": {
            "success_rate": [0.75, 0.8, 0.85],
            "response_time": [1.1, 1.0, 0.8],
            "confidence": [0.8, 0.85, 0.9]
        }
    }

@pytest.fixture
def mock_consensus_metrics():
    """Mock consensus metrics for testing."""
    return ConsensusMetrics(
        agreement_matrix=np.array([[1.0, 0.8, 0.7],
                                 [0.8, 1.0, 0.9],
                                 [0.7, 0.9, 1.0]]),
        agent_ids=["agent1", "agent2", "agent3"],
        resolution_path=[
            ("agent1", 0.9),
            ("agent2", 0.85),
            ("agent3", 0.8)
        ]
    )

@pytest.fixture
def mock_thought_tree():
    """Mock thought tree for testing."""
    return ThoughtNode(
        id="root",
        content="Initial thought",
        confidence=0.9,
        children=[
            ThoughtNode(
                id="child1",
                content="First branch",
                confidence=0.8,
                children=[]
            ),
            ThoughtNode(
                id="child2",
                content="Second branch",
                confidence=0.7,
                children=[]
            )
        ]
    )

@pytest.fixture
def mock_resource_utilization():
    """Mock resource utilization metrics for testing."""
    return {
        "agent1": {
            "cpu": 0.45,
            "memory": 0.6,
            "tokens": 0.3
        },
        "agent2": {
            "cpu": 0.55,
            "memory": 0.4,
            "tokens": 0.5
        }
    }
