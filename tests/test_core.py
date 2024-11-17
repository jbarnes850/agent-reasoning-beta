"""Tests for core functionality."""

import pytest
from unittest.mock import Mock, patch
import json

from agent_reasoning_beta.core.models import ModelConfig
from agent_reasoning_beta.core.agents import Agent
from agent_reasoning_beta.core.reasoning import ReasoningEngine
from agent_reasoning_beta.core.types import AgentResponse, Confidence

@pytest.mark.asyncio
async def test_agent_initialization(mock_env, mock_config):
    """Test agent initialization."""
    config = ModelConfig(**mock_config["model"])
    agent = Agent(config)
    assert agent.model_config == config
    assert agent.provider == mock_config["model"]["provider"]

@pytest.mark.asyncio
async def test_reasoning_engine(mock_env, mock_config):
    """Test reasoning engine functionality."""
    config = ModelConfig(**mock_config["model"])
    engine = ReasoningEngine(config)
    
    response = AgentResponse(
        content="Test response",
        confidence=Confidence.HIGH,
        metadata={"test": "data"}
    )
    
    with patch.object(engine, '_process_response', return_value=response):
        result = await engine.process_query("test query")
        assert result == response
        assert result.confidence == Confidence.HIGH
        assert result.metadata["test"] == "data"

@pytest.mark.asyncio
async def test_agent_response_validation():
    """Test agent response validation."""
    config = ModelConfig(
        name="test",
        provider="test",
        model_id="test-model",
        temperature=0.7
    )
    engine = ReasoningEngine(config)
    
    # Test invalid response
    with pytest.raises(ValueError):
        await engine.process_response(None)
        
    # Test empty content
    with pytest.raises(ValueError):
        response = AgentResponse(
            content="",
            confidence=Confidence.LOW,
            metadata={}
        )
        await engine.process_response(response)
