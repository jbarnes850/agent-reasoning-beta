"""Tests for core functionality."""

import pytest
from unittest.mock import Mock, patch
import json

from src.core.models import ModelConfig
from src.core.agents import Agent
from src.core.reasoning import ReasoningEngine
from src.core.types import AgentResponse, Confidence

@pytest.mark.asyncio
async def test_agent_initialization(mock_env, mock_config):
    """Test agent initialization."""
    config = ModelConfig(**mock_config["model"])
    agent = Agent(config)
    assert agent.model_config == config
    assert agent.provider == mock_config["model"]["provider"]

@pytest.mark.asyncio
async def test_reasoning_engine(mock_env, mock_config):
    """Test reasoning engine initialization and basic functionality."""
    engine = ReasoningEngine()
    assert engine is not None
    
    # Test confidence calculation
    confidence = Confidence(
        score=0.8,
        reasoning="Test reasoning",
        evidence=["Test evidence"]
    )
    assert confidence.score == 0.8
    assert confidence.reasoning == "Test reasoning"
    
@pytest.mark.asyncio
async def test_agent_response_validation():
    """Test agent response validation."""
    # Valid response
    response = AgentResponse(
        content="Test response",
        confidence=Confidence(
            score=0.9,
            reasoning="High confidence due to test",
            evidence=["Test evidence"]
        ),
        metadata={
            "model": "test-model",
            "tokens": 100
        }
    )
    assert response.content == "Test response"
    assert response.confidence.score == 0.9
    
    # Invalid confidence score
    with pytest.raises(ValueError):
        AgentResponse(
            content="Test response",
            confidence=Confidence(
                score=1.5,  # Invalid: > 1.0
                reasoning="Invalid confidence",
                evidence=[]
            ),
            metadata={}
        )
