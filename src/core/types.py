"""
Core type definitions for the Agent Reasoning Beta platform.

This module defines the fundamental types and data structures used throughout the system,
including agent states, reasoning primitives, and visualization data structures.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ReasoningType(Enum):
    """Enumeration of supported reasoning primitives."""
    MCTS = auto()  # Monte Carlo Tree Search exploration
    VERIFICATION = auto()  # Verification and validation
    CONSENSUS = auto()  # Multi-agent consensus building


class AgentRole(Enum):
    """Defines the possible roles an agent can take in the system."""
    EXPLORER = auto()  # Agents that perform MCTS exploration
    VERIFIER = auto()  # Agents that verify reasoning paths
    COORDINATOR = auto()  # Agents that coordinate consensus building
    OBSERVER = auto()  # Agents that monitor and log system behavior


class ModelProvider(Enum):
    """Supported LLM providers."""
    GROQ = auto()
    OPENAI = auto()
    ANTHROPIC = auto()
    LOCAL = auto()


class ThoughtNode(BaseModel):
    """Represents a single node in the reasoning tree."""
    id: UUID = Field(default_factory=uuid4)
    parent_id: Optional[UUID] = None
    content: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_type: ReasoningType
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        frozen = True


class ReasoningPath(BaseModel):
    """Represents a complete path of reasoning from root to leaf."""
    id: UUID = Field(default_factory=uuid4)
    nodes: List[ThoughtNode]
    total_confidence: float = Field(ge=0.0, le=1.0)
    verified: bool = False
    consensus_reached: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentState(BaseModel):
    """Represents the current state of an agent."""
    id: UUID = Field(default_factory=uuid4)
    role: AgentRole
    current_task: Optional[str] = None
    active_reasoning_paths: List[ReasoningPath] = Field(default_factory=list)
    model_provider: ModelProvider
    model_config: Dict[str, Any] = Field(default_factory=dict)
    performance_metrics: Dict[str, float] = Field(default_factory=dict)


class VisualizationData(BaseModel):
    """Container for data needed by visualization components."""
    reasoning_trees: Dict[UUID, List[ThoughtNode]] = Field(default_factory=dict)
    agent_states: Dict[UUID, AgentState] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SystemConfig(BaseModel):
    """Global system configuration."""
    max_agents: int = Field(default=10, ge=1)
    default_model_provider: ModelProvider = ModelProvider.GROQ
    logging_level: str = "INFO"
    visualization_refresh_rate: float = 1.0  # seconds
    api_keys: Dict[ModelProvider, str] = Field(default_factory=dict)


# Type aliases for improved readability
ThoughtNodeID = UUID
AgentID = UUID
ReasoningPathID = UUID
Confidence = float
Metadata = Dict[str, Any]
ModelResponse = Dict[str, Any]

# Complex type definitions
ReasoningResult = Union[ThoughtNode, ReasoningPath, List[ThoughtNode]]
VisualizationUpdate = Dict[str, Union[ThoughtNode, AgentState, Dict[str, Any]]]
