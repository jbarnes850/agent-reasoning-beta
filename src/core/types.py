"""
Core type definitions for the Agent Reasoning Beta platform.

This module defines the fundamental types and data structures used throughout the system,
including agent states, reasoning primitives, and visualization data structures.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    TypeAlias,
    TypeGuard,
    TypeVar,
    Union,
    cast,
)
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, PositiveInt

# Type aliases for improved readability and type safety
ThoughtNodeID: TypeAlias = UUID
AgentID: TypeAlias = UUID
ReasoningPathID: TypeAlias = UUID
VisualizationID: TypeAlias = UUID

# Domain-specific type aliases
Confidence: TypeAlias = NonNegativeFloat  # Constrained to [0.0, 1.0]
ModelResponse: TypeAlias = Dict[str, Any]  # TODO: Make this more specific based on provider responses

# Generic type variables for type-safe collections
T = TypeVar("T")
NodeType = TypeVar("NodeType", bound="ThoughtNode")

# Type aliases for common data structures
ThoughtGraph = Dict[Literal["nodes", "edges"], Union[List[ThoughtNode], List[Dict[str, Any]]]]
MetricsData = Dict[str, List[Dict[Literal["timestamp", "value"], Union[str, float]]]]
ConfigDict = Dict[str, Any]

__all__ = [
    "ReasoningType",
    "AgentRole",
    "ModelProvider",
    "AgentVote",
    "ThoughtNode",
    "ReasoningPath",
    "AgentState",
    "VisualizationData",
    "SystemConfig",
    "ConsensusMetrics",
    "ToolMetrics",
    "Evidence",
    "ProofNode",
    "VerificationChain",
    "VerificationStatus",
    "ThoughtNodeID",
    "AgentID",
    "ReasoningPathID",
    "VisualizationID",
    "Confidence",
    "ModelResponse",
    "ReasoningResult",
]


class ModelProvider(str, Enum):
    """Supported model providers."""

    GROQ = "groq"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class AgentRole(str, Enum):
    """Agent roles in the system."""

    EXPLORER = "explorer"
    VERIFIER = "verifier"
    COORDINATOR = "coordinator"
    OBSERVER = "observer"
    CONSENSUS = "consensus"


class ReasoningType(str, Enum):
    """Types of reasoning processes."""

    MCTS = "mcts"
    VERIFICATION = "verification"
    CONSENSUS = "consensus"


class ThoughtNode(BaseModel):
    """Represents a single node in the reasoning tree."""

    id: ThoughtNodeID = Field(default_factory=uuid4)
    parent_id: Optional[ThoughtNodeID] = None
    content: str
    confidence: Confidence = Field(ge=0.0, le=1.0)
    reasoning_type: ReasoningType
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(frozen=True)


def is_valid_thought_node(obj: Any) -> TypeGuard[ThoughtNode]:
    """Runtime type check for ThoughtNode.
    
    Args:
        obj: Any object to check
        
    Returns:
        bool: True if obj is a valid ThoughtNode
    """
    if not isinstance(obj, dict):
        return False
        
    required_attrs = {
        "id": lambda x: isinstance(x, UUID),
        "content": lambda x: isinstance(x, str),
        "confidence": lambda x: isinstance(x, (int, float)) and 0.0 <= x <= 1.0,
        "reasoning_type": lambda x: isinstance(x, ReasoningType),
        "metadata": lambda x: isinstance(x, dict),
        "created_at": lambda x: isinstance(x, datetime)
    }
    
    return all(
        attr in obj and check(obj[attr])
        for attr, check in required_attrs.items()
    )


class ReasoningPath(BaseModel):
    """Represents a complete path of reasoning from root to leaf."""

    id: ReasoningPathID = Field(default_factory=uuid4)
    nodes: List[ThoughtNode]
    total_confidence: Confidence = Field(ge=0.0, le=1.0)
    verified: bool = False
    consensus_reached: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class AgentState(BaseModel):
    """Represents the current state of an agent."""

    id: AgentID = Field(default_factory=uuid4)
    role: AgentRole
    current_task: Optional[str] = None
    active_reasoning_paths: List[ReasoningPath] = Field(default_factory=list)
    model_provider: ModelProvider
    model_config: Dict[str, Any] = Field(default_factory=dict)
    performance_metrics: Dict[str, float] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class VisualizationData(BaseModel):
    """Container for data needed by visualization components."""

    reasoning_trees: Dict[UUID, List[ThoughtNode]] = Field(default_factory=dict)
    agent_states: Dict[UUID, AgentState] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(frozen=True)


class SystemConfig(BaseModel):
    """Global system configuration."""

    max_agents: PositiveInt = Field(default=10)
    default_model_provider: ModelProvider = ModelProvider.GROQ
    logging_level: str = "INFO"
    visualization_refresh_rate: float = 1.0  # seconds
    api_keys: Dict[ModelProvider, str] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class ConsensusMetrics(BaseModel):
    """Metrics for consensus tracking."""

    agreement_matrix: List[List[float]]
    agent_ids: List[str]
    resolution_path: List[Tuple[str, float]]

    model_config = ConfigDict(frozen=True)


class ToolMetrics(BaseModel):
    """Metrics for tool usage and performance tracking."""

    total_calls: int = Field(default=0)
    successful_calls: int = Field(default=0)
    failed_calls: int = Field(default=0)
    average_latency: float = Field(default=0.0)
    last_call_timestamp: Optional[datetime] = None
    error_counts: Dict[str, int] = Field(default_factory=dict)
    performance_history: List[float] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True)


class Evidence(BaseModel):
    """Represents a piece of evidence in verification."""

    id: UUID = Field(default_factory=uuid4)
    content: str
    source: str
    confidence: Confidence = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(frozen=True)


class ProofNode(BaseModel):
    """Represents a node in the verification proof tree."""

    id: UUID = Field(default_factory=uuid4)
    parent_id: Optional[UUID] = None
    statement: str
    evidence: List[Evidence] = Field(default_factory=list)
    confidence: Confidence = Field(ge=0.0, le=1.0)
    verified: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(frozen=True)


class VerificationStatus(Enum):
    """Status of verification process."""

    PENDING = auto()
    IN_PROGRESS = auto()
    VERIFIED = auto()
    REJECTED = auto()


class VerificationChain(BaseModel):
    """Represents a complete verification chain."""

    id: UUID = Field(default_factory=uuid4)
    root_node: ProofNode
    nodes: List[ProofNode] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
    proof_nodes: List[ProofNode] = Field(default_factory=list)
    evidence_map: Dict[UUID, List[Evidence]] = Field(default_factory=dict)
    total_confidence: Confidence = Field(ge=0.0, le=1.0)
    mean_confidence: Confidence = Field(ge=0.0, le=1.0)
    status: VerificationStatus = Field(default=VerificationStatus.PENDING)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(frozen=True)


# Define ReasoningResult after all required types are defined
ReasoningResult: TypeAlias = Union[ThoughtNode, ReasoningPath, List[ThoughtNode]]


def validate_thought_graph(graph: ThoughtGraph) -> bool:
    """Validate thought graph structure and types.
    
    Args:
        graph: The graph structure to validate
        
    Returns:
        bool: True if the graph is valid
    """
    if not isinstance(graph, dict):
        return False

    required_keys = {"nodes", "edges"}
    if not all(key in graph for key in required_keys):
        return False

    nodes = cast(List[Any], graph["nodes"])
    edges = cast(List[Dict[str, Any]], graph["edges"])

    if not isinstance(nodes, list) or not isinstance(edges, list):
        return False

    return all(is_valid_thought_node(node) for node in nodes)


def validate_metrics_data(metrics: MetricsData) -> bool:
    """Validate metrics data structure and types.
    
    Args:
        metrics: The metrics data to validate
        
    Returns:
        bool: True if the metrics data is valid
    """
    if not isinstance(metrics, dict):
        return False

    for metric_list in metrics.values():
        if not isinstance(metric_list, list):
            return False

        for entry in metric_list:
            if not isinstance(entry, dict):
                return False

            required_keys = {"timestamp", "value"}
            if not all(key in entry for key in required_keys):
                return False

            timestamp = entry.get("timestamp")
            value = entry.get("value")

            if not isinstance(timestamp, str) or not isinstance(value, (int, float)):
                return False

    return True
