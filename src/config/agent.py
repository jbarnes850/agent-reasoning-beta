"""Agent configuration for the Agent Reasoning Beta platform."""

from typing import Dict, Optional
from pydantic import BaseModel, Field

from core.types import AgentRole, ModelProvider
from .model import ModelConfig, SearchConfig

class SystemConfig(BaseModel):
    """System-wide configuration."""
    
    max_agents: int = Field(
        default=10,
        ge=1,
        description="Maximum number of concurrent agents"
    )
    default_model_provider: ModelProvider = Field(
        default=ModelProvider.GROQ,
        description="Default model provider"
    )
    logging_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    visualization_refresh_rate: float = Field(
        default=1.0,
        ge=0.1,
        description="Visualization update frequency in seconds"
    )
    api_keys: Dict[ModelProvider, str] = Field(
        default_factory=dict,
        description="API keys for different providers"
    )
    search_config: SearchConfig = Field(
        default_factory=SearchConfig,
        description="Search configuration"
    )

class AgentConfig(BaseModel):
    """Configuration for individual agents."""
    
    role: AgentRole = Field(
        ...,
        description="Agent's role in the system"
    )
    model_config: ModelConfig = Field(
        ...,
        description="Model configuration"
    )
    search_enabled: bool = Field(
        default=True,
        description="Whether to enable search capabilities"
    )

class ExplorerConfig(AgentConfig):
    """Configuration for explorer agents."""
    
    role: AgentRole = Field(
        default=AgentRole.EXPLORER,
        const=True
    )
    exploration_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "c_puct": 1.0,
            "max_depth": 10,
            "min_visits": 3
        },
        description="MCTS exploration parameters"
    )
    
    @validator("exploration_params")
    def validate_exploration_params(cls, v):
        """Validate exploration parameters."""
        required_keys = {"c_puct", "max_depth", "min_visits"}
        if not all(key in v for key in required_keys):
            raise ValueError(f"Exploration params must contain all required keys: {required_keys}")
        if v["c_puct"] <= 0:
            raise ValueError("c_puct must be positive")
        if v["max_depth"] < 1:
            raise ValueError("max_depth must be at least 1")
        if v["min_visits"] < 1:
            raise ValueError("min_visits must be at least 1")
        return v


class VerifierConfig(AgentConfig):
    """Configuration for verifier agents."""
    
    role: AgentRole = Field(
        default=AgentRole.VERIFIER,
        const=True
    )
    verification_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for verification"
    )


class CoordinatorConfig(AgentConfig):
    """Configuration for coordinator agents."""
    
    role: AgentRole = Field(
        default=AgentRole.COORDINATOR,
        const=True
    )
    min_paths: int = Field(
        default=3,
        ge=1,
        description="Minimum paths required for consensus"
    )
