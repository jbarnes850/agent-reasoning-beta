"""Agent configuration for the Agent Reasoning Beta platform."""

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.core.types import AgentRole, ModelProvider

from .model import ModelConfig, SearchConfig


class SystemConfig(BaseModel):
    """System-wide configuration."""

    max_agents: int = Field(
        default=10, ge=1, description="Maximum number of concurrent agents"
    )
    default_model_provider: ModelProvider = Field(
        default=ModelProvider.GROQ, description="Default model provider"
    )
    logging_level: str = Field(default="INFO", description="Logging level")
    visualization_refresh_rate: float = Field(
        default=1.0, ge=0.1, description="Visualization update frequency in seconds"
    )
    api_keys: Dict[ModelProvider, str] = Field(
        default_factory=dict, description="API keys for different providers"
    )
    search_config: SearchConfig = Field(
        default_factory=SearchConfig, description="Search configuration"
    )

    model_config = ConfigDict(frozen=True)


class AgentConfig(BaseModel):
    """Configuration for individual agents."""

    role: AgentRole = Field(..., description="Agent's role in the system")
    model_config: ModelConfig = Field(..., description="Model configuration")
    search_enabled: bool = Field(default=True, description="Enable search capabilities")

    model_config = ConfigDict(validate_assignment=True)


class ExplorerConfig(AgentConfig):
    """Configuration for explorer agents."""

    role: Literal[AgentRole.EXPLORER] = Field(
        default=AgentRole.EXPLORER, description="Explorer agent role"
    )
    exploration_params: Dict[str, Any] = Field(
        default_factory=lambda: {"c_puct": 1.0, "max_depth": 10, "min_visits": 3},
        description="MCTS exploration parameters",
    )

    @classmethod
    def validate_exploration_params(cls, v):
        """Validate exploration parameters."""
        required = {"c_puct", "max_depth", "min_visits"}
        if not all(k in v for k in required):
            raise ValueError(
                f"Missing required exploration parameters: {required - set(v.keys())}"
            )
        return v

    model_config = ConfigDict(validate_assignment=True)


class VerifierConfig(AgentConfig):
    """Configuration for verifier agents."""

    role: Literal[AgentRole.VERIFIER] = Field(
        default=AgentRole.VERIFIER, description="Verifier agent role"
    )
    verification_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum confidence for verification"
    )

    model_config = ConfigDict(validate_assignment=True)


class CoordinatorConfig(AgentConfig):
    """Configuration for coordinator agents."""

    role: Literal[AgentRole.COORDINATOR] = Field(
        default=AgentRole.COORDINATOR, description="Coordinator agent role"
    )
    min_paths: int = Field(
        default=3, ge=1, description="Minimum paths required for consensus"
    )

    model_config = ConfigDict(validate_assignment=True)
