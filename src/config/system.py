"""System-wide configuration for the Agent Reasoning Beta platform."""

from typing import Dict, Optional
from pydantic import BaseModel, Field

from core.types import ModelProvider


class SystemConfig(BaseModel):
    """Global system configuration."""
    
    # System settings
    max_agents: int = Field(
        default=10,
        ge=1,
        description="Maximum number of concurrent agents"
    )
    default_model_provider: ModelProvider = Field(
        default=ModelProvider.GROQ,
        description="Default LLM provider"
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
    
    # API keys
    api_keys: Dict[ModelProvider, str] = Field(
        default_factory=dict,
        description="API keys for different providers"
    )
    
    class Config:
        """Pydantic model configuration."""
        use_enum_values = True
        validate_assignment = True
        
    @classmethod
    def from_env(cls) -> "SystemConfig":
        """Create configuration from environment variables."""
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        
        return cls(
            max_agents=int(os.getenv("MAX_AGENTS", "10")),
            default_model_provider=ModelProvider[
                os.getenv("DEFAULT_MODEL_PROVIDER", "GROQ").upper()
            ],
            logging_level=os.getenv("LOG_LEVEL", "INFO"),
            visualization_refresh_rate=float(
                os.getenv("VISUALIZATION_REFRESH_RATE", "1.0")
            ),
            api_keys={
                ModelProvider.GROQ: os.getenv("GROQ_API_KEY", ""),
                ModelProvider.OPENAI: os.getenv("OPENAI_API_KEY", ""),
                ModelProvider.ANTHROPIC: os.getenv("ANTHROPIC_API_KEY", "")
            }
        )
