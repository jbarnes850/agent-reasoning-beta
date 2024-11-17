"""System-wide configuration for the Agent Reasoning Beta platform."""

import os
from enum import Enum
from typing import Dict, Optional

import streamlit as st
from pydantic import BaseModel, ConfigDict, Field

from src.core.types import ModelProvider


class SystemConfig(BaseModel):
    """Global system configuration."""

    # System settings
    max_agents: int = Field(
        default=10, ge=1, description="Maximum number of concurrent agents"
    )
    default_model_provider: ModelProvider = Field(
        default=ModelProvider.GROQ, description="Default LLM provider"
    )
    logging_level: str = Field(default="INFO", description="Logging level")
    visualization_refresh_rate: float = Field(
        default=1.0, ge=0.1, description="Visualization update frequency in seconds"
    )

    # API keys
    api_keys: Dict[ModelProvider, str] = Field(
        default_factory=dict, description="API keys for different providers"
    )

    model_config = ConfigDict(use_enum_values=True, validate_assignment=True)

    def validate_api_keys(self) -> None:
        """Validate required API keys are present."""
        required_keys = {
            ModelProvider.GROQ: "GROQ_API_KEY",
            ModelProvider.OPENAI: "OPENAI_API_KEY",
            ModelProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
        }
        missing_keys = []
        for provider, key in required_keys.items():
            if not os.getenv(key):
                missing_keys.append(key)
        if missing_keys:
            st.warning(f"Missing API keys: {', '.join(missing_keys)}")
            st.info("Add missing keys to your .env file to enable all model providers.")

    @classmethod
    def from_env(cls) -> "SystemConfig":
        """Create configuration from environment variables."""
        import dotenv

        dotenv.load_dotenv()

        return cls(
            max_agents=int(os.getenv("MAX_AGENTS", "10")),
            default_model_provider=ModelProvider[
                os.getenv("DEFAULT_MODEL_PROVIDER", "GROQ")
            ],
            logging_level=os.getenv("LOGGING_LEVEL", "INFO"),
            visualization_refresh_rate=float(
                os.getenv("VISUALIZATION_REFRESH_RATE", "1.0")
            ),
            api_keys={
                ModelProvider.GROQ: os.getenv("GROQ_API_KEY", ""),
                ModelProvider.OPENAI: os.getenv("OPENAI_API_KEY", ""),
                ModelProvider.ANTHROPIC: os.getenv("ANTHROPIC_API_KEY", ""),
            },
        )
