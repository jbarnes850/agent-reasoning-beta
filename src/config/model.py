"""Model configuration for the Agent Reasoning Beta platform."""

from typing import Optional
from pydantic import BaseModel, Field, validator

from core.types import ModelProvider


class ModelConfig(BaseModel):
    """Base configuration for all model providers."""
    
    provider: ModelProvider = Field(
        ...,
        description="Model provider"
    )
    model_name: str = Field(
        ...,
        description="Name of the model to use"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    max_tokens: int = Field(
        default=1000,
        ge=1,
        description="Maximum tokens to generate"
    )
    top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter"
    )
    presence_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Presence penalty"
    )
    frequency_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="Frequency penalty"
    )
    timeout: float = Field(
        default=30.0,
        ge=0.1,
        description="Request timeout in seconds"
    )
    retry_attempts: int = Field(
        default=3,
        ge=0,
        description="Number of retry attempts"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for the provider"
    )
    
    @validator("model_name")
    def validate_model_name(cls, v, values):
        """Validate model name based on provider."""
        provider = values.get("provider")
        if provider == ModelProvider.GROQ:
            valid_models = {"llama-3.1-70b-versatile"}
            if v not in valid_models:
                raise ValueError(f"Invalid Groq model. Must be one of: {valid_models}")
        elif provider == ModelProvider.OPENAI:
            valid_models = {"gpt-4o", "gpt-4o-mini"}
            if v not in valid_models:
                raise ValueError(f"Invalid OpenAI model. Must be one of: {valid_models}")
        elif provider == ModelProvider.ANTHROPIC:
            valid_models = {"claude-3-5-sonnet-latest", "claude-3-5-haiku-latest"}
            if v not in valid_models:
                raise ValueError(f"Invalid Anthropic model. Must be one of: {valid_models}")
        return v
    
    class Config:
        """Pydantic model configuration."""
        use_enum_values = True
        validate_assignment = True


class GroqConfig(ModelConfig):
    """Configuration for Groq models."""
    
    provider: ModelProvider = Field(
        default=ModelProvider.GROQ,
        const=True
    )
    model_name: str = Field(
        default="llama-3.1-70b-versatile"
    )


class OpenAIConfig(ModelConfig):
    """Configuration for OpenAI models."""
    
    provider: ModelProvider = Field(
        default=ModelProvider.OPENAI,
        const=True
    )
    model_name: str = Field(
        default="gpt-4o"
    )


class AnthropicConfig(ModelConfig):
    """Configuration for Anthropic models."""
    
    provider: ModelProvider = Field(
        default=ModelProvider.ANTHROPIC,
        const=True
    )
    model_name: str = Field(
        default="claude-3-5-sonnet-latest"
    )


class SearchConfig(BaseModel):
    """Configuration for search providers."""
    
    tavily_api_key: Optional[str] = Field(
        default=None,
        description="Tavily API key"
    )
    cache_duration: int = Field(
        default=3600,  # 1 hour
        ge=60,  # Minimum 1 minute
        description="Cache duration in seconds"
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of search results"
    )
