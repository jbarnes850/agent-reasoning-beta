"""Model configuration for the Agent Reasoning Beta platform."""

from typing import Optional, Literal
from pydantic import BaseModel, Field, validator, ConfigDict
from typing_extensions import Annotated

from src.core.types import ModelProvider


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
        if not provider:
            return v
            
        valid_models = {
            ModelProvider.GROQ: [
                "llama-3.1-70b-versatile",
            ],
            ModelProvider.OPENAI: [
                "gpt-4o",
                "gpt-4o-mini",
            ],
            ModelProvider.ANTHROPIC: [
                "claude-3-5-sonnet-latest",
                "claude-3-5-haiku-latest",
            ],
        }
        
        if v not in valid_models.get(provider, []):
            raise ValueError(
                f"Invalid model name '{v}' for provider {provider}. "
                f"Valid models are: {', '.join(valid_models.get(provider, []))}"
            )
        return v
    
    class Config:
        """Pydantic model configuration."""
        model_config = ConfigDict(
            validate_assignment=True,
            protected_namespaces=()
        )
        use_enum_values = True
        validate_assignment = True


class GroqConfig(ModelConfig):
    """Configuration for Groq models."""
    
    provider: Literal[ModelProvider.GROQ] = Field(
        default=ModelProvider.GROQ,
        description="Groq model provider"
    )
    model_name: str = Field(
        default="llama-3.1-70b-versatile",
        description="Default Groq model"
    )


class OpenAIConfig(ModelConfig):
    """Configuration for OpenAI models."""
    
    provider: Literal[ModelProvider.OPENAI] = Field(
        default=ModelProvider.OPENAI,
        description="OpenAI model provider"
    )
    model_name: str = Field(
        default="gpt-4o",
        description="Default OpenAI model"
    )


class AnthropicConfig(ModelConfig):
    """Configuration for Anthropic models."""
    
    provider: Literal[ModelProvider.ANTHROPIC] = Field(
        default=ModelProvider.ANTHROPIC,
        description="Anthropic model provider"
    )
    model_name: str = Field(
        default="claude-3-5-sonnet-latest",
        description="Default Anthropic model"
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
    
    class Config:
        """Pydantic model configuration."""
        use_enum_values = True
        validate_assignment = True
