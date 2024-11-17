"""Configuration management utilities for the Agent Reasoning Beta platform."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from core.types import ModelProvider, SystemConfig


class ConfigError(Exception):
    """Base exception for configuration errors."""

    pass


class ConfigNotFoundError(ConfigError):
    """Raised when a configuration file is not found."""

    pass


class ConfigValidationError(ConfigError):
    """Raised when configuration validation fails."""

    pass


class EnvConfig(BaseModel):
    """Environment-specific configuration."""

    env_name: str = Field(default="development")
    debug: bool = Field(default=True)
    api_keys: Dict[str, str] = Field(default_factory=dict)
    cache_dir: str = Field(default=".cache")
    log_dir: str = Field(default="logs")


class AppConfig(BaseModel):
    """Application configuration combining system and environment settings."""

    system: SystemConfig
    env: EnvConfig

    class Config:
        """Pydantic model configuration."""

        arbitrary_types_allowed = True


class ConfigManager:
    """Manages application configuration loading and validation."""

    DEFAULT_CONFIG_PATH = "config/default.json"
    ENV_CONFIG_PREFIX = "config/env"

    def __init__(self, config_dir: Optional[str] = None):
        """Initialize configuration manager.

        Args:
            config_dir: Optional directory containing configuration files.
                       Defaults to project root config directory.
        """
        self.config_dir = (
            Path(config_dir)
            if config_dir
            else Path(__file__).parent.parent.parent / "config"
        )
        self._config: Optional[AppConfig] = None

    @property
    def config(self) -> AppConfig:
        """Get the current application configuration."""
        if not self._config:
            self._config = self.load_config()
        return self._config

    def load_config(self, env: str = None) -> AppConfig:
        """Load configuration for the specified environment.

        Args:
            env: Environment name. If not specified, uses ENV environment variable
                or defaults to 'development'.

        Returns:
            Loaded and validated configuration.

        Raises:
            ConfigNotFoundError: If configuration files cannot be found.
            ConfigValidationError: If configuration validation fails.
        """
        # Determine environment
        env = env or os.getenv("ENV", "development")

        try:
            # Load default config
            default_config = self._load_json_file(
                self.config_dir / self.DEFAULT_CONFIG_PATH
            )

            # Load environment-specific config
            env_config_path = self.config_dir / f"{self.ENV_CONFIG_PREFIX}/{env}.json"
            env_config = (
                self._load_json_file(env_config_path)
                if env_config_path.exists()
                else {}
            )

            # Merge configurations
            merged_config = self._merge_configs(default_config, env_config)

            # Create and validate config objects
            system_config = SystemConfig(**merged_config.get("system", {}))
            env_config = EnvConfig(env_name=env, **merged_config.get("env", {}))

            return AppConfig(system=system_config, env=env_config)

        except FileNotFoundError as e:
            raise ConfigNotFoundError(f"Configuration file not found: {e.filename}")
        except ValueError as e:
            raise ConfigValidationError(f"Configuration validation failed: {str(e)}")

    @staticmethod
    def _load_json_file(path: Path) -> Dict[str, Any]:
        """Load and parse a JSON configuration file."""
        try:
            with open(path) as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigValidationError(f"Invalid JSON in {path}: {str(e)}")

    @staticmethod
    def _merge_configs(
        base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deep merge two configuration dictionaries."""
        merged = base.copy()

        for key, value in override.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = ConfigManager._merge_configs(merged[key], value)
            else:
                merged[key] = value

        return merged

    def get_api_key(self, provider: ModelProvider) -> Optional[str]:
        """Get API key for the specified provider.

        Args:
            provider: Model provider to get API key for.

        Returns:
            API key if found, None otherwise.
        """
        # Check environment variables first
        env_var = f"{provider.name.upper()}_API_KEY"
        if api_key := os.getenv(env_var):
            return api_key

        # Then check configuration
        return self.config.env.api_keys.get(provider.name.lower())

    @lru_cache()
    def get_cache_dir(self) -> Path:
        """Get the cache directory path, creating it if necessary."""
        cache_dir = Path(self.config.env.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    @lru_cache()
    def get_log_dir(self) -> Path:
        """Get the log directory path, creating it if necessary."""
        log_dir = Path(self.config.env.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir
