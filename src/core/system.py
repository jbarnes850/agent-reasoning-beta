"""
System management for the Agent Reasoning Beta platform.

This module provides:
- Global system state management
- Resource management
- API key handling
- Concurrent agent management
- System configuration
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Set
from uuid import UUID

from pydantic import BaseModel, Field, SecretStr, ConfigDict

from .agents import AgentFactory, BaseAgent
from .metrics import MetricsManager
from .models import ModelRegistry
from .search import SearchManager
from .types import AgentRole, ModelProvider, SystemConfig

logger = logging.getLogger(__name__)

class SystemState(BaseModel):
    """Current state of the system."""
    active_agents: Dict[UUID, BaseAgent] = Field(default_factory=dict)
    api_keys: Dict[ModelProvider, SecretStr] = Field(default_factory=dict)
    config: SystemConfig
    start_time: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        arbitrary_types_allowed = True

class SystemManager:
    """Manager for system-wide operations."""
    
    def __init__(self, config: SystemConfig):
        self._state = SystemState(config=config)
        self._metrics = MetricsManager()
        self._model_registry = ModelRegistry()
        self._search_manager = SearchManager()
        self._agent_factory = AgentFactory()
        self._setup_logging()
    
    async def initialize(self) -> None:
        """Initialize the system."""
        logger.info("Initializing system...")
        
        # Set up API keys
        for provider, key in self._state.config.api_keys.items():
            self.set_api_key(provider, key)
        
        # Initialize model registry
        await self._model_registry.initialize()
        
        logger.info("System initialized successfully")
    
    def set_api_key(self, provider: ModelProvider, key: str) -> None:
        """Set API key for a provider."""
        self._state.api_keys[provider] = SecretStr(key)
        logger.info(f"API key set for provider: {provider.value}")
    
    async def create_agent(
        self,
        role: AgentRole,
        model_provider: Optional[ModelProvider] = None,
        model_config: Optional[Dict] = None
    ) -> UUID:
        """Create and start a new agent."""
        if len(self._state.active_agents) >= self._state.config.max_agents:
            raise RuntimeError("Maximum number of agents reached")
        
        provider = model_provider or self._state.config.default_model_provider
        agent = self._agent_factory.create_agent(
            role=role,
            model_provider=provider,
            model_config=model_config
        )
        
        self._state.active_agents[agent.id] = agent
        await agent.start()
        
        self._metrics.update_system_metrics(len(self._state.active_agents))
        logger.info(f"Created agent {agent.id} with role {role.value}")
        
        return agent.id
    
    async def stop_agent(self, agent_id: UUID) -> None:
        """Stop and remove an agent."""
        if agent_id in self._state.active_agents:
            agent = self._state.active_agents[agent_id]
            await agent.stop()
            del self._state.active_agents[agent_id]
            
            self._metrics.update_system_metrics(len(self._state.active_agents))
            logger.info(f"Stopped agent {agent_id}")
    
    async def stop_all_agents(self) -> None:
        """Stop all active agents."""
        agent_ids = list(self._state.active_agents.keys())
        for agent_id in agent_ids:
            await self.stop_agent(agent_id)
    
    def get_agent(self, agent_id: UUID) -> Optional[BaseAgent]:
        """Get an agent by ID."""
        return self._state.active_agents.get(agent_id)
    
    def get_active_agents(self) -> Dict[UUID, BaseAgent]:
        """Get all active agents."""
        return self._state.active_agents.copy()
    
    @property
    def metrics(self) -> MetricsManager:
        """Get system metrics."""
        return self._metrics
    
    @property
    def model_registry(self) -> ModelRegistry:
        """Get model registry."""
        return self._model_registry
    
    @property
    def search_manager(self) -> SearchManager:
        """Get search manager."""
        return self._search_manager
    
    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        logging.basicConfig(
            level=self._state.config.logging_level,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("system.log")
            ]
        )
