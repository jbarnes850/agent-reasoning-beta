"""
Metrics and monitoring for the Agent Reasoning Beta platform.

This module provides:
- Cost tracking for API usage
- Performance monitoring
- Usage statistics
- API call tracking
- Visualization data preparation
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID

from pydantic import BaseModel, Field

from .types import AgentRole, ModelProvider, ReasoningType

class APICallMetrics(BaseModel):
    """Metrics for API calls."""
    total_calls: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    average_latency: float = 0.0
    error_rate: float = 0.0
    last_call: datetime = Field(default_factory=datetime.utcnow)
    provider: ModelProvider
    model_name: str

class AgentMetrics(BaseModel):
    """Metrics for agent performance."""
    total_thoughts: int = 0
    successful_verifications: int = 0
    failed_verifications: int = 0
    consensus_participations: int = 0
    average_confidence: float = 0.0
    total_messages: int = 0
    agent_role: AgentRole
    agent_id: UUID
    last_active: datetime = Field(default_factory=datetime.utcnow)

class SystemMetrics(BaseModel):
    """System-wide metrics."""
    total_agents: int = 0
    active_agents: int = 0
    total_reasoning_paths: int = 0
    average_path_length: float = 0.0
    total_api_cost: float = 0.0
    uptime: timedelta = Field(default_factory=lambda: timedelta())
    start_time: datetime = Field(default_factory=datetime.utcnow)

class MetricsManager:
    """Manager for system-wide metrics collection and analysis."""
    
    def __init__(self):
        self._api_metrics: Dict[str, APICallMetrics] = {}
        self._agent_metrics: Dict[UUID, AgentMetrics] = {}
        self._system_metrics = SystemMetrics()
        self._cost_per_token: Dict[Tuple[ModelProvider, str], float] = {
            (ModelProvider.GROQ, "llama-3.1-70b-versatile"): 0.0002,
            (ModelProvider.OPENAI, "gpt-4o"): 0.03,
            (ModelProvider.ANTHROPIC, "claude-3-5-sonnet-latest"): 0.015
        }
    
    def track_api_call(
        self,
        provider: ModelProvider,
        model_name: str,
        tokens: int,
        latency: float,
        success: bool
    ) -> None:
        """Track an API call."""
        key = f"{provider.value}:{model_name}"
        if key not in self._api_metrics:
            self._api_metrics[key] = APICallMetrics(
                provider=provider,
                model_name=model_name
            )
        
        metrics = self._api_metrics[key]
        metrics.total_calls += 1
        metrics.total_tokens += tokens
        metrics.total_cost += self._calculate_cost(provider, model_name, tokens)
        metrics.average_latency = (
            (metrics.average_latency * (metrics.total_calls - 1) + latency)
            / metrics.total_calls
        )
        if not success:
            metrics.error_rate = (
                (metrics.error_rate * (metrics.total_calls - 1) + 1)
                / metrics.total_calls
            )
        metrics.last_call = datetime.utcnow()
        
        # Update system metrics
        self._system_metrics.total_api_cost += metrics.total_cost
    
    def track_agent_activity(
        self,
        agent_id: UUID,
        role: AgentRole,
        activity_type: str,
        success: bool = True,
        confidence: Optional[float] = None
    ) -> None:
        """Track agent activity."""
        if agent_id not in self._agent_metrics:
            self._agent_metrics[agent_id] = AgentMetrics(
                agent_id=agent_id,
                agent_role=role
            )
        
        metrics = self._agent_metrics[agent_id]
        metrics.last_active = datetime.utcnow()
        
        if activity_type == "thought":
            metrics.total_thoughts += 1
        elif activity_type == "verification":
            if success:
                metrics.successful_verifications += 1
            else:
                metrics.failed_verifications += 1
        elif activity_type == "consensus":
            metrics.consensus_participations += 1
        elif activity_type == "message":
            metrics.total_messages += 1
        
        if confidence is not None:
            metrics.average_confidence = (
                (metrics.average_confidence * (metrics.total_thoughts - 1) + confidence)
                / metrics.total_thoughts
            )
    
    def update_system_metrics(self, active_agents: int) -> None:
        """Update system-wide metrics."""
        self._system_metrics.active_agents = active_agents
        self._system_metrics.total_agents = len(self._agent_metrics)
        self._system_metrics.uptime = datetime.utcnow() - self._system_metrics.start_time
    
    def _calculate_cost(
        self,
        provider: ModelProvider,
        model_name: str,
        tokens: int
    ) -> float:
        """Calculate cost for API usage."""
        cost_per_token = self._cost_per_token.get((provider, model_name.lower()), 0.0)
        return cost_per_token * tokens
    
    def get_api_metrics(self) -> Dict[str, APICallMetrics]:
        """Get all API metrics."""
        return self._api_metrics
    
    def get_agent_metrics(self) -> Dict[UUID, AgentMetrics]:
        """Get all agent metrics."""
        return self._agent_metrics
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get system-wide metrics."""
        return self._system_metrics
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self._api_metrics.clear()
        self._agent_metrics.clear()
        self._system_metrics = SystemMetrics()
