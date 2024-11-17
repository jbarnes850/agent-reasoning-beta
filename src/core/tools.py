"""
Tools and utilities for the Agent Reasoning Beta platform.

This module provides:
- Environment interaction tools
- Data transformation utilities
- Metrics collection and analysis
- Visualization helpers
- Common operations for agents
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union
from uuid import UUID

import numpy as np
from pydantic import BaseModel, Field, ValidationError

from .types import (
    AgentState,
    ModelProvider,
    ReasoningPath,
    ReasoningType,
    ThoughtNode,
    VisualizationData,
)

# Type variables for generic functions
T = TypeVar("T")
R = TypeVar("R")


class ToolMetrics(BaseModel):
    """Metrics for tool usage and performance."""

    calls: int = 0
    total_duration: float = 0.0
    average_duration: float = 0.0
    success_rate: float = 1.0
    last_used: datetime = Field(default_factory=datetime.utcnow)
    errors: List[str] = Field(default_factory=list)


class ToolRegistry:
    """Registry for tracking tool usage and metrics."""

    def __init__(self):
        self._metrics: Dict[str, ToolMetrics] = {}

    def record_usage(
        self,
        tool_name: str,
        duration: float,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """Record tool usage metrics."""
        if tool_name not in self._metrics:
            self._metrics[tool_name] = ToolMetrics()

        metrics = self._metrics[tool_name]
        metrics.calls += 1
        metrics.total_duration += duration
        metrics.average_duration = metrics.total_duration / metrics.calls

        if not success and error:
            metrics.errors.append(error)
            metrics.success_rate = (metrics.calls - len(metrics.errors)) / metrics.calls

        metrics.last_used = datetime.utcnow()

    def get_metrics(self, tool_name: str) -> ToolMetrics:
        """Get metrics for a specific tool."""
        return self._metrics.get(tool_name, ToolMetrics())

    def get_all_metrics(self) -> Dict[str, ToolMetrics]:
        """Get metrics for all tools."""
        return self._metrics.copy()


def measure_performance(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to measure tool performance."""

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> T:
        start_time = datetime.now()
        try:
            result = await func(*args, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            registry.record_usage(func.__name__, duration, True)
            return result
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            registry.record_usage(func.__name__, duration, False, str(e))
            raise

    return wrapper


# Initialize global registry
registry = ToolRegistry()


class DataTransformation:
    """Tools for data transformation and analysis."""

    @staticmethod
    @measure_performance
    async def normalize_confidence(
        confidences: List[float],
        min_confidence: float = 0.0,
        max_confidence: float = 1.0,
    ) -> List[float]:
        """Normalize confidence scores to specified range."""
        if not confidences:
            return []

        arr = np.array(confidences)
        arr_norm = (arr - arr.min()) / (arr.max() - arr.min())
        return list(arr_norm * (max_confidence - min_confidence) + min_confidence)

    @staticmethod
    @measure_performance
    async def merge_thought_nodes(
        nodes: List[ThoughtNode], strategy: str = "weighted_average"
    ) -> ThoughtNode:
        """Merge multiple thought nodes into one."""
        if not nodes:
            raise ValueError("Cannot merge empty list of nodes")

        if strategy == "weighted_average":
            # Weight by confidence
            weights = [node.confidence for node in nodes]
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]

            merged_content = " | ".join(
                f"{w:.2f}: {n.content}" for w, n in zip(weights, nodes)
            )
            merged_confidence = sum(w * n.confidence for w, n in zip(weights, nodes))

            return ThoughtNode(
                content=merged_content,
                confidence=merged_confidence,
                reasoning_type=nodes[0].reasoning_type,
                metadata={"merged_from": [str(n.id) for n in nodes]},
            )
        else:
            raise ValueError(f"Unsupported merge strategy: {strategy}")


class EnvironmentTools:
    """Tools for interacting with the environment."""

    @staticmethod
    @measure_performance
    async def save_reasoning_path(path: ReasoningPath, filepath: str) -> None:
        """Save reasoning path to file."""
        try:
            with open(filepath, "w") as f:
                json.dump(path.dict(), f, indent=2)
        except Exception as e:
            raise ToolError(f"Failed to save reasoning path: {str(e)}")

    @staticmethod
    @measure_performance
    async def load_reasoning_path(filepath: str) -> ReasoningPath:
        """Load reasoning path from file."""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            return ReasoningPath(**data)
        except Exception as e:
            raise ToolError(f"Failed to load reasoning path: {str(e)}")


class VisualizationTools:
    """Tools for visualization preparation."""

    @staticmethod
    @measure_performance
    async def prepare_tree_data(path: ReasoningPath) -> Dict[str, Any]:
        """Prepare reasoning path for tree visualization."""
        nodes = []
        edges = []

        for i, node in enumerate(path.nodes):
            nodes.append(
                {
                    "id": str(node.id),
                    "label": node.content[:50] + "...",
                    "confidence": node.confidence,
                    "type": node.reasoning_type.value,
                }
            )

            if i > 0:
                edges.append({"from": str(path.nodes[i - 1].id), "to": str(node.id)})

        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "total_confidence": path.total_confidence,
                "verified": path.verified,
                "consensus_reached": path.consensus_reached,
            },
        }

    @staticmethod
    @measure_performance
    async def prepare_metrics_data(metrics: Dict[str, ToolMetrics]) -> Dict[str, Any]:
        """Prepare metrics for visualization."""
        return {
            "tool_usage": {
                name: {
                    "calls": m.calls,
                    "avg_duration": m.average_duration,
                    "success_rate": m.success_rate,
                }
                for name, m in metrics.items()
            },
            "total_calls": sum(m.calls for m in metrics.values()),
            "overall_success_rate": np.mean([m.success_rate for m in metrics.values()]),
        }


class AgentTools:
    """Tools specifically for agent operations."""

    @staticmethod
    @measure_performance
    async def validate_reasoning_path(
        path: ReasoningPath, min_confidence: float = 0.5, max_steps: int = 10
    ) -> Tuple[bool, List[str]]:
        """Validate a reasoning path."""
        issues = []

        # Check path length
        if len(path.nodes) > max_steps:
            issues.append(f"Path too long: {len(path.nodes)} > {max_steps}")

        # Check confidence scores
        low_confidence_nodes = [
            (i, node)
            for i, node in enumerate(path.nodes)
            if node.confidence < min_confidence
        ]
        if low_confidence_nodes:
            issues.extend(
                [
                    f"Low confidence at step {i}: {node.confidence}"
                    for i, node in low_confidence_nodes
                ]
            )

        # Check logical flow
        for i in range(len(path.nodes) - 1):
            current = path.nodes[i]
            next_node = path.nodes[i + 1]

            if current.reasoning_type != next_node.reasoning_type:
                issues.append(
                    f"Reasoning type mismatch at steps {i} and {i+1}: "
                    f"{current.reasoning_type} -> {next_node.reasoning_type}"
                )

        return len(issues) == 0, issues

    @staticmethod
    @measure_performance
    async def extract_key_insights(
        path: ReasoningPath, max_insights: int = 5
    ) -> List[str]:
        """Extract key insights from a reasoning path."""
        insights = []

        # Sort nodes by confidence
        sorted_nodes = sorted(path.nodes, key=lambda n: n.confidence, reverse=True)

        # Extract top insights
        for node in sorted_nodes[:max_insights]:
            # Split content into sentences and take first one as key insight
            sentences = node.content.split(". ")
            if sentences:
                insights.append(sentences[0])

        return insights


class ToolError(Exception):
    """Custom exception for tool-related errors."""

    pass
