"""Visualization configuration for the Agent Reasoning Beta platform."""

from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, ConfigDict, Field

from src.core.types import AgentRole, ReasoningType


class BaseVisualizationConfig(BaseModel):
    """Base configuration for all visualizations."""

    LAYOUT_WIDTH: int = Field(
        default=1200, ge=100, description="Visualization width in pixels"
    )
    LAYOUT_HEIGHT: int = Field(
        default=800, ge=100, description="Visualization height in pixels"
    )

    model_config = ConfigDict(validate_assignment=True)


class TreeConfig(BaseVisualizationConfig):
    """Configuration for tree visualization."""

    MIN_NODE_DISTANCE: int = Field(
        default=100, ge=10, description="Minimum distance between nodes"
    )
    NODE_SIZE_RANGE: Tuple[int, int] = Field(
        default=(20, 50), description="Range for node sizes (min, max)"
    )
    EDGE_WIDTH_RANGE: Tuple[int, int] = Field(
        default=(1, 5), description="Range for edge widths (min, max)"
    )
    COLOR_SCHEMES: Dict[ReasoningType, Dict[str, str]] = Field(
        default_factory=lambda: {
            ReasoningType.MCTS: {
                "node": "#1f77b4",
                "edge": "#7fcdbb",
                "text": "#2c3e50",
            },
            ReasoningType.VERIFICATION: {
                "node": "#2ca02c",
                "edge": "#98df8a",
                "text": "#2c3e50",
            },
            ReasoningType.CONSENSUS: {
                "node": "#ff7f0e",
                "edge": "#ffbb78",
                "text": "#2c3e50",
            },
        },
        description="Color schemes for different reasoning types",
    )


class GraphConfig(BaseVisualizationConfig):
    """Configuration for graph visualization."""

    MIN_NODE_DISTANCE: int = Field(
        default=100, ge=10, description="Minimum distance between nodes"
    )
    NODE_SIZE_RANGE: Tuple[int, int] = Field(
        default=(20, 50), description="Range for node sizes (min, max)"
    )
    EDGE_WIDTH_RANGE: Tuple[int, int] = Field(
        default=(1, 5), description="Range for edge widths (min, max)"
    )
    COLOR_SCHEMES: Dict[AgentRole, Dict[str, str]] = Field(
        default_factory=lambda: {
            AgentRole.EXPLORER: {
                "node": "#1f77b4",
                "edge": "#7fcdbb",
                "text": "#2c3e50",
            },
            AgentRole.VERIFIER: {
                "node": "#2ca02c",
                "edge": "#98df8a",
                "text": "#2c3e50",
            },
            AgentRole.COORDINATOR: {
                "node": "#ff7f0e",
                "edge": "#ffbb78",
                "text": "#2c3e50",
            },
        },
        description="Color schemes for different agent roles",
    )


class MetricsConfig(BaseVisualizationConfig):
    """Configuration for metrics visualization."""

    COLOR_SCHEMES: Dict[str, str] = Field(
        default_factory=lambda: {
            "success": "#2ecc71",
            "failure": "#e74c3c",
            "neutral": "#3498db",
            "warning": "#f1c40f",
        },
        description="Color schemes for different metrics",
    )
    CHART_TYPES: List[str] = Field(
        default=["line", "bar", "scatter", "area", "histogram"],
        description="Available chart types",
    )
    TIME_WINDOWS: List[str] = Field(
        default=["1h", "6h", "24h", "7d", "30d"], description="Available time windows"
    )


class HeatmapConfig(BaseVisualizationConfig):
    """Configuration for heatmap visualization."""

    COLOR_SCALES: Dict[str, str] = Field(
        default_factory=lambda: {
            "activity": "Viridis",
            "performance": "RdYlGn",
            "interaction": "Blues",
            "resource": "Oranges",
        },
        description="Color scales for different metrics",
    )
    TIME_BINS: Dict[str, int] = Field(
        default_factory=lambda: {"hourly": 24, "daily": 7, "weekly": 4, "monthly": 12},
        description="Time bins for different scales",
    )
    THRESHOLD_LEVELS: Dict[str, float] = Field(
        default_factory=lambda: {"low": 0.3, "medium": 0.6, "high": 0.8},
        description="Threshold levels for heatmap",
    )


class VisualizationConfig(BaseModel):
    """Global visualization configuration."""

    tree: TreeConfig = Field(
        default_factory=TreeConfig, description="Tree visualization configuration"
    )
    graph: GraphConfig = Field(
        default_factory=GraphConfig, description="Graph visualization configuration"
    )
    metrics: MetricsConfig = Field(
        default_factory=MetricsConfig, description="Metrics visualization configuration"
    )
    heatmap: HeatmapConfig = Field(
        default_factory=HeatmapConfig, description="Heatmap visualization configuration"
    )

    model_config = ConfigDict(validate_assignment=True)
