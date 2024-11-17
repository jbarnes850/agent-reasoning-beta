"""
Metrics visualization components for the Agent Reasoning Beta platform.

This module provides visualizations for:
- Agent performance metrics
- Confidence distributions
- Success rate trends
- Resource utilization
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core.tools import VisualizationTools
from core.types import AgentRole, ReasoningType


class MetricsConfig:
    """Configuration for metrics visualization."""

    # Layout
    LAYOUT_WIDTH = 1200
    LAYOUT_HEIGHT = 800

    # Color schemes
    COLOR_SCHEMES = {
        "success": "#2ecc71",
        "failure": "#e74c3c",
        "neutral": "#3498db",
        "warning": "#f1c40f",
    }

    # Chart types
    CHART_TYPES = ["line", "bar", "scatter", "area", "histogram"]

    # Time windows
    TIME_WINDOWS = ["1h", "6h", "24h", "7d", "30d"]


class MetricsVisualizer:
    """Interactive metrics visualization component."""

    def __init__(self, config: Optional[MetricsConfig] = None):
        self.config = config or MetricsConfig()

    async def visualize_performance_metrics(
        self,
        metrics: Dict[str, Dict[str, List[float]]],
        container: st.container,
        title: str = "Agent Performance Metrics",
        chart_type: str = "line",
    ) -> None:
        """
        Create an interactive visualization of agent performance metrics over time.

        Args:
            metrics: Dictionary of agent metrics time series
            container: Streamlit container to render in
            title: Title for the visualization
            chart_type: Type of chart to display
        """
        if chart_type not in self.config.CHART_TYPES:
            raise ValueError(f"Unsupported chart type: {chart_type}")

        # Create figure
        fig = self._create_figure(title)

        # Add metric traces
        for agent_id, agent_metrics in metrics.items():
            for metric_name, metric_values in agent_metrics.items():
                self._add_metric_trace(
                    fig, agent_id, metric_name, metric_values, chart_type
                )

        # Configure layout
        self._configure_layout(fig)

        # Add summary statistics
        self._add_summary_stats(fig, metrics)

        # Render in Streamlit
        container.plotly_chart(fig, use_container_width=True)

    async def visualize_confidence_distribution(
        self,
        confidence_scores: List[float],
        container: st.container,
        title: str = "Confidence Score Distribution",
    ) -> None:
        """
        Create a visualization of confidence score distribution.

        Args:
            confidence_scores: List of confidence scores
            container: Streamlit container to render in
            title: Title for the visualization
        """
        fig = go.Figure()

        # Add histogram
        fig.add_trace(
            go.Histogram(
                x=confidence_scores,
                nbinsx=20,
                name="Confidence Distribution",
                marker_color=self.config.COLOR_SCHEMES["neutral"],
            )
        )

        # Add mean and median lines
        mean_conf = np.mean(confidence_scores)
        median_conf = np.median(confidence_scores)

        fig.add_vline(
            x=mean_conf,
            line_dash="dash",
            line_color=self.config.COLOR_SCHEMES["success"],
            annotation_text=f"Mean: {mean_conf:.2f}",
        )

        fig.add_vline(
            x=median_conf,
            line_dash="dash",
            line_color=self.config.COLOR_SCHEMES["warning"],
            annotation_text=f"Median: {median_conf:.2f}",
        )

        # Configure layout
        fig.update_layout(
            title=title,
            xaxis_title="Confidence Score",
            yaxis_title="Frequency",
            showlegend=True,
            width=self.config.LAYOUT_WIDTH,
            height=self.config.LAYOUT_HEIGHT,
        )

        # Render in Streamlit
        container.plotly_chart(fig, use_container_width=True)

    async def visualize_success_rate_trends(
        self,
        success_rates: Dict[str, List[Tuple[float, float]]],
        container: st.container,
        title: str = "Success Rate Trends",
        time_window: str = "24h",
    ) -> None:
        """
        Create a visualization of success rate trends over time.

        Args:
            success_rates: Dictionary of agent success rates over time
            container: Streamlit container to render in
            title: Title for the visualization
            time_window: Time window to display
        """
        if time_window not in self.config.TIME_WINDOWS:
            raise ValueError(f"Unsupported time window: {time_window}")

        fig = go.Figure()

        # Add success rate lines for each agent
        for agent_id, rates in success_rates.items():
            timestamps, values = zip(*rates)

            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=values,
                    mode="lines+markers",
                    name=f"Agent {agent_id}",
                    line=dict(width=2),
                    marker=dict(size=8),
                )
            )

        # Add target success rate line
        fig.add_hline(
            y=0.8,
            line_dash="dash",
            line_color=self.config.COLOR_SCHEMES["success"],
            annotation_text="Target Success Rate",
        )

        # Configure layout
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Success Rate",
            showlegend=True,
            width=self.config.LAYOUT_WIDTH,
            height=self.config.LAYOUT_HEIGHT,
        )

        # Render in Streamlit
        container.plotly_chart(fig, use_container_width=True)

    async def visualize_resource_utilization(
        self,
        utilization: Dict[str, Dict[str, float]],
        container: st.container,
        title: str = "Resource Utilization",
    ) -> None:
        """
        Create a visualization of resource utilization metrics.

        Args:
            utilization: Dictionary of resource utilization metrics
            container: Streamlit container to render in
            title: Title for the visualization
        """
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("CPU Usage", "Memory Usage", "API Calls", "Token Usage"),
        )

        # Add CPU usage gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=utilization["cpu"]["usage"] * 100,
                title="CPU Usage %",
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {
                        "color": self._get_utilization_color(
                            utilization["cpu"]["usage"]
                        )
                    },
                },
            ),
            row=1,
            col=1,
        )

        # Add memory usage gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=utilization["memory"]["usage"] * 100,
                title="Memory Usage %",
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {
                        "color": self._get_utilization_color(
                            utilization["memory"]["usage"]
                        )
                    },
                },
            ),
            row=1,
            col=2,
        )

        # Add API calls bar chart
        fig.add_trace(
            go.Bar(
                x=list(utilization["api"]["calls_by_endpoint"].keys()),
                y=list(utilization["api"]["calls_by_endpoint"].values()),
                name="API Calls",
                marker_color=self.config.COLOR_SCHEMES["neutral"],
            ),
            row=2,
            col=1,
        )

        # Add token usage pie chart
        fig.add_trace(
            go.Pie(
                labels=list(utilization["tokens"]["usage_by_model"].keys()),
                values=list(utilization["tokens"]["usage_by_model"].values()),
                name="Token Usage",
            ),
            row=2,
            col=2,
        )

        # Configure layout
        fig.update_layout(
            title=title,
            showlegend=True,
            width=self.config.LAYOUT_WIDTH,
            height=self.config.LAYOUT_HEIGHT,
        )

        # Render in Streamlit
        container.plotly_chart(fig, use_container_width=True)

    def _create_figure(self, title: str) -> go.Figure:
        """Create base plotly figure."""
        fig = go.Figure()

        fig.update_layout(
            title=title,
            showlegend=True,
            width=self.config.LAYOUT_WIDTH,
            height=self.config.LAYOUT_HEIGHT,
            template="plotly_white",
        )

        return fig

    def _add_metric_trace(
        self,
        fig: go.Figure,
        agent_id: str,
        metric_name: str,
        metric_values: List[float],
        chart_type: str,
    ) -> None:
        """Add a metric trace to the figure."""
        if chart_type == "line":
            fig.add_trace(
                go.Scatter(
                    y=metric_values,
                    mode="lines+markers",
                    name=f"{agent_id} - {metric_name}",
                    line=dict(width=2),
                    marker=dict(size=8),
                )
            )
        elif chart_type == "bar":
            fig.add_trace(go.Bar(y=metric_values, name=f"{agent_id} - {metric_name}"))
        elif chart_type == "scatter":
            fig.add_trace(
                go.Scatter(
                    y=metric_values,
                    mode="markers",
                    name=f"{agent_id} - {metric_name}",
                    marker=dict(size=10),
                )
            )
        elif chart_type == "area":
            fig.add_trace(
                go.Scatter(
                    y=metric_values,
                    fill="tonexty",
                    name=f"{agent_id} - {metric_name}",
                    line=dict(width=2),
                )
            )
        elif chart_type == "histogram":
            fig.add_trace(
                go.Histogram(
                    x=metric_values, name=f"{agent_id} - {metric_name}", nbinsx=20
                )
            )

    def _configure_layout(self, fig: go.Figure) -> None:
        """Configure figure layout."""
        fig.update_layout(
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)"),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)"),
            plot_bgcolor="white",
        )

    def _add_summary_stats(
        self, fig: go.Figure, metrics: Dict[str, Dict[str, List[float]]]
    ) -> None:
        """Add summary statistics to the figure."""
        stats_text = "Summary Statistics:<br>"

        for agent_id, agent_metrics in metrics.items():
            stats_text += f"<br>{agent_id}:<br>"
            for metric_name, metric_values in agent_metrics.items():
                mean_val = np.mean(metric_values)
                std_val = np.std(metric_values)
                max_val = np.max(metric_values)
                min_val = np.min(metric_values)

                stats_text += (
                    f"  {metric_name}:<br>"
                    f"    Mean: {mean_val:.2f}<br>"
                    f"    Std: {std_val:.2f}<br>"
                    f"    Max: {max_val:.2f}<br>"
                    f"    Min: {min_val:.2f}<br>"
                )

        fig.add_annotation(
            text=stats_text,
            xref="paper",
            yref="paper",
            x=1.15,
            y=0.5,
            showarrow=False,
            font=dict(size=10),
            align="left",
        )

    def _get_utilization_color(self, utilization: float) -> str:
        """Get color based on utilization level."""
        if utilization >= 0.8:
            return self.config.COLOR_SCHEMES["failure"]
        elif utilization >= 0.6:
            return self.config.COLOR_SCHEMES["warning"]
        else:
            return self.config.COLOR_SCHEMES["success"]
