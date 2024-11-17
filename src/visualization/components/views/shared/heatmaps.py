"""
Heatmap visualization components for the Agent Reasoning Beta platform.

This module provides heatmap visualizations for:
- Agent activity patterns
- Interaction density
- Performance hotspots
- Resource usage patterns
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


class HeatmapConfig:
    """Configuration for heatmap visualization."""

    # Layout
    LAYOUT_WIDTH = 1200
    LAYOUT_HEIGHT = 800

    # Color scales
    COLOR_SCALES = {
        "activity": "Viridis",
        "performance": "RdYlGn",
        "interaction": "Blues",
        "resource": "Oranges",
    }

    # Time bins
    TIME_BINS = {"hourly": 24, "daily": 7, "weekly": 4, "monthly": 12}

    # Threshold levels
    THRESHOLD_LEVELS = {"low": 0.3, "medium": 0.6, "high": 0.8}


class HeatmapVisualizer:
    """Interactive heatmap visualization component."""

    def __init__(self, config: Optional[HeatmapConfig] = None):
        self.config = config or HeatmapConfig()

    async def visualize_activity_patterns(
        self,
        activity_data: Dict[str, Dict[str, List[float]]],
        container: st.container,
        title: str = "Agent Activity Patterns",
        time_scale: str = "hourly",
    ) -> None:
        """
        Create a heatmap visualization of agent activity patterns.

        Args:
            activity_data: Dictionary of agent activity data
            container: Streamlit container to render in
            title: Title for the visualization
            time_scale: Time scale for binning ('hourly', 'daily', 'weekly', 'monthly')
        """
        if time_scale not in self.config.TIME_BINS:
            raise ValueError(f"Unsupported time scale: {time_scale}")

        # Process activity data
        processed_data = self._process_activity_data(activity_data, time_scale)

        # Create figure
        fig = go.Figure()

        # Add heatmap
        fig.add_trace(
            go.Heatmap(
                z=processed_data["values"],
                x=processed_data["time_labels"],
                y=processed_data["agent_labels"],
                colorscale=self.config.COLOR_SCALES["activity"],
                colorbar=dict(title="Activity Level", tickformat=".2f"),
            )
        )

        # Configure layout
        fig.update_layout(
            title=title,
            xaxis_title="Time Period",
            yaxis_title="Agent",
            width=self.config.LAYOUT_WIDTH,
            height=self.config.LAYOUT_HEIGHT,
        )

        # Render in Streamlit
        container.plotly_chart(fig, use_container_width=True)

    async def visualize_interaction_density(
        self,
        interaction_data: Dict[str, Dict[str, int]],
        container: st.container,
        title: str = "Agent Interaction Density",
    ) -> None:
        """
        Create a heatmap visualization of agent interaction density.

        Args:
            interaction_data: Dictionary of agent interaction counts
            container: Streamlit container to render in
            title: Title for the visualization
        """
        # Create interaction matrix
        agents = sorted(interaction_data.keys())
        matrix = np.zeros((len(agents), len(agents)))

        for i, agent1 in enumerate(agents):
            for j, agent2 in enumerate(agents):
                matrix[i][j] = interaction_data[agent1].get(agent2, 0)

        # Create figure
        fig = go.Figure()

        # Add heatmap
        fig.add_trace(
            go.Heatmap(
                z=matrix,
                x=agents,
                y=agents,
                colorscale=self.config.COLOR_SCALES["interaction"],
                colorbar=dict(title="Interaction Count", tickformat="d"),
            )
        )

        # Configure layout
        fig.update_layout(
            title=title,
            xaxis_title="Agent",
            yaxis_title="Agent",
            width=self.config.LAYOUT_WIDTH,
            height=self.config.LAYOUT_HEIGHT,
        )

        # Render in Streamlit
        container.plotly_chart(fig, use_container_width=True)

    async def visualize_performance_hotspots(
        self,
        performance_data: Dict[str, Dict[str, float]],
        container: st.container,
        title: str = "Performance Hotspots",
    ) -> None:
        """
        Create a heatmap visualization of performance metrics.

        Args:
            performance_data: Dictionary of agent performance metrics
            container: Streamlit container to render in
            title: Title for the visualization
        """
        # Process performance data
        agents = []
        metrics = []
        values = []

        for agent_id, metrics_dict in performance_data.items():
            for metric_name, value in metrics_dict.items():
                agents.append(agent_id)
                metrics.append(metric_name)
                values.append(value)

        # Create figure
        fig = go.Figure()

        # Add heatmap
        fig.add_trace(
            go.Heatmap(
                z=values,
                x=metrics,
                y=agents,
                colorscale=self.config.COLOR_SCALES["performance"],
                colorbar=dict(title="Performance Score", tickformat=".2f"),
            )
        )

        # Configure layout
        fig.update_layout(
            title=title,
            xaxis_title="Metric",
            yaxis_title="Agent",
            width=self.config.LAYOUT_WIDTH,
            height=self.config.LAYOUT_HEIGHT,
        )

        # Render in Streamlit
        container.plotly_chart(fig, use_container_width=True)

    async def visualize_resource_patterns(
        self,
        resource_data: Dict[str, Dict[str, List[float]]],
        container: st.container,
        title: str = "Resource Usage Patterns",
        time_scale: str = "hourly",
    ) -> None:
        """
        Create a heatmap visualization of resource usage patterns.

        Args:
            resource_data: Dictionary of resource usage data
            container: Streamlit container to render in
            title: Title for the visualization
            time_scale: Time scale for binning
        """
        if time_scale not in self.config.TIME_BINS:
            raise ValueError(f"Unsupported time scale: {time_scale}")

        # Process resource data
        processed_data = self._process_resource_data(resource_data, time_scale)

        # Create subplots for different resource types
        fig = make_subplots(
            rows=len(processed_data), cols=1, subplot_titles=list(processed_data.keys())
        )

        # Add heatmaps for each resource type
        for i, (resource_type, data) in enumerate(processed_data.items(), 1):
            fig.add_trace(
                go.Heatmap(
                    z=data["values"],
                    x=data["time_labels"],
                    y=data["agent_labels"],
                    colorscale=self.config.COLOR_SCALES["resource"],
                    colorbar=dict(title=f"{resource_type} Usage", tickformat=".2f"),
                ),
                row=i,
                col=1,
            )

        # Configure layout
        fig.update_layout(
            title=title,
            showlegend=False,
            width=self.config.LAYOUT_WIDTH,
            height=self.config.LAYOUT_HEIGHT * len(processed_data),
        )

        # Render in Streamlit
        container.plotly_chart(fig, use_container_width=True)

    def _process_activity_data(
        self, activity_data: Dict[str, Dict[str, List[float]]], time_scale: str
    ) -> Dict[str, Any]:
        """Process activity data for visualization."""
        num_bins = self.config.TIME_BINS[time_scale]

        # Initialize arrays
        agents = sorted(activity_data.keys())
        values = np.zeros((len(agents), num_bins))

        # Process data for each agent
        for i, agent_id in enumerate(agents):
            agent_activities = activity_data[agent_id]
            for time_period in range(num_bins):
                values[i][time_period] = np.mean(
                    [
                        activities[time_period]
                        for activities in agent_activities.values()
                    ]
                )

        # Generate time labels
        if time_scale == "hourly":
            time_labels = [f"{i:02d}:00" for i in range(24)]
        elif time_scale == "daily":
            time_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        elif time_scale == "weekly":
            time_labels = [f"Week {i+1}" for i in range(4)]
        else:  # monthly
            time_labels = [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]

        return {"values": values, "agent_labels": agents, "time_labels": time_labels}

    def _process_resource_data(
        self, resource_data: Dict[str, Dict[str, List[float]]], time_scale: str
    ) -> Dict[str, Dict[str, Any]]:
        """Process resource data for visualization."""
        processed_data = {}
        num_bins = self.config.TIME_BINS[time_scale]

        for resource_type, agent_data in resource_data.items():
            agents = sorted(agent_data.keys())
            values = np.zeros((len(agents), num_bins))

            # Process data for each agent
            for i, agent_id in enumerate(agents):
                usage_data = agent_data[agent_id]
                for time_period in range(num_bins):
                    values[i][time_period] = usage_data[time_period]

            # Generate time labels
            if time_scale == "hourly":
                time_labels = [f"{i:02d}:00" for i in range(24)]
            elif time_scale == "daily":
                time_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            elif time_scale == "weekly":
                time_labels = [f"Week {i+1}" for i in range(4)]
            else:  # monthly
                time_labels = [
                    "Jan",
                    "Feb",
                    "Mar",
                    "Apr",
                    "May",
                    "Jun",
                    "Jul",
                    "Aug",
                    "Sep",
                    "Oct",
                    "Nov",
                    "Dec",
                ]

            processed_data[resource_type] = {
                "values": values,
                "agent_labels": agents,
                "time_labels": time_labels,
            }

        return processed_data
