"""
Consensus visualization for the Agent Reasoning Beta platform.

This module provides a simplified visualization for agent consensus:
- Agreement matrix
- Resolution path tracking
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.core.types import AgentVote, ConsensusMetrics


class ConsensusConfig:
    """Configuration for consensus visualization."""

    # Layout
    LAYOUT_WIDTH = 1000
    LAYOUT_HEIGHT = 500

    # Colors
    HIGH_AGREEMENT = "#2ecc71"
    MED_AGREEMENT = "#f1c40f"
    LOW_AGREEMENT = "#e74c3c"
    PATH_COLOR = "#3498db"


class ConsensusVisualizer:
    """Simple consensus visualization component."""

    def __init__(self, config: Optional[ConsensusConfig] = None):
        """Initialize visualizer with config."""
        self.config = config or ConsensusConfig()

    def visualize_consensus(
        self,
        metrics: ConsensusMetrics,
        container: st.container,
        title: str = "Agent Consensus",
    ):
        """
        Create a simple visualization of agent consensus.

        Args:
            metrics: Consensus metrics data
            container: Streamlit container to render in
            title: Title for the visualization
        """
        # Create figure with two subplots
        fig = make_subplots(
            rows=1, cols=2, subplot_titles=("Agreement Matrix", "Resolution Path")
        )

        # Add agreement matrix
        self._add_agreement_matrix(fig, metrics, row=1, col=1)

        # Add resolution path
        self._add_resolution_path(fig, metrics, row=1, col=2)

        # Update layout
        fig.update_layout(
            title=title,
            showlegend=True,
            width=self.config.LAYOUT_WIDTH,
            height=self.config.LAYOUT_HEIGHT,
        )

        # Add metrics summary
        metrics_col1, metrics_col2 = container.columns(2)
        with metrics_col1:
            st.write("Agreement Statistics:")
            st.write(f"Mean Agreement: {metrics.mean_agreement:.3f}")

        with metrics_col2:
            st.write("Resolution Status:")
            st.write(f"Status: {metrics.resolution_status}")

        container.plotly_chart(fig, use_container_width=True)

    def _add_agreement_matrix(
        self, fig: go.Figure, metrics: ConsensusMetrics, row: int, col: int
    ):
        """Add agreement matrix subplot."""
        # Create agreement matrix
        agents = list(metrics.agent_pairs.keys())
        matrix = np.zeros((len(agents), len(agents)))

        for i, agent1 in enumerate(agents):
            for j, agent2 in enumerate(agents):
                if i != j:
                    pair = tuple(sorted([agent1, agent2]))
                    matrix[i][j] = metrics.agent_pairs[pair].agreement

        # Add heatmap
        heatmap = go.Heatmap(
            z=matrix,
            x=agents,
            y=agents,
            colorscale=[
                [0, self.config.LOW_AGREEMENT],
                [0.5, self.config.MED_AGREEMENT],
                [1, self.config.HIGH_AGREEMENT],
            ],
            showscale=True,
            colorbar=dict(title="Agreement"),
        )

        fig.add_trace(heatmap, row=row, col=col)

    def _add_resolution_path(
        self, fig: go.Figure, metrics: ConsensusMetrics, row: int, col: int
    ):
        """Add resolution path subplot."""
        # Create path visualization
        times = sorted(metrics.resolution_path.keys())
        agreements = [metrics.resolution_path[t].agreement for t in times]

        scatter = go.Scatter(
            x=times,
            y=agreements,
            mode="lines+markers",
            name="Resolution",
            line=dict(color=self.config.PATH_COLOR),
            hovertemplate="Time: %{x}<br>Agreement: %{y:.3f}",
        )

        fig.add_trace(scatter, row=row, col=col)
        fig.update_yaxes(title_text="Agreement", range=[0, 1], row=row, col=col)
        fig.update_xaxes(title_text="Time", row=row, col=col)


class ConsensusView:
    """Main consensus view component."""

    def __init__(self):
        """Initialize consensus view."""
        self.visualizer = ConsensusVisualizer()

    def render(
        self,
        metrics: Optional[ConsensusMetrics] = None,
        container: Optional[st.container] = None,
    ):
        """
        Render the consensus view.

        Args:
            metrics: Consensus metrics to visualize
            container: Streamlit container to render in
        """
        container = container or st.container()

        with container:
            st.header("Consensus View")

            if metrics is None:
                st.info(
                    "No consensus data available. Start consensus building to analyze agent agreement."
                )
                return

            # Add consensus controls
            control_col1, control_col2 = st.columns(2)
            with control_col1:
                st.subheader("Visualization Controls")
                show_matrix = st.checkbox("Show Agreement Matrix", value=True)
                show_path = st.checkbox("Show Resolution Path", value=True)

            with control_col2:
                st.subheader("Analysis Controls")
                min_agreement = st.slider(
                    "Min Agreement", min_value=0.0, max_value=1.0, value=0.5
                )
                highlight_conflicts = st.checkbox("Highlight Conflicts", value=True)

            # Create visualization container
            viz_container = st.container()
            self.visualizer.visualize_consensus(
                metrics=metrics, container=viz_container, title="Consensus Analysis"
            )
