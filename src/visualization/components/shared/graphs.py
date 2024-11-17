"""
Graph visualization components for the Agent Reasoning Beta platform.

This module provides network visualizations for:
- Agent interaction networks
- Reasoning path dependencies
- Confidence flow graphs
- Performance metrics networks
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core.types import AgentRole, ReasoningType
from core.tools import VisualizationTools

class GraphConfig:
    """Configuration for graph visualization."""
    # Layout
    LAYOUT_WIDTH = 1200
    LAYOUT_HEIGHT = 800
    MIN_NODE_DISTANCE = 100
    
    # Node styling
    NODE_SIZE_RANGE = (20, 50)
    EDGE_WIDTH_RANGE = (1, 5)
    
    # Color schemes
    COLOR_SCHEMES = {
        AgentRole.EXPLORER: {
            "node": "#1f77b4",
            "edge": "#7fcdbb",
            "text": "#2c3e50"
        },
        AgentRole.VERIFIER: {
            "node": "#2ca02c",
            "edge": "#98df8a",
            "text": "#2c3e50"
        },
        AgentRole.COORDINATOR: {
            "node": "#ff7f0e",
            "edge": "#ffbb78",
            "text": "#2c3e50"
        },
        AgentRole.OBSERVER: {
            "node": "#9467bd",
            "edge": "#c5b0d5",
            "text": "#2c3e50"
        }
    }
    
    # Animation
    ANIMATION_DURATION = 500
    TRANSITION_EASING = "cubic-in-out"

class GraphVisualizer:
    """Interactive graph visualization component."""
    
    def __init__(self, config: Optional[GraphConfig] = None):
        self.config = config or GraphConfig()
        self._graph = nx.Graph()
        self._layout: Optional[Dict[str, Tuple[float, float]]] = None
    
    async def visualize_agent_network(
        self,
        agents: List[Dict[str, Any]],
        interactions: List[Dict[str, Any]],
        container: st.container,
        title: str = "Agent Interaction Network",
        layout_type: str = "spring"
    ) -> None:
        """
        Create an interactive visualization of agent interactions.
        
        Args:
            agents: List of agent data
            interactions: List of agent interactions
            container: Streamlit container to render in
            title: Title for the visualization
            layout_type: Type of layout algorithm ('spring', 'circular', 'kamada_kawai')
        """
        # Build networkx graph
        self._build_graph(agents, interactions)
        
        # Calculate layout
        self._layout = self._compute_layout(layout_type)
        
        # Create figure
        fig = self._create_figure(title)
        
        # Add nodes and edges
        self._add_nodes(fig, agents)
        self._add_edges(fig, interactions)
        
        # Configure layout
        self._configure_layout(fig)
        
        # Add metrics summary
        self._add_metrics_summary(fig, agents)
        
        # Render in Streamlit
        container.plotly_chart(fig, use_container_width=True)
    
    async def visualize_performance_network(
        self,
        metrics: Dict[str, Dict[str, float]],
        container: st.container,
        title: str = "Agent Performance Network",
        layout_type: str = "circular"
    ) -> None:
        """
        Create an interactive visualization of agent performance metrics.
        
        Args:
            metrics: Dictionary of agent metrics
            container: Streamlit container to render in
            title: Title for the visualization
            layout_type: Type of layout algorithm
        """
        # Create graph from metrics
        self._graph.clear()
        for agent_id, agent_metrics in metrics.items():
            self._graph.add_node(
                agent_id,
                metrics=agent_metrics
            )
        
        # Add edges based on metric correlations
        for agent1 in metrics:
            for agent2 in metrics:
                if agent1 < agent2:  # Avoid duplicate edges
                    correlation = self._compute_metric_correlation(
                        metrics[agent1],
                        metrics[agent2]
                    )
                    if correlation > 0.5:  # Only show strong correlations
                        self._graph.add_edge(
                            agent1,
                            agent2,
                            weight=correlation
                        )
        
        # Calculate layout
        self._layout = self._compute_layout(layout_type)
        
        # Create figure
        fig = self._create_figure(title)
        
        # Add visualization elements
        self._add_performance_nodes(fig, metrics)
        self._add_performance_edges(fig)
        
        # Configure layout
        self._configure_layout(fig)
        
        # Render in Streamlit
        container.plotly_chart(fig, use_container_width=True)
    
    def _build_graph(
        self,
        agents: List[Dict[str, Any]],
        interactions: List[Dict[str, Any]]
    ) -> None:
        """Build networkx graph from agent and interaction data."""
        self._graph.clear()
        
        # Add agent nodes
        for agent in agents:
            self._graph.add_node(
                agent["id"],
                **agent
            )
        
        # Add interaction edges
        for interaction in interactions:
            self._graph.add_edge(
                interaction["from"],
                interaction["to"],
                **interaction
            )
    
    def _compute_layout(self, layout_type: str) -> Dict[str, Tuple[float, float]]:
        """Compute graph layout using specified algorithm."""
        if layout_type == "spring":
            return nx.spring_layout(
                self._graph,
                k=self.config.MIN_NODE_DISTANCE / 1000,
                iterations=50
            )
        elif layout_type == "circular":
            return nx.circular_layout(self._graph)
        elif layout_type == "kamada_kawai":
            return nx.kamada_kawai_layout(self._graph)
        else:
            raise ValueError(f"Unsupported layout type: {layout_type}")
    
    def _create_figure(self, title: str) -> go.Figure:
        """Create base plotly figure."""
        fig = go.Figure()
        
        fig.update_layout(
            title=title,
            showlegend=True,
            hovermode="closest",
            width=self.config.LAYOUT_WIDTH,
            height=self.config.LAYOUT_HEIGHT,
            template="plotly_white"
        )
        
        return fig
    
    def _add_nodes(self, fig: go.Figure, agents: List[Dict[str, Any]]) -> None:
        """Add agent nodes to the figure."""
        x_coords = []
        y_coords = []
        node_sizes = []
        node_colors = []
        hover_texts = []
        
        for agent in agents:
            pos = self._layout[agent["id"]]
            x_coords.append(pos[0])
            y_coords.append(pos[1])
            
            # Size based on activity level
            size = 30 + (agent["metrics"]["total_interactions"] * 2)
            node_sizes.append(size)
            
            # Color based on role
            node_colors.append(
                self.config.COLOR_SCHEMES[AgentRole(agent["role"])]["node"]
            )
            
            # Hover text
            hover_texts.append(
                f"Agent: {agent['name']}<br>"
                f"Role: {agent['role']}<br>"
                f"Success Rate: {agent['metrics']['success_rate']:.2f}<br>"
                f"Total Interactions: {agent['metrics']['total_interactions']}"
            )
        
        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="markers+text",
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    symbol="circle",
                    line=dict(width=2, color="#ffffff")
                ),
                text=[a["name"] for a in agents],
                hovertext=hover_texts,
                hoverinfo="text",
                name="Agents"
            )
        )
    
    def _add_edges(
        self,
        fig: go.Figure,
        interactions: List[Dict[str, Any]]
    ) -> None:
        """Add interaction edges to the figure."""
        for interaction in interactions:
            start_pos = self._layout[interaction["from"]]
            end_pos = self._layout[interaction["to"]]
            
            # Edge width based on interaction strength
            width = (
                interaction.get("strength", 0.5) *
                (self.config.EDGE_WIDTH_RANGE[1] - self.config.EDGE_WIDTH_RANGE[0]) +
                self.config.EDGE_WIDTH_RANGE[0]
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[start_pos[0], end_pos[0]],
                    y=[start_pos[1], end_pos[1]],
                    mode="lines",
                    line=dict(
                        width=width,
                        color=self.config.COLOR_SCHEMES[
                            AgentRole(interaction["type"])
                        ]["edge"]
                    ),
                    hoverinfo="text",
                    hovertext=(
                        f"Type: {interaction['type']}<br>"
                        f"Strength: {interaction.get('strength', 0.5):.2f}<br>"
                        f"Time: {interaction['timestamp']}"
                    ),
                    showlegend=False
                )
            )
    
    def _configure_layout(self, fig: go.Figure) -> None:
        """Configure figure layout."""
        fig.update_layout(
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            plot_bgcolor="white"
        )
    
    def _add_metrics_summary(
        self,
        fig: go.Figure,
        agents: List[Dict[str, Any]]
    ) -> None:
        """Add summary of agent metrics."""
        metrics_text = "Network Metrics:<br>"
        
        # Calculate network-level metrics
        total_agents = len(agents)
        total_interactions = sum(
            agent["metrics"]["total_interactions"]
            for agent in agents
        )
        avg_success_rate = sum(
            agent["metrics"]["success_rate"]
            for agent in agents
        ) / total_agents
        
        metrics_text += (
            f"Total Agents: {total_agents}<br>"
            f"Total Interactions: {total_interactions}<br>"
            f"Avg Success Rate: {avg_success_rate:.2f}"
        )
        
        fig.add_annotation(
            text=metrics_text,
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            showarrow=False,
            font=dict(size=12),
            align="left"
        )
    
    def _add_performance_nodes(
        self,
        fig: go.Figure,
        metrics: Dict[str, Dict[str, float]]
    ) -> None:
        """Add performance metric nodes."""
        x_coords = []
        y_coords = []
        node_sizes = []
        node_colors = []
        hover_texts = []
        
        for agent_id, agent_metrics in metrics.items():
            pos = self._layout[agent_id]
            x_coords.append(pos[0])
            y_coords.append(pos[1])
            
            # Size based on overall performance
            size = 30 + (agent_metrics["performance_score"] * 20)
            node_sizes.append(size)
            
            # Color based on success rate
            node_colors.append(
                self._get_performance_color(agent_metrics["success_rate"])
            )
            
            # Hover text with all metrics
            hover_texts.append(
                "<br>".join(
                    f"{k}: {v:.2f}"
                    for k, v in agent_metrics.items()
                )
            )
        
        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="markers+text",
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    symbol="diamond",
                    line=dict(width=2, color="#ffffff")
                ),
                text=list(metrics.keys()),
                hovertext=hover_texts,
                hoverinfo="text",
                name="Performance Metrics"
            )
        )
    
    def _add_performance_edges(self, fig: go.Figure) -> None:
        """Add edges between correlated performance metrics."""
        for edge in self._graph.edges(data=True):
            start_pos = self._layout[edge[0]]
            end_pos = self._layout[edge[1]]
            
            # Edge width based on correlation strength
            width = edge[2]["weight"] * 5
            
            fig.add_trace(
                go.Scatter(
                    x=[start_pos[0], end_pos[0]],
                    y=[start_pos[1], end_pos[1]],
                    mode="lines",
                    line=dict(
                        width=width,
                        color="rgba(100, 100, 100, 0.3)"
                    ),
                    hoverinfo="text",
                    hovertext=f"Correlation: {edge[2]['weight']:.2f}",
                    showlegend=False
                )
            )
    
    def _compute_metric_correlation(
        self,
        metrics1: Dict[str, float],
        metrics2: Dict[str, float]
    ) -> float:
        """Compute correlation between two sets of metrics."""
        # Simple correlation based on common metrics
        common_metrics = set(metrics1.keys()) & set(metrics2.keys())
        if not common_metrics:
            return 0.0
        
        values1 = [metrics1[k] for k in common_metrics]
        values2 = [metrics2[k] for k in common_metrics]
        
        # Compute correlation coefficient
        mean1 = sum(values1) / len(values1)
        mean2 = sum(values2) / len(values2)
        
        num = sum((x - mean1) * (y - mean2) for x, y in zip(values1, values2))
        den1 = sum((x - mean1) ** 2 for x in values1)
        den2 = sum((y - mean2) ** 2 for y in values2)
        
        if den1 == 0 or den2 == 0:
            return 0.0
            
        return num / (den1 * den2) ** 0.5
    
    def _get_performance_color(self, success_rate: float) -> str:
        """Get color based on performance level."""
        if success_rate >= 0.8:
            return "#2ecc71"  # Green
        elif success_rate >= 0.6:
            return "#f1c40f"  # Yellow
        elif success_rate >= 0.4:
            return "#e67e22"  # Orange
        else:
            return "#e74c3c"  # Red
