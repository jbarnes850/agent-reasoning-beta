"""
Tree visualization components for the Agent Reasoning Beta platform.

This module provides interactive tree visualizations for:
- MCTS exploration trees
- Reasoning path hierarchies
- Agent interaction networks
- Decision process flows
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import networkx as nx
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core.types import ReasoningPath, ReasoningType, ThoughtNode
from core.tools import VisualizationTools

class TreeConfig:
    """Configuration for tree visualization."""
    # Tree layout
    LAYOUT_WIDTH = 1200
    LAYOUT_HEIGHT = 800
    MIN_NODE_DISTANCE = 100
    
    # Node styling
    NODE_SIZE_RANGE = (20, 50)
    EDGE_WIDTH_RANGE = (1, 5)
    
    # Color schemes
    COLOR_SCHEMES = {
        ReasoningType.MCTS: {
            "node": "#1f77b4",
            "edge": "#7fcdbb",
            "text": "#2c3e50"
        },
        ReasoningType.VERIFICATION: {
            "node": "#2ca02c",
            "edge": "#98df8a",
            "text": "#2c3e50"
        },
        ReasoningType.CONSENSUS: {
            "node": "#ff7f0e",
            "edge": "#ffbb78",
            "text": "#2c3e50"
        }
    }
    
    # Animation
    ANIMATION_DURATION = 500
    TRANSITION_EASING = "cubic-in-out"

class TreeVisualizer:
    """Interactive tree visualization component."""
    
    def __init__(self, config: Optional[TreeConfig] = None):
        self.config = config or TreeConfig()
        self._graph = nx.DiGraph()
        self._layout: Optional[Dict[UUID, Tuple[float, float]]] = None
    
    async def visualize_reasoning_path(
        self,
        path: ReasoningPath,
        container: st.container,
        title: str = "Reasoning Path Visualization"
    ) -> None:
        """
        Create an interactive visualization of a reasoning path.
        
        Args:
            path: ReasoningPath to visualize
            container: Streamlit container to render in
            title: Title for the visualization
        """
        # Prepare data
        tree_data = await VisualizationTools.prepare_tree_data(path)
        
        # Build networkx graph
        self._build_graph(tree_data)
        
        # Calculate layout
        self._layout = nx.spring_layout(
            self._graph,
            k=self.config.MIN_NODE_DISTANCE / 1000,
            iterations=50
        )
        
        # Create plotly figure
        fig = self._create_figure(title)
        
        # Add nodes
        self._add_nodes(fig, tree_data["nodes"])
        
        # Add edges
        self._add_edges(fig, tree_data["edges"])
        
        # Add metadata
        self._add_metadata(fig, tree_data["metadata"])
        
        # Configure layout
        self._configure_layout(fig)
        
        # Render in Streamlit
        container.plotly_chart(fig, use_container_width=True)
    
    async def visualize_agent_network(
        self,
        agents: List[Dict[str, Any]],
        interactions: List[Dict[str, Any]],
        container: st.container,
        title: str = "Agent Interaction Network"
    ) -> None:
        """
        Create an interactive visualization of agent interactions.
        
        Args:
            agents: List of agent data
            interactions: List of agent interactions
            container: Streamlit container to render in
            title: Title for the visualization
        """
        # Build networkx graph
        self._graph.clear()
        for agent in agents:
            self._graph.add_node(
                agent["id"],
                label=agent["name"],
                role=agent["role"],
                metrics=agent["metrics"]
            )
        
        for interaction in interactions:
            self._graph.add_edge(
                interaction["from"],
                interaction["to"],
                type=interaction["type"],
                timestamp=interaction["timestamp"]
            )
        
        # Calculate layout
        self._layout = nx.spring_layout(
            self._graph,
            k=self.config.MIN_NODE_DISTANCE / 1000,
            iterations=50
        )
        
        # Create figure
        fig = self._create_figure(title)
        
        # Add agent nodes
        self._add_agent_nodes(fig, agents)
        
        # Add interaction edges
        self._add_interaction_edges(fig, interactions)
        
        # Configure layout
        self._configure_layout(fig)
        
        # Render in Streamlit
        container.plotly_chart(fig, use_container_width=True)
    
    def _build_graph(self, tree_data: Dict[str, Any]) -> None:
        """Build networkx graph from tree data."""
        self._graph.clear()
        
        # Add nodes
        for node in tree_data["nodes"]:
            self._graph.add_node(
                node["id"],
                label=node["label"],
                confidence=node["confidence"],
                type=node["type"]
            )
        
        # Add edges
        for edge in tree_data["edges"]:
            self._graph.add_edge(edge["from"], edge["to"])
    
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
    
    def _add_nodes(self, fig: go.Figure, nodes: List[Dict[str, Any]]) -> None:
        """Add nodes to the figure."""
        x_coords = []
        y_coords = []
        node_sizes = []
        node_colors = []
        hover_texts = []
        
        for node in nodes:
            pos = self._layout[node["id"]]
            x_coords.append(pos[0])
            y_coords.append(pos[1])
            
            # Scale node size by confidence
            size = (node["confidence"] * 
                   (self.config.NODE_SIZE_RANGE[1] - self.config.NODE_SIZE_RANGE[0]) +
                   self.config.NODE_SIZE_RANGE[0])
            node_sizes.append(size)
            
            # Get color from scheme
            color_scheme = self.config.COLOR_SCHEMES[ReasoningType(node["type"])]
            node_colors.append(color_scheme["node"])
            
            # Create hover text
            hover_texts.append(
                f"ID: {node['id']}<br>"
                f"Confidence: {node['confidence']:.2f}<br>"
                f"Type: {node['type']}<br>"
                f"Content: {node['label']}"
            )
        
        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="markers+text",
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    line=dict(width=1, color="#ffffff")
                ),
                text=[n["label"] for n in nodes],
                hovertext=hover_texts,
                hoverinfo="text",
                name="Nodes"
            )
        )
    
    def _add_edges(self, fig: go.Figure, edges: List[Dict[str, Any]]) -> None:
        """Add edges to the figure."""
        for edge in edges:
            start_pos = self._layout[edge["from"]]
            end_pos = self._layout[edge["to"]]
            
            fig.add_trace(
                go.Scatter(
                    x=[start_pos[0], end_pos[0]],
                    y=[start_pos[1], end_pos[1]],
                    mode="lines",
                    line=dict(
                        width=self.config.EDGE_WIDTH_RANGE[0],
                        color=self.config.COLOR_SCHEMES[ReasoningType.MCTS]["edge"]
                    ),
                    hoverinfo="none",
                    showlegend=False
                )
            )
    
    def _add_metadata(self, fig: go.Figure, metadata: Dict[str, Any]) -> None:
        """Add metadata annotations to the figure."""
        fig.add_annotation(
            text=(
                f"Total Confidence: {metadata['total_confidence']:.2f}<br>"
                f"Verified: {'Yes' if metadata['verified'] else 'No'}<br>"
                f"Consensus: {'Yes' if metadata['consensus_reached'] else 'No'}"
            ),
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            showarrow=False,
            font=dict(size=12),
            align="left"
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
    
    def _add_agent_nodes(
        self,
        fig: go.Figure,
        agents: List[Dict[str, Any]]
    ) -> None:
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
            
            # Size based on activity
            size = 30 + (agent["metrics"]["total_thoughts"] * 2)
            node_sizes.append(size)
            
            # Color based on role
            node_colors.append(self.config.COLOR_SCHEMES[agent["role"]]["node"])
            
            # Hover text
            hover_texts.append(
                f"Agent: {agent['name']}<br>"
                f"Role: {agent['role']}<br>"
                f"Thoughts: {agent['metrics']['total_thoughts']}<br>"
                f"Success Rate: {agent['metrics']['success_rate']:.2f}"
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
                text=[a["name"] for a in agents],
                hovertext=hover_texts,
                hoverinfo="text",
                name="Agents"
            )
        )
    
    def _add_interaction_edges(
        self,
        fig: go.Figure,
        interactions: List[Dict[str, Any]]
    ) -> None:
        """Add interaction edges to the figure."""
        for interaction in interactions:
            start_pos = self._layout[interaction["from"]]
            end_pos = self._layout[interaction["to"]]
            
            fig.add_trace(
                go.Scatter(
                    x=[start_pos[0], end_pos[0]],
                    y=[start_pos[1], end_pos[1]],
                    mode="lines+text",
                    line=dict(
                        width=2,
                        color=self.config.COLOR_SCHEMES[interaction["type"]]["edge"]
                    ),
                    text=[interaction["type"]],
                    textposition="middle center",
                    hoverinfo="text",
                    hovertext=f"Type: {interaction['type']}<br>"
                             f"Time: {interaction['timestamp']}",
                    showlegend=False
                )
            )
