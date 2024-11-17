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
import math
import networkx as nx
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from src.core.types import AgentRole, ReasoningType

class GraphConfig:
    """Configuration for graph visualization."""
    
    # Layout
    LAYOUT_WIDTH = 1200
    LAYOUT_HEIGHT = 800
    MIN_NODE_DISTANCE = 100
    
    # Node styling
    NODE_SIZE_RANGE = (20, 50)
    NODE_LINE_WIDTH = 2
    
    # Edge styling
    EDGE_WIDTH_RANGE = (1, 5)
    EDGE_OPACITY = 0.6
    
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

class GraphVisualizer:
    """Interactive graph visualization component with performance optimizations."""
    
    def __init__(self, config: Optional[GraphConfig] = None):
        self.config = config or GraphConfig()
        self._graph = nx.Graph()
        self._layout = None
        self._node_positions = {}
        self._edge_traces = []
        self._node_traces = []
        self.performance_mode = False
    
    def optimize_layout(self) -> None:
        """Optimize graph layout for performance."""
        node_count = len(self._graph.nodes)
        
        # Enable performance mode for large graphs
        self.performance_mode = node_count > 100
        
        if self.performance_mode:
            # Use faster layout algorithm for large graphs
            self._layout = nx.spring_layout(
                self._graph,
                k=1/math.sqrt(node_count),  # Optimal node spacing
                iterations=50,  # Reduced iterations
                seed=42  # Consistent layout
            )
        else:
            # Use more aesthetically pleasing layout for smaller graphs
            self._layout = nx.kamada_kawai_layout(self._graph)
        
        # Cache node positions
        self._node_positions = {
            node: (pos[0], pos[1]) 
            for node, pos in self._layout.items()
        }
    
    async def visualize_agent_network(
        self,
        agents: List[Dict[str, Any]],
        interactions: List[Dict[str, Any]],
        container: Any,
        title: str = "Agent Interaction Network",
        layout_type: str = "spring"
    ) -> None:
        """
        Visualize agent interaction network with performance optimizations.
        
        Args:
            agents: List of agent data
            interactions: List of interaction data
            container: Streamlit container to render in
            title: Title for the visualization
            layout_type: Type of layout algorithm
        """
        # Build graph
        self._graph.clear()
        for agent in agents:
            self._graph.add_node(
                agent["id"],
                **agent
            )
        for interaction in interactions:
            self._graph.add_edge(
                interaction["from"],
                interaction["to"],
                **interaction
            )
        
        # Optimize layout
        self.optimize_layout()
        
        # Create traces with batching
        self._create_agent_traces(agents)
        self._create_interaction_traces(interactions)
        
        # Create and configure figure
        fig = go.Figure(
            data=self._edge_traces + self._node_traces,
            layout=go.Layout(
                title=title,
                showlegend=True,
                hovermode="closest",
                width=self.config.LAYOUT_WIDTH,
                height=self.config.LAYOUT_HEIGHT,
                template="plotly_white",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        
        # Add metrics summary if not in performance mode
        if not self.performance_mode:
            self._add_metrics_summary(fig, agents)
        
        # Render in Streamlit
        container.plotly_chart(fig, use_container_width=True)
    
    def _create_agent_traces(self, agents: List[Dict[str, Any]]) -> None:
        """Create agent node traces with batching."""
        if not self._layout:
            self.optimize_layout()
        
        # Batch node creation
        x_coords = []
        y_coords = []
        node_sizes = []
        node_colors = []
        hover_texts = []
        
        for agent in agents:
            pos = self._node_positions[agent["id"]]
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
        
        self._node_traces = [go.Scatter(
            x=x_coords,
            y=y_coords,
            mode="markers+text" if not self.performance_mode else "markers",
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
        )]
    
    def _create_interaction_traces(
        self,
        interactions: List[Dict[str, Any]]
    ) -> None:
        """Create interaction edge traces with batching."""
        if not self._layout:
            self.optimize_layout()
        
        # Batch edge creation
        edge_x = []
        edge_y = []
        edge_colors = []
        edge_widths = []
        hover_texts = []
        
        for interaction in interactions:
            start_pos = self._node_positions[interaction["from"]]
            end_pos = self._node_positions[interaction["to"]]
            
            edge_x.extend([start_pos[0], end_pos[0], None])
            edge_y.extend([start_pos[1], end_pos[1], None])
            
            # Edge styling
            color = self.config.COLOR_SCHEMES[
                AgentRole(interaction["type"])
            ]["edge"]
            edge_colors.extend([color, color, color])
            
            width = (
                interaction.get("strength", 0.5) *
                (self.config.EDGE_WIDTH_RANGE[1] - self.config.EDGE_WIDTH_RANGE[0]) +
                self.config.EDGE_WIDTH_RANGE[0]
            )
            edge_widths.extend([width, width, width])
            
            hover_text = (
                f"Type: {interaction['type']}<br>"
                f"Strength: {interaction.get('strength', 0.5):.2f}<br>"
                f"Time: {interaction['timestamp']}"
            )
            hover_texts.extend([hover_text, hover_text, None])
        
        self._edge_traces = [go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(
                width=edge_widths,
                color=edge_colors,
                shape="spline" if not self.performance_mode else "linear"
            ),
            hoverinfo="text",
            hovertext=hover_texts,
            opacity=self.config.EDGE_OPACITY,
            showlegend=False
        )]
    
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
