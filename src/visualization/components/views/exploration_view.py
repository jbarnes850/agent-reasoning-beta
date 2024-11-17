"""
MCTS exploration visualization for the Agent Reasoning Beta platform.

This module provides a simplified visualization for Monte Carlo Tree Search:
- Tree structure visualization
- Node confidence coloring
- Path highlighting
"""

from __future__ import annotations

from typing import Dict, List, Optional
from uuid import UUID

import networkx as nx
import plotly.graph_objects as go
import streamlit as st

from core.types import ThoughtNode
from visualization.components.shared.trees import TreeVisualizer, TreeConfig

class MCTSConfig(TreeConfig):
    """Configuration for MCTS visualization."""
    # Layout
    LAYOUT_WIDTH = 1000
    LAYOUT_HEIGHT = 600
    
    # Node styling
    NODE_SIZE = 20
    EDGE_WIDTH = 1
    
    # Colors
    HIGH_CONFIDENCE = "#2ecc71"
    MED_CONFIDENCE = "#f1c40f"
    LOW_CONFIDENCE = "#e74c3c"
    EDGE_COLOR = "#888"
    
    # Hover text
    HOVER_TEMPLATE = """
    Node: %{customdata[0]}
    Confidence: %{customdata[1]:.3f}
    """

class MCTSVisualizer(TreeVisualizer):
    """Simple MCTS visualization component."""
    
    def __init__(self, config: Optional[MCTSConfig] = None):
        """Initialize visualizer with config."""
        super().__init__(config or MCTSConfig())
        self._graph = None
        self._layout = None
    
    def visualize_mcts(
        self,
        root_node: ThoughtNode,
        container: st.container,
        title: str = "MCTS Exploration"
    ):
        """
        Create a simple visualization of MCTS exploration.
        
        Args:
            root_node: Root node of the MCTS tree
            container: Streamlit container to render in
            title: Title for the visualization
        """
        # Build graph
        self._build_graph(root_node)
        
        # Create figure
        fig = go.Figure()
        
        # Add visualization elements
        self._add_nodes(fig)
        self._add_edges(fig)
        
        # Configure layout
        fig.update_layout(
            title=title,
            showlegend=False,
            width=self.config.LAYOUT_WIDTH,
            height=self.config.LAYOUT_HEIGHT,
            hovermode="closest"
        )
        
        # Add metrics summary
        metrics_col1, metrics_col2 = container.columns(2)
        with metrics_col1:
            st.write("Tree Statistics:")
            st.write(f"Total Nodes: {len(self._graph.nodes)}")
            st.write(f"Max Depth: {self._get_max_depth()}")
        
        with metrics_col2:
            st.write("Confidence Summary:")
            confidences = [
                data["confidence"]
                for _, data in self._graph.nodes(data=True)
            ]
            if confidences:
                st.write(f"Mean Confidence: {sum(confidences)/len(confidences):.3f}")
                st.write(f"Max Confidence: {max(confidences):.3f}")
        
        container.plotly_chart(fig, use_container_width=True)
    
    def _build_graph(self, root_node: ThoughtNode):
        """Build networkx graph from MCTS tree."""
        self._graph = nx.DiGraph()
        self._add_node_recursive(root_node)
        self._layout = nx.spring_layout(self._graph)
    
    def _add_node_recursive(self, node: ThoughtNode):
        """Add node and its children recursively."""
        self._graph.add_node(
            node.id,
            confidence=node.confidence
        )
        
        for child in node.children:
            self._graph.add_edge(node.id, child.id)
            self._add_node_recursive(child)
    
    def _add_nodes(self, fig: go.Figure):
        """Add nodes to figure with confidence-based coloring."""
        node_x = []
        node_y = []
        node_color = []
        node_data = []
        
        for node_id in self._graph.nodes():
            pos = self._layout[node_id]
            confidence = self._graph.nodes[node_id]["confidence"]
            
            node_x.append(pos[0])
            node_y.append(pos[1])
            node_color.append(self._get_confidence_color(confidence))
            node_data.append([str(node_id), confidence])
        
        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers",
                marker=dict(
                    size=self.config.NODE_SIZE,
                    color=node_color
                ),
                customdata=node_data,
                hovertemplate=self.config.HOVER_TEMPLATE
            )
        )
    
    def _add_edges(self, fig: go.Figure):
        """Add edges to figure."""
        edge_x = []
        edge_y = []
        
        for edge in self._graph.edges():
            x0, y0 = self._layout[edge[0]]
            x1, y1 = self._layout[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(
                    width=self.config.EDGE_WIDTH,
                    color=self.config.EDGE_COLOR
                ),
                hoverinfo="none"
            )
        )
    
    def _get_confidence_color(self, confidence: float) -> str:
        """Get color based on confidence value."""
        if confidence >= 0.7:
            return self.config.HIGH_CONFIDENCE
        elif confidence >= 0.4:
            return self.config.MED_CONFIDENCE
        else:
            return self.config.LOW_CONFIDENCE
    
    def _get_max_depth(self) -> int:
        """Calculate maximum depth of the tree."""
        if not self._graph:
            return 0
            
        root = [n for n, d in self._graph.in_degree() if d == 0][0]
        return max(len(path) for path in nx.all_simple_paths(self._graph, root))
