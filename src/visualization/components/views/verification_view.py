"""
Verification visualization for the Agent Reasoning Beta platform.

This module provides specialized visualization for reasoning verification:
- Proof trees
- Verification chains
- Evidence mapping
- Confidence scoring
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Union
from uuid import UUID

import networkx as nx
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core.types import VerificationChain, Evidence, ProofNode
from visualization.components.shared.trees import TreeVisualizer, TreeConfig

class VerificationConfig(TreeConfig):
    """Configuration for verification visualization."""
    # Layout
    LAYOUT_WIDTH = 1200
    LAYOUT_HEIGHT = 800
    
    # Node styling
    EVIDENCE_TYPES = {
        "direct": "#2ecc71",
        "indirect": "#f1c40f",
        "inferred": "#e74c3c"
    }
    
    CONFIDENCE_COLORS = {
        "high": "#2ecc71",
        "medium": "#f1c40f",
        "low": "#e74c3c"
    }
    
    # Hover text
    HOVER_TEMPLATE = """
    Node: %{customdata[0]}
    Type: %{customdata[1]}
    Confidence: %{customdata[2]:.3f}
    Evidence Count: %{customdata[3]}
    """

class VerificationVisualizer(TreeVisualizer):
    """Interactive verification visualization component."""
    
    def __init__(self, config: Optional[VerificationConfig] = None):
        """Initialize visualizer with config."""
        super().__init__(config or VerificationConfig())
    
    def visualize_verification(
        self,
        chain: VerificationChain,
        container: st.container,
        title: str = "Verification Chain"
    ):
        """
        Create an interactive visualization of verification chain.
        
        Args:
            chain: Verification chain to visualize
            container: Streamlit container to render in
            title: Title for the visualization
        """
        # Create figure with subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Proof Tree",
                "Evidence Map",
                "Confidence Distribution",
                "Verification Timeline"
            )
        )
        
        # Add proof tree
        self._add_proof_tree(fig, chain, row=1, col=1)
        
        # Add evidence map
        self._add_evidence_map(fig, chain, row=1, col=2)
        
        # Add confidence distribution
        self._add_confidence_distribution(fig, chain, row=2, col=1)
        
        # Add verification timeline
        self._add_verification_timeline(fig, chain, row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title=title,
            showlegend=True,
            width=self.config.LAYOUT_WIDTH,
            height=self.config.LAYOUT_HEIGHT
        )
        
        # Add filters and controls
        filters_col1, filters_col2 = container.columns(2)
        
        with filters_col1:
            evidence_types = st.multiselect(
                "Evidence Types",
                list(self.config.EVIDENCE_TYPES.keys()),
                default=list(self.config.EVIDENCE_TYPES.keys())
            )
        
        with filters_col2:
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1
            )
        
        # Add chain summary
        st.write("### Chain Summary")
        summary_col1, summary_col2 = container.columns(2)
        
        with summary_col1:
            st.write(f"Total Nodes: {len(chain.nodes)}")
            st.write(f"Evidence Count: {len(chain.evidence)}")
        
        with summary_col2:
            st.write(f"Mean Confidence: {chain.mean_confidence:.3f}")
            st.write(f"Verification Status: {chain.status}")
        
        container.plotly_chart(fig, use_container_width=True)
    
    def _add_proof_tree(
        self,
        fig: go.Figure,
        chain: VerificationChain,
        row: int,
        col: int
    ):
        """Add proof tree subplot."""
        # Build graph
        graph = nx.DiGraph()
        self._build_proof_tree(chain.root_node, graph)
        layout = nx.spring_layout(graph)
        
        # Create node traces
        node_x = []
        node_y = []
        node_color = []
        node_size = []
        node_text = []
        node_data = []
        
        for node in graph.nodes():
            pos = layout[node]
            attrs = graph.nodes[node]
            
            node_x.append(pos[0])
            node_y.append(pos[1])
            node_color.append(self._get_confidence_color(attrs["confidence"]))
            node_size.append(20 + 10 * len(attrs["evidence"]))
            node_text.append(f"Node {node}")
            node_data.append([
                str(node),
                attrs["type"],
                attrs["confidence"],
                len(attrs["evidence"])
            ])
        
        scatter = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker=dict(
                size=node_size,
                color=node_color,
                showscale=False
            ),
            text=node_text,
            customdata=node_data,
            hovertemplate=self.config.HOVER_TEMPLATE,
            name="Nodes"
        )
        
        fig.add_trace(scatter, row=row, col=col)
        
        # Create edge traces
        edge_x = []
        edge_y = []
        
        for edge in graph.edges():
            x0, y0 = layout[edge[0]]
            x1, y1 = layout[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edges = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1, color="#888"),
            hoverinfo="none",
            name="Edges"
        )
        
        fig.add_trace(edges, row=row, col=col)
    
    def _build_proof_tree(
        self,
        node: ProofNode,
        graph: nx.DiGraph
    ):
        """Build networkx graph from proof tree."""
        graph.add_node(
            node.id,
            type=node.type,
            confidence=node.confidence,
            evidence=node.evidence
        )
        
        for child in node.children:
            graph.add_edge(node.id, child.id)
            self._build_proof_tree(child, graph)
    
    def _add_evidence_map(
        self,
        fig: go.Figure,
        chain: VerificationChain,
        row: int,
        col: int
    ):
        """Add evidence map subplot."""
        # Create evidence network
        graph = nx.Graph()
        
        # Add nodes for evidence and claims
        for evidence in chain.evidence:
            graph.add_node(
                evidence.id,
                type="evidence",
                subtype=evidence.type,
                confidence=evidence.confidence
            )
            
            for claim_id in evidence.supports:
                if not graph.has_node(claim_id):
                    graph.add_node(
                        claim_id,
                        type="claim",
                        confidence=chain.nodes[claim_id].confidence
                    )
                graph.add_edge(evidence.id, claim_id)
        
        # Create layout
        layout = nx.spring_layout(graph)
        
        # Create node traces for evidence and claims
        for node_type in ["evidence", "claim"]:
            nodes = [
                n for n, d in graph.nodes(data=True)
                if d["type"] == node_type
            ]
            
            if not nodes:
                continue
            
            x = [layout[node][0] for node in nodes]
            y = [layout[node][1] for node in nodes]
            
            if node_type == "evidence":
                color = [
                    self.config.EVIDENCE_TYPES[graph.nodes[n]["subtype"]]
                    for n in nodes
                ]
            else:
                color = [
                    self._get_confidence_color(graph.nodes[n]["confidence"])
                    for n in nodes
                ]
            
            scatter = go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(
                    size=15,
                    color=color,
                    showscale=False
                ),
                name=node_type.capitalize()
            )
            
            fig.add_trace(scatter, row=row, col=col)
        
        # Create edge trace
        edge_x = []
        edge_y = []
        
        for edge in graph.edges():
            x0, y0 = layout[edge[0]]
            x1, y1 = layout[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edges = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1, color="#888"),
            hoverinfo="none",
            name="Support"
        )
        
        fig.add_trace(edges, row=row, col=col)
    
    def _add_confidence_distribution(
        self,
        fig: go.Figure,
        chain: VerificationChain,
        row: int,
        col: int
    ):
        """Add confidence distribution subplot."""
        # Collect confidence values
        node_confidence = [
            node.confidence for node in chain.nodes.values()
        ]
        evidence_confidence = [
            evidence.confidence for evidence in chain.evidence
        ]
        
        # Create histograms
        nodes = go.Histogram(
            x=node_confidence,
            name="Nodes",
            opacity=0.75
        )
        
        evidence = go.Histogram(
            x=evidence_confidence,
            name="Evidence",
            opacity=0.75
        )
        
        fig.add_trace(nodes, row=row, col=col)
        fig.add_trace(evidence, row=row, col=col)
        
        fig.update_xaxes(title_text="Confidence", row=row, col=col)
        fig.update_yaxes(title_text="Count", row=row, col=col)
    
    def _add_verification_timeline(
        self,
        fig: go.Figure,
        chain: VerificationChain,
        row: int,
        col: int
    ):
        """Add verification timeline subplot."""
        # Create timeline of verification events
        events = chain.verification_history
        times = list(events.keys())
        confidences = [event.confidence for event in events.values()]
        statuses = [event.status for event in events.values()]
        
        scatter = go.Scatter(
            x=times,
            y=confidences,
            mode="lines+markers",
            marker=dict(
                size=10,
                color=[
                    self._get_confidence_color(conf)
                    for conf in confidences
                ]
            ),
            text=statuses,
            name="Verification"
        )
        
        fig.add_trace(scatter, row=row, col=col)
        fig.update_xaxes(title_text="Time", row=row, col=col)
        fig.update_yaxes(
            title_text="Confidence",
            range=[0, 1],
            row=row,
            col=col
        )
    
    def _get_confidence_color(self, confidence: float) -> str:
        """Get color based on confidence value."""
        if confidence >= 0.7:
            return self.config.CONFIDENCE_COLORS["high"]
        elif confidence >= 0.4:
            return self.config.CONFIDENCE_COLORS["medium"]
        else:
            return self.config.CONFIDENCE_COLORS["low"]
