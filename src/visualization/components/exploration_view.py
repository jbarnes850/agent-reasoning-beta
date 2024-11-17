"""Exploration View Component for visualizing MCTS exploration processes."""

import streamlit as st
from typing import Dict, List, Optional
import plotly.graph_objects as go
from datetime import datetime

from core.types import ReasoningType, AgentRole
from core.reasoning import MCTSNode, ReasoningPath
from visualization.components.shared.trees import TreeVisualizer
from visualization.components.shared.metrics import MetricsVisualizer


class ExplorationView:
    """Interactive view for MCTS exploration visualization and analysis."""

    def __init__(self):
        """Initialize exploration view component."""
        self.tree_viz = TreeVisualizer()
        self.metrics_viz = MetricsVisualizer()
        
        # Initialize session state for exploration
        if "exploration_history" not in st.session_state:
            st.session_state.exploration_history = []
        if "selected_node" not in st.session_state:
            st.session_state.selected_node = None

    def render_header(self):
        """Render the exploration view header with controls."""
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.title("🔍 MCTS Exploration")
            st.markdown("""
                Visualize and analyze the Monte Carlo Tree Search exploration process
                in real-time. Monitor agent decisions, confidence scores, and exploration paths.
            """)
        
        with col2:
            st.button("Reset Exploration", 
                     on_click=self._reset_exploration,
                     help="Clear the current exploration state and start fresh")
        
        with col3:
            st.download_button(
                "Export Data",
                data=self._get_export_data(),
                file_name=f"exploration_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                help="Download the current exploration data as JSON"
            )

    def render_main_view(self, root_node: MCTSNode, current_path: Optional[ReasoningPath] = None):
        """Render the main exploration visualization and controls.
        
        Args:
            root_node: The root node of the MCTS tree
            current_path: Optional current reasoning path being explored
        """
        # Layout columns
        tree_col, details_col = st.columns([2, 1])
        
        with tree_col:
            # Interactive tree visualization
            st.subheader("Exploration Tree")
            
            # Tree visualization container with custom styling
            tree_container = st.container()
            tree_container.markdown("""
                <style>
                    .tree-container {
                        border: 1px solid #e0e0e0;
                        border-radius: 5px;
                        padding: 10px;
                        background-color: #ffffff;
                    }
                </style>
                """, unsafe_allow_html=True)
            
            with tree_container:
                # Render interactive tree
                selected_node = self.tree_viz.visualize_reasoning_path(
                    root_node,
                    current_path,
                    container=tree_container,
                    title="MCTS Exploration Tree",
                    on_node_click=self._handle_node_selection
                )
                
                # Update session state
                if selected_node:
                    st.session_state.selected_node = selected_node
        
        with details_col:
            self._render_node_details()
            self._render_exploration_metrics()

    def _render_node_details(self):
        """Render details panel for the selected node."""
        st.subheader("Node Details")
        
        if st.session_state.selected_node:
            node = st.session_state.selected_node
            
            # Node information card
            with st.expander("Node Information", expanded=True):
                st.markdown(f"""
                    **Confidence Score:** {node.confidence:.2f}
                    
                    **Visit Count:** {node.visits}
                    
                    **Depth:** {node.depth}
                    
                    **Status:** {node.status}
                """)
                
                # Action details if available
                if node.action:
                    st.markdown("#### Action Details")
                    st.json(node.action)
                
                # State visualization if available
                if node.state:
                    st.markdown("#### State")
                    st.json(node.state)
        else:
            st.info("Select a node in the tree to view details")

    def _render_exploration_metrics(self):
        """Render exploration metrics and statistics."""
        st.subheader("Exploration Metrics")
        
        # Metrics tabs
        tab1, tab2 = st.tabs(["Performance", "Statistics"])
        
        with tab1:
            # Performance metrics
            self.metrics_viz.render_performance_metrics(
                metrics={
                    "Average Confidence": 0.75,
                    "Exploration Rate": 0.85,
                    "Path Success Rate": 0.65
                },
                container=st
            )
        
        with tab2:
            # Exploration statistics
            self.metrics_viz.render_exploration_stats(
                stats={
                    "Total Nodes": len(st.session_state.exploration_history),
                    "Max Depth": 8,
                    "Branching Factor": 3.5
                },
                container=st
            )

    def _reset_exploration(self):
        """Reset the exploration state."""
        st.session_state.exploration_history = []
        st.session_state.selected_node = None
        st.experimental_rerun()

    def _handle_node_selection(self, node: MCTSNode):
        """Handle node selection in the tree visualization.
        
        Args:
            node: The selected MCTSNode
        """
        st.session_state.selected_node = node
        
        # Add to exploration history
        if node not in st.session_state.exploration_history:
            st.session_state.exploration_history.append(node)

    def _get_export_data(self) -> str:
        """Get exploration data for export.
        
        Returns:
            JSON string of exploration data
        """
        import json
        
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "exploration_history": [
                {
                    "id": str(node.id),
                    "confidence": node.confidence,
                    "visits": node.visits,
                    "depth": node.depth,
                    "status": node.status
                }
                for node in st.session_state.exploration_history
            ]
        }
        
        return json.dumps(export_data, indent=2)

    def render(self, root_node: MCTSNode, current_path: Optional[ReasoningPath] = None):
        """Render the complete exploration view.
        
        Args:
            root_node: The root node of the MCTS tree
            current_path: Optional current reasoning path being explored
        """
        self.render_header()
        self.render_main_view(root_node, current_path)
