"""Consensus View Component for visualizing multi-agent consensus building."""

import streamlit as st
from typing import Dict, List, Optional
import plotly.graph_objects as go
from datetime import datetime

from core.types import ReasoningType, AgentRole
from core.reasoning import ConsensusResult, ReasoningPath
from visualization.components.shared.graphs import GraphVisualizer
from visualization.components.shared.metrics import MetricsVisualizer
from visualization.components.shared.heatmaps import HeatmapVisualizer


class ConsensusView:
    """Interactive view for multi-agent consensus visualization and analysis."""

    def __init__(self):
        """Initialize consensus view component."""
        self.graph_viz = GraphVisualizer()
        self.metrics_viz = MetricsVisualizer()
        self.heatmap_viz = HeatmapVisualizer()
        
        # Initialize session state
        if "consensus_history" not in st.session_state:
            st.session_state.consensus_history = []
        if "selected_consensus" not in st.session_state:
            st.session_state.selected_consensus = None
        if "view_mode" not in st.session_state:
            st.session_state.view_mode = "graph"  # or "heatmap"

    def render_header(self):
        """Render the consensus view header with controls."""
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.title("🤝 Multi-Agent Consensus")
            st.markdown("""
                Visualize and analyze the consensus building process between agents.
                Monitor agreement levels, decision paths, and agent interactions.
            """)
        
        with col2:
            # View mode selector
            st.radio(
                "View Mode",
                options=["graph", "heatmap"],
                key="view_mode",
                horizontal=True,
                help="Switch between graph and heatmap visualization"
            )
        
        with col3:
            st.download_button(
                "Export Consensus",
                data=self._get_export_data(),
                file_name=f"consensus_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                help="Download consensus data as JSON"
            )

    def render_main_view(self, consensus_results: List[ConsensusResult]):
        """Render the main consensus visualization and analysis view.
        
        Args:
            consensus_results: List of consensus results to visualize
        """
        # Layout columns
        viz_col, details_col = st.columns([2, 1])
        
        with viz_col:
            if st.session_state.view_mode == "graph":
                self._render_consensus_graph(consensus_results)
            else:
                self._render_consensus_heatmap(consensus_results)
        
        with details_col:
            self._render_consensus_details()
            self._render_consensus_metrics(consensus_results)

    def _render_consensus_graph(self, results: List[ConsensusResult]):
        """Render the consensus process graph visualization.
        
        Args:
            results: Consensus results to visualize
        """
        st.subheader("Consensus Process Graph")
        
        # Graph container with custom styling
        graph_container = st.container()
        graph_container.markdown("""
            <style>
                .graph-container {
                    border: 1px solid #e0e0e0;
                    border-radius: 5px;
                    padding: 10px;
                    background-color: #ffffff;
                }
            </style>
            """, unsafe_allow_html=True)
        
        with graph_container:
            # Render interactive consensus graph
            selected_consensus = self.graph_viz.visualize_consensus_process(
                results,
                container=graph_container,
                title="Agent Consensus Graph",
                on_node_click=self._handle_consensus_selection
            )
            
            # Update session state
            if selected_consensus:
                st.session_state.selected_consensus = selected_consensus

    def _render_consensus_heatmap(self, results: List[ConsensusResult]):
        """Render the consensus heatmap visualization.
        
        Args:
            results: Consensus results to visualize
        """
        st.subheader("Consensus Heatmap")
        
        # Heatmap container with custom styling
        heatmap_container = st.container()
        heatmap_container.markdown("""
            <style>
                .heatmap-container {
                    border: 1px solid #e0e0e0;
                    border-radius: 5px;
                    padding: 10px;
                    background-color: #ffffff;
                }
            </style>
            """, unsafe_allow_html=True)
        
        with heatmap_container:
            # Render interactive consensus heatmap
            self.heatmap_viz.visualize_consensus_heatmap(
                results,
                container=heatmap_container,
                title="Agent Agreement Heatmap"
            )

    def _render_consensus_details(self):
        """Render details panel for the selected consensus result."""
        st.subheader("Consensus Details")
        
        if st.session_state.selected_consensus:
            consensus = st.session_state.selected_consensus
            
            # Consensus information card
            with st.expander("Consensus Information", expanded=True):
                st.markdown(f"""
                    **Status:** {consensus.status}
                    
                    **Agreement Level:** {consensus.agreement_level:.2f}
                    
                    **Participating Agents:** {len(consensus.participating_agents)}
                    
                    **Time to Consensus:** {consensus.time_to_consensus:.2f}s
                """)
                
                # Agent contributions
                if consensus.agent_contributions:
                    st.markdown("#### Agent Contributions")
                    for agent_id, contribution in consensus.agent_contributions.items():
                        st.markdown(f"- **{agent_id}**: {contribution:.2f}")
                
                # Decision path
                if consensus.decision_path:
                    st.markdown("#### Decision Path")
                    for step in consensus.decision_path:
                        st.markdown(f"- {step}")
        else:
            st.info("Select a consensus node in the graph to view details")

    def _render_consensus_metrics(self, results: List[ConsensusResult]):
        """Render consensus metrics and statistics.
        
        Args:
            results: Consensus results for metrics calculation
        """
        st.subheader("Consensus Metrics")
        
        # Metrics tabs
        tab1, tab2, tab3 = st.tabs(["Performance", "Agreement", "Time"])
        
        with tab1:
            # Performance metrics
            total = len(results)
            successful = len([r for r in results if r.status == "success"])
            
            metrics = {
                "Success Rate": successful / total if total > 0 else 0,
                "Average Agreement": sum(r.agreement_level for r in results) / total if total > 0 else 0,
                "Agent Participation": sum(len(r.participating_agents) for r in results) / total if total > 0 else 0
            }
            
            self.metrics_viz.render_performance_metrics(metrics, container=st)
        
        with tab2:
            # Agreement distribution
            agreement_levels = [r.agreement_level for r in results]
            fig = go.Figure(data=[go.Histogram(x=agreement_levels, nbinsx=20)])
            fig.update_layout(title="Agreement Level Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Time metrics
            times = [r.time_to_consensus for r in results]
            stats = {
                "Average Time": sum(times) / len(times) if times else 0,
                "Min Time": min(times) if times else 0,
                "Max Time": max(times) if times else 0
            }
            
            self.metrics_viz.render_time_metrics(stats, container=st)

    def _handle_consensus_selection(self, consensus: ConsensusResult):
        """Handle consensus selection in the visualization.
        
        Args:
            consensus: The selected consensus result
        """
        st.session_state.selected_consensus = consensus
        
        # Add to consensus history
        if consensus not in st.session_state.consensus_history:
            st.session_state.consensus_history.append(consensus)

    def _get_export_data(self) -> str:
        """Get consensus data for export.
        
        Returns:
            JSON string of consensus data
        """
        import json
        
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "consensus_history": [
                {
                    "id": str(consensus.id),
                    "status": consensus.status,
                    "agreement_level": consensus.agreement_level,
                    "participating_agents": [str(a) for a in consensus.participating_agents],
                    "time_to_consensus": consensus.time_to_consensus,
                    "agent_contributions": {
                        str(k): v for k, v in consensus.agent_contributions.items()
                    } if consensus.agent_contributions else {},
                    "decision_path": consensus.decision_path
                }
                for consensus in st.session_state.consensus_history
            ]
        }
        
        return json.dumps(export_data, indent=2)

    def render(self, consensus_results: List[ConsensusResult]):
        """Render the complete consensus view.
        
        Args:
            consensus_results: List of consensus results to visualize
        """
        self.render_header()
        self.render_main_view(consensus_results)
