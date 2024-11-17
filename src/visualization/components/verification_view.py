"""Verification View Component for visualizing reasoning verification processes."""

import streamlit as st
from typing import Dict, List, Optional
import plotly.graph_objects as go
from datetime import datetime

from core.types import ReasoningType, AgentRole
from core.reasoning import ReasoningPath, VerificationResult
from visualization.components.shared.graphs import GraphVisualizer
from visualization.components.shared.metrics import MetricsVisualizer


class VerificationView:
    """Interactive view for reasoning verification visualization and analysis."""

    def __init__(self):
        """Initialize verification view component."""
        self.graph_viz = GraphVisualizer()
        self.metrics_viz = MetricsVisualizer()
        
        # Initialize session state
        if "verification_history" not in st.session_state:
            st.session_state.verification_history = []
        if "selected_path" not in st.session_state:
            st.session_state.selected_path = None
        if "verification_filter" not in st.session_state:
            st.session_state.verification_filter = "all"

    def render_header(self):
        """Render the verification view header with controls."""
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.title("✓ Reasoning Verification")
            st.markdown("""
                Monitor and analyze the verification process of reasoning paths.
                Track confidence scores, verification chains, and agent interactions.
            """)
        
        with col2:
            # Verification filters
            st.selectbox(
                "Filter Paths",
                options=["all", "verified", "pending", "rejected"],
                key="verification_filter",
                help="Filter verification paths by status"
            )
        
        with col3:
            st.download_button(
                "Export Results",
                data=self._get_export_data(),
                file_name=f"verification_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                help="Download verification results as JSON"
            )

    def render_main_view(self, verification_results: List[VerificationResult]):
        """Render the main verification visualization and analysis view.
        
        Args:
            verification_results: List of verification results to visualize
        """
        # Filter results based on selected filter
        filtered_results = self._filter_results(verification_results)
        
        # Layout columns
        graph_col, details_col = st.columns([2, 1])
        
        with graph_col:
            self._render_verification_graph(filtered_results)
        
        with details_col:
            self._render_verification_details()
            self._render_verification_metrics(filtered_results)

    def _render_verification_graph(self, results: List[VerificationResult]):
        """Render the verification process graph visualization.
        
        Args:
            results: Filtered verification results to visualize
        """
        st.subheader("Verification Process")
        
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
            # Render interactive verification graph
            selected_path = self.graph_viz.visualize_verification_process(
                results,
                container=graph_container,
                title="Verification Graph",
                on_node_click=self._handle_path_selection
            )
            
            # Update session state
            if selected_path:
                st.session_state.selected_path = selected_path

    def _render_verification_details(self):
        """Render details panel for the selected verification path."""
        st.subheader("Verification Details")
        
        if st.session_state.selected_path:
            path = st.session_state.selected_path
            
            # Path information card
            with st.expander("Path Information", expanded=True):
                st.markdown(f"""
                    **Status:** {path.status}
                    
                    **Confidence Score:** {path.confidence:.2f}
                    
                    **Verifier Agent:** {path.verifier_id}
                    
                    **Verification Time:** {path.verification_time:.2f}s
                """)
                
                # Verification steps
                if path.verification_steps:
                    st.markdown("#### Verification Steps")
                    for step in path.verification_steps:
                        st.markdown(f"- {step}")
                
                # Failure reasons if rejected
                if path.status == "rejected" and path.failure_reasons:
                    st.markdown("#### Failure Reasons")
                    for reason in path.failure_reasons:
                        st.error(reason)
        else:
            st.info("Select a path in the graph to view details")

    def _render_verification_metrics(self, results: List[VerificationResult]):
        """Render verification metrics and statistics.
        
        Args:
            results: Filtered verification results for metrics calculation
        """
        st.subheader("Verification Metrics")
        
        # Metrics tabs
        tab1, tab2 = st.tabs(["Performance", "Statistics"])
        
        with tab1:
            # Calculate and display performance metrics
            total = len(results)
            verified = len([r for r in results if r.status == "verified"])
            rejected = len([r for r in results if r.status == "rejected"])
            
            metrics = {
                "Verification Rate": verified / total if total > 0 else 0,
                "Average Confidence": sum(r.confidence for r in results) / total if total > 0 else 0,
                "Rejection Rate": rejected / total if total > 0 else 0
            }
            
            self.metrics_viz.render_performance_metrics(metrics, container=st)
        
        with tab2:
            # Calculate and display statistics
            stats = {
                "Total Paths": total,
                "Verified Paths": verified,
                "Rejected Paths": rejected,
                "Average Time": sum(r.verification_time for r in results) / total if total > 0 else 0
            }
            
            self.metrics_viz.render_verification_stats(stats, container=st)

    def _filter_results(self, results: List[VerificationResult]) -> List[VerificationResult]:
        """Filter verification results based on selected filter.
        
        Args:
            results: List of verification results to filter
            
        Returns:
            Filtered list of verification results
        """
        filter_type = st.session_state.verification_filter
        if filter_type == "all":
            return results
        return [r for r in results if r.status == filter_type]

    def _handle_path_selection(self, path: ReasoningPath):
        """Handle path selection in the verification graph.
        
        Args:
            path: The selected reasoning path
        """
        st.session_state.selected_path = path
        
        # Add to verification history
        if path not in st.session_state.verification_history:
            st.session_state.verification_history.append(path)

    def _get_export_data(self) -> str:
        """Get verification data for export.
        
        Returns:
            JSON string of verification data
        """
        import json
        
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "verification_history": [
                {
                    "id": str(path.id),
                    "status": path.status,
                    "confidence": path.confidence,
                    "verifier_id": str(path.verifier_id),
                    "verification_time": path.verification_time,
                    "verification_steps": path.verification_steps,
                    "failure_reasons": path.failure_reasons if path.status == "rejected" else []
                }
                for path in st.session_state.verification_history
            ]
        }
        
        return json.dumps(export_data, indent=2)

    def render(self, verification_results: List[VerificationResult]):
        """Render the complete verification view.
        
        Args:
            verification_results: List of verification results to visualize
        """
        self.render_header()
        self.render_main_view(verification_results)
