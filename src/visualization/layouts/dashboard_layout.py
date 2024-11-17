"""Dashboard layout component for the Agent Reasoning Beta platform."""

import streamlit as st
from typing import Dict, List, Optional
import plotly.graph_objects as go
import plotly.express as px

from core.types import ReasoningType, AgentRole
from core.reasoning import MCTSNode, ReasoningPath, VerificationResult, ConsensusResult
from ui.state.session import SessionState


class DashboardLayout:
    """Dashboard layout manager for analytics and metrics visualization."""
    
    def __init__(self, session: SessionState):
        """Initialize dashboard layout.
        
        Args:
            session: Global session state manager
        """
        self.session = session
    
    def render_performance_metrics(self):
        """Render performance metrics section."""
        st.subheader("Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Success rate over time
            success_data = self.session.metrics["success_rate"]
            if success_data:
                fig = px.line(
                    success_data,
                    x="timestamp",
                    y="value",
                    title="Success Rate Over Time"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No success rate data available")
        
        with col2:
            # Confidence distribution
            confidence_data = self.session.metrics["confidence_scores"]
            if confidence_data:
                fig = px.histogram(
                    confidence_data,
                    x="value",
                    title="Confidence Score Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No confidence score data available")
    
    def render_cost_tracking(self):
        """Render cost tracking section."""
        st.subheader("Cost Tracking")
        
        costs = self.session.model_state.cost_tracking
        if any(costs.values()):
            fig = px.bar(
                x=list(costs.keys()),
                y=list(costs.values()),
                title="API Costs by Provider"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Cost breakdown
            st.markdown("### Cost Breakdown")
            for provider, cost in costs.items():
                st.metric(
                    f"{provider.title()} Cost",
                    f"${cost:.4f}"
                )
        else:
            st.info("No cost data available")
    
    def render_error_analysis(self):
        """Render error analysis section."""
        st.subheader("Error Analysis")
        
        errors = self.session.metrics["error_counts"]
        if errors:
            # Error distribution
            fig = px.pie(
                values=list(errors.values()),
                names=list(errors.keys()),
                title="Error Distribution by Type"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Error details
            st.markdown("### Error Details")
            for error_type, count in errors.items():
                st.metric(
                    error_type,
                    count,
                    delta=None
                )
        else:
            st.info("No error data available")
    
    def render_response_times(self):
        """Render response time analysis section."""
        st.subheader("Response Times")
        
        times = self.session.metrics["response_times"]
        if times:
            # Response time trend
            fig = px.line(
                times,
                x="timestamp",
                y="value",
                title="Response Time Trend"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            values = [t["value"] for t in times]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Response Time", f"{sum(values)/len(values):.2f}s")
            with col2:
                st.metric("Min Response Time", f"{min(values):.2f}s")
            with col3:
                st.metric("Max Response Time", f"{max(values):.2f}s")
        else:
            st.info("No response time data available")
    
    def render(self):
        """Render the complete dashboard layout."""
        st.title("Analytics Dashboard")
        
        # Render metrics sections
        self.render_performance_metrics()
        st.markdown("---")
        
        self.render_cost_tracking()
        st.markdown("---")
        
        self.render_error_analysis()
        st.markdown("---")
        
        self.render_response_times()
        
        # Export controls
        st.markdown("---")
        st.subheader("Export Data")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export Metrics (JSON)"):
                st.download_button(
                    "Download JSON",
                    data=str(self.session.export_state()),
                    file_name="agent_metrics.json",
                    mime="application/json"
                )
        with col2:
            if st.button("Reset Metrics"):
                self.session._initialize_state()
                st.experimental_rerun()
