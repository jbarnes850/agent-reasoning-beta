"""
Analytics dashboard for the Agent Reasoning Beta platform.

This module provides comprehensive analytics for:
- Agent performance metrics
- Resource utilization
- Success rate analysis
- Cost tracking
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from src.core.metrics import MetricsManager
from src.visualization.components.shared.metrics import MetricsVisualizer
from src.visualization.components.shared.graphs import GraphVisualizer

def render_analytics():
    """Render the analytics dashboard."""
    st.title("Analytics Dashboard 📊")
    
    # Initialize visualizers
    metrics_viz = MetricsVisualizer()
    graph_viz = GraphVisualizer()
    metrics_manager = MetricsManager()
    
    # Time range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now().date() - timedelta(days=7)
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now().date()
        )
    
    # Main metrics dashboard
    st.header("Performance Overview")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Total Experiments",
            metrics_manager.get_total_experiments(start_date, end_date)
        )
    with col2:
        st.metric(
            "Success Rate",
            f"{metrics_manager.get_success_rate(start_date, end_date):.1f}%"
        )
    with col3:
        st.metric(
            "Avg. Response Time",
            f"{metrics_manager.get_avg_response_time(start_date, end_date):.2f}s"
        )
    with col4:
        st.metric(
            "Total Cost",
            f"${metrics_manager.get_total_cost(start_date, end_date):.2f}"
        )
    
    # Detailed metrics
    tab1, tab2, tab3 = st.tabs([
        "Performance Analysis",
        "Agent Interactions",
        "Resource Usage"
    ])
    
    with tab1:
        st.subheader("Performance Metrics")
        
        # Performance over time
        metrics = metrics_manager.get_performance_metrics(start_date, end_date)
        fig = px.line(
            metrics,
            x="timestamp",
            y=["success_rate", "confidence", "response_time"],
            title="Performance Trends"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Success rate distribution
        success_rates = metrics_manager.get_success_rate_distribution(start_date, end_date)
        fig = px.histogram(
            success_rates,
            x="success_rate",
            title="Success Rate Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Model performance comparison
        model_metrics = metrics_manager.get_model_performance(start_date, end_date)
        fig = px.bar(
            model_metrics,
            x="model",
            y=["success_rate", "avg_confidence"],
            title="Model Performance Comparison"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Agent Interaction Analysis")
        
        # Agent network visualization
        agents = metrics_manager.get_active_agents(start_date, end_date)
        interactions = metrics_manager.get_agent_interactions(start_date, end_date)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            container = st.container()
            graph_viz.visualize_agent_network(
                agents=agents,
                interactions=interactions,
                container=container,
                layout_type=st.selectbox(
                    "Network Layout",
                    ["spring", "circular", "kamada_kawai"]
                )
            )
        
        with col2:
            # Agent statistics
            st.write("Agent Statistics")
            agent_stats = metrics_manager.get_agent_statistics(start_date, end_date)
            st.dataframe(agent_stats)
        
        # Interaction patterns
        st.subheader("Interaction Patterns")
        interaction_patterns = metrics_manager.get_interaction_patterns(start_date, end_date)
        fig = px.scatter(
            interaction_patterns,
            x="time",
            y="interaction_count",
            color="agent_role",
            title="Agent Interaction Patterns"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Resource Utilization")
        
        # Resource usage over time
        resource_usage = metrics_manager.get_resource_usage(start_date, end_date)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=resource_usage["timestamp"],
            y=resource_usage["cpu_usage"],
            name="CPU Usage"
        ))
        fig.add_trace(go.Scatter(
            x=resource_usage["timestamp"],
            y=resource_usage["memory_usage"],
            name="Memory Usage"
        ))
        fig.update_layout(title="Resource Usage Over Time")
        st.plotly_chart(fig, use_container_width=True)
        
        # Cost analysis
        st.subheader("Cost Analysis")
        costs = metrics_manager.get_cost_breakdown(start_date, end_date)
        
        # Cost by model
        fig = px.pie(
            costs["model_costs"],
            values="cost",
            names="model",
            title="Cost Distribution by Model"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Cost trends
        fig = px.line(
            costs["daily_costs"],
            x="date",
            y="cost",
            title="Daily Cost Trends"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Export options
    st.sidebar.header("Export Options")
    export_format = st.sidebar.selectbox(
        "Export Format",
        ["CSV", "JSON", "PDF"]
    )
    if st.sidebar.button("Export Analytics"):
        export_analytics(
            metrics_manager=metrics_manager,
            start_date=start_date,
            end_date=end_date,
            format=export_format
        )

def export_analytics(
    metrics_manager: MetricsManager,
    start_date: datetime.date,
    end_date: datetime.date,
    format: str
):
    """
    Export analytics data in the specified format.
    
    Args:
        metrics_manager: Metrics manager instance
        start_date: Start date for the export
        end_date: End date for the export
        format: Export format (CSV, JSON, or PDF)
    """
    data = metrics_manager.export_data(start_date, end_date)
    
    if format == "CSV":
        st.download_button(
            "Download CSV",
            data.to_csv(),
            "analytics.csv",
            "text/csv"
        )
    elif format == "JSON":
        st.download_button(
            "Download JSON",
            data.to_json(),
            "analytics.json",
            "application/json"
        )
    else:  # PDF
        st.download_button(
            "Download PDF",
            metrics_manager.generate_pdf_report(data),
            "analytics.pdf",
            "application/pdf"
        )

if __name__ == "__main__":
    render_analytics()
