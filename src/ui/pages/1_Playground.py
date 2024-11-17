"""
Playground interface for the Agent Reasoning Beta platform.

This module provides an interactive playground for:
- Running agent reasoning experiments
- Visualizing decision trees and graphs
- Testing different agent configurations
- Real-time monitoring of agent interactions
"""

import streamlit as st
from typing import Dict, List, Optional

from src.core.agents import AgentFactory, AgentRole, ModelProvider
from src.core.types import ReasoningType
from src.core.models import ModelConfig
from src.visualization.components.shared.trees import TreeVisualizer
from src.visualization.components.shared.graphs import GraphVisualizer
from src.visualization.components.shared.metrics import MetricsVisualizer
from src.visualization.components.exploration_view import ExplorationView
from src.visualization.components.verification_view import VerificationView
from src.visualization.components.consensus_view import ConsensusView

def render_playground():
    """Render the main playground interface."""
    st.title("Agent Reasoning Playground 🎮")
    
    # Initialize visualizers
    tree_viz = TreeVisualizer()
    graph_viz = GraphVisualizer()
    metrics_viz = MetricsVisualizer()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Agent configuration
        st.subheader("Agent Setup")
        num_agents = st.slider("Number of Agents", 1, 10, 3)
        
        # Model configuration
        st.subheader("Model Settings")
        model_provider = st.selectbox(
            "Model Provider",
            options=[
                ModelProvider.GROQ,
                ModelProvider.OPENAI,
                ModelProvider.ANTHROPIC
            ],
            format_func=lambda x: x.value
        )
        
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7)
        max_tokens = st.number_input("Max Tokens", 100, 4000, 1000)
        
        # Reasoning configuration
        st.subheader("Reasoning Settings")
        reasoning_type = st.selectbox(
            "Reasoning Type",
            options=[
                ReasoningType.EXPLORATION,
                ReasoningType.VERIFICATION,
                ReasoningType.CONSENSUS
            ],
            format_func=lambda x: x.value
        )
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            top_p = st.slider("Top P", 0.0, 1.0, 0.9)
            presence_penalty = st.slider("Presence Penalty", -2.0, 2.0, 0.0)
            frequency_penalty = st.slider("Frequency Penalty", -2.0, 2.0, 0.0)
            
            st.subheader("Search Settings")
            enable_search = st.checkbox("Enable Web Search", value=True)
            if enable_search:
                search_depth = st.slider("Search Depth", 1, 5, 3)
                max_results = st.number_input("Max Results", 1, 20, 5)
        
        if st.button("Start Experiment"):
            run_experiment(
                num_agents=num_agents,
                model_provider=model_provider,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_type=reasoning_type,
                advanced_settings={
                    "top_p": top_p,
                    "presence_penalty": presence_penalty,
                    "frequency_penalty": frequency_penalty,
                    "enable_search": enable_search,
                    "search_depth": search_depth if enable_search else None,
                    "max_results": max_results if enable_search else None
                }
            )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Visualization tabs
        tab1, tab2, tab3 = st.tabs([
            "Reasoning Process",
            "Agent Network",
            "Metrics"
        ])
        
        with tab1:
            st.subheader("Reasoning Process Visualization")
            if reasoning_type == ReasoningType.EXPLORATION:
                ExplorationView().render()
            elif reasoning_type == ReasoningType.VERIFICATION:
                VerificationView().render()
            else:
                ConsensusView().render()
            
            # Additional visualization controls
            with st.expander("Visualization Controls"):
                st.selectbox(
                    "Layout",
                    ["Tree", "Force-Directed", "Hierarchical"],
                    key="viz_layout"
                )
                st.checkbox("Show Confidence Scores", value=True)
                st.checkbox("Show Edge Weights", value=True)
                st.slider("Update Interval (s)", 0.1, 5.0, 1.0)
        
        with tab2:
            st.subheader("Agent Interaction Network")
            container = st.container()
            graph_viz.visualize_agent_network(
                agents=st.session_state.get("agents", []),
                interactions=st.session_state.get("interactions", []),
                container=container
            )
        
        with tab3:
            st.subheader("Performance Metrics")
            container = st.container()
            metrics_viz.visualize_performance_metrics(
                metrics=st.session_state.get("metrics", {}),
                container=container
            )
    
    with col2:
        # Real-time monitoring
        st.subheader("Live Updates")
        status_container = st.container()
        with status_container:
            display_status()
        
        # Agent details
        st.subheader("Agent Details")
        agent_container = st.container()
        with agent_container:
            display_agent_details()

def run_experiment(
    num_agents: int,
    model_provider: ModelProvider,
    temperature: float,
    max_tokens: int,
    reasoning_type: ReasoningType,
    advanced_settings: Dict
):
    """
    Run a new reasoning experiment with the specified configuration.
    
    Args:
        num_agents: Number of agents to create
        model_provider: Selected model provider
        temperature: Model temperature setting
        max_tokens: Maximum tokens per response
        reasoning_type: Type of reasoning to perform
        advanced_settings: Additional configuration options
    """
    # Initialize experiment
    st.session_state["experiment_running"] = True
    st.session_state["agents"] = []
    st.session_state["interactions"] = []
    st.session_state["metrics"] = {}
    
    # Create model configuration
    model_config = ModelConfig(
        provider=model_provider,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=advanced_settings["top_p"],
        presence_penalty=advanced_settings["presence_penalty"],
        frequency_penalty=advanced_settings["frequency_penalty"]
    )
    
    # Initialize agents
    factory = AgentFactory()
    for _ in range(num_agents):
        if reasoning_type == ReasoningType.EXPLORATION:
            agent = factory.create_agent(
                role=AgentRole.EXPLORER,
                model_provider=model_provider,
                model_config=model_config,
                search_config={
                    "enabled": advanced_settings["enable_search"],
                    "depth": advanced_settings.get("search_depth"),
                    "max_results": advanced_settings.get("max_results")
                }
            )
        elif reasoning_type == ReasoningType.VERIFICATION:
            agent = factory.create_agent(
                role=AgentRole.VERIFIER,
                model_provider=model_provider,
                model_config=model_config
            )
        else:
            agent = factory.create_agent(
                role=AgentRole.COORDINATOR,
                model_provider=model_provider,
                model_config=model_config
            )
        
        st.session_state["agents"].append(agent)
        agent.start()
    
    st.toast("Experiment started!")

def display_status():
    """Display real-time status updates."""
    if st.session_state.get("experiment_running", False):
        st.success("Experiment in progress")
        
        # Display current phase
        st.write("Current Phase:", st.session_state.get("current_phase", "Initializing"))
        
        # Show progress
        progress = st.progress(0)
        status = st.empty()
        
        # Update progress (in practice, this would be updated by the experiment)
        progress.progress(50)
        status.write("Processing agent interactions...")
    else:
        st.info("No experiment running")

def display_agent_details():
    """Display detailed information about active agents."""
    agents = st.session_state.get("agents", [])
    if agents:
        for agent in agents:
            with st.expander(f"Agent {agent.id}"):
                st.write(f"Role: {agent.role.value}")
                st.write(f"Status: {'Active' if agent._active else 'Inactive'}")
                st.write(f"Confidence: {agent.metrics.average_confidence:.2f}")
                
                # Show recent thoughts
                st.write("Recent Thoughts:")
                for thought in agent.state.thought_history[-5:]:
                    st.info(thought.content)
                
                # Show metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Thoughts", agent.metrics.total_thoughts)
                    st.metric("Successful Verifications", agent.metrics.successful_verifications)
                with col2:
                    st.metric("Failed Verifications", agent.metrics.failed_verifications)
                    st.metric("Consensus Participations", agent.metrics.consensus_participations)
    else:
        st.write("No active agents")

if __name__ == "__main__":
    render_playground()
