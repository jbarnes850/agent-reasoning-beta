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
from src.visualization.components.views.shared.tree_viz import TreeVisualizer
from src.visualization.components.views.shared.graphs import GraphVisualizer
from src.visualization.components.views.shared.metrics import MetricsVisualizer
from src.visualization.components.views.exploration_view import ExplorationView
from src.visualization.components.views.verification_view import VerificationView
from src.visualization.components.views.consensus_view import ConsensusView

def render_playground():
    """Render the main playground interface."""
    st.markdown("<h1 class='main-header'>🎮 Agent Reasoning Playground</h1>", unsafe_allow_html=True)
    
    # Welcome message and description
    st.markdown("""
        <div class='info-box'>
            <h3 style='margin-top:0'>👋 Welcome to the Agent Reasoning Playground!</h3>
            <p>
                This is your space to explore and experiment with multi-agent reasoning systems. 
                Create agent teams, observe their decision-making processes, and analyze their 
                collaborative problem-solving abilities in real-time.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize visualizers
    tree_viz = TreeVisualizer()
    graph_viz = GraphVisualizer()
    metrics_viz = MetricsVisualizer()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<h2 class='section-header'>🔍 Experiment Visualization</h2>", unsafe_allow_html=True)
        
        # Visualization tabs
        tab1, tab2, tab3 = st.tabs(["🌳 Decision Tree", "🕸️ Agent Graph", "📊 Metrics"])
        
        with tab1:
            tree_viz.render()
        with tab2:
            graph_viz.render()
        with tab3:
            metrics_viz.render()
    
    with col2:
        st.markdown("<h2 class='section-header'>⚙️ Configuration</h2>", unsafe_allow_html=True)
        
        with st.form("experiment_config"):
            # Agent configuration
            st.markdown("#### 🤖 Agent Setup")
            num_agents = st.slider(
                "Number of Agents",
                min_value=1,
                max_value=10,
                value=3,
                help="Select the number of agents to participate in the reasoning process"
            )
            
            # Model configuration
            st.markdown("#### 🧠 Model Settings")
            model_provider = st.selectbox(
                "Model Provider",
                options=[
                    ModelProvider.GROQ,
                    ModelProvider.OPENAI,
                    ModelProvider.ANTHROPIC
                ],
                format_func=lambda x: {
                    ModelProvider.GROQ: "🚀 Groq",
                    ModelProvider.OPENAI: "🌟 OpenAI",
                    ModelProvider.ANTHROPIC: "🔮 Anthropic"
                }[x]
            )
            
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Controls randomness in the model's output"
            )
            
            # Advanced settings in expander
            with st.expander("🔧 Advanced Settings"):
                max_tokens = st.number_input(
                    "Max Tokens",
                    min_value=100,
                    max_value=4000,
                    value=1000,
                    help="Maximum number of tokens in model responses"
                )
                
                reasoning_type = st.selectbox(
                    "Reasoning Type",
                    options=list(ReasoningType),
                    format_func=lambda x: {
                        ReasoningType.EXPLORATION: "🔍 Exploration",
                        ReasoningType.VERIFICATION: "✅ Verification",
                        ReasoningType.CONSENSUS: "🤝 Consensus"
                    }[x]
                )
            
            # Submit button with loading state
            submitted = st.form_submit_button(
                "🚀 Start Experiment",
                use_container_width=True,
                type="primary"
            )
            
            if submitted:
                with st.spinner("Initializing experiment..."):
                    run_experiment(
                        num_agents=num_agents,
                        model_provider=model_provider,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        reasoning_type=reasoning_type,
                        advanced_settings={}
                    )

    # Status and monitoring section
    st.markdown("<h2 class='section-header'>📊 Experiment Status</h2>", unsafe_allow_html=True)
    
    status_col1, status_col2 = st.columns(2)
    
    with status_col1:
        display_status()
    
    with status_col2:
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
