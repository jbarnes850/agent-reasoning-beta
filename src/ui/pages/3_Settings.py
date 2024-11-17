"""
Settings management interface for the Agent Reasoning Beta platform.

This module provides configuration management for:
- Model providers and API keys
- Agent behavior settings
- System-wide configurations
- Search and resource limits
"""

import streamlit as st
import os
from typing import Dict, Optional

from src.core.types import ModelProvider, AgentRole
from src.config.agent import SystemConfig, AgentConfig
from src.config.model import ModelConfig, SearchConfig

def render_settings():
    """Render the settings management interface."""
    st.title("Settings ⚙️")
    
    # Load current configurations
    system_config = st.session_state.get("system_config", SystemConfig())
    
    # Create tabs for different settings categories
    tab1, tab2, tab3, tab4 = st.tabs([
        "Model Providers",
        "Agent Settings",
        "System Settings",
        "Search Settings"
    ])
    
    with tab1:
        render_model_settings(system_config)
    
    with tab2:
        render_agent_settings(system_config)
    
    with tab3:
        render_system_settings(system_config)
    
    with tab4:
        render_search_settings(system_config)
    
    # Save settings button
    if st.sidebar.button("Save Settings"):
        save_settings(system_config)
        st.sidebar.success("Settings saved successfully!")

def render_model_settings(config: SystemConfig):
    """Render model provider settings."""
    st.header("Model Provider Settings")
    
    # Default model provider
    st.subheader("Default Provider")
    config.default_model_provider = st.selectbox(
        "Default Model Provider",
        options=[provider.value for provider in ModelProvider],
        index=list(ModelProvider).index(config.default_model_provider)
    )
    
    # API Keys
    st.subheader("API Keys")
    col1, col2 = st.columns(2)
    
    with col1:
        groq_key = st.text_input(
            "Groq API Key",
            value=config.api_keys.get(ModelProvider.GROQ, ""),
            type="password"
        )
        if groq_key:
            config.api_keys[ModelProvider.GROQ] = groq_key
        
        openai_key = st.text_input(
            "OpenAI API Key",
            value=config.api_keys.get(ModelProvider.OPENAI, ""),
            type="password"
        )
        if openai_key:
            config.api_keys[ModelProvider.OPENAI] = openai_key
    
    with col2:
        anthropic_key = st.text_input(
            "Anthropic API Key",
            value=config.api_keys.get(ModelProvider.ANTHROPIC, ""),
            type="password"
        )
        if anthropic_key:
            config.api_keys[ModelProvider.ANTHROPIC] = anthropic_key
        
        tavily_key = st.text_input(
            "Tavily API Key",
            value=config.search_config.api_key or "",
            type="password"
        )
        if tavily_key:
            config.search_config.api_key = tavily_key
    
    # Model-specific settings
    st.subheader("Model Settings")
    for provider in ModelProvider:
        with st.expander(f"{provider.value} Settings"):
            model_config = config.model_configs.get(provider, ModelConfig(provider=provider))
            
            # Temperature
            model_config.temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=model_config.temperature,
                key=f"temp_{provider.value}"
            )
            
            # Max tokens
            model_config.max_tokens = st.number_input(
                "Max Tokens",
                min_value=1,
                max_value=32000,
                value=model_config.max_tokens,
                key=f"tokens_{provider.value}"
            )
            
            # Top P
            model_config.top_p = st.slider(
                "Top P",
                min_value=0.0,
                max_value=1.0,
                value=model_config.top_p,
                key=f"top_p_{provider.value}"
            )
            
            # Presence penalty
            model_config.presence_penalty = st.slider(
                "Presence Penalty",
                min_value=-2.0,
                max_value=2.0,
                value=model_config.presence_penalty,
                key=f"presence_{provider.value}"
            )
            
            # Frequency penalty
            model_config.frequency_penalty = st.slider(
                "Frequency Penalty",
                min_value=-2.0,
                max_value=2.0,
                value=model_config.frequency_penalty,
                key=f"frequency_{provider.value}"
            )
            
            config.model_configs[provider] = model_config

def render_agent_settings(config: SystemConfig):
    """Render agent-specific settings."""
    st.header("Agent Settings")
    
    # Maximum agents
    config.max_agents = st.number_input(
        "Maximum Concurrent Agents",
        min_value=1,
        max_value=50,
        value=config.max_agents
    )
    
    # Agent roles and capabilities
    st.subheader("Role Settings")
    for role in AgentRole:
        with st.expander(f"{role.value} Settings"):
            agent_config = config.agent_configs.get(role, AgentConfig())
            
            # Basic settings
            agent_config.enable_search = st.checkbox(
                "Enable Search",
                value=agent_config.enable_search,
                key=f"search_{role.value}"
            )
            
            agent_config.response_timeout = st.number_input(
                "Response Time Limit (s)",
                min_value=1,
                max_value=300,
                value=agent_config.response_timeout,
                key=f"timeout_{role.value}"
            )
            
            agent_config.confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=agent_config.confidence_threshold,
                key=f"confidence_{role.value}"
            )
            
            # Advanced settings
            with st.expander("Advanced Settings"):
                agent_config.max_retries = st.number_input(
                    "Max Retries",
                    min_value=0,
                    max_value=10,
                    value=agent_config.max_retries,
                    key=f"retries_{role.value}"
                )
                
                agent_config.backoff_factor = st.number_input(
                    "Backoff Factor",
                    min_value=0.1,
                    max_value=5.0,
                    value=agent_config.backoff_factor,
                    step=0.1,
                    key=f"backoff_{role.value}"
                )
                
                agent_config.max_context_length = st.number_input(
                    "Max Context Length",
                    min_value=100,
                    max_value=32000,
                    value=agent_config.max_context_length,
                    key=f"context_{role.value}"
                )
            
            config.agent_configs[role] = agent_config

def render_system_settings(config: SystemConfig):
    """Render system-wide settings."""
    st.header("System Settings")
    
    # Logging
    st.subheader("Logging")
    config.logging_level = st.selectbox(
        "Logging Level",
        options=["DEBUG", "INFO", "WARNING", "ERROR"],
        index=["DEBUG", "INFO", "WARNING", "ERROR"].index(config.logging_level)
    )
    
    # Visualization
    st.subheader("Visualization")
    config.visualization_refresh_rate = st.number_input(
        "Refresh Rate (seconds)",
        min_value=0.1,
        max_value=60.0,
        value=config.visualization_refresh_rate,
        step=0.1
    )
    
    config.visualization_layout = st.selectbox(
        "Default Layout",
        options=["Tree", "Force-Directed", "Hierarchical"],
        index=["Tree", "Force-Directed", "Hierarchical"].index(config.visualization_layout)
    )
    
    config.primary_color = st.color_picker(
        "Primary Color",
        value=config.primary_color
    )
    
    config.show_confidence = st.checkbox(
        "Show Confidence Scores",
        value=config.show_confidence
    )
    
    config.realtime_updates = st.checkbox(
        "Real-time Updates",
        value=config.realtime_updates
    )
    
    # Resource limits
    st.subheader("Resource Limits")
    config.max_memory_mb = st.number_input(
        "Max Memory Usage (MB)",
        min_value=100,
        max_value=10000,
        value=config.max_memory_mb
    )
    
    config.max_cpu_percent = st.number_input(
        "Max CPU Usage (%)",
        min_value=10,
        max_value=100,
        value=config.max_cpu_percent
    )

def render_search_settings(config: SystemConfig):
    """Render search-related settings."""
    st.header("Search Settings")
    
    search_config = config.search_config
    
    # Search parameters
    st.subheader("Search Parameters")
    search_config.max_results = st.number_input(
        "Maximum Results per Search",
        min_value=1,
        max_value=100,
        value=search_config.max_results
    )
    
    search_config.cache_duration = st.number_input(
        "Cache Duration (minutes)",
        min_value=1,
        max_value=1440,
        value=search_config.cache_duration
    )
    
    # Advanced settings
    st.subheader("Advanced Settings")
    search_config.include_news = st.checkbox(
        "Include News Sources",
        value=search_config.include_news
    )
    
    search_config.include_academic = st.checkbox(
        "Include Academic Sources",
        value=search_config.include_academic
    )
    
    search_config.excluded_domains = st.multiselect(
        "Excluded Domains",
        options=search_config.available_domains,
        default=search_config.excluded_domains
    )
    
    # Custom search settings
    with st.expander("Custom Search Settings"):
        search_config.max_depth = st.slider(
            "Max Search Depth",
            min_value=1,
            max_value=10,
            value=search_config.max_depth
        )
        
        search_config.relevance_threshold = st.slider(
            "Relevance Threshold",
            min_value=0.0,
            max_value=1.0,
            value=search_config.relevance_threshold
        )
        
        search_config.max_tokens_per_result = st.number_input(
            "Max Tokens per Result",
            min_value=100,
            max_value=2000,
            value=search_config.max_tokens_per_result
        )

def save_settings(config: SystemConfig):
    """
    Save the current configuration.
    
    Args:
        config: Current system configuration
    """
    st.session_state["system_config"] = config
    
    # Save to configuration file
    config_path = "config/system_config.json"
    with open(config_path, "w") as f:
        import json
        json.dump(config.dict(), f, indent=2)
    
    # Update environment variables
    for provider, key in config.api_keys.items():
        if key:
            os.environ[f"{provider.value.upper()}_API_KEY"] = key
    
    if config.search_config.api_key:
        os.environ["TAVILY_API_KEY"] = config.search_config.api_key

if __name__ == "__main__":
    render_settings()
