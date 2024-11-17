## Agent Reasoning Beta - Configuration Guide

This document details all configuration options for the Agent Reasoning Beta platform, including system-wide settings, agent configurations, model settings, and visualization parameters.

### System Configuration

The `SystemConfig` class provides global configuration options:

```python
from core.types import SystemConfig, ModelProvider

config = SystemConfig(
    max_agents=10,                    # Maximum number of concurrent agents
    default_model_provider=ModelProvider.GROQ,
    logging_level="INFO",
    visualization_refresh_rate=1.0,   # Visualization update frequency in seconds
    api_keys={                        # API keys for different providers
        ModelProvider.GROQ: "your-groq-key",
        ModelProvider.OPENAI: "your-openai-key",
        ModelProvider.ANTHROPIC: "your-anthropic-key"
    }
)
```

### Agent Configuration

#### Base Agent Configuration
All agents share these base configuration options:

```python
from core.types import AgentRole, ModelProvider

base_config = {
    "role": AgentRole.EXPLORER,       # Agent's role in the system
    "model_provider": ModelProvider.GROQ,
    "model_config": {
        "temperature": 0.7,
        "max_tokens": 1000,
        "top_p": 1.0
    }
}
```

#### Role-Specific Configurations

##### Explorer Agent
```python
explorer_config = {
    **base_config,
    "role": AgentRole.EXPLORER,
    "exploration_params": {
        "c_puct": 1.0,               # Exploration constant
        "max_depth": 10,             # Maximum exploration depth
        "min_visits": 3              # Minimum visits before expansion
    }
}
```

##### Verifier Agent
```python
verifier_config = {
    **base_config,
    "role": AgentRole.VERIFIER,
    "verification_threshold": 0.7     # Minimum confidence for verification
}
```

##### Coordinator Agent
```python
coordinator_config = {
    **base_config,
    "role": AgentRole.COORDINATOR,
    "min_paths": 3                   # Minimum paths for consensus
}
```

### Model Configuration

Each LLM provider has specific configuration options:

#### Groq Configuration
```python
groq_config = ModelConfig(
    provider=ModelProvider.GROQ,
    model_name="mixtral-8x7b-32768",
    temperature=0.7,
    max_tokens=1000,
    top_p=1.0,
    presence_penalty=0.0,
    frequency_penalty=0.0,
    timeout=30.0,
    retry_attempts=3
)
```

#### OpenAI Configuration
```python
openai_config = ModelConfig(
    provider=ModelProvider.OPENAI,
    model_name="gpt-4-turbo",
    temperature=0.7,
    max_tokens=1000,
    top_p=1.0,
    presence_penalty=0.0,
    frequency_penalty=0.0,
    timeout=30.0,
    retry_attempts=3
)
```

#### Anthropic Configuration
```python
anthropic_config = ModelConfig(
    provider=ModelProvider.ANTHROPIC,
    model_name="claude-3-opus",
    temperature=0.7,
    max_tokens=1000,
    top_p=1.0,
    presence_penalty=0.0,
    frequency_penalty=0.0,
    timeout=30.0,
    retry_attempts=3
)
```

### Visualization Configuration

#### Tree Visualization
```python
tree_config = TreeConfig(
    # Layout
    LAYOUT_WIDTH=1200,
    LAYOUT_HEIGHT=800,
    MIN_NODE_DISTANCE=100,
    
    # Node styling
    NODE_SIZE_RANGE=(20, 50),
    EDGE_WIDTH_RANGE=(1, 5),
    
    # Color schemes by reasoning type
    COLOR_SCHEMES={
        ReasoningType.MCTS: {
            "node": "#1f77b4",
            "edge": "#7fcdbb",
            "text": "#2c3e50"
        },
        ReasoningType.VERIFICATION: {
            "node": "#2ca02c",
            "edge": "#98df8a",
            "text": "#2c3e50"
        },
        ReasoningType.CONSENSUS: {
            "node": "#ff7f0e",
            "edge": "#ffbb78",
            "text": "#2c3e50"
        }
    }
)
```

#### Graph Visualization
```python
graph_config = GraphConfig(
    # Layout
    LAYOUT_WIDTH=1200,
    LAYOUT_HEIGHT=800,
    MIN_NODE_DISTANCE=100,
    
    # Node styling
    NODE_SIZE_RANGE=(20, 50),
    EDGE_WIDTH_RANGE=(1, 5),
    
    # Color schemes by agent role
    COLOR_SCHEMES={
        AgentRole.EXPLORER: {
            "node": "#1f77b4",
            "edge": "#7fcdbb",
            "text": "#2c3e50"
        },
        AgentRole.VERIFIER: {
            "node": "#2ca02c",
            "edge": "#98df8a",
            "text": "#2c3e50"
        },
        AgentRole.COORDINATOR: {
            "node": "#ff7f0e",
            "edge": "#ffbb78",
            "text": "#2c3e50"
        }
    }
)
```

#### Metrics Visualization
```python
metrics_config = MetricsConfig(
    # Layout
    LAYOUT_WIDTH=1200,
    LAYOUT_HEIGHT=800,
    
    # Color schemes
    COLOR_SCHEMES={
        "success": "#2ecc71",
        "failure": "#e74c3c",
        "neutral": "#3498db",
        "warning": "#f1c40f"
    },
    
    # Chart types
    CHART_TYPES=[
        "line",
        "bar",
        "scatter",
        "area",
        "histogram"
    ],
    
    # Time windows
    TIME_WINDOWS=[
        "1h",
        "6h",
        "24h",
        "7d",
        "30d"
    ]
)
```

#### Heatmap Visualization
```python
heatmap_config = HeatmapConfig(
    # Layout
    LAYOUT_WIDTH=1200,
    LAYOUT_HEIGHT=800,
    
    # Color scales
    COLOR_SCALES={
        "activity": "Viridis",
        "performance": "RdYlGn",
        "interaction": "Blues",
        "resource": "Oranges"
    },
    
    # Time bins
    TIME_BINS={
        "hourly": 24,
        "daily": 7,
        "weekly": 4,
        "monthly": 12
    },
    
    # Threshold levels
    THRESHOLD_LEVELS={
        "low": 0.3,
        "medium": 0.6,
        "high": 0.8
    }
)
```

## Environment Variables

Required environment variables:
```bash
# API Keys
GROQ_API_KEY=your-groq-key
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key

# System Settings
MAX_AGENTS=10
LOG_LEVEL=INFO
VISUALIZATION_REFRESH_RATE=1.0

# Model Settings
DEFAULT_MODEL_PROVIDER=groq
DEFAULT_TEMPERATURE=0.7
DEFAULT_MAX_TOKENS=1000
```

## Configuration Best Practices

1. API Key Management
   - Store API keys in environment variables
   - Never commit API keys to version control
   - Use a `.env` file for local development

2. Model Configuration
   - Start with conservative temperature values (0.7)
   - Adjust based on task requirements
   - Monitor token usage and costs

3. Agent Configuration
   - Scale max_agents based on system resources
   - Adjust exploration parameters based on task complexity
   - Set appropriate verification thresholds

4. Visualization Configuration
   - Adjust layout dimensions to display size
   - Use colorblind-friendly color schemes
   - Set appropriate refresh rates for performance

5. Error Handling
   - Set appropriate timeout values
   - Configure retry attempts
   - Implement proper error logging