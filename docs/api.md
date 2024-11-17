# API Reference

## Core Components

### Agent API

#### `Agent`
```python
class Agent:
    def __init__(self, model_config: ModelConfig)
    async def generate_response(self, prompt: str) -> AgentResponse
    async def analyze_confidence(self, response: str) -> Confidence
```

The `Agent` class represents an individual reasoning agent that can process prompts and generate responses.

#### `ReasoningEngine`
```python
class ReasoningEngine:
    async def analyze_response(self, question: str, response: AgentResponse) -> ReasoningTree
    async def build_consensus(self, responses: List[AgentResponse]) -> ConsensusResult
```

The `ReasoningEngine` processes agent responses and builds reasoning trees and consensus data.

### Model Configuration

#### `ModelConfig`
```python
class ModelConfig:
    provider: str  # "groq", "openai", or "anthropic"
    name: str      # Model name
    temperature: float
    max_tokens: int
```

Configuration for language models and their parameters.

### Types

#### `AgentResponse`
```python
class AgentResponse:
    content: str
    confidence: Confidence
    metadata: Dict[str, Any]
```

Represents a structured response from an agent.

#### `Confidence`
```python
class Confidence:
    score: float  # 0.0 to 1.0
    reasoning: str
    evidence: List[str]
```

Confidence scoring and reasoning information.

### Visualization Components

#### `MainLayout`
```python
class MainLayout:
    def render(self, mcts_data, verification_data, consensus_data)
```

Main application layout with visualization components.

#### `DashboardLayout`
```python
class DashboardLayout:
    def render(self)
```

Analytics dashboard layout.

## Usage Examples

### Basic Agent Usage
```python
from src.core.models import ModelConfig
from src.core.agents import Agent

config = ModelConfig(
    provider="groq",
    name="llama-3.1-70b-versatile",
    temperature=0.7,
    max_tokens=1000
)

agent = Agent(config)
response = await agent.generate_response("What is quantum computing?")
```

### Multi-Agent Consensus
```python
from src.core.reasoning import ReasoningEngine

engine = ReasoningEngine()
consensus = await engine.build_consensus([response1, response2, response3])
```

## Utility Functions

### Configuration
```python
from src.utils.config import load_config, save_config

config = load_config()
save_config(config)
```

### Logging
```python
from src.utils.logger import get_logger

logger = get_logger(__name__)
logger.info("Message")
```

### Helper Functions
```python
from src.utils.helpers import retry, timed, truncate_text

@retry(max_attempts=3)
@timed
async def my_function():
    pass
```

## Error Handling

The API uses standard Python exceptions with additional context:

- `ConfigurationError`: Configuration-related errors
- `ModelError`: Model API errors
- `ValidationError`: Input validation errors
- `VisualizationError`: Visualization-related errors

Example error handling:
```python
try:
    response = await agent.generate_response(prompt)
except ModelError as e:
    logger.error(f"Model error: {str(e)}")
    # Handle error appropriately
```

## Best Practices

1. **Configuration Management**
   - Use environment variables for sensitive data
   - Load configuration from files for other settings
   - Validate all configuration before use

2. **Error Handling**
   - Always catch and handle exceptions appropriately
   - Log errors with context
   - Provide meaningful error messages to users

3. **Performance**
   - Use async functions for I/O operations
   - Implement caching where appropriate
   - Monitor and optimize token usage

4. **Security**
   - Never expose API keys in code
   - Validate all inputs
   - Sanitize outputs before display

## Rate Limits and Quotas

Different model providers have different rate limits:

- Groq: Varies by subscription
- OpenAI: Varies by model and subscription
- Anthropic: Varies by subscription

Use appropriate retry logic and respect rate limits:
```python
@retry(max_attempts=3, delay=1.0, backoff=2.0)
async def rate_limited_call():
    pass
```
