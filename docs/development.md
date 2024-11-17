# Development Guide

## Getting Started

### Prerequisites
- Python 3.12+
- pip
- git
- virtualenv (recommended)

### Local Setup
```bash
# Clone repository
git clone https://github.com/jbarnes850/agent-reasoning-beta.git
cd agent-reasoning-beta

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Copy and configure environment
cp env.example .env
# Edit .env with your API keys
```

## Development Workflow

### 1. Code Organization

```
agent-reasoning-beta/
├── src/
│   ├── core/           # Core business logic
│   ├── visualization/  # Visualization components
│   ├── config/        # Configuration management
│   └── utils/         # Utility functions
├── tests/             # Test suite
├── examples/          # Example code
└── docs/             # Documentation
```

### 2. Development Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Follow code style guidelines
   - Add tests
   - Update documentation

3. **Run Tests**
   ```bash
   # Run all tests
   pytest

   # Run with coverage
   pytest --cov=src

   # Run specific test
   pytest tests/test_core.py
   ```

4. **Format Code**
   ```bash
   # Format code
   black src tests

   # Sort imports
   isort src tests

   # Type check
   mypy src
   ```

5. **Submit PR**
   - Create pull request
   - Add description
   - Link related issues

## Code Style Guidelines

### Python Style

1. **Formatting**
   - Use black for code formatting
   - 88 character line length
   - Follow PEP 8

2. **Type Hints**
   ```python
   def process_data(data: List[str]) -> Dict[str, Any]:
       pass
   ```

3. **Docstrings**
   ```python
   def complex_function(param1: str, param2: int) -> bool:
       """Process data with specific parameters.
       
       Args:
           param1: Description of param1
           param2: Description of param2
           
       Returns:
           bool: Description of return value
           
       Raises:
           ValueError: When parameters are invalid
       """
       pass
   ```

### Testing Guidelines

1. **Test Structure**
   ```python
   @pytest.mark.asyncio
   async def test_feature():
       # Arrange
       data = setup_test_data()
       
       # Act
       result = await process_data(data)
       
       # Assert
       assert result.status == "success"
   ```

2. **Mock Usage**
   ```python
   @patch("src.core.models.ModelAPI")
   async def test_with_mock(mock_api):
       mock_api.return_value = test_response
       result = await function_under_test()
       assert result == expected_result
   ```

3. **Fixtures**
   ```python
   @pytest.fixture
   def test_data():
       return {
           "key": "value"
       }
   ```

## Common Development Tasks

### 1. Adding a New Model

1. **Update Configuration**
   ```python
   # src/config/model.py
   SUPPORTED_MODELS = {
       "new-provider": ["model-name"]
   }
   ```

2. **Implement Interface**
   ```python
   # src/core/models.py
   class NewProviderModel(BaseModel):
       async def generate(self, prompt: str) -> str:
           pass
   ```

### 2. Adding a Visualization

1. **Create Component**
   ```python
   # src/visualization/components/new_view.py
   class NewVisualization:
       def render(self, data: Any) -> None:
           pass
   ```

2. **Update Layout**
   ```python
   # src/visualization/layouts/main_layout.py
   class MainLayout:
       def add_visualization(self):
           NewVisualization().render(data)
   ```

### 3. Adding Configuration

1. **Update Environment**
   ```python
   # env.example
   NEW_CONFIG_VALUE="default"
   ```

2. **Add to Config**
   ```python
   # src/config/system.py
   class SystemConfig:
       new_value: str = Field(...)
   ```

## Debugging

### 1. Logging

```python
from src.utils.logger import get_logger

logger = get_logger(__name__)
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
```

### 2. Debug Configuration

```python
# .env
DEBUG_MODE=true
LOG_LEVEL=DEBUG
```

### 3. Streamlit Debugging

```python
import streamlit as st

st.write("Debug info:")
st.json(debug_data)
```

## Performance Optimization

### 1. Caching

```python
@st.cache_data
def expensive_computation():
    pass

@st.cache_resource
def load_model():
    pass
```

### 2. Async Operations

```python
async def batch_process(items: List[str]) -> List[Result]:
    tasks = [process_item(item) for item in items]
    return await asyncio.gather(*tasks)
```

### 3. Memory Management

```python
@contextmanager
def manage_resources():
    try:
        yield resource
    finally:
        resource.cleanup()
```

## Deployment

### 1. Local Development
```bash
streamlit run src/app.py
```

### 2. Streamlit Cloud
1. Push to GitHub
2. Connect to Streamlit Cloud
3. Configure secrets

### 3. Custom Server
```bash
# Production setup
gunicorn src.app:server
```

## Troubleshooting

### Common Issues

1. **API Connection**
   - Check API keys
   - Verify network connection
   - Check rate limits

2. **Memory Issues**
   - Monitor resource usage
   - Implement cleanup
   - Use streaming responses

3. **Performance**
   - Profile code
   - Optimize queries
   - Use caching

## Contributing

1. **Issue Creation**
   - Use templates
   - Provide reproduction steps
   - Include system info

2. **Pull Requests**
   - Follow guidelines
   - Include tests
   - Update docs

3. **Code Review**
   - Review checklist
   - Performance impact
   - Security considerations
