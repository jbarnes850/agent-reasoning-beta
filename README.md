# Agent Reasoning Beta 🤖🔍

A powerful open-source platform for visualizing and understanding multi-agent reasoning processes. This project serves as both an educational tool and a practical sandbox for exploring how AI agents think, reason, and collaborate.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🎯 Purpose

Agent Reasoning Beta is designed to:
- **Visualize** complex reasoning processes of AI agents
- **Teach** concepts of multi-agent systems and LLM reasoning
- **Explore** different agent architectures and decision-making patterns
- **Verify** agent behavior and consensus-building mechanisms
- **Learn** through interactive experimentation and visualization

## 🌟 Key Features

- **Interactive Visualization**
  - Real-time reasoning process trees
  - Agent interaction networks
  - Decision confidence scoring
  - Step-by-step reasoning playback

- **Multi-Model Support**
  - Groq: `llama-3.1-70b-versatile`
  - OpenAI: `gpt-4o`, `gpt-4o-mini`
  - Anthropic: `claude-3-5-sonnet-latest`, `claude-3-5-haiku-latest`

- **Educational Tools**
  - Built-in reasoning patterns and examples
  - Detailed logging and explanation of agent decisions
  - Configurable visualization layouts
  - Export and share capabilities

## 🚀 Quick Start

1. **Installation**
   ```bash
   # Clone the repository
   git clone https://github.com/jbarnes850/agent-reasoning-beta.git
   cd agent-reasoning-beta

   # Install with development dependencies
   pip install -e ".[dev]"
   ```

2. **Configure API Keys**
   ```bash
   # Copy the example environment file
   cp env.example .env

   # Edit .env with your API keys
   nano .env
   ```

   Required API keys:
   - Groq: [Get API Key](https://console.groq.com)
   - OpenAI: [Get API Key](https://platform.openai.com/api-keys)
   - Anthropic: [Get API Key](https://console.anthropic.com/account/keys)
   - Tavily: [Get API Key](https://tavily.com/#api-keys)

   For Streamlit Cloud deployment, add these as secrets in your Streamlit dashboard.

3. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## 📚 Learning Resources

### Tutorials
1. **Basic Agent Visualization**
   - Understanding the reasoning tree
   - Interpreting confidence scores
   - Analyzing agent decisions

2. **Multi-Agent Interactions**
   - Setting up agent teams
   - Configuring communication patterns
   - Analyzing consensus building

3. **Custom Reasoning Patterns**
   - Implementing new reasoning strategies
   - Adding custom visualization layouts
   - Creating specialized agents

### Example Use Cases
- Academic research on agent behavior
- Educational demonstrations of LLM reasoning
- Debugging complex agent systems
- Exploring different reasoning architectures

## 🛠️ Development

### Project Structure
```
agent-reasoning-beta/
├── src/
│   ├── core/           # Core reasoning and agent logic
│   ├── visualization/  # Visualization components
│   ├── config/        # Configuration management
│   └── utils/         # Utility functions
├── tests/             # Test suite
├── examples/          # Example notebooks and scripts
└── docs/             # Documentation
```

### Running Tests
```bash
# Run the full test suite
pytest

# Run with coverage
pytest --cov=src
```

### Code Style
We use:
- `black` for code formatting
- `isort` for import sorting
- `mypy` for type checking

```bash
# Format code
black src tests
isort src tests

# Type check
mypy src
```

## 🤝 Contributing

We welcome contributions! Whether you're:
- Adding new visualization features
- Implementing novel reasoning patterns
- Improving documentation
- Fixing bugs
- Adding educational content

Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📖 Documentation

- [API Reference](docs/api.md)
- [Architecture Guide](docs/architecture.md)
- [Educational Resources](docs/education.md)
- [Development Guide](docs/development.md)

## 🔬 Research & Education

This platform is actively used in:
- Academic research on multi-agent systems
- Educational courses on AI and LLMs
- Workshops and tutorials
- Agent behavior analysis

We encourage using this tool for:
- Teaching AI concepts
- Research projects
- Educational demonstrations
- Experimental exploration

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Built with ❤️ by NEAR AI
