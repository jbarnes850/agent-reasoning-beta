# Contributing to Agent Reasoning Beta

Thank you for your interest in contributing to Agent Reasoning Beta! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

This project adheres to the Contributor Covenant code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to engineering@codeium.com.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/agent-reasoning-beta.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate the environment: 
   - Windows: `venv\Scripts\activate`
   - Unix/macOS: `source venv/bin/activate`
5. Install dependencies: `pip install -e ".[dev]"`
6. Create a new branch: `git checkout -b feature/your-feature-name`

## Development Guidelines

### Code Style

- We use `black` for code formatting
- Sort imports with `isort`
- Type hints are required for all functions
- Use docstrings for all public functions and classes

### Testing

- Write tests for all new features
- Maintain or improve code coverage
- Run tests locally before submitting: `pytest tests/`

### Documentation

- Update relevant documentation for any changes
- Include docstrings for new functions and classes
- Add type hints for all function parameters and returns

### Commit Messages

Follow the conventional commits specification:
- feat: new feature
- fix: bug fix
- docs: documentation changes
- style: formatting changes
- refactor: code restructuring
- test: test additions or modifications
- chore: maintenance tasks

### Pull Request Process

1. Update documentation
2. Add or update tests
3. Run linting and tests locally
4. Create a pull request with a clear description
5. Link any related issues
6. Wait for review and address feedback

## Development Setup

### Required Environment Variables

Copy `env.example` to `.env` and fill in your API keys:
```bash
cp env.example .env
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src

# Run specific test file
pytest tests/test_core.py
```

### Running Linting

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/
```

## Questions or Problems?

- Check existing issues
- Create a new issue with a clear description
- Join our community discussions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
