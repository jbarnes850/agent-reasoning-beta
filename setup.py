"""Setup configuration for agent-reasoning-beta package."""

from setuptools import setup, find_packages

setup(
    name="agent-reasoning-beta",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src", exclude=["tests"]),
    include_package_data=True,
    python_requires=">=3.11",
    install_requires=[
        "pydantic>=2.9.0",  # Using latest Pydantic for modern configuration style
        "numpy>=1.21.0",
        "backoff>=2.2.1",
        "aiohttp>=3.8.0",
        "groq>=0.4.0",
        "openai>=1.0.0",
        "anthropic>=0.8.0",
        "streamlit>=1.32.0",
        "plotly>=5.18.0",
        "networkx>=3.2.1",
        "pyvis>=0.3.2",
        "pandas>=2.2.0",
        "tavily-python>=0.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=24.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
        ]
    }
)
