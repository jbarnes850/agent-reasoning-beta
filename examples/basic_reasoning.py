"""Basic example of using the Agent Reasoning Beta platform."""

import asyncio
from dotenv import load_dotenv
import os

from src.core.models import ModelConfig
from src.core.agents import Agent
from src.core.reasoning import ReasoningEngine
from src.utils.logger import get_logger

# Load environment variables
load_dotenv()

# Initialize logger
logger = get_logger(__name__)

async def main():
    """Run a basic reasoning example."""
    try:
        # Initialize configuration
        config = ModelConfig(
            provider="groq",
            name="llama-3.1-70b-versatile",
            temperature=0.7,
            max_tokens=1000
        )
        
        # Initialize agent and reasoning engine
        agent = Agent(config)
        engine = ReasoningEngine()
        
        # Example reasoning task
        question = "What are the potential implications of quantum computing on cryptography?"
        
        # Get agent's response
        response = await agent.generate_response(question)
        
        # Process reasoning
        reasoning_tree = await engine.analyze_response(
            question=question,
            response=response
        )
        
        # Print results
        logger.info(f"Question: {question}")
        logger.info(f"Response: {response.content}")
        logger.info(f"Confidence: {response.confidence.score}")
        logger.info(f"Reasoning: {response.confidence.reasoning}")
        logger.info("Evidence:")
        for evidence in response.confidence.evidence:
            logger.info(f"- {evidence}")
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
