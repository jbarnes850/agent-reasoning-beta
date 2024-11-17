"""
Agent implementation for the Agent Reasoning Beta platform.

This module implements the core agent system, including:
- Base agent class and role-specific implementations
- Agent coordination and communication
- LLM integration for different providers
- Agent state management and monitoring
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Type
from uuid import UUID, uuid4

import aiohttp
import backoff
from pydantic import BaseModel, ValidationError

from .reasoning import ReasoningEngine
from .types import (
    AgentRole,
    AgentState,
    ModelProvider,
    ReasoningPath,
    ReasoningType,
    ThoughtNode,
    VisualizationData,
)

class AgentMessage(BaseModel):
    """Message format for inter-agent communication."""
    id: UUID = Field(default_factory=uuid4)
    sender_id: UUID
    receiver_id: Optional[UUID] = None
    content: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    message_type: str
    priority: int = 0

class AgentMetrics(BaseModel):
    """Metrics tracked for each agent."""
    total_thoughts: int = 0
    successful_verifications: int = 0
    failed_verifications: int = 0
    consensus_participations: int = 0
    average_confidence: float = 0.0
    total_processing_time: float = 0.0
    last_active: datetime = Field(default_factory=datetime.utcnow)

class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""
    
    def __init__(
        self,
        role: AgentRole,
        model_provider: ModelProvider,
        model_config: Optional[Dict[str, Any]] = None,
    ):
        self.id: UUID = uuid4()
        self.role = role
        self.model_provider = model_provider
        self.model_config = model_config or {}
        self.state = AgentState(
            id=self.id,
            role=role,
            model_provider=model_provider,
            model_config=self.model_config
        )
        self.metrics = AgentMetrics()
        self.message_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self.reasoning_engine = ReasoningEngine()
        self._active = False
        self._last_thought: Optional[ThoughtNode] = None
        
    @abstractmethod
    async def process_message(self, message: AgentMessage) -> None:
        """Process incoming message based on agent role."""
        pass
        
    @abstractmethod
    async def generate_thought(self, context: Dict[str, Any]) -> ThoughtNode:
        """Generate next thought using LLM."""
        pass
        
    async def start(self) -> None:
        """Start agent's processing loop."""
        self._active = True
        try:
            await self._processing_loop()
        except Exception as e:
            self._active = False
            raise
            
    async def stop(self) -> None:
        """Stop agent's processing loop."""
        self._active = False
        
    async def send_message(
        self,
        receiver_id: Optional[UUID],
        content: Dict[str, Any],
        message_type: str,
        priority: int = 0
    ) -> None:
        """Send message to another agent or broadcast."""
        message = AgentMessage(
            sender_id=self.id,
            receiver_id=receiver_id,
            content=content,
            message_type=message_type,
            priority=priority
        )
        await self.message_queue.put(message)
        
    @backoff.on_exception(backoff.expo, aiohttp.ClientError, max_tries=3)
    async def _call_llm(self, prompt: str) -> str:
        """Make API call to LLM provider with exponential backoff."""
        async with aiohttp.ClientSession() as session:
            if self.model_provider == ModelProvider.GROQ:
                return await self._call_groq(session, prompt)
            elif self.model_provider == ModelProvider.OPENAI:
                return await self._call_openai(session, prompt)
            elif self.model_provider == ModelProvider.ANTHROPIC:
                return await self._call_anthropic(session, prompt)
            else:
                raise ValueError(f"Unsupported model provider: {self.model_provider}")
                
    async def _processing_loop(self) -> None:
        """Main processing loop for handling messages."""
        while self._active:
            try:
                message = await self.message_queue.get()
                await self.process_message(message)
                self.message_queue.task_done()
            except Exception as e:
                # Log error but continue processing
                continue

class ExplorerAgent(BaseAgent):
    """Agent specialized in MCTS exploration of reasoning paths."""
    
    def __init__(
        self,
        model_provider: ModelProvider,
        model_config: Optional[Dict[str, Any]] = None,
        exploration_params: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            role=AgentRole.EXPLORER,
            model_provider=model_provider,
            model_config=model_config
        )
        self.exploration_params = exploration_params or {}
        
    async def process_message(self, message: AgentMessage) -> None:
        """Process exploration-related messages."""
        if message.message_type == "explore_request":
            context = message.content.get("context", {})
            root_thought = await self.generate_thought(context)
            
            # Perform MCTS exploration
            path = await self.reasoning_engine.explore(
                root_thought,
                self.state,
                num_simulations=self.exploration_params.get("num_simulations", 100)
            )
            
            # Send results back
            await self.send_message(
                message.sender_id,
                {"path": path.dict()},
                "exploration_result"
            )
            
    async def generate_thought(self, context: Dict[str, Any]) -> ThoughtNode:
        """Generate next thought using LLM."""
        prompt = self._create_exploration_prompt(context)
        response = await self._call_llm(prompt)
        
        return ThoughtNode(
            content=response,
            confidence=self._calculate_confidence(response),
            reasoning_type=ReasoningType.MCTS
        )
        
    def _create_exploration_prompt(self, context: Dict[str, Any]) -> str:
        """Create prompt for exploration."""
        return f"Given the context: {json.dumps(context)}\nExplore the next possible step in reasoning."

class VerifierAgent(BaseAgent):
    """Agent specialized in verifying reasoning paths."""
    
    def __init__(
        self,
        model_provider: ModelProvider,
        model_config: Optional[Dict[str, Any]] = None,
        verification_threshold: float = 0.7
    ):
        super().__init__(
            role=AgentRole.VERIFIER,
            model_provider=model_provider,
            model_config=model_config
        )
        self.verification_threshold = verification_threshold
        
    async def process_message(self, message: AgentMessage) -> None:
        """Process verification-related messages."""
        if message.message_type == "verify_request":
            path = ReasoningPath(**message.content["path"])
            
            # Perform verification
            is_valid, confidence, notes = await self.reasoning_engine.verify(
                path,
                self.state
            )
            
            # Update metrics
            if is_valid:
                self.metrics.successful_verifications += 1
            else:
                self.metrics.failed_verifications += 1
                
            # Send results back
            await self.send_message(
                message.sender_id,
                {
                    "is_valid": is_valid,
                    "confidence": confidence,
                    "notes": notes
                },
                "verification_result"
            )
            
    async def generate_thought(self, context: Dict[str, Any]) -> ThoughtNode:
        """Generate verification thought using LLM."""
        prompt = self._create_verification_prompt(context)
        response = await self._call_llm(prompt)
        
        return ThoughtNode(
            content=response,
            confidence=self._calculate_confidence(response),
            reasoning_type=ReasoningType.VERIFICATION
        )
        
    def _create_verification_prompt(self, context: Dict[str, Any]) -> str:
        """Create prompt for verification."""
        return f"Verify the logical consistency of: {json.dumps(context)}"

class CoordinatorAgent(BaseAgent):
    """Agent specialized in building consensus among multiple reasoning paths."""
    
    def __init__(
        self,
        model_provider: ModelProvider,
        model_config: Optional[Dict[str, Any]] = None,
        min_paths: int = 3
    ):
        super().__init__(
            role=AgentRole.COORDINATOR,
            model_provider=model_provider,
            model_config=model_config
        )
        self.min_paths = min_paths
        self._collected_paths: List[ReasoningPath] = []
        self._participating_agents: Set[UUID] = set()
        
    async def process_message(self, message: AgentMessage) -> None:
        """Process consensus-related messages."""
        if message.message_type == "consensus_contribution":
            path = ReasoningPath(**message.content["path"])
            self._collected_paths.append(path)
            self._participating_agents.add(message.sender_id)
            
            # Check if we have enough paths
            if len(self._collected_paths) >= self.min_paths:
                consensus_path = await self.reasoning_engine.build_consensus(
                    self._collected_paths,
                    [self.state]  # In future, could include states of all participants
                )
                
                # Broadcast consensus result
                await self.send_message(
                    None,  # Broadcast
                    {"consensus_path": consensus_path.dict()},
                    "consensus_result",
                    priority=1
                )
                
                # Reset collection
                self._collected_paths = []
                self._participating_agents = set()
                
    async def generate_thought(self, context: Dict[str, Any]) -> ThoughtNode:
        """Generate consensus-related thought using LLM."""
        prompt = self._create_consensus_prompt(context)
        response = await self._call_llm(prompt)
        
        return ThoughtNode(
            content=response,
            confidence=self._calculate_confidence(response),
            reasoning_type=ReasoningType.CONSENSUS
        )
        
    def _create_consensus_prompt(self, context: Dict[str, Any]) -> str:
        """Create prompt for consensus building."""
        return f"Build consensus among these perspectives: {json.dumps(context)}"

class AgentFactory:
    """Factory for creating and managing agents."""
    
    _agent_types: Dict[AgentRole, Type[BaseAgent]] = {
        AgentRole.EXPLORER: ExplorerAgent,
        AgentRole.VERIFIER: VerifierAgent,
        AgentRole.COORDINATOR: CoordinatorAgent,
    }
    
    @classmethod
    def create_agent(
        cls,
        role: AgentRole,
        model_provider: ModelProvider,
        model_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> BaseAgent:
        """Create an agent of the specified role."""
        agent_class = cls._agent_types.get(role)
        if not agent_class:
            raise ValueError(f"Unsupported agent role: {role}")
            
        return agent_class(model_provider, model_config, **kwargs)
