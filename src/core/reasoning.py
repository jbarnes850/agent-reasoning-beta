"""
Core reasoning implementations for the Agent Reasoning Beta platform.

This module implements the fundamental reasoning primitives:
- Monte Carlo Tree Search (MCTS) for exploration
- Verification of reasoning paths
- Consensus building among multiple agents
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import numpy as np
from pydantic import ValidationError

from .types import (
    AgentState,
    Confidence,
    ModelResponse,
    ReasoningPath,
    ReasoningResult,
    ReasoningType,
    ThoughtNode,
    VisualizationData,
)

# MCTS Constants
C_PUCT = 1.0  # Exploration constant
MAX_DEPTH = 10
MIN_VISITS = 3


@dataclass
class MCTSStats:
    """Statistics tracked for each node in the MCTS tree."""
    visits: int = 0
    total_value: float = 0.0
    prior_probability: float = 1.0
    
    @property
    def mean_value(self) -> float:
        """Calculate mean value of the node."""
        return self.total_value / max(1, self.visits)


class ReasoningEngine:
    """Core reasoning engine implementing MCTS, verification, and consensus algorithms."""
    
    def __init__(self, max_depth: int = MAX_DEPTH):
        self.max_depth = max_depth
        self.stats: Dict[UUID, MCTSStats] = {}
        self._reasoning_cache: Dict[UUID, ReasoningResult] = {}
    
    async def explore(
        self,
        root_node: ThoughtNode,
        agent_state: AgentState,
        num_simulations: int = 100
    ) -> ReasoningPath:
        """
        Perform MCTS exploration starting from the root node.
        
        Args:
            root_node: Starting point for exploration
            agent_state: Current state of the exploring agent
            num_simulations: Number of MCTS simulations to run
            
        Returns:
            ReasoningPath containing the most promising sequence of thoughts
        """
        try:
            self.stats[root_node.id] = MCTSStats()
            
            for _ in range(num_simulations):
                node = root_node
                search_path = [node]
                
                # Selection phase
                while node.id in self._reasoning_cache:
                    node = self._select_child(node)
                    search_path.append(node)
                    
                    if len(search_path) >= self.max_depth:
                        break
                
                # Expansion and simulation
                if len(search_path) < self.max_depth:
                    child_node = await self._expand_node(node, agent_state)
                    search_path.append(child_node)
                    
                    # Simulate from the new node
                    value = await self._simulate(child_node, agent_state)
                else:
                    value = await self._evaluate(node, agent_state)
                
                # Backpropagation
                self._backpropagate(search_path, value)
            
            # Extract best path
            best_path = self._extract_best_path(root_node)
            return ReasoningPath(
                nodes=best_path,
                total_confidence=self._calculate_path_confidence(best_path)
            )
            
        except Exception as e:
            # Log error and return minimal valid path
            return ReasoningPath(nodes=[root_node], total_confidence=0.0)

    async def verify(
        self,
        reasoning_path: ReasoningPath,
        agent_state: AgentState,
    ) -> Tuple[bool, float, List[str]]:
        """
        Verify a reasoning path for logical consistency and soundness.
        
        Args:
            reasoning_path: Path to verify
            agent_state: State of the verifying agent
            
        Returns:
            Tuple of (is_valid, confidence_score, verification_notes)
        """
        verification_notes = []
        total_confidence = 0.0
        
        # Verify each transition in the path
        for i in range(len(reasoning_path.nodes) - 1):
            current = reasoning_path.nodes[i]
            next_node = reasoning_path.nodes[i + 1]
            
            # Check logical connection
            transition_score = await self._verify_transition(
                current, next_node, agent_state
            )
            total_confidence += transition_score
            
            if transition_score < 0.5:
                verification_notes.append(
                    f"Weak transition between nodes {current.id} and {next_node.id}"
                )
        
        # Calculate final confidence
        avg_confidence = total_confidence / max(1, len(reasoning_path.nodes) - 1)
        is_valid = avg_confidence >= 0.7
        
        return is_valid, avg_confidence, verification_notes

    async def build_consensus(
        self,
        paths: List[ReasoningPath],
        agent_states: List[AgentState],
    ) -> ReasoningPath:
        """
        Build consensus among multiple reasoning paths.
        
        Args:
            paths: List of reasoning paths to consider
            agent_states: States of participating agents
            
        Returns:
            Consensus reasoning path
        """
        if not paths:
            raise ValueError("Cannot build consensus with empty paths")
            
        # Weight paths by their confidence and verification status
        weighted_paths = []
        for path in paths:
            weight = path.total_confidence
            if path.verified:
                weight *= 1.5  # Boost verified paths
            weighted_paths.append((weight, path))
            
        # Sort by weight
        weighted_paths.sort(reverse=True, key=lambda x: x[0])
        
        # Take weighted average of top paths
        top_k = min(3, len(weighted_paths))
        consensus_nodes = []
        
        # Merge nodes from top paths
        max_length = max(len(p[1].nodes) for p in weighted_paths[:top_k])
        for i in range(max_length):
            merged_content = []
            merged_confidence = 0.0
            
            for weight, path in weighted_paths[:top_k]:
                if i < len(path.nodes):
                    node = path.nodes[i]
                    merged_content.append(node.content)
                    merged_confidence += weight * node.confidence
                    
            if merged_content:
                consensus_node = ThoughtNode(
                    content=" | ".join(merged_content),
                    confidence=merged_confidence / sum(w for w, _ in weighted_paths[:top_k]),
                    reasoning_type=ReasoningType.CONSENSUS,
                )
                consensus_nodes.append(consensus_node)
        
        return ReasoningPath(
            nodes=consensus_nodes,
            total_confidence=sum(w * p.total_confidence for w, p in weighted_paths[:top_k]) / sum(w for w, _ in weighted_paths[:top_k]),
            consensus_reached=True
        )

    async def _select_child(self, node: ThoughtNode) -> ThoughtNode:
        """Select most promising child node using PUCT algorithm."""
        children = self._reasoning_cache.get(node.id, [])
        if not children:
            return node
            
        # Calculate UCT scores
        scores = []
        total_visits = sum(self.stats[c.id].visits for c in children)
        
        for child in children:
            stats = self.stats[child.id]
            exploitation = stats.mean_value
            exploration = (C_PUCT * stats.prior_probability * 
                         math.sqrt(total_visits) / (1 + stats.visits))
            scores.append(exploitation + exploration)
            
        return children[np.argmax(scores)]

    async def _expand_node(
        self,
        node: ThoughtNode,
        agent_state: AgentState
    ) -> ThoughtNode:
        """
        Expand a node by generating a new child thought using the agent's LLM.
        
        This implements the 'Chain of Thought' approach where each expansion
        considers both the semantic context and potential code implications.
        """
        try:
            # Prepare context from parent nodes
            context = self._build_context(node)
            
            # Generate next thought using agent's LLM
            response = await self._generate_thought(
                context=context,
                agent_state=agent_state
            )
            
            # Create new thought node
            child_node = ThoughtNode(
                content=response["thought"],
                confidence=response["confidence"],
                reasoning_type=node.reasoning_type,
                parent_id=node.id,
                metadata={
                    "code_context": response.get("code_context"),
                    "validation_steps": response.get("validation_steps", [])
                }
            )
            
            # Update reasoning cache
            if node.id not in self._reasoning_cache:
                self._reasoning_cache[node.id] = []
            self._reasoning_cache[node.id].append(child_node)
            
            # Initialize stats
            self.stats[child_node.id] = MCTSStats(
                prior_probability=response["confidence"]
            )
            
            return child_node
            
        except Exception as e:
            # Fallback to safe expansion
            return ThoughtNode(
                content=f"Error in expansion: {str(e)}",
                confidence=0.1,
                reasoning_type=node.reasoning_type,
                parent_id=node.id
            )

    async def _simulate(
        self,
        node: ThoughtNode,
        agent_state: AgentState,
        depth: int = 3
    ) -> float:
        """
        Simulate reasoning path from node to estimate value.
        
        Uses fast rollout policy with lightweight LLM calls to estimate path promise.
        """
        current = node
        value = current.confidence
        
        for _ in range(depth):
            # Quick simulation using lightweight model
            response = await self._quick_simulate(
                current,
                agent_state
            )
            
            if not response["success"]:
                break
                
            value *= 0.9  # Decay factor
            value += response["value"] * 0.1  # New information value
            
            if response["terminal"]:
                break
                
        return value

    async def _evaluate(
        self,
        node: ThoughtNode,
        agent_state: AgentState
    ) -> float:
        """
        Evaluate the current node's promise using multiple criteria:
        1. Semantic coherence with reasoning goal
        2. Code executability (if code context exists)
        3. Validation against known constraints
        """
        eval_scores = []
        
        # Semantic evaluation
        semantic_score = await self._evaluate_semantics(node, agent_state)
        eval_scores.append(semantic_score)
        
        # Code evaluation if code context exists
        if "code_context" in node.metadata:
            code_score = await self._evaluate_code(node, agent_state)
            eval_scores.append(code_score)
        
        # Constraint validation
        validation_score = await self._evaluate_constraints(node, agent_state)
        eval_scores.append(validation_score)
        
        # Weighted average of scores
        weights = [0.4, 0.4, 0.2]  # Adjust based on reasoning type
        return sum(s * w for s, w in zip(eval_scores, weights))

    async def _verify_transition(
        self,
        node1: ThoughtNode,
        node2: ThoughtNode,
        agent_state: AgentState
    ) -> float:
        """
        Verify logical connection between two thoughts using Chain-of-Code.
        
        This implements a robust verification that considers both semantic
        and code-level transitions between thoughts.
        """
        # Semantic verification
        semantic_score = await self._verify_semantic_link(node1, node2, agent_state)
        
        # Code verification if code context exists
        code_score = 1.0
        if "code_context" in node1.metadata and "code_context" in node2.metadata:
            code_score = await self._verify_code_transition(node1, node2, agent_state)
        
        # Validation steps verification
        validation_score = await self._verify_validation_steps(node1, node2, agent_state)
        
        # Weighted verification score
        weights = {
            ReasoningType.MCTS: (0.4, 0.4, 0.2),
            ReasoningType.VERIFICATION: (0.3, 0.5, 0.2),
            ReasoningType.CONSENSUS: (0.5, 0.3, 0.2)
        }
        
        w1, w2, w3 = weights.get(node1.reasoning_type, (0.4, 0.4, 0.2))
        return w1 * semantic_score + w2 * code_score + w3 * validation_score

    async def build_consensus(
        self,
        paths: List[ReasoningPath],
        agent_states: List[AgentState],
    ) -> ReasoningPath:
        """
        Build consensus among multiple reasoning paths using Mixture of Agents.
        
        This enhanced implementation:
        1. Uses multiple models to evaluate each path
        2. Implements weighted voting based on model confidence
        3. Synthesizes new paths when consensus is low
        """
        if not paths:
            raise ValueError("Cannot build consensus with empty paths")
            
        # Multi-model evaluation of each path
        path_evaluations = []
        for path in paths:
            model_scores = []
            for agent in agent_states:
                score = await self._evaluate_path_with_model(path, agent)
                model_scores.append((score, agent.model_provider))
            path_evaluations.append((path, model_scores))
        
        # Calculate consensus weights
        weighted_paths = []
        for path, scores in path_evaluations:
            # Base weight from path confidence
            weight = path.total_confidence
            
            # Adjust by model diversity bonus
            unique_models = len(set(model for _, model in scores))
            diversity_bonus = unique_models / len(agent_states)
            weight *= (1 + 0.2 * diversity_bonus)
            
            # Adjust by verification status
            if path.verified:
                weight *= 1.5
                
            weighted_paths.append((weight, path))
            
        # Sort by weight
        weighted_paths.sort(reverse=True, key=lambda x: x[0])
        
        # Check consensus strength
        top_weight = weighted_paths[0][0]
        runner_up_weight = weighted_paths[1][0] if len(weighted_paths) > 1 else 0
        consensus_strength = (top_weight - runner_up_weight) / top_weight
        
        if consensus_strength < 0.3 and len(agent_states) >= 2:
            # Weak consensus - synthesize new path
            return await self._synthesize_consensus_path(
                weighted_paths[:3],
                agent_states
            )
        
        # Strong consensus - merge top paths with weights
        return await self._merge_consensus_paths(
            weighted_paths[:3],
            agent_states
        )

    def _backpropagate(self, path: List[ThoughtNode], value: float) -> None:
        """Backpropagate simulation results through the path."""
        for node in reversed(path):
            stats = self.stats[node.id]
            stats.visits += 1
            stats.total_value += value

    def _extract_best_path(self, root_node: ThoughtNode) -> List[ThoughtNode]:
        """Extract the most promising path from the MCTS tree."""
        path = [root_node]
        current = root_node
        
        while current.id in self._reasoning_cache:
            children = self._reasoning_cache[current.id]
            if not children:
                break
                
            # Select child with most visits
            current = max(
                children,
                key=lambda c: self.stats[c.id].visits
            )
            path.append(current)
            
            if len(path) >= self.max_depth:
                break
                
        return path

    def _calculate_path_confidence(self, path: List[ThoughtNode]) -> float:
        """Calculate overall confidence of a reasoning path."""
        if not path:
            return 0.0
            
        confidences = [node.confidence for node in path]
        return sum(confidences) / len(confidences)

    # Helper methods for LLM integration
    
    async def _build_context(self, node: ThoughtNode) -> Dict[str, Any]:
        """Build context for LLM from current node and its ancestors."""
        context = {
            "current_thought": node.content,
            "reasoning_type": node.reasoning_type.name,
            "confidence": node.confidence,
            "metadata": node.metadata
        }
        
        # Add ancestor context
        ancestors = []
        current = node
        while current.parent_id and len(ancestors) < 3:  # Limit context window
            if current.parent_id in self._reasoning_cache:
                parent = next(
                    (n for n in self._reasoning_cache[current.parent_id] 
                     if n.id == current.parent_id),
                    None
                )
                if parent:
                    ancestors.append(parent.content)
                    current = parent
        
        context["ancestor_thoughts"] = ancestors
        return context

    async def _generate_thought(
        self,
        context: Dict[str, Any],
        agent_state: AgentState
    ) -> Dict[str, Any]:
        """Generate next thought using agent's LLM."""
        # Prepare prompt
        prompt = self._build_thought_prompt(context)
        
        try:
            # Call LLM with appropriate provider
            if agent_state.model_provider == ModelProvider.GROQ:
                response = await self._call_groq(prompt, agent_state)
            elif agent_state.model_provider == ModelProvider.OPENAI:
                response = await self._call_openai(prompt, agent_state)
            elif agent_state.model_provider == ModelProvider.ANTHROPIC:
                response = await self._call_anthropic(prompt, agent_state)
            else:
                raise ValueError(f"Unsupported model provider: {agent_state.model_provider}")
            
            return {
                "thought": response["content"],
                "confidence": response["confidence"],
                "code_context": response.get("code_context"),
                "validation_steps": response.get("validation_steps", [])
            }
            
        except Exception as e:
            raise ValueError(f"Error generating thought: {str(e)}")

    async def _quick_simulate(
        self,
        node: ThoughtNode,
        agent_state: AgentState
    ) -> Dict[str, Any]:
        """Quick simulation using lightweight model configuration."""
        # Use faster model settings
        quick_config = agent_state.model_config.copy()
        quick_config.update({
            "temperature": 0.9,  # Increase exploration
            "max_tokens": 100,   # Shorter responses
            "top_p": 0.7        # More focused sampling
        })
        
        context = self._build_context(node)
        prompt = self._build_simulation_prompt(context)
        
        try:
            # Use fastest available model
            response = await self._call_groq(prompt, agent_state, quick_config)
            
            return {
                "success": True,
                "value": response["confidence"],
                "terminal": response.get("terminal", False)
            }
        except Exception:
            return {"success": False, "value": 0.0, "terminal": True}

    # Evaluation helper methods
    
    async def _evaluate_semantics(
        self,
        node: ThoughtNode,
        agent_state: AgentState
    ) -> float:
        """Evaluate semantic coherence of thought."""
        context = self._build_context(node)
        prompt = self._build_semantic_eval_prompt(context)
        
        try:
            response = await self._call_model(prompt, agent_state)
            return float(response["score"])
        except Exception:
            return 0.5  # Neutral score on failure

    async def _evaluate_code(
        self,
        node: ThoughtNode,
        agent_state: AgentState
    ) -> float:
        """Evaluate code context for executability and correctness."""
        if "code_context" not in node.metadata:
            return 1.0
            
        code_context = node.metadata["code_context"]
        prompt = self._build_code_eval_prompt(code_context)
        
        try:
            response = await self._call_model(prompt, agent_state)
            return float(response["score"])
        except Exception:
            return 0.5  # Neutral score on failure

    async def _evaluate_constraints(
        self,
        node: ThoughtNode,
        agent_state: AgentState
    ) -> float:
        """Evaluate adherence to known constraints."""
        context = self._build_context(node)
        prompt = self._build_constraint_eval_prompt(context)
        
        try:
            response = await self._call_model(prompt, agent_state)
            return float(response["score"])
        except Exception:
            return 0.5  # Neutral score on failure

    async def _verify_semantic_link(
        self,
        node1: ThoughtNode,
        node2: ThoughtNode,
        agent_state: AgentState
    ) -> float:
        """Verify semantic connection between thoughts."""
        context = {
            "thought1": node1.content,
            "thought2": node2.content,
            "metadata1": node1.metadata,
            "metadata2": node2.metadata
        }
        prompt = self._build_semantic_verify_prompt(context)
        
        try:
            response = await self._call_model(prompt, agent_state)
            return float(response["score"])
        except Exception:
            return 0.5  # Neutral score on failure

    async def _verify_code_transition(
        self,
        node1: ThoughtNode,
        node2: ThoughtNode,
        agent_state: AgentState
    ) -> float:
        """Verify code-level transition between thoughts."""
        context = {
            "code1": node1.metadata["code_context"],
            "code2": node2.metadata["code_context"]
        }
        prompt = self._build_code_verify_prompt(context)
        
        try:
            response = await self._call_model(prompt, agent_state)
            return float(response["score"])
        except Exception:
            return 0.5  # Neutral score on failure

    async def _verify_validation_steps(
        self,
        node1: ThoughtNode,
        node2: ThoughtNode,
        agent_state: AgentState
    ) -> float:
        """Verify validation steps between thoughts."""
        steps1 = node1.metadata.get("validation_steps", [])
        steps2 = node2.metadata.get("validation_steps", [])
        
        if not steps1 or not steps2:
            return 1.0
            
        context = {
            "steps1": steps1,
            "steps2": steps2
        }
        prompt = self._build_validation_verify_prompt(context)
        
        try:
            response = await self._call_model(prompt, agent_state)
            return float(response["score"])
        except Exception:
            return 0.5  # Neutral score on failure

    async def _evaluate_path_with_model(
        self,
        path: ReasoningPath,
        agent_state: AgentState
    ) -> float:
        """Evaluate complete reasoning path with a specific model."""
        context = {
            "nodes": [{"content": n.content, "metadata": n.metadata}
                     for n in path.nodes],
            "total_confidence": path.total_confidence,
            "verified": path.verified
        }
        prompt = self._build_path_eval_prompt(context)
        
        try:
            response = await self._call_model(prompt, agent_state)
            return float(response["score"])
        except Exception:
            return 0.5  # Neutral score on failure

    async def _synthesize_consensus_path(
        self,
        weighted_paths: List[Tuple[float, ReasoningPath]],
        agent_states: List[AgentState]
    ) -> ReasoningPath:
        """Synthesize new consensus path when agreement is low."""
        # Extract key elements from top paths
        path_elements = []
        for _, path in weighted_paths:
            for node in path.nodes:
                path_elements.append({
                    "content": node.content,
                    "metadata": node.metadata
                })
        
        # Use strongest model to synthesize
        primary_agent = max(
            agent_states,
            key=lambda a: len(a.active_reasoning_paths)
        )
        
        context = {
            "path_elements": path_elements,
            "num_agents": len(agent_states)
        }
        prompt = self._build_synthesis_prompt(context)
        
        try:
            response = await self._call_model(prompt, primary_agent)
            
            # Create new consensus path
            nodes = []
            for thought in response["thoughts"]:
                node = ThoughtNode(
                    content=thought["content"],
                    confidence=thought["confidence"],
                    reasoning_type=ReasoningType.CONSENSUS,
                    metadata=thought.get("metadata", {})
                )
                nodes.append(node)
            
            return ReasoningPath(
                nodes=nodes,
                total_confidence=response["total_confidence"],
                consensus_reached=True
            )
            
        except Exception as e:
            # Fallback to highest weighted path
            return weighted_paths[0][1]

    async def _merge_consensus_paths(
        self,
        weighted_paths: List[Tuple[float, ReasoningPath]],
        agent_states: List[AgentState]
    ) -> ReasoningPath:
        """Merge top paths when strong consensus exists."""
        if not weighted_paths:
            raise ValueError("No paths to merge")
            
        # Normalize weights
        total_weight = sum(w for w, _ in weighted_paths)
        weights = [w / total_weight for w, _ in weighted_paths]
        
        # Get max length
        max_length = max(len(p[1].nodes) for p in weighted_paths)
        
        # Merge nodes
        merged_nodes = []
        for i in range(max_length):
            node_contents = []
            node_confidences = []
            node_metadata = []
            
            for (weight, path), path_weight in zip(weighted_paths, weights):
                if i < len(path.nodes):
                    node = path.nodes[i]
                    node_contents.append(node.content)
                    node_confidences.append(node.confidence * path_weight)
                    node_metadata.append(node.metadata)
            
            if node_contents:
                merged_node = ThoughtNode(
                    content=" | ".join(node_contents),
                    confidence=sum(node_confidences),
                    reasoning_type=ReasoningType.CONSENSUS,
                    metadata=self._merge_metadata(node_metadata)
                )
                merged_nodes.append(merged_node)
        
        return ReasoningPath(
            nodes=merged_nodes,
            total_confidence=sum(w * p.total_confidence 
                               for (w, p), weight in zip(weighted_paths, weights)),
            consensus_reached=True
        )

    def _merge_metadata(self, metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge metadata from multiple nodes."""
        if not metadata_list:
            return {}
            
        merged = {}
        for key in set().union(*metadata_list):
            values = [m[key] for m in metadata_list if key in m]
            if all(isinstance(v, (int, float)) for v in values):
                merged[key] = sum(values) / len(values)
            elif all(isinstance(v, list) for v in values):
                merged[key] = list(set().union(*values))
            else:
                merged[key] = values[0]  # Take first non-numeric value
                
        return merged

    # Prompt building methods
    
    def _build_thought_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for thought generation."""
        return f"""Given the current reasoning context:
Current thought: {context['current_thought']}
Reasoning type: {context['reasoning_type']}
Previous thoughts: {' -> '.join(context['ancestor_thoughts'])}

Generate the next logical thought in this reasoning chain. Consider:
1. Semantic coherence with previous thoughts
2. Logical progression towards a solution
3. Any relevant code implications

Your response should include:
- A clear, concise thought
- A confidence score (0-1)
- Any relevant code context
- Validation steps for verification

Response format:
{{
    "content": "your thought here",
    "confidence": 0.0-1.0,
    "code_context": "optional code snippet",
    "validation_steps": ["step1", "step2", ...]
}}"""

    def _build_simulation_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for quick simulation."""
        return f"""Given the current thought:
{context['current_thought']}

Quickly assess:
1. Is this a promising direction? (confidence)
2. Is this a terminal state? (terminal)

Response format:
{{
    "confidence": 0.0-1.0,
    "terminal": true/false
}}"""

    def _build_semantic_eval_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for semantic evaluation."""
        return f"""Evaluate the semantic coherence of this thought:
{context['current_thought']}

Consider:
1. Clarity and specificity
2. Logical connection to context
3. Progress towards goal

Response format:
{{
    "score": 0.0-1.0,
    "explanation": "brief explanation"
}}"""

    def _build_code_eval_prompt(self, code_context: str) -> str:
        """Build prompt for code evaluation."""
        return f"""Evaluate this code context:
{code_context}

Consider:
1. Syntactic correctness
2. Best practices
3. Potential issues

Response format:
{{
    "score": 0.0-1.0,
    "issues": ["issue1", "issue2", ...]
}}"""

    def _build_constraint_eval_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for constraint evaluation."""
        return f"""Evaluate adherence to constraints:
Thought: {context['current_thought']}
Type: {context['reasoning_type']}

Consider:
1. Type-specific constraints
2. System limitations
3. Resource usage

Response format:
{{
    "score": 0.0-1.0,
    "violations": ["violation1", "violation2", ...]
}}"""

    def _build_semantic_verify_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for semantic verification."""
        return f"""Verify the logical connection between these thoughts:
Thought 1: {context['thought1']}
Thought 2: {context['thought2']}

Consider:
1. Logical progression
2. Information preservation
3. Semantic relevance

Response format:
{{
    "score": 0.0-1.0,
    "explanation": "verification explanation"
}}"""

    def _build_code_verify_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for code verification."""
        return f"""Verify the code transition:
Code 1:
{context['code1']}

Code 2:
{context['code2']}

Consider:
1. Compatibility
2. Dependency preservation
3. State consistency

Response format:
{{
    "score": 0.0-1.0,
    "issues": ["issue1", "issue2", ...]
}}"""

    def _build_validation_verify_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for validation step verification."""
        return f"""Verify validation steps consistency:
Steps 1: {context['steps1']}
Steps 2: {context['steps2']}

Consider:
1. Step coverage
2. Logical progression
3. Completeness

Response format:
{{
    "score": 0.0-1.0,
    "missing_validations": ["validation1", "validation2", ...]
}}"""

    def _build_path_eval_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for complete path evaluation."""
        return f"""Evaluate this complete reasoning path:
{json.dumps(context['nodes'], indent=2)}

Consider:
1. Overall coherence
2. Goal achievement
3. Efficiency

Response format:
{{
    "score": 0.0-1.0,
    "strengths": ["strength1", "strength2", ...],
    "weaknesses": ["weakness1", "weakness2", ...]
}}"""

    def _build_synthesis_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for path synthesis."""
        return f"""Synthesize a new consensus path from these elements:
{json.dumps(context['path_elements'], indent=2)}

Number of agents: {context['num_agents']}

Create a new path that:
1. Incorporates key insights
2. Resolves conflicts
3. Maintains coherence

Response format:
{{
    "thoughts": [
        {{
            "content": "thought1",
            "confidence": 0.0-1.0,
            "metadata": {{...}}
        }},
        ...
    ],
    "total_confidence": 0.0-1.0
}}"""

    # Model calling methods
    
    async def _call_model(
        self,
        prompt: str,
        agent_state: AgentState,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Call appropriate model based on agent state."""
        if agent_state.model_provider == ModelProvider.GROQ:
            return await self._call_groq(prompt, agent_state, config)
        elif agent_state.model_provider == ModelProvider.OPENAI:
            return await self._call_openai(prompt, agent_state, config)
        elif agent_state.model_provider == ModelProvider.ANTHROPIC:
            return await self._call_anthropic(prompt, agent_state, config)
        else:
            raise ValueError(f"Unsupported model provider: {agent_state.model_provider}")

    async def _call_groq(
        self,
        prompt: str,
        agent_state: AgentState,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Call Groq API."""
        from groq import AsyncGroq
        
        try:
            client = AsyncGroq()
            response = await client.chat.completions.create(
                model=agent_state.model_config.get("model_name", "llama-3.1-70b-versatile"),
                messages=[{"role": "user", "content": prompt}],
                **(config or agent_state.model_config)
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            raise ValueError(f"Groq API error: {str(e)}")

    async def _call_openai(
        self,
        prompt: str,
        agent_state: AgentState,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Call OpenAI API."""
        from openai import AsyncOpenAI
        
        try:
            client = AsyncOpenAI()
            response = await client.chat.completions.create(
                model=agent_state.model_config.get("model_name", "gpt-4o"),
                messages=[{"role": "user", "content": prompt}],
                **(config or agent_state.model_config)
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            raise ValueError(f"OpenAI API error: {str(e)}")

    async def _call_anthropic(
        self,
        prompt: str,
        agent_state: AgentState,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Call Anthropic API."""
        from anthropic import AsyncAnthropic
        
        try:
            client = AsyncAnthropic()
            response = await client.messages.create(
                model=agent_state.model_config.get("model_name", "claude-3-5-sonnet-latest"),
                messages=[{"role": "user", "content": prompt}],
                **(config or agent_state.model_config)
            )
            
            return json.loads(response.content[0].text)
            
        except Exception as e:
            raise ValueError(f"Anthropic API error: {str(e)}")
