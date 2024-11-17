"""Tests for visualization components."""

from unittest.mock import Mock, patch

import networkx as nx
import plotly.graph_objects as go
import pytest

from core.types import ConsensusMetrics, ThoughtNode
from visualization.components.views.consensus_view import ConsensusVisualizer
from visualization.components.views.exploration_view import MCTSVisualizer


@pytest.fixture
def mock_graph_data():
    """Mock graph data for testing."""
    G = nx.DiGraph()
    G.add_nodes_from([1, 2, 3])
    G.add_edges_from([(1, 2), (2, 3)])
    return G


@pytest.fixture
def mock_consensus_metrics():
    """Mock consensus metrics for testing."""
    return ConsensusMetrics(
        agreement_matrix=[[1.0, 0.8], [0.8, 1.0]],
        agent_ids=["agent1", "agent2"],
        resolution_path=[("agent1", 0.9), ("agent2", 0.85)],
    )


@pytest.fixture
def mock_thought_tree():
    """Mock thought tree for testing."""
    root_node = ThoughtNode(
        id="root", content="Root thought", confidence=0.9, children=[]
    )
    return root_node


def test_consensus_visualization(mock_graph_data, mock_consensus_metrics):
    """Test consensus visualization component."""
    visualizer = ConsensusVisualizer()

    # Test visualization creation
    container = Mock()
    visualizer.visualize_consensus(mock_consensus_metrics, container)
    container.plotly_chart.assert_called()


def test_mcts_visualization(mock_graph_data, mock_thought_tree):
    """Test MCTS visualization component."""
    visualizer = MCTSVisualizer()

    # Test visualization creation
    container = Mock()
    visualizer.visualize_mcts(mock_thought_tree, container)
    container.plotly_chart.assert_called()
