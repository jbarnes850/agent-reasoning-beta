"""Tests for visualization components."""

import pytest
from unittest.mock import Mock, patch
import networkx as nx
from src.visualization.components.shared import graphs, trees, metrics, heatmaps
from src.visualization.layouts import main_layout, dashboard_layout

@pytest.fixture
def mock_graph_data():
    """Mock graph data for testing."""
    G = nx.DiGraph()
    G.add_nodes_from([
        (1, {"label": "Root", "type": "decision"}),
        (2, {"label": "Child1", "type": "reasoning"}),
        (3, {"label": "Child2", "type": "evidence"})
    ])
    G.add_edges_from([(1, 2), (1, 3)])
    return G

def test_tree_layout(mock_graph_data):
    """Test tree layout generation."""
    layout = trees.generate_tree_layout(mock_graph_data)
    assert len(layout) == 3  # One position for each node
    
    # Test node positioning
    root_pos = layout[1]
    child1_pos = layout[2]
    child2_pos = layout[3]
    
    # Root should be above children
    assert root_pos[1] > child1_pos[1]
    assert root_pos[1] > child2_pos[1]

def test_graph_rendering(mock_graph_data):
    """Test graph rendering functionality."""
    renderer = graphs.GraphRenderer()
    
    # Test node styling
    styles = renderer.get_node_styles(mock_graph_data)
    assert "decision" in styles
    assert "reasoning" in styles
    assert "evidence" in styles
    
    # Test edge styling
    edge_styles = renderer.get_edge_styles(mock_graph_data)
    assert len(edge_styles) == 2  # Two edges

def test_metrics_calculation():
    """Test metrics calculation and visualization."""
    data = {
        "confidence_scores": [0.8, 0.9, 0.7],
        "response_times": [1.2, 1.0, 1.5],
        "success_rates": [0.85, 0.90, 0.80]
    }
    
    # Test metric aggregation
    aggregated = metrics.aggregate_metrics(data)
    assert "avg_confidence" in aggregated
    assert "avg_response_time" in aggregated
    assert "overall_success_rate" in aggregated
    
    # Test visualization generation
    viz = metrics.generate_metrics_visualization(data)
    assert viz is not None

def test_heatmap_generation():
    """Test heatmap generation for interaction patterns."""
    interaction_data = [
        {"source": "Agent1", "target": "Agent2", "weight": 0.8},
        {"source": "Agent2", "target": "Agent3", "weight": 0.6},
        {"source": "Agent1", "target": "Agent3", "weight": 0.4}
    ]
    
    heatmap = heatmaps.generate_interaction_heatmap(interaction_data)
    assert heatmap is not None
    assert len(heatmap.data) == 3  # Three interactions

def test_dashboard_layout():
    """Test dashboard layout configuration."""
    config = {
        "metrics_position": "top",
        "graph_position": "center",
        "controls_position": "sidebar"
    }
    
    layout = dashboard_layout.DashboardLayout(config)
    assert layout.metrics_position == "top"
    assert layout.graph_position == "center"
    assert layout.controls_position == "sidebar"
    
    # Test layout rendering
    rendered = layout.render()
    assert rendered is not None

def test_main_layout_responsiveness():
    """Test main layout responsiveness."""
    layout = main_layout.MainLayout()
    
    # Test desktop layout
    desktop_layout = layout.get_layout("desktop")
    assert desktop_layout["sidebar_width"] > 200
    
    # Test mobile layout
    mobile_layout = layout.get_layout("mobile")
    assert mobile_layout["sidebar_width"] <= 200
    
    # Test layout adaptation
    assert layout.is_responsive
