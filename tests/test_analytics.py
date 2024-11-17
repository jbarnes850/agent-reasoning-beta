"""Tests for analytics functionality."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from agent_reasoning_beta.core.metrics import MetricsCollector
from agent_reasoning_beta.core.types import AgentRole, AgentVote, ConsensusMetrics


@pytest.fixture
def mock_experiment_data():
    """Mock experiment data for testing."""
    return [
        ConsensusMetrics(
            agreement_matrix=[[1.0, 0.8], [0.8, 1.0]],
            agent_ids=["agent1", "agent2"],
            resolution_path=[("decision1", 0.9), ("decision2", 0.85)],
        ),
        ConsensusMetrics(
            agreement_matrix=[[0.9, 0.7], [0.7, 0.9]],
            agent_ids=["agent3", "agent4"],
            resolution_path=[("decision3", 0.8), ("decision4", 0.75)],
        ),
    ]


@pytest.fixture
def mock_votes():
    """Mock agent votes for testing."""
    return [
        AgentVote(
            agent_id="agent1",
            choice="option1",
            confidence=0.9,
            reasoning="This is the best option because...",
        ),
        AgentVote(
            agent_id="agent2",
            choice="option2",
            confidence=0.85,
            reasoning="I think this alternative is better...",
        ),
    ]


@pytest.mark.asyncio
async def test_metrics_collection(mock_experiment_data):
    """Test metrics collection and aggregation."""
    collector = MetricsCollector()

    for exp in mock_experiment_data:
        collector.add_metrics(exp)

    aggregated = collector.get_aggregated_metrics()
    assert len(aggregated) > 0
    assert "agreement_score" in aggregated
    assert 0 <= aggregated["agreement_score"] <= 1


def test_consensus_analysis(mock_votes):
    """Test consensus analysis functionality."""
    collector = MetricsCollector()
    analysis = collector.analyze_consensus(mock_votes)

    assert "consensus_reached" in analysis
    assert "confidence_level" in analysis
    assert isinstance(analysis["consensus_reached"], bool)
    assert 0 <= analysis["confidence_level"] <= 1


def test_time_series_analysis(mock_experiment_data):
    """Test time series analysis of metrics."""
    collector = MetricsCollector()
    for exp in mock_experiment_data:
        collector.add_metrics(exp)

    time_series = collector.get_time_series_data("agreement_score")
    assert len(time_series) == 2
    assert time_series[0]["value"] > 0
    assert time_series[1]["value"] > 0


def test_performance_comparison():
    """Test performance comparison between agent configurations."""
    collector = MetricsCollector()

    # Add mock data for different configurations
    config1_data = ConsensusMetrics(
        agreement_matrix=[[1.0, 0.8], [0.8, 1.0]],
        agent_ids=["agent1", "agent2"],
        resolution_path=[("decision1", 0.9), ("decision2", 0.85)],
    )

    config2_data = ConsensusMetrics(
        agreement_matrix=[[0.9, 0.7], [0.7, 0.9]],
        agent_ids=["agent3", "agent4"],
        resolution_path=[("decision3", 0.8), ("decision4", 0.75)],
    )

    collector.add_metrics(config1_data)
    collector.add_metrics(config2_data)

    comparison = collector.compare_configurations()
    assert len(comparison) == 2
    assert comparison[0]["config"] != comparison[1]["config"]


def test_resource_utilization(mock_experiment_data):
    """Test resource utilization analysis."""
    collector = MetricsCollector()
    for exp in mock_experiment_data:
        collector.add_metrics(exp)

    utilization = collector.get_resource_utilization()
    assert "memory_usage" in utilization
    assert "cpu_usage" in utilization
    assert len(utilization["memory_usage"]) == 2


def test_analytics_visualization():
    """Test analytics visualization generation."""
    page = AnalyticsPage()

    # Mock data
    data = {
        "agreement_score": [0.8, 0.7, 0.6],
        "confidence": [0.9, 0.8, 0.7],
        "consensus_reached": [True, False, True],
    }

    # Test different visualization types
    time_series = page.create_time_series_plot(data["agreement_score"])
    assert time_series is not None

    histogram = page.create_histogram(data["confidence"])
    assert histogram is not None

    heatmap = page.create_correlation_heatmap(data)
    assert heatmap is not None

    scatter = page.create_scatter_plot(data["agreement_score"], data["confidence"])
    assert scatter is not None


def test_export_functionality(mock_experiment_data):
    """Test data export functionality."""
    collector = MetricsCollector()
    for exp in mock_experiment_data:
        collector.add_metrics(exp)

    # Test CSV export
    csv_data = collector.export_to_csv()
    assert isinstance(csv_data, str)
    assert "agent1" in csv_data
    assert "agent2" in csv_data

    # Test JSON export
    json_data = collector.export_to_json()
    assert isinstance(json_data, dict)
    assert len(json_data["experiments"]) == 2


def test_metrics_collection():
    """Test metrics collection functionality."""
    collector = MetricsCollector()

    # Add test metrics
    collector.add_metric("agreement_score", 0.8)
    collector.add_metric("confidence", 0.9)
    collector.add_metric("consensus_reached", True)

    # Test metrics retrieval
    metrics = collector.get_metrics()
    assert metrics["agreement_score"] == 0.8
    assert metrics["confidence"] == 0.9
    assert metrics["consensus_reached"] is True

    # Test metrics aggregation
    collector.add_metric("agreement_score", 0.7)
    collector.add_metric("agreement_score", 0.6)

    stats = collector.get_statistics("agreement_score")
    assert stats["mean"] == pytest.approx(0.7, rel=1e-3)
    assert stats["std"] > 0


def test_metrics_analysis():
    """Test metrics analysis functionality."""
    collector = MetricsCollector()

    # Add test data
    for _ in range(10):
        collector.add_metric("confidence", np.random.random())
        collector.add_metric("agreement_score", np.random.random() * 2)
        collector.add_metric("consensus_reached", bool(np.random.randint(0, 2)))

    # Test analysis functions
    analysis = collector.analyze_metrics()
    assert "confidence" in analysis
    assert "agreement_score" in analysis
    assert "consensus_reached" in analysis

    # Test correlation analysis
    corr = collector.get_correlations()
    assert isinstance(corr, pd.DataFrame)
    assert corr.shape == (3, 3)  # 3x3 correlation matrix
