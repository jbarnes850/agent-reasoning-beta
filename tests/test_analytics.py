"""Tests for analytics functionality."""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from src.core.metrics import MetricsCollector
from src.ui.pages.Analytics import AnalyticsPage
from src.core.types import ExperimentResult, AgentRole

@pytest.fixture
def mock_experiment_data():
    """Mock experiment data for testing."""
    return [
        ExperimentResult(
            id="exp1",
            name="Test Experiment 1",
            start_time="2024-03-17T10:00:00",
            end_time="2024-03-17T10:01:00",
            agent_roles=[AgentRole.RESEARCHER, AgentRole.CRITIC],
            success=True,
            metrics={
                "confidence": 0.85,
                "response_time": 1.2,
                "token_usage": 1500
            }
        ),
        ExperimentResult(
            id="exp2",
            name="Test Experiment 2",
            start_time="2024-03-17T10:02:00",
            end_time="2024-03-17T10:03:00",
            agent_roles=[AgentRole.RESEARCHER],
            success=False,
            metrics={
                "confidence": 0.65,
                "response_time": 1.5,
                "token_usage": 1200
            }
        )
    ]

def test_metrics_collection(mock_experiment_data):
    """Test metrics collection and aggregation."""
    collector = MetricsCollector()
    
    # Test adding experiment results
    for result in mock_experiment_data:
        collector.add_result(result)
    
    # Test basic metrics
    metrics = collector.get_metrics()
    assert metrics["total_experiments"] == 2
    assert metrics["success_rate"] == 0.5  # 1 out of 2 successful
    
    # Test average metrics
    assert 0.7 < metrics["avg_confidence"] < 0.8  # (0.85 + 0.65) / 2
    assert 1.3 < metrics["avg_response_time"] < 1.4  # (1.2 + 1.5) / 2

def test_time_series_analysis(mock_experiment_data):
    """Test time series analysis of metrics."""
    collector = MetricsCollector()
    for result in mock_experiment_data:
        collector.add_result(result)
    
    # Test time series data
    time_series = collector.get_time_series()
    assert len(time_series) == 2  # Two data points
    assert "timestamp" in time_series.columns
    assert "confidence" in time_series.columns
    assert "response_time" in time_series.columns

def test_performance_comparison():
    """Test performance comparison between agent configurations."""
    data = {
        "single_agent": {
            "success_rate": 0.75,
            "avg_confidence": 0.8,
            "avg_response_time": 1.2
        },
        "multi_agent": {
            "success_rate": 0.85,
            "avg_confidence": 0.9,
            "avg_response_time": 1.5
        }
    }
    
    analytics = AnalyticsPage()
    comparison = analytics.compare_configurations(data)
    
    assert comparison["multi_agent"]["success_rate"] > comparison["single_agent"]["success_rate"]
    assert comparison["multi_agent"]["avg_confidence"] > comparison["single_agent"]["avg_confidence"]

def test_resource_utilization(mock_experiment_data):
    """Test resource utilization analysis."""
    collector = MetricsCollector()
    for result in mock_experiment_data:
        collector.add_result(result)
    
    # Test token usage analysis
    token_stats = collector.analyze_token_usage()
    assert token_stats["total_tokens"] == 2700  # 1500 + 1200
    assert token_stats["avg_tokens_per_experiment"] == 1350  # 2700 / 2

def test_analytics_visualization():
    """Test analytics visualization generation."""
    data = pd.DataFrame({
        "timestamp": pd.date_range(start="2024-03-17", periods=5),
        "confidence": np.random.uniform(0.6, 0.9, 5),
        "response_time": np.random.uniform(1.0, 2.0, 5),
        "success": [True, True, False, True, True]
    })
    
    analytics = AnalyticsPage()
    
    # Test confidence trend
    confidence_plot = analytics.plot_confidence_trend(data)
    assert confidence_plot is not None
    
    # Test response time distribution
    response_plot = analytics.plot_response_distribution(data)
    assert response_plot is not None
    
    # Test success rate over time
    success_plot = analytics.plot_success_rate(data)
    assert success_plot is not None

def test_export_functionality(mock_experiment_data):
    """Test data export functionality."""
    collector = MetricsCollector()
    for result in mock_experiment_data:
        collector.add_result(result)
    
    # Test CSV export
    csv_data = collector.export_to_csv()
    assert isinstance(csv_data, str)
    assert "id,name,start_time" in csv_data
    
    # Test JSON export
    json_data = collector.export_to_json()
    assert isinstance(json_data, str)
    assert "exp1" in json_data
    assert "exp2" in json_data
