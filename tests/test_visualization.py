"""
Unit tests for visualization functions.
"""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from calibration_toolbox import visualization


class TestReliabilityDiagram:
    """Tests for reliability diagram visualization."""
    
    def test_basic_plot(self):
        """Test basic reliability diagram creation."""
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(2), size=50)
        labels = np.random.randint(0, 2, size=50)
        
        fig, ax = visualization.reliability_diagram(probs, labels, return_fig=True)
        assert fig is not None
        assert ax is not None
        plt.close(fig)
    
    def test_custom_bins(self):
        """Test reliability diagram with custom number of bins."""
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(2), size=50)
        labels = np.random.randint(0, 2, size=50)
        
        fig, ax = visualization.reliability_diagram(probs, labels, n_bins=10, return_fig=True)
        assert fig is not None
        plt.close(fig)
    
    def test_with_title(self):
        """Test reliability diagram with custom title."""
        probs = np.array([[0.8, 0.2], [0.6, 0.4], [0.9, 0.1]])
        labels = np.array([0, 1, 0])
        
        fig, ax = visualization.reliability_diagram(
            probs, labels, title="Custom Title", return_fig=True
        )
        assert ax.get_title() == "Custom Title"
        plt.close(fig)
    
    def test_logits_input(self):
        """Test reliability diagram with logits."""
        logits = np.array([[2.0, -1.0], [1.0, 0.5], [-0.5, 1.5]])
        labels = np.array([0, 0, 1])
        
        fig, ax = visualization.reliability_diagram(logits, labels, logits=True, return_fig=True)
        assert fig is not None
        plt.close(fig)
    
    def test_custom_figsize(self):
        """Test reliability diagram with custom figure size."""
        probs = np.array([[0.8, 0.2], [0.6, 0.4]])
        labels = np.array([0, 1])
        
        fig, ax = visualization.reliability_diagram(
            probs, labels, figsize=(8, 8), return_fig=True
        )
        assert fig.get_size_inches()[0] == 8
        assert fig.get_size_inches()[1] == 8
        plt.close(fig)


class TestConfidenceHistogram:
    """Tests for confidence histogram visualization."""
    
    def test_basic_plot(self):
        """Test basic confidence histogram creation."""
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(2), size=50)
        labels = np.random.randint(0, 2, size=50)
        
        fig, ax = visualization.confidence_histogram(probs, labels, return_fig=True)
        assert fig is not None
        assert ax is not None
        plt.close(fig)
    
    def test_custom_bins(self):
        """Test confidence histogram with custom number of bins."""
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(2), size=50)
        labels = np.random.randint(0, 2, size=50)
        
        fig, ax = visualization.confidence_histogram(probs, labels, n_bins=20, return_fig=True)
        assert fig is not None
        plt.close(fig)
    
    def test_with_title(self):
        """Test confidence histogram with custom title."""
        probs = np.array([[0.8, 0.2], [0.6, 0.4], [0.9, 0.1]])
        labels = np.array([0, 1, 0])
        
        fig, ax = visualization.confidence_histogram(
            probs, labels, title="Custom Histogram", return_fig=True
        )
        assert ax.get_title() == "Custom Histogram"
        plt.close(fig)
    
    def test_logits_input(self):
        """Test confidence histogram with logits."""
        logits = np.array([[2.0, -1.0], [1.0, 0.5], [-0.5, 1.5]])
        labels = np.array([0, 0, 1])
        
        fig, ax = visualization.confidence_histogram(logits, labels, logits=True, return_fig=True)
        assert fig is not None
        plt.close(fig)


class TestClassWiseCalibrationCurve:
    """Tests for class-wise calibration curve visualization."""
    
    def test_basic_plot(self):
        """Test basic class-wise calibration curve."""
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(3), size=50)
        labels = np.random.randint(0, 3, size=50)
        
        fig, ax = visualization.class_wise_calibration_curve(probs, labels, return_fig=True)
        assert fig is not None
        assert ax is not None
        plt.close(fig)
    
    def test_binary_classification(self):
        """Test class-wise curve with binary classification."""
        probs = np.array([[0.8, 0.2], [0.6, 0.4], [0.9, 0.1], [0.3, 0.7]])
        labels = np.array([0, 1, 0, 1])
        
        fig, ax = visualization.class_wise_calibration_curve(probs, labels, return_fig=True)
        assert fig is not None
        plt.close(fig)
    
    def test_many_classes(self):
        """Test class-wise curve with many classes."""
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(15), size=100)
        labels = np.random.randint(0, 15, size=100)
        
        # Should limit to max_classes
        fig, ax = visualization.class_wise_calibration_curve(
            probs, labels, max_classes=10, return_fig=True
        )
        assert fig is not None
        plt.close(fig)
    
    def test_custom_title(self):
        """Test class-wise curve with custom title."""
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(3), size=50)
        labels = np.random.randint(0, 3, size=50)
        
        fig, ax = visualization.class_wise_calibration_curve(
            probs, labels, title="Per-Class Calibration", return_fig=True
        )
        assert ax.get_title() == "Per-Class Calibration"
        plt.close(fig)


class TestCalibrationErrorDecomposition:
    """Tests for calibration error decomposition visualization."""
    
    def test_basic_plot(self):
        """Test basic calibration error decomposition."""
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(3), size=50)
        labels = np.random.randint(0, 3, size=50)
        
        fig, ax = visualization.calibration_error_decomposition(probs, labels, return_fig=True)
        assert fig is not None
        assert ax is not None
        plt.close(fig)
    
    def test_binary_classification(self):
        """Test decomposition with binary classification."""
        probs = np.array([[0.8, 0.2], [0.6, 0.4], [0.9, 0.1], [0.7, 0.3]])
        labels = np.array([0, 1, 0, 0])
        
        fig, ax = visualization.calibration_error_decomposition(probs, labels, return_fig=True)
        assert fig is not None
        plt.close(fig)
    
    def test_custom_bins(self):
        """Test decomposition with custom number of bins."""
        np.random.seed(42)
        probs = np.random.dirichlet(np.ones(2), size=50)
        labels = np.random.randint(0, 2, size=50)
        
        fig, ax = visualization.calibration_error_decomposition(
            probs, labels, n_bins=10, return_fig=True
        )
        assert fig is not None
        plt.close(fig)
    
    def test_logits_input(self):
        """Test decomposition with logits."""
        logits = np.array([[2.0, -1.0], [1.0, 0.5], [-0.5, 1.5], [0.0, 0.0]])
        labels = np.array([0, 0, 1, 1])
        
        fig, ax = visualization.calibration_error_decomposition(
            logits, labels, logits=True, return_fig=True
        )
        assert fig is not None
        plt.close(fig)


class TestEdgeCases:
    """Tests for edge cases in visualization."""
    
    def test_single_sample(self):
        """Test visualizations with single sample."""
        probs = np.array([[0.8, 0.2]])
        labels = np.array([0])
        
        fig1, ax1 = visualization.reliability_diagram(probs, labels, return_fig=True)
        assert fig1 is not None
        plt.close(fig1)
        
        fig2, ax2 = visualization.confidence_histogram(probs, labels, return_fig=True)
        assert fig2 is not None
        plt.close(fig2)
    
    def test_few_samples(self):
        """Test visualizations with few samples."""
        probs = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
        labels = np.array([0, 1, 0])
        
        fig1, ax1 = visualization.reliability_diagram(probs, labels, return_fig=True)
        assert fig1 is not None
        plt.close(fig1)
        
        fig2, ax2 = visualization.confidence_histogram(probs, labels, return_fig=True)
        assert fig2 is not None
        plt.close(fig2)
        
        fig3, ax3 = visualization.calibration_error_decomposition(probs, labels, return_fig=True)
        assert fig3 is not None
        plt.close(fig3)
    
    def test_perfect_predictions(self):
        """Test visualizations with perfect predictions."""
        probs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        labels = np.array([0, 1, 0])
        
        fig1, ax1 = visualization.reliability_diagram(probs, labels, return_fig=True)
        assert fig1 is not None
        plt.close(fig1)
        
        fig2, ax2 = visualization.confidence_histogram(probs, labels, return_fig=True)
        assert fig2 is not None
        plt.close(fig2)
