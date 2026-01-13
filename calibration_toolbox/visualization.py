"""
Visualization tools for calibration analysis.

This module provides plotting functions for visualizing model calibration,
including reliability diagrams and confidence histograms.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from typing import Optional, Tuple


def reliability_diagram(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    logits: bool = False,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (6, 6),
    return_fig: bool = False
):
    """
    Plot a reliability diagram (calibration curve).
    
    A reliability diagram visualizes the relationship between predicted
    confidence and actual accuracy across bins. Well-calibrated models
    should have points close to the diagonal identity line.
    
    Args:
        probabilities: Array of shape (n_samples, n_classes) containing
            predicted probabilities for each class.
        labels: Array of shape (n_samples,) containing true class labels.
        n_bins: Number of bins for grouping predictions. Default: 15.
        logits: If True, input is logits and will be converted to probabilities.
            Default: False.
        title: Plot title. If None, uses default title. Default: None.
        figsize: Figure size as (width, height). Default: (6, 6).
        return_fig: If True, return figure and axis objects. Default: False.
    
    Returns:
        If return_fig is True, returns (fig, ax) tuple. Otherwise, displays plot.
    
    Example:
        >>> probs = np.array([[0.8, 0.2], [0.6, 0.4], [0.9, 0.1]])
        >>> labels = np.array([0, 1, 0])
        >>> reliability_diagram(probs, labels)
    """
    # Convert logits to probabilities if needed
    if logits:
        probabilities = softmax(probabilities, axis=1)
    
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)
    
    # Get predictions and confidences
    predictions = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)
    accuracies = (predictions == labels).astype(float)
    
    # Compute bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bin_centers = (bin_lowers + bin_uppers) / 2
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(
            confidences > bin_lower,
            confidences <= bin_upper
        )
        
        bin_size = np.sum(in_bin)
        bin_counts.append(bin_size)
        
        if bin_size > 0:
            bin_accuracy = np.mean(accuracies[in_bin])
            bin_confidence = np.mean(confidences[in_bin])
            bin_accuracies.append(bin_accuracy)
            bin_confidences.append(bin_confidence)
        else:
            bin_accuracies.append(0)
            bin_confidences.append(0)
    
    bin_accuracies = np.array(bin_accuracies)
    bin_confidences = np.array(bin_confidences)
    bin_counts = np.array(bin_counts)
    
    # Compute calibration gap for coloring
    gaps = np.abs(bin_confidences - bin_accuracies)
    
    # Create plot
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot grid
    ax.grid(color='tab:grey', linestyle=(0, (1, 5)), linewidth=1, zorder=0, alpha=0.5)
    
    # Plot bars
    delta = 1.0 / n_bins
    x_positions = bin_lowers
    
    # Main bars (accuracy)
    ax.bar(x_positions, bin_accuracies, width=delta, align='edge',
           edgecolor='black', color='steelblue', alpha=0.8, label='Accuracy', zorder=5)
    
    # Gap bars (calibration error)
    for i, (x, acc, conf, gap, count) in enumerate(zip(x_positions, bin_accuracies, 
                                                         bin_confidences, gaps, bin_counts)):
        if count > 0 and gap > 0:
            bottom = min(acc, conf)
            ax.bar(x, gap, width=delta, bottom=bottom, align='edge',
                   edgecolor='red', color='mistyrose', alpha=0.7, 
                   linewidth=1.5, hatch='//', zorder=10)
    
    # Add gap to legend (only once)
    ax.bar([], [], edgecolor='red', color='mistyrose', alpha=0.7,
           linewidth=1.5, hatch='//', label='Gap')
    
    # Plot identity line
    ax.plot([0, 1], [0, 1], '--', color='tab:grey', linewidth=2, label='Perfect Calibration', zorder=15)
    
    # Labels and styling
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.legend(loc='upper left', framealpha=0.95, fontsize=10)
    
    if title:
        ax.set_title(title, fontsize=14)
    else:
        avg_conf = np.mean(confidences)
        avg_acc = np.mean(accuracies)
        ax.set_title(f'Reliability Diagram\n(Avg. Confidence: {avg_conf:.3f}, Avg. Accuracy: {avg_acc:.3f})', 
                    fontsize=12)
    
    plt.tight_layout()
    
    if return_fig:
        return fig, ax
    else:
        plt.show()


def confidence_histogram(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    logits: bool = False,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (6, 6),
    return_fig: bool = False
):
    """
    Plot a confidence histogram showing the distribution of model confidences.
    
    The histogram shows how confident the model is across predictions, with
    vertical lines indicating average accuracy and average confidence.
    
    Args:
        probabilities: Array of shape (n_samples, n_classes) containing
            predicted probabilities for each class.
        labels: Array of shape (n_samples,) containing true class labels.
        n_bins: Number of bins for the histogram. Default: 15.
        logits: If True, input is logits and will be converted to probabilities.
            Default: False.
        title: Plot title. If None, uses default title. Default: None.
        figsize: Figure size as (width, height). Default: (6, 6).
        return_fig: If True, return figure and axis objects. Default: False.
    
    Returns:
        If return_fig is True, returns (fig, ax) tuple. Otherwise, displays plot.
    
    Example:
        >>> probs = np.array([[0.8, 0.2], [0.6, 0.4], [0.9, 0.1]])
        >>> labels = np.array([0, 1, 0])
        >>> confidence_histogram(probs, labels)
    """
    # Convert logits to probabilities if needed
    if logits:
        probabilities = softmax(probabilities, axis=1)
    
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)
    
    # Get predictions and confidences
    predictions = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)
    accuracies = (predictions == labels).astype(float)
    
    # Compute average accuracy and confidence
    avg_accuracy = np.mean(accuracies)
    avg_confidence = np.mean(confidences)
    
    # Create plot
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot grid
    ax.grid(color='tab:grey', linestyle=(0, (1, 5)), linewidth=1, zorder=0, alpha=0.5)
    
    # Plot histogram
    n_samples = len(confidences)
    weights = np.ones(n_samples) / n_samples  # Normalize to show proportions
    
    ax.hist(confidences, bins=n_bins, range=(0.0, 1.0), weights=weights,
            color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.2, zorder=5)
    
    # Plot vertical lines for accuracy and confidence
    ax.axvline(x=avg_accuracy, color='darkgreen', linestyle='--', linewidth=2.5,
               label=f'Accuracy ({avg_accuracy:.3f})', zorder=10)
    ax.axvline(x=avg_confidence, color='darkred', linestyle='--', linewidth=2.5,
               label=f'Avg. Confidence ({avg_confidence:.3f})', zorder=10)
    
    # Labels and styling
    ax.set_xlim(0, 1)
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Proportion of Samples', fontsize=12)
    ax.legend(loc='upper left', framealpha=0.95, fontsize=10)
    
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title('Confidence Histogram', fontsize=12)
    
    plt.tight_layout()
    
    if return_fig:
        return fig, ax
    else:
        plt.show()


def class_wise_calibration_curve(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    logits: bool = False,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    max_classes: Optional[int] = 10,
    return_fig: bool = False
):
    """
    Plot class-wise calibration curves.
    
    Shows calibration curves for each class separately, useful for
    understanding per-class calibration behavior.
    
    Args:
        probabilities: Array of shape (n_samples, n_classes) containing
            predicted probabilities for each class.
        labels: Array of shape (n_samples,) containing true class labels.
        n_bins: Number of bins for grouping predictions. Default: 15.
        logits: If True, input is logits and will be converted to probabilities.
            Default: False.
        title: Plot title. If None, uses default title. Default: None.
        figsize: Figure size as (width, height). Default: (8, 6).
        max_classes: Maximum number of classes to plot. If None, plot all.
            Default: 10.
        return_fig: If True, return figure and axis objects. Default: False.
    
    Returns:
        If return_fig is True, returns (fig, ax) tuple. Otherwise, displays plot.
    
    Example:
        >>> probs = np.array([[0.8, 0.15, 0.05], [0.6, 0.3, 0.1]])
        >>> labels = np.array([0, 1])
        >>> class_wise_calibration_curve(probs, labels)
    """
    # Convert logits to probabilities if needed
    if logits:
        probabilities = softmax(probabilities, axis=1)
    
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)
    
    n_classes = probabilities.shape[1]
    
    # Limit number of classes to plot
    if max_classes and n_classes > max_classes:
        classes_to_plot = range(max_classes)
        print(f"Plotting first {max_classes} of {n_classes} classes")
    else:
        classes_to_plot = range(n_classes)
    
    # Create plot
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot grid
    ax.grid(color='tab:grey', linestyle=(0, (1, 5)), linewidth=1, zorder=0, alpha=0.3)
    
    # Plot identity line
    ax.plot([0, 1], [0, 1], '--', color='black', linewidth=2, 
            label='Perfect Calibration', zorder=5)
    
    # Color map for classes
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes_to_plot)))
    
    # Compute calibration curve for each class
    for class_idx, color in zip(classes_to_plot, colors):
        class_probs = probabilities[:, class_idx]
        class_correct = (labels == class_idx).astype(float)
        
        # Compute bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_accuracies = []
        bin_confidences = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = np.logical_and(
                class_probs > bin_lower,
                class_probs <= bin_upper
            )
            
            bin_size = np.sum(in_bin)
            
            if bin_size > 0:
                bin_accuracy = np.mean(class_correct[in_bin])
                bin_confidence = np.mean(class_probs[in_bin])
                bin_accuracies.append(bin_accuracy)
                bin_confidences.append(bin_confidence)
        
        if bin_confidences:  # Only plot if there are points
            ax.plot(bin_confidences, bin_accuracies, 'o-', color=color,
                   linewidth=2, markersize=6, label=f'Class {class_idx}', 
                   alpha=0.8, zorder=10)
    
    # Labels and styling
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.legend(loc='upper left', framealpha=0.95, fontsize=9, ncol=2)
    
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title('Class-wise Calibration Curves', fontsize=12)
    
    plt.tight_layout()
    
    if return_fig:
        return fig, ax
    else:
        plt.show()


def calibration_error_decomposition(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    logits: bool = False,
    figsize: Tuple[float, float] = (10, 6),
    return_fig: bool = False
):
    """
    Plot a comparison of different calibration error metrics.
    
    Creates a bar chart comparing ECE, MCE, RMSCE, ACE, and SCE
    for the given predictions.
    
    Args:
        probabilities: Array of shape (n_samples, n_classes) containing
            predicted probabilities for each class.
        labels: Array of shape (n_samples,) containing true class labels.
        n_bins: Number of bins for computing metrics. Default: 15.
        logits: If True, input is logits and will be converted to probabilities.
            Default: False.
        figsize: Figure size as (width, height). Default: (10, 6).
        return_fig: If True, return figure and axis objects. Default: False.
    
    Returns:
        If return_fig is True, returns (fig, ax) tuple. Otherwise, displays plot.
    
    Example:
        >>> probs = np.array([[0.8, 0.2], [0.6, 0.4], [0.9, 0.1]])
        >>> labels = np.array([0, 1, 0])
        >>> calibration_error_decomposition(probs, labels)
    """
    from .metrics import (expected_calibration_error, maximum_calibration_error,
                          root_mean_square_calibration_error, 
                          adaptive_calibration_error, static_calibration_error)
    
    # Compute different metrics
    metrics = {
        'ECE': expected_calibration_error(probabilities, labels, n_bins, logits),
        'MCE': maximum_calibration_error(probabilities, labels, n_bins, logits),
        'RMSCE': root_mean_square_calibration_error(probabilities, labels, n_bins, logits),
        'ACE': adaptive_calibration_error(probabilities, labels, n_bins, logits),
        'SCE': static_calibration_error(probabilities, labels, n_bins, logits),
    }
    
    # Create plot
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=figsize)
    
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    colors = ['steelblue', 'coral', 'lightgreen', 'plum', 'lightsalmon']
    bars = ax.bar(metric_names, metric_values, color=colors, edgecolor='black', 
                  linewidth=1.5, alpha=0.8)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{value:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Labels and styling
    ax.set_ylabel('Calibration Error', fontsize=12)
    ax.set_title('Calibration Error Metrics Comparison', fontsize=14)
    ax.grid(axis='y', color='tab:grey', linestyle=(0, (1, 5)), linewidth=1, alpha=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if return_fig:
        return fig, ax
    else:
        plt.show()
