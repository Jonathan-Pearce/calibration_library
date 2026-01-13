"""
Calibration Toolbox - A Python library for evaluating model calibration.

This package provides a comprehensive collection of binning-based calibration
metrics and visualization tools for classification models.
"""

from .metrics import (
    general_calibration_error,
    expected_calibration_error,
    maximum_calibration_error,
    root_mean_square_calibration_error,
    static_calibration_error,
    adaptive_calibration_error,
    top_k_calibration_error,
    thresholded_adaptive_calibration_error,
    overconfidence_error,
    # Aliases
    GCE,
    ECE,
    MCE,
    RMSCE,
    SCE,
    ACE,
    TACE,
    OE,
)

from .visualization import (
    reliability_diagram,
    confidence_histogram,
    class_wise_calibration_curve,
    calibration_error_decomposition,
)

__version__ = "0.1.0"

__all__ = [
    # Metrics
    "general_calibration_error",
    "expected_calibration_error",
    "maximum_calibration_error",
    "root_mean_square_calibration_error",
    "static_calibration_error",
    "adaptive_calibration_error",
    "top_k_calibration_error",
    "thresholded_adaptive_calibration_error",
    "overconfidence_error",
    # Aliases
    "GCE",
    "ECE",
    "MCE",
    "RMSCE",
    "SCE",
    "ACE",
    "TACE",
    "OE",
    # Visualization
    "reliability_diagram",
    "confidence_histogram",
    "class_wise_calibration_curve",
    "calibration_error_decomposition",
]
