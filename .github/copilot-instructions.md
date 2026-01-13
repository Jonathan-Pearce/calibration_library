# Copilot Instructions for Calibration Toolbox

## Project Overview

Calibration Toolbox is a Python package for evaluating machine learning model calibration using binning-based metrics. The package provides a comprehensive collection of calibration metrics and visualization tools for research in deep learning and uncertainty quantification.

### Project Goals
- Create a lightweight, framework-agnostic calibration metrics library for ML research
- Provide implementations of popular calibration metrics from recent research papers
- Offer visualization tools for calibration analysis (reliability diagrams, confidence histograms)
- Maintain compatibility with pure NumPy/SciPy (avoid PyTorch/TensorFlow dependencies)
- Follow the structure and quality standards of [uncertainty-toolbox](https://github.com/uncertainty-toolbox/uncertainty-toolbox)

### Target Audience
Machine learning researchers evaluating model calibration and uncertainty quantification for their models.

## Technical Stack

### Core Dependencies
- **NumPy**: Primary numerical computing library
- **SciPy**: Scientific computing utilities (e.g., `softmax`)
- **Matplotlib**: Visualization and plotting

### Avoided Dependencies
- **PyTorch**: Keep framework-agnostic
- **TensorFlow**: Keep framework-agnostic
- **JAX**: Keep framework-agnostic

## Code Style and Conventions

### Python Style
- Follow PEP 8 style guidelines
- Use clear, descriptive variable names
- Add comprehensive docstrings to all public functions and classes
- Include type hints for function parameters and returns
- Keep functions focused and modular

### Documentation Standards
- Every metric should include:
  - Mathematical formula (LaTeX format for README)
  - Citation to original paper
  - Example usage code
  - Parameter descriptions
  - Return value description
- Use NumPy-style docstrings

### Testing Requirements
- Write unit tests for all metrics
- Include edge case testing (empty arrays, single samples, etc.)
- Test numerical accuracy against known results
- Verify bin boundary calculations
- Test both uniform and adaptive binning

### Example Docstring Format
```python
def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, 
                              n_bins: int = 15) -> float:
    """
    Calculate Expected Calibration Error (ECE).
    
    ECE measures the difference between model confidence and accuracy
    across uniformly-spaced bins. Lower values indicate better calibration.
    
    Reference:
        Naeini et al. (2015). "Obtaining Well Calibrated Probabilities 
        Using Bayesian Binning." AAAI.
    
    Args:
        probabilities: Array of shape (n_samples, n_classes) containing
            predicted probabilities for each class.
        labels: Array of shape (n_samples,) containing true class labels.
        n_bins: Number of bins for confidence discretization. Default: 15.
    
    Returns:
        float: ECE value between 0 and 1.
    
    Example:
        >>> probs = np.array([[0.8, 0.2], [0.6, 0.4]])
        >>> labels = np.array([0, 1])
        >>> ece = expected_calibration_error(probs, labels)
    """
```

## Architecture

### Module Organization
```
calibration-toolbox/
├── calibration_toolbox/          # Main package
│   ├── __init__.py               # Package initialization
│   ├── metrics.py                # Calibration metrics
│   ├── visualization.py          # Plotting functions
│   ├── utils.py                  # Helper functions
│   └── binning.py                # Binning strategies
├── tests/                        # Test suite
│   ├── test_metrics.py
│   ├── test_visualization.py
│   └── test_binning.py
├── docs/                         # Documentation
│   ├── conf.py                   # Sphinx configuration
│   ├── index.rst
│   └── api/                      # API documentation
├── examples/                     # Example notebooks
│   └── basic_usage.ipynb
├── setup.py                      # Package setup
├── requirements.txt              # Dependencies
├── README.md                     # Main documentation
└── LICENSE                       # License file
```

### Key Metrics to Implement

Based on the General Calibration Error (GCE) framework:

1. **Expected Calibration Error (ECE)**: `GCE(norm=1, class_conditional=False)`
2. **Root Mean Squared Calibration Error (RMSCE)**: `GCE(norm=2, class_conditional=False)`
3. **Maximum Calibration Error (MCE)**: `GCE(norm='inf', class_conditional=False)`
4. **Static Calibration Error (SCE)**: `GCE(norm=1, class_conditional=True, adaptive_bins=False)`
5. **Adaptive Calibration Error (ACE)**: `GCE(norm=1, class_conditional=True, adaptive_bins=True)`
6. **Top-r Calibration Error (ToprCE)**: `GCE(norm=1, class_conditional=True, top_k_classes=r)`

Future metrics:
- Class-wise Calibration Error (CWCE)
- Overconfidence Error (OE)
- Thresholded Adaptive Calibration Error (TACE)

## General Calibration Error (GCE) Framework

The GCE is the central metric with configurable parameters:

```
GCE = (Σ_k Σ_b (n_bk / NK) |acc(b,k) - conf(b,k)|^p)^(1/p)
```

### Parameters
- `n_bins`: Number of bins (default: 15)
- `class_conditional`: Whether to compute class-specific calibration
- `adaptive_bins`: Use adaptive binning based on data distribution
- `top_k_classes`: Consider only top-k predicted classes ('all' or integer)
- `norm`: Lp norm to use (1, 2, 'inf')
- `thresholding`: Ignore probabilities below this threshold

## Visualization Guidelines

### Supported Plots
1. **Reliability Diagram**: Shows calibration gaps between confidence and accuracy
2. **Confidence Histogram**: Distribution of model confidence levels

### Plot Style
- Use serif fonts (`plt.rcParams["font.family"] = "serif"`)
- Square figures (3x3 inches by default)
- Grid lines: grey, dashed `(0, (1, 5))`
- Clean, publication-ready aesthetics
- Include legends and axis labels

## Development Workflow

### Adding New Metrics
1. Implement the metric function with proper docstring
2. Add wrapper function if it's a special case of GCE
3. Write unit tests with known test cases
4. Add documentation to README.md with:
   - Mathematical formula
   - Citation
   - Code example
5. Update API documentation

### Code Review Checklist
- [ ] Docstrings complete with parameters, returns, examples
- [ ] Type hints added
- [ ] Unit tests written and passing
- [ ] Works with NumPy arrays of various shapes
- [ ] Handles edge cases (empty arrays, single samples)
- [ ] README updated with usage example
- [ ] No deep learning framework dependencies introduced

## Input/Output Conventions

### Expected Input Formats
- `probabilities`: NumPy array of shape `(n_samples, n_classes)` containing predicted class probabilities
- `labels`: NumPy array of shape `(n_samples,)` containing integer class labels (0-indexed)
- `logits`: Boolean flag indicating if input is logits (requires softmax) or probabilities

### Output Formats
- Metrics return scalar float values (typically between 0 and 1)
- Visualization functions return matplotlib figure/axis objects
- Always validate input shapes and types

## Best Practices

### Numerical Stability
- Use SciPy's softmax when converting logits to probabilities
- Handle division by zero in empty bins
- Use appropriate NumPy comparison functions (`np.greater`, `np.less_equal`)

### Performance
- Vectorize operations using NumPy
- Avoid Python loops where possible
- Pre-allocate arrays for bin statistics

### Error Handling
- Validate input array shapes and types
- Check for valid probability distributions (sum to 1, non-negative)
- Provide clear error messages for invalid inputs

## Research Paper Integration

When adding metrics from papers:
1. Add paper reference to `papers.md`
2. Include arXiv link and publication venue
3. Note which specific metrics/concepts were introduced
4. Reference in metric docstring

## Package Distribution Goals

Following uncertainty-toolbox model:
- PyPI package distribution
- Sphinx documentation with ReadTheDocs hosting
- Example Jupyter notebooks
- Comprehensive API documentation
- GitHub Actions CI/CD for testing
- Code coverage reporting

## Common Patterns

### Binning Strategy
```python
# Uniform binning
bin_boundaries = np.linspace(0, 1, n_bins + 1)

# Adaptive binning (equal sample counts)
bin_n = int(n_data / n_bins)
probabilities_sort = np.sort(probabilities)
bin_boundaries = probabilities_sort[::bin_n]
```

### Bin Assignment
```python
in_bin = np.logical_and(
    confidences > bin_lower,
    confidences <= bin_upper
)
```

### Binary Matrix Creation (for class-conditional metrics)
```python
pred_matrix = np.zeros([n_data, n_class])
pred_matrix[np.arange(n_data), predictions] = 1
```

## Future Enhancements

1. Add more calibration metrics from recent papers
2. Support for regression calibration metrics
3. Calibration curve fitting utilities
4. Multi-label classification calibration
5. Hierarchical/structured prediction calibration
6. Integration examples with popular ML frameworks (scikit-learn, etc.)

## Questions to Consider When Contributing

- Does this metric require a deep learning framework, or can it work with NumPy?
- Is the implementation numerically stable?
- Are edge cases handled appropriately?
- Is the API consistent with existing metrics?
- Does the documentation clearly explain the metric's purpose and interpretation?

## References

Key papers that informed this package's design are cataloged in `papers.md`.
