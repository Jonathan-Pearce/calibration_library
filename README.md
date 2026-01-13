# Calibration Toolbox

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Calibration Toolbox** is a Python library for evaluating machine learning model calibration using binning-based metrics. The package provides a comprehensive collection of calibration metrics and visualization tools for research in deep learning and uncertainty quantification.

## Features

- **Comprehensive Metrics**: ECE, MCE, RMSCE, ACE, SCE, and more
- **General Calibration Error Framework**: Flexible GCE function for custom metrics
- **Framework-Agnostic**: Works with NumPy arrays - no PyTorch or TensorFlow required
- **Visualization Tools**: Reliability diagrams, confidence histograms, and class-wise calibration curves
- **Well-Tested**: Extensive test coverage with edge case handling
- **Research-Oriented**: Implementations based on recent research papers

## Installation

### From PyPI (once published)

```bash
pip install calibration-toolbox
```

### From Source

```bash
git clone https://github.com/Jonathan-Pearce/calibration-toolbox.git
cd calibration-toolbox
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from calibration_toolbox import expected_calibration_error, reliability_diagram

# Your model's predicted probabilities
probabilities = np.array([[0.8, 0.2], [0.6, 0.4], [0.9, 0.1]])
labels = np.array([0, 1, 0])

# Compute Expected Calibration Error
ece = expected_calibration_error(probabilities, labels)
print(f"ECE: {ece:.4f}")

# Visualize calibration
reliability_diagram(probabilities, labels)
```

## Metrics

### General Calibration Error (GCE)

The GCE is a flexible framework that can compute various calibration metrics through parameter configuration:

```math
\text{GCE} = \left(\sum_{k=1}^{K} \sum_{b=1}^{B} \frac{n_{bk}}{NK} |acc(b,k) - conf(b,k)|^p\right)^{1/p}
```

Where:
- $`acc(b,k)`$ and $`conf(b,k)`$ are the accuracy and confidence of bin $`b`$ for class $`k`$
- $`n_{bk}`$ is the number of predictions in bin $`b`$ for class $`k`$
- $`N`$ is the total number of data points
- $`K`$ is the number of classes
- $`p`$ is the norm parameter (1, 2, or ∞)

**Usage:**

```python
from calibration_toolbox import general_calibration_error

gce = general_calibration_error(
    probabilities, labels,
    n_bins=15,
    class_conditional=True,
    adaptive_bins=False,
    top_k_classes='all',
    norm=1,
    thresholding=0.0
)
```

### Expected Calibration Error (ECE)

**Reference:** Naeini et al. (2015). "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.

ECE measures the difference between model confidence and accuracy across uniformly-spaced bins:

```math
\text{ECE} = \sum_{b=1}^{B} \frac{n_b}{N} |acc(b) - conf(b)|
```

```python
from calibration_toolbox import ECE

ece = ECE(probabilities, labels, n_bins=15)
```

### Maximum Calibration Error (MCE)

**Reference:** Naeini et al. (2015). "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.

MCE is the maximum calibration error across all bins:

```math
\text{MCE} = \max_{b} |acc(b) - conf(b)|
```

```python
from calibration_toolbox import MCE

mce = MCE(probabilities, labels, n_bins=15)
```

### Root Mean Square Calibration Error (RMSCE)

**Reference:** Hendrycks et al. (2019). "Deep Anomaly Detection with Outlier Exposure." ICLR.

RMSCE is the root mean square of calibration errors:

```math
\text{RMSCE} = \sqrt{\sum_{b=1}^{B} \frac{n_b}{N} (acc(b) - conf(b))^2}
```

```python
from calibration_toolbox import RMSCE

rmsce = RMSCE(probabilities, labels, n_bins=15)
```

### Static Calibration Error (SCE)

**Reference:** Nixon et al. (2020). "Measuring Calibration in Deep Learning." CVPR Workshops.

SCE is the class-conditional calibration error with uniform binning:

```python
from calibration_toolbox import SCE

sce = SCE(probabilities, labels, n_bins=15)
```

### Adaptive Calibration Error (ACE)

**Reference:** Nixon et al. (2020). "Measuring Calibration in Deep Learning." CVPR Workshops.

ACE uses adaptive binning (equal number of samples per bin):

```python
from calibration_toolbox import ACE

ace = ACE(probabilities, labels, n_bins=15)
```

### Top-k Calibration Error

**Reference:** Gupta et al. (2021). "Calibration of Neural Networks using Splines." ICLR.

Computes calibration error for the top-k predicted classes:

```python
from calibration_toolbox import top_k_calibration_error

top2_ce = top_k_calibration_error(probabilities, labels, k=2)
```

### Overconfidence Error (OE)

**Reference:** Thulasidasan et al. (2019). "On Mixup Training." NeurIPS.

OE measures the degree of overconfidence:

```python
from calibration_toolbox import overconfidence_error

oe = overconfidence_error(probabilities, labels)
```

## Visualization

### Reliability Diagram

Shows the relationship between predicted confidence and actual accuracy:

```python
from calibration_toolbox import reliability_diagram

reliability_diagram(probabilities, labels, n_bins=15)
```

### Confidence Histogram

Shows the distribution of model confidences:

```python
from calibration_toolbox import confidence_histogram

confidence_histogram(probabilities, labels, n_bins=15)
```

### Class-wise Calibration Curves

Shows calibration curves for each class separately:

```python
from calibration_toolbox import class_wise_calibration_curve

class_wise_calibration_curve(probabilities, labels)
```

### Calibration Error Decomposition

Compares multiple calibration metrics in one plot:

```python
from calibration_toolbox import calibration_error_decomposition

calibration_error_decomposition(probabilities, labels)
```

## Working with Logits

If your model outputs logits instead of probabilities, set `logits=True`:

```python
from calibration_toolbox import ECE

# Model outputs logits
logits = np.array([[2.0, -1.0], [1.0, 0.5], [-0.5, 1.5]])
labels = np.array([0, 0, 1])

# Compute ECE (will apply softmax internally)
ece = ECE(logits, labels, logits=True)
```

## Examples

Check out the [examples](examples/) directory for Jupyter notebooks demonstrating:
- Basic usage and metric computation
- Visualization creation
- Comparing models with different calibration qualities
- Advanced usage of the GCE framework

## Documentation

Full documentation is available at [Read the Docs](https://calibration-toolbox.readthedocs.io/) (once published).

To build the documentation locally:

```bash
cd docs
make html
```

## Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

Or use the test script:

```bash
bash run_tests.sh
```

With coverage (requires `pytest-cov`):

```bash
pip install pytest-cov
pytest tests/ -v --cov=calibration_toolbox --cov-report=html
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Citation

If you use Calibration Toolbox in your research, please cite:

```bibtex
@software{calibration_toolbox,
  author = {Pearce, Jonathan},
  title = {Calibration Toolbox: A Python Library for Model Calibration Evaluation},
  year = {2026},
  url = {https://github.com/Jonathan-Pearce/calibration-toolbox}
}
```

## Related Work

This package is inspired by and follows the structure of:
- [Uncertainty Toolbox](https://github.com/uncertainty-toolbox/uncertainty-toolbox) - Comprehensive uncertainty quantification library

## References

Key papers that informed this package's design:

- **Naeini et al. (2015)**: "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
- **Guo et al. (2017)**: "On Calibration of Modern Neural Networks." ICML.
- **Kull et al. (2019)**: "Beyond temperature scaling: Obtaining well-calibrated multiclass probabilities with Dirichlet calibration." NeurIPS.
- **Nixon et al. (2020)**: "Measuring Calibration in Deep Learning." CVPR Workshops.
- **Gupta et al. (2021)**: "Calibration of Neural Networks using Splines." ICLR.

See [papers.md](papers.md) for a complete list of references.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The calibration research community for developing these metrics
- The Uncertainty Toolbox team for inspiration on package structure
- Contributors and users of this library

---

**Maintained by:** Jonathan Pearce  
**Repository:** [https://github.com/Jonathan-Pearce/calibration-toolbox](https://github.com/Jonathan-Pearce/calibration-toolbox)
