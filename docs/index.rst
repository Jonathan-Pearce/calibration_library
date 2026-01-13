Calibration Toolbox Documentation
===================================

**Calibration Toolbox** is a Python library for evaluating machine learning model calibration using binning-based metrics.

The package provides a comprehensive collection of calibration metrics and visualization tools for research in deep learning and uncertainty quantification.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/index
   examples/index
   references

Overview
--------

Calibration Toolbox focuses on **binning-based calibration metrics**, which are widely used in machine learning research to evaluate how well predicted probabilities match actual outcomes.

Key Features
------------

* **Comprehensive Metrics**: ECE, MCE, RMSCE, ACE, SCE, and more
* **General Calibration Error (GCE)**: Flexible framework for computing various calibration metrics
* **Framework-Agnostic**: Works with NumPy arrays - no PyTorch or TensorFlow required
* **Visualization Tools**: Reliability diagrams, confidence histograms, and class-wise calibration curves
* **Well-Tested**: Extensive test coverage with edge case handling
* **Research-Oriented**: Implementations based on recent research papers

Quick Example
-------------

.. code-block:: python

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

Installation
------------

Install from PyPI (once published):

.. code-block:: bash

   pip install calibration-toolbox

Or install from source:

.. code-block:: bash

   git clone https://github.com/Jonathan-Pearce/calibration-toolbox.git
   cd calibration-toolbox
   pip install -e .

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
