Metrics Module
==============

The metrics module provides calibration error metrics for classification models.

.. automodule:: calibration_toolbox.metrics
   :members:
   :undoc-members:
   :show-inheritance:

General Calibration Error
--------------------------

.. autofunction:: calibration_toolbox.metrics.general_calibration_error

Standard Metrics
----------------

.. autofunction:: calibration_toolbox.metrics.expected_calibration_error

.. autofunction:: calibration_toolbox.metrics.maximum_calibration_error

.. autofunction:: calibration_toolbox.metrics.root_mean_square_calibration_error

Class-Conditional Metrics
--------------------------

.. autofunction:: calibration_toolbox.metrics.static_calibration_error

.. autofunction:: calibration_toolbox.metrics.adaptive_calibration_error

.. autofunction:: calibration_toolbox.metrics.top_k_calibration_error

.. autofunction:: calibration_toolbox.metrics.thresholded_adaptive_calibration_error

Other Metrics
-------------

.. autofunction:: calibration_toolbox.metrics.overconfidence_error
