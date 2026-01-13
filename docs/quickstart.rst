Quick Start Guide
=================

This guide will help you get started with Calibration Toolbox.

Basic Usage
-----------

Computing Calibration Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The simplest way to compute calibration metrics is to use the wrapper functions:

.. code-block:: python

   import numpy as np
   from calibration_toolbox import expected_calibration_error

   # Your model's predicted probabilities (n_samples, n_classes)
   probabilities = np.array([
       [0.8, 0.2],
       [0.6, 0.4],
       [0.9, 0.1],
       [0.3, 0.7]
   ])
   
   # True labels
   labels = np.array([0, 1, 0, 1])
   
   # Compute Expected Calibration Error
   ece = expected_calibration_error(probabilities, labels)
   print(f"ECE: {ece:.4f}")

Multiple Metrics
^^^^^^^^^^^^^^^^

You can compute multiple calibration metrics:

.. code-block:: python

   from calibration_toolbox import ECE, MCE, RMSCE, ACE, SCE
   
   ece = ECE(probabilities, labels)
   mce = MCE(probabilities, labels)
   rmsce = RMSCE(probabilities, labels)
   ace = ACE(probabilities, labels)
   sce = SCE(probabilities, labels)
   
   print(f"ECE:   {ece:.4f}")
   print(f"MCE:   {mce:.4f}")
   print(f"RMSCE: {rmsce:.4f}")
   print(f"ACE:   {ace:.4f}")
   print(f"SCE:   {sce:.4f}")

General Calibration Error
^^^^^^^^^^^^^^^^^^^^^^^^^^

The General Calibration Error (GCE) is a flexible framework that can compute various metrics:

.. code-block:: python

   from calibration_toolbox import general_calibration_error
   
   # ECE: L1 norm, not class-conditional
   ece = general_calibration_error(
       probabilities, labels,
       norm=1,
       class_conditional=False
   )
   
   # MCE: L-infinity norm
   mce = general_calibration_error(
       probabilities, labels,
       norm='inf',
       class_conditional=False
   )
   
   # ACE: Class-conditional with adaptive bins
   ace = general_calibration_error(
       probabilities, labels,
       norm=1,
       class_conditional=True,
       adaptive_bins=True,
       top_k_classes='all'
   )

Visualization
-------------

Reliability Diagram
^^^^^^^^^^^^^^^^^^^

A reliability diagram shows the relationship between predicted confidence and actual accuracy:

.. code-block:: python

   from calibration_toolbox import reliability_diagram
   
   reliability_diagram(probabilities, labels, n_bins=10)

Confidence Histogram
^^^^^^^^^^^^^^^^^^^^

A confidence histogram shows the distribution of model confidences:

.. code-block:: python

   from calibration_toolbox import confidence_histogram
   
   confidence_histogram(probabilities, labels, n_bins=15)

Class-wise Calibration Curves
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For multi-class problems, you can visualize per-class calibration:

.. code-block:: python

   from calibration_toolbox import class_wise_calibration_curve
   
   # Multi-class probabilities
   probs = np.random.dirichlet(np.ones(5), size=100)
   labels = np.random.randint(0, 5, size=100)
   
   class_wise_calibration_curve(probs, labels)

Metric Comparison
^^^^^^^^^^^^^^^^^

Compare multiple calibration metrics in one plot:

.. code-block:: python

   from calibration_toolbox import calibration_error_decomposition
   
   calibration_error_decomposition(probabilities, labels)

Working with Logits
-------------------

If your model outputs logits instead of probabilities, set ``logits=True``:

.. code-block:: python

   import numpy as np
   from calibration_toolbox import expected_calibration_error
   
   # Model outputs (logits)
   logits = np.array([
       [2.0, -1.0],
       [1.0, 0.5],
       [-0.5, 1.5]
   ])
   labels = np.array([0, 0, 1])
   
   # Compute ECE (will apply softmax internally)
   ece = expected_calibration_error(logits, labels, logits=True)

Common Patterns
---------------

Evaluating a Trained Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from calibration_toolbox import ECE, reliability_diagram
   
   # Get predictions from your model
   # predictions = model.predict_proba(X_test)
   
   # For this example, use dummy data
   predictions = np.random.dirichlet(np.ones(3), size=200)
   true_labels = np.random.randint(0, 3, size=200)
   
   # Compute calibration error
   ece = ECE(predictions, true_labels)
   print(f"Model ECE: {ece:.4f}")
   
   # Visualize calibration
   reliability_diagram(predictions, true_labels, 
                      title=f"Model Calibration (ECE: {ece:.4f})")

Comparing Multiple Models
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from calibration_toolbox import ECE
   
   # Predictions from different models
   model1_probs = np.random.dirichlet(np.ones(2), size=100)
   model2_probs = np.random.dirichlet(np.ones(2), size=100)
   labels = np.random.randint(0, 2, size=100)
   
   ece1 = ECE(model1_probs, labels)
   ece2 = ECE(model2_probs, labels)
   
   print(f"Model 1 ECE: {ece1:.4f}")
   print(f"Model 2 ECE: {ece2:.4f}")
   
   if ece1 < ece2:
       print("Model 1 is better calibrated")
   else:
       print("Model 2 is better calibrated")

Next Steps
----------

- Check out the :doc:`api/index` for detailed documentation of all functions
- See the :doc:`examples/index` for more comprehensive examples
- Read about the :doc:`references` for the research papers behind each metric
