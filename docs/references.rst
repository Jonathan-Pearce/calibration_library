References
==========

This page lists the research papers that informed the design and implementation of Calibration Toolbox.

Key Papers
----------

General Calibration Error Framework
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Kull et al. (2019)**: "Beyond temperature scaling: Obtaining well-calibrated multiclass probabilities with Dirichlet calibration." *NeurIPS 2019*. `arXiv:1910.12656 <https://arxiv.org/abs/1910.12656>`_

- **Nixon et al. (2020)**: "Measuring Calibration in Deep Learning." *CVPR Workshops 2020*. `arXiv:1904.01685 <https://arxiv.org/abs/1904.01685>`_

Foundational Metrics
^^^^^^^^^^^^^^^^^^^^

Expected Calibration Error (ECE)
"""""""""""""""""""""""""""""""""

- **Naeini et al. (2015)**: "Obtaining Well Calibrated Probabilities Using Bayesian Binning." *AAAI 2015*. `Paper <https://ojs.aaai.org/index.php/AAAI/article/view/9602>`_

- **Guo et al. (2017)**: "On Calibration of Modern Neural Networks." *ICML 2017*. `arXiv:1706.04599 <https://arxiv.org/abs/1706.04599>`_

Root Mean Square Calibration Error (RMSCE)
"""""""""""""""""""""""""""""""""""""""""""

- **Hendrycks et al. (2019)**: "Deep Anomaly Detection with Outlier Exposure." *ICLR 2019*. `arXiv:1812.04606 <https://arxiv.org/abs/1812.04606>`_

Class-Conditional Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^

Static and Adaptive Calibration Error (SCE, ACE)
"""""""""""""""""""""""""""""""""""""""""""""""""

- **Nixon et al. (2020)**: "Measuring Calibration in Deep Learning." *CVPR Workshops 2020*. `arXiv:1904.01685 <https://arxiv.org/abs/1904.01685>`_

Top-k Calibration Error
""""""""""""""""""""""""

- **Gupta et al. (2021)**: "Calibration of Neural Networks using Splines." *ICLR 2021*. `arXiv:2006.12800 <https://arxiv.org/abs/2006.12800>`_

Overconfidence Error
^^^^^^^^^^^^^^^^^^^^

- **Thulasidasan et al. (2019)**: "On Mixup Training: Improved Calibration and Predictive Uncertainty for Deep Neural Networks." *NeurIPS 2019*. `arXiv:1905.11001 <https://arxiv.org/abs/1905.11001>`_

Additional Resources
--------------------

Calibration Methods
^^^^^^^^^^^^^^^^^^^

- **Platt (1999)**: "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods."

- **Zadrozny & Elkan (2002)**: "Transforming Classifier Scores into Accurate Multiclass Probability Estimates." *KDD 2002*.

- **Kull et al. (2017)**: "Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers." *AISTATS 2017*. `arXiv <https://proceedings.mlr.press/v54/kull17a.html>`_

Review Papers
^^^^^^^^^^^^^

- **Roelofs et al. (2022)**: "Mitigating Bias in Calibration Error Estimation." *AISTATS 2022*. `arXiv:2012.08668 <https://arxiv.org/abs/2012.08668>`_

- **Minderer et al. (2021)**: "Revisiting the Calibration of Modern Neural Networks." *NeurIPS 2021*. `arXiv:2106.07998 <https://arxiv.org/abs/2106.07998>`_

- **Kumar et al. (2019)**: "Verified Uncertainty Calibration." *NeurIPS 2019*. `arXiv:1909.10155 <https://arxiv.org/abs/1909.10155>`_

Related Tools
-------------

- **Uncertainty Toolbox**: Comprehensive uncertainty quantification library. `GitHub <https://github.com/uncertainty-toolbox/uncertainty-toolbox>`_

Citation
--------

If you use Calibration Toolbox in your research, please cite:

.. code-block:: bibtex

   @software{calibration_toolbox,
     author = {Pearce, Jonathan},
     title = {Calibration Toolbox: A Python Library for Model Calibration Evaluation},
     year = {2026},
     url = {https://github.com/Jonathan-Pearce/calibration-toolbox}
   }

Contributing
------------

We welcome contributions! Please see the repository's `CONTRIBUTING.md` file for guidelines.
