Installation
============

Requirements
------------

Calibration Toolbox requires:

- Python >= 3.8
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- Matplotlib >= 3.5.0

Install from PyPI
-----------------

Once published, you can install Calibration Toolbox using pip:

.. code-block:: bash

   pip install calibration-toolbox

Install from Source
-------------------

To install the latest development version:

.. code-block:: bash

   git clone https://github.com/Jonathan-Pearce/calibration-toolbox.git
   cd calibration-toolbox
   pip install -e .

Development Installation
------------------------

If you want to contribute to the project or run tests:

.. code-block:: bash

   git clone https://github.com/Jonathan-Pearce/calibration-toolbox.git
   cd calibration-toolbox
   pip install -e ".[dev]"

This installs additional development dependencies including pytest, sphinx, and code quality tools.

Verify Installation
-------------------

You can verify the installation by importing the package:

.. code-block:: python

   import calibration_toolbox as cal
   print(cal.__version__)

Or run the test suite:

.. code-block:: bash

   pytest tests/
