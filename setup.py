"""Setup configuration for calibration-toolbox."""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="calibration-toolbox",
    version="0.1.0",
    author="Jonathan Pearce",
    author_email="",
    description="A Python library for evaluating machine learning model calibration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Jonathan-Pearce/calibration-toolbox",
    packages=find_packages(exclude=["tests", "tests.*", "docs", "examples"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "nbsphinx>=0.9.0",
        ],
    },
    keywords="calibration, uncertainty, machine-learning, deep-learning, classification, evaluation",
    project_urls={
        "Bug Reports": "https://github.com/Jonathan-Pearce/calibration-toolbox/issues",
        "Source": "https://github.com/Jonathan-Pearce/calibration-toolbox",
        "Documentation": "https://calibration-toolbox.readthedocs.io/",
    },
)
