# Contributing to Calibration Toolbox

Thank you for considering contributing to Calibration Toolbox! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork the repository
2. Clone your fork:
```bash
git clone https://github.com/YOUR_USERNAME/calibration-toolbox.git
cd calibration-toolbox
```

3. Install in development mode:
```bash
pip install -e ".[dev]"
```

## Code Style

We follow PEP 8 style guidelines. Before submitting:

```bash
# Format code with black
black calibration_toolbox tests

# Check with flake8
flake8 calibration_toolbox tests

# Type check with mypy
mypy calibration_toolbox
```

## Testing

All contributions must include tests. Run the test suite:

```bash
pytest tests/
```

With coverage:

```bash
pytest tests/ --cov=calibration_toolbox --cov-report=html
```

## Adding New Metrics

When adding a new calibration metric:

1. Implement the metric in `calibration_toolbox/metrics.py`
2. Add comprehensive docstring with:
   - Mathematical formula
   - Reference to original paper
   - Example usage
3. Write unit tests in `tests/test_metrics.py`
4. Update `README.md` with the new metric
5. Add the paper reference to `papers.md`
6. Update documentation in `docs/api/metrics.rst`

## Pull Request Process

1. Create a new branch for your feature:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and commit:
```bash
git add .
git commit -m "Add feature: description"
```

3. Push to your fork:
```bash
git push origin feature/your-feature-name
```

4. Open a Pull Request with:
   - Clear description of changes
   - Reference to any related issues
   - Test results showing all tests pass

## Questions?

Open an issue for questions or discussions about contributions.
