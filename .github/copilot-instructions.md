# Copilot Instructions for InferAGNI

## Quick Start

### Build & Test Commands

**Setup:**
```bash
pip install -e '.[develop]'  # Install package + dev dependencies
```

**Running Tests:**
- Full suite: `pytest tests/`
- Single test file: `pytest tests/test_util.py`
- Single test: `pytest tests/test_util.py::test_calc_scaleheight_basic`
- With coverage (matches CI): `coverage run -m pytest tests/`
- With markers (unit tests only): `pytest -m unit tests/`
- Exclude slow tests: `pytest -m "not slow" tests/`

**Test Markers:** The codebase uses pytest markers:
- `@pytest.mark.unit` - fast unit tests
- `@pytest.mark.slow` - slow tests (deselect with `-m "not slow"`)
- `@pytest.mark.integration` - integration tests
- `@pytest.mark.smoke` - quick validation with real binaries

**Linting & Formatting:**
- Lint with ruff: `ruff check src/ tests/`
- Format with ruff: `ruff format src/ tests/`
- Pre-commit: `pre-commit run --all-files`

**Code Coverage:**
- HTML report: `coverage run -m pytest tests/ && coverage html`
- Report to console: `coverage report`
- Minimum threshold: 10% (enforced)

## Project Architecture

**InferAGNI** is a Bayesian inference framework for exoplanet characterization using AGNI as a static structure model. It couples planetary interior models with atmospheric retrieval.

### Core Modules

- **`cli.py`**: Click-based CLI with commands like `infer`, `listplanets`, `planet`, `listvars`
- **`planets.py`**: Exoplanet database integration using the `exoatlas` library; handles both solar system and exoplanet data
- **`retrieve.py`**: MCMC-based atmospheric retrieval logic using `emcee`
- **`grid.py`**: Grid data management for AGNI model outputs
- **`data.py`**: Data loading and caching utilities
- **`plot.py`**: Plotting utilities using matplotlib and cmcrameri colormaps
- **`util.py`**: Helper functions (scale height, dimensioning, latex formatting)

### Entry Points

The package exports a CLI command `inferagni` (defined in `pyproject.toml` under `[project.scripts]`). Primary use case: `inferagni infer "planet_name"`

### Exoplanet Data

- Exoplanet data comes from `exoatlas` package
- Solar system data is bundled in `src/inferagni/exoatlas-data/`
- The `EXOATLAS_DATA` environment variable is set in `planets.py` to point to the bundled data

## Key Conventions

### Python Style
- **Required import**: All modules must start with `from __future__ import annotations` (enforced by ruff isort rules)
- **Formatting**: Ruff formatter with double quotes, 96-char line length
- **Code structure**: 4-space indentation; target Python 3.12+
- **Type hints**: Use them; `from __future__ import annotations` allows forward references

### Module Exports
- Modules explicitly define `__all__` to control public API (see `__init__.py`)
- Many submodules are currently commented out in `__init__.py` as they're still in development

### Testing
- Tests live in `tests/` directory, matching source structure where applicable
- Test functions prefixed with `test_`, classes with `Test*`
- Always mark tests with appropriate pytest markers (`@pytest.mark.unit`, etc.)
- Use `numpy.testing.assert_*` or `numpy.isclose()` for float comparisons

### Project Metadata
- Version follows `YY.MM.DD` format (defined in `pyproject.toml` and `__init__.py`)
- Must be kept in sync between these two files
- License is GPLv3; author is Harrison Nicholls

### CI/CD
- Tests run on ubuntu-latest and macos-latest across Python 3.12, 3.13, 3.14
- Coverage reports are appended to GitHub step summary
- Publishing to PyPI only on release; uses trusted publishing (OIDC)

## Important Notes

- The package has a minimum coverage threshold of 10% (see `[tool.coverage.report]` fail_under)
- Draft PRs skip CI tests
- Exoplanet database filtering is performed in `planets.py` (radius > 0 filter applied to exoplanets)
- AGNI grid documentation: https://www.h-nicholls.space/AGNI/
