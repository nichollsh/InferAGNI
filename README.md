# AGNI Inference Package

Inferring planet properties using AGNI as a static structure model. This package provides a Bayesian inference framework for exoplanet characterization by coupling AGNI (a static interior structure model) with atmospheric retrieval via MCMC.

### Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Getting Started with Development](#getting-started-with-development)
- [Running Tests](#running-tests)
- [Key Modules](#key-modules)
- [Resources](#resources)

### External links

- [AGNI Model (atmosphere climate model)](https://www.h-nicholls.space/AGNI/)
- [AGNI GitHub](https://github.com/nichollsh/AGNI/)
- [Zalmoxis (interior structure model)](https://proteus-framework.org/Zalmoxis/)
- [Research Paper](https://www.overleaf.com/project/6853d410bda854791be86cd7)

### Citation

```bibtex
@article{nicholls_volatile_2026,
	author = {Nicholls, Harrison and Lichtenberg, Tim and Chatterjee, Richard D. and Guimond, Claire Marie and Postolec, Emma and Pierrehumbert, Raymond T.},
	title = {{Volatile-rich evolution of molten super-Earth L 98-59 d}},
	journal = {Nat. Astron.},
	pages = {1--9},
	year = {2026},
	month = mar,
	issn = {2397-3366},
	publisher = {Nature Publishing Group},
	doi = {10.1038/s41550-026-02815-8}
}
```

-------------

## Installation

### Prerequisites

- Python 3.12 or later
- A conda distribution (e.g., [Miniforge](https://github.com/conda-forge/miniforge))

### Setup

1. Clone the repository and navigate to the directory:
   ```bash
   git clone https://github.com/nichollsh/InferAGNI.git
   cd InferAGNI
   ```

2. Install the package and its dependencies:
   ```bash
   pip install -e .
   ```

3. Download required data files:
   ```bash
   inferagni update
   ```

## Quick Start

### Running Your First Inference

Once installed, retrieve properties for an exoplanet:

```bash
inferagni infer "L 98-59 d"
```

### Available Commands

- `inferagni infer <planet_name>` - Run atmospheric retrieval for a planet
- `inferagni listplanets` - List all available planets in the database
- `inferagni planet <planet_name>` - Get observed parameters for a specific planet
- `inferagni listvars` - Display available variables in the AGNI grid


## Running Tests

### Quick Test Commands

```bash
# Run all tests
pytest tests/

# Run a specific test file
pytest tests/test_util.py

# Run a single test
pytest tests/test_util.py::test_calc_scaleheight_basic

# Run only unit tests (fast)
pytest -m unit tests/

# Run with coverage report
coverage run -m pytest tests/
coverage report
```

### Test Organization

Tests are organized by functionality with pytest markers:
- `@pytest.mark.unit` - Fast unit tests (~seconds)
- `@pytest.mark.integration` - Integration tests requiring external resources
- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.smoke` - Quick validation tests

Exclude slow tests during development with `pytest -m "not slow" tests/`.

### Code Quality

```bash
# Lint code
ruff check src/ tests/

# Format code
ruff format src/ tests/

# Run pre-commit hooks
pre-commit run --all-files
```

The project enforces a minimum 10% code coverage threshold in CI.

## Project Structure
```
src/inferagni/
├── cli.py              # Command-line interface (Click-based)
├── planets.py          # Exoplanet database integration
├── retrieve.py         # MCMC atmospheric retrieval logic
├── grid.py             # AGNI grid data management
├── data.py             # Data loading and caching
├── plot.py             # Visualization utilities
├── util.py             # Helper functions
├── data/               # Grid data and other dependencies
└── exoatlas-data/      # Bundled exoplanet database
```

### `cli.py`
The command-line interface providing user-facing commands for inferring planet properties, listing available exoplanets, and inspecting grid variables.

### `planets.py`
Integrates with the [exoatlas](https://zkbt.github.io/exoatlas/) library to manage exoplanet data. Solar system planets are bundled separately; exoplanet data is filtered to include only those with measured radii.

### `retrieve.py`
Implements Bayesian atmospheric retrieval using MCMC sampling. Uses [emcee](https://emcee.readthedocs.io/) for efficient ensemble sampling.

### `grid.py`
Manages static grid data from AGNI model outputs. Provides interpolation and data access for interior structure calculations.

### `util.py`
Utility functions for common calculations (scale height, dimensionless parameter conversion, LaTeX formatting) and general helpers.

### `plot.py`
Plotting utilities using [matplotlib](https://matplotlib.org/) and [cmcrameri](https://www.fabiocrameri.ch/plotting/) colormaps for scientific visualization.

### External Libraries

- [emcee](https://emcee.readthedocs.io/) - MCMC sampling
- [exoatlas](https://zkbt.github.io/exoatlas/) - Exoplanet database
- [Click](https://click.palletsprojects.com/) - CLI framework
- [netCDF4](https://unidata.github.io/netcdf4-python/) - Data file format
- [NumPy/SciPy](https://scipy.org/) - Scientific computing

---

Available under GPLv3. Copyright (c) 2026 Harrison Nicholls.

