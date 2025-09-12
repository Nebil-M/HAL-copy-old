# HAL: Hardware-Aware Layout for Quantum Error Correction

[![CI](https://github.com/EQuS/HAL/workflows/CI/badge.svg)](https://github.com/your-username/hal/actions)
[![Code Coverage](https://codecov.io/gh/EQuS/HAL/branch/main/graph/badge.svg)](https://codecov.io/gh/your-username/hal)
[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Abstract

Quantum error correcting codes (QECCs) with asymptotically lower overheads than the surface code require non-local connectivity. Leveraging multi-layer routing and long-range coupling capabilities in superconducting qubit hardware, we develop Hardware-Aware Layout, HAL: a robust, runtime-efficient heuristic algorithm that automates and optimizes the placement and routing of arbitrary QECCs. Using HAL, we perform a comparative study of hardware cost across various families of QECCs, including the bivariate bicycle codes, the open-boundary tile codes, and the constant-depth-decodable radial codes. The layouts produced by HAL confirm that open boundaries significantly reduce the hardware cost, while incurring reductions in logical efficiency. Among the best-performing codes were low-weight radial codes, despite lacking topological structure. Overall, HAL provides a valuable framework for evaluating the hardware feasibility of existing QECCs and guiding the discovery of new codes compatible with realistic hardware constraints.

## Overview

Hardware-Aware Layout (HAL) is a Python library for placing and routing Tanner graphs of arbitrary quantum error correcting codes onto tiered superconducting hardware. HAL provides a comprehensive framework for evaluating the hardware feasibility of quantum error correction codes and optimizing their physical layout.

## Features

- **Hardware-Aware Layout**: Place and route Tanner graphs onto tiered hardware grids
- **Multi-Tier Routing**: Support for complex routing across multiple hardware layers
- **Flexible Configuration**: Comprehensive `Settings` class for customizing placement and routing behavior
- **Multiple Code Families**: Support for comparison of surface codes, toric codes, bicycle codes, tile codes, and radial codes
- **Performance Analysis**: Built-in benchmarking and cost analysis tools
- **Visualization**: Grid and routing visualization capabilities
- **Extensible Architecture**: Modular design for adding new routing algorithms and code types

## Installation

### Prerequisites

- Python 3.12+
- pip

### Basic Installation

```bash
# Clone the repository
git clone <repository-url>
cd HAL

# Install package with runtime dependencies (from pyproject.toml)
pip install .
```

### Development Installation

For development and testing:

```bash
# Install in editable mode with dev dependencies
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Local docs (served at http://127.0.0.1:8000)
mkdocs serve
```

## Quick Start

### Basic Usage

```python
from hal.hal import HardwareAwareLayout
from hal.codes.radial_codes import RadialCode

# Create a radial code
radial_code = RadialCode(r=2, s=2)

# Initialize HAL with default settings
hal = HardwareAwareLayout(
    name="radial_code_example_[[16,2,4]]",
    directory_path=PATH_TO_DATABASE,
    tanner_graph=radial_code.graph
)

# Place and route
hal.place()
hal.route()

# Benchmark results
results = hal.benchmark()
print(results)
```

### Output Structure

When you run HAL with `place()`, `route()`, and `benchmark()` methods, it generates a timestamped folder containing comprehensive results and visualizations. Here's what gets created:

```
yyymmdd_aahbbmccs_radial_code_example/
├── benchmark.csv           # Performance metrics and hardware complexity scores
├── perf.json              # Timing information for placement and routing
├── settings.json          # Complete configuration used for the run
├── tanner                 # Binary Tanner graph representation
├── grid_view/             # Grid visualization outputs
│   ├── tier_grid_0.png    # Visual representation of tier 0 grid
│   ├── tier_grid_1.png    # Visual representation of tier 1 grid
│   └── tier_grids.pdf     # Combined PDF of all tier visualizations
├── tier_interactive_0.png # Interactive visualization for tier 0
├── tier_interactive_0.svg # Interactive SVG for tier 0
├── tier_interactive_1.png # Interactive visualization for tier 1
├── tier_interactive_1.svg # Interactive SVG for tier 1
├── tiers/                 # Detailed tier-specific data
│   ├── expanded_grid_0    # Expanded grid representation for tier 0
│   ├── expanded_grid_1    # Expanded grid representation for tier 1
│   ├── graph_0            # Graph structure for tier 0
│   ├── graph_1            # Graph structure for tier 1
│   ├── grid_0             # Grid layout for tier 0
│   ├── grid_1             # Grid layout for tier 1
│   ├── metrics_0.csv      # Detailed routing metrics for tier 0
│   └── metrics_1.csv      # Detailed routing metrics for tier 1
└── layers/                # Layer-specific routing data (if applicable)
```

- **`benchmark.csv`**: Contains normalized performance metrics including number of tiers, average coupler length, face switches, TSVs per edge, and overall hardware complexity score
- **`perf.json`**: Timing breakdown showing placement time, benchmark time, and total execution time
- **`settings.json`**: Complete snapshot of all configuration parameters used during execution
- **`tier_interactive_*.png/svg`**: High-quality visualizations showing the final routed layout with interactive elements
- **`tiers/metrics_*.csv`**: Detailed per-tier routing statistics including edge routing information, face switches, and routing paths

### Custom Configuration

```python
from hal import Settings

# Create custom settings
settings = Settings(
    grid_size=600,
    node_expansion_val=2,
    edge_expansion_val=1,
    max_bump_transitions_per_coupler=15,
    max_tsvs_per_coupler=3,
    max_coupler_length=50,
    verbose=True
)

# Use custom settings
hal = HardwareAwareLayout(
    name="custom_example",
    directory_path="./output",
    tanner_graph=code.graph,
    settings=settings
)
```

## Configuration with Settings

The `Settings` class provides comprehensive control over HAL's behavior:

### Key Parameters

| Category | Parameter | Description | Default |
|----------|-----------|-------------|---------|
| **Placement** | `grid_size` | Overall grid dimension | 500 |
| | `node_expansion_val` | Node expansion radius | 1 |
| | `place_margin` | Grid normalization margin | 0.02 |
| | `layout` | Initial placement | "community" |
| | `custom_positions` | Final node positions | None |
| **Routing** | `edge_expansion_val` | Edge safety margin | 1 |
| | `max_bump_transitions_per_coupler` | Max bump transitions | 10 |
| | `max_tsvs_per_coupler` | Max TSVs per connection | None |
| | `max_coupler_length` | Max connection length | 1000 |
| | `route_edge_order` | Routing Edge Order | "length_asc" |
| **Benchmarking** | `baseline_defaults` | Baseline thresholds | num_tiers: 1<br>avg_coupler_length: 1<br>max_avg_face_switches: 0<br>avg_tsvs_per_edge: 0 |
| | `bad_defaults` | Bad thresholds | num_tiers: 5<br>avg_coupler_length: 10<br>max_avg_face_switches: 4<br>avg_tsvs_per_edge: 3 |

### Method-Level Overrides

You can override settings for specific operations:

```python
# Override settings for placement
hal.place(
    grid_size=800,
    node_expansion_val=3,
    layout="kamada_kawai"
)

# Override settings for routing
hal.route(
    max_bump_transitions_per_coupler=20,
    edge_expansion_val=2
)

# Override benchmark thresholds
hal.benchmark(
    baseline={"num_tiers": 2.0, "avg_coupler_length": 2.0},
    bad={"num_tiers": 8.0, "avg_coupler_length": 15.0}
)
```

## Development

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=hal --cov-report=html

# Run specific test file
python -m pytest tests/test_settings.py -v
```

### Code Quality

```bash
# Lint with Pylint (config in pyproject.toml)
python -m pylint hal/

# Type checking with MyPy (config in pyproject.toml)
python -m mypy hal/

# Or use Makefile helpers
make lint      # black --check, isort --check, pylint
make typecheck # mypy
make format    # black + isort
```

### Pre-commit Hooks

The project uses pre-commit to ensure code quality. Mypy currently runs as a non-blocking hook on commit/push (it reports issues but will not block your commit). You can run the blocking mypy manually when you’re ready to enforce type checks.

```bash
# Install hooks
pre-commit install

# Run all hooks on the full tree
pre-commit run --all-files

# Run blocking mypy manually (if configured as manual stage)
pre-commit run mypy --all-files || true  # non-zero exit indicates type issues
```

## Documentation

This repository includes an MkDocs site configured with Material theme and mkdocstrings for automatic API reference.

- Build and serve locally:

```bash
mkdocs serve
```

- Site configuration: `mkdocs.yml`
- Content:
  - Overview: `docs/index.md`
  - API pages under `docs/api/**` render Python docstrings via mkdocstrings.

If you intend to publish docs (e.g., GitHub Pages), we can add a small CI workflow to deploy on every push to main.

## Project Structure

```
HAL/
├── hal/                    # Main library code
│   ├── __init__.py
│   ├── hal.py              # Core HardwareAwareLayout class
│   ├── settings.py         # Configuration Settings class
│   ├── tier.py             # Tier representation
│   ├── codes/              # Code implementations
│   │   ├── radial_codes.py
│   │   ├── tile_codes.py
│   │   └── generalized_toric_codes.py
│   ├── routing/            # Routing algorithms
│   │   ├── routing_algo.py
│   │   ├── astar.py
│   │   └── straightline.py
│   ├── placing/            # Placement utilities
│   └── website/            # Web interface
├── tests/                  # Test suite
├── example_notebooks/      # Example notebooks
├── pyproject.toml          # Project configuration
├── .pre-commit-config.yaml # Pre-commit hooks
└── README.md               # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write comprehensive docstrings
- Maintain test coverage above 50%
- Use pre-commit hooks for code quality

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use HAL in your research, please cite our paper:

```bibtex
@article{hal2025,
  title={Placing and Routing Non-Local Quantum Error Correcting Codes in Multi-Layer Superconducting Qubit Hardware},
  author={Melvin Mathews, Lukas Pahl, David Pahl, Vaishnavi L. Addala, Catherine Tang},
  journal={arXiv preprint},
  year={2025},
  url={https://github.com/EQuS/HAL}
}
```
