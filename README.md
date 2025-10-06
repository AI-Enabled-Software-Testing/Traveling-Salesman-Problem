# Heuristic TSP Solver for CSI 5186 Assignment 1

This repository contains the implementation of a Travelling Salesman Problem (TSP) solver for the CSI 5186 – AI-enabled Software Verification and Testing course, Assignment 1, Autumn 2025. 

The project is developed by Fernando Nogueira and Kelvin Mock.

## Project Overview

The aim is to implement a solver for the Travelling Salesman Problem (TSP) using a selection of metaheuristic search algorithms discussed in the lectures or an algorithm researched independently. 

The solver targets TSP instances from TSPLIB, specifically symmetric TSP (TYPE: TSP) with Euclidean distances in 2D (EDGE WEIGHT TYPE: EUC 2D).

Problem instances can be found at http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/. 

The dataset structure is explained at http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp95.pdf.

The program takes a .tsp file as input and produces:
- A single output file named `solution.csv` containing a single column of city (node) indices in the order of the solution.
- The total distance travelled printed to standard output.

For example, for a solution visiting cities 5, 4, 1, 3, 2 with distance 8934.12:
```
> python main.py aaa.tsp
8934.12
> cat solution.csv
5
4
1
3
2
```
Optionally, you could select an algorithm in the third command-line argument. Here are the allowed argument names: 
* SimulatedAnnealing_random
* SimulatedAnnealing_NearestNeighbor
* GeneticAlgorithm_NearestNeighbor
* Baseline_Random

By default, **Simulated Annealing with Nearest Neighbor** search is chosen given its lowest cost.

The submission includes the implementation files and a detailed report (as .pdf) describing the solution, approach, and optimizations implemented.

Note: The submission must be self-contained, with no dependencies on external files. Solvers should work out of the box, with reasonable documentation.

## Dataset Setup

This project includes a dataset setup script that downloads and filters TSP instances from TSPLIB95. The script automatically:

1. Downloads the complete TSPLIB95 dataset
2. Filters for TSP instances with `TYPE: TSP` and `EDGE_WEIGHT_TYPE: EUC_2D`
3. Extracts corresponding optimal tour files (`.opt.tour`) when available
4. Saves all files to the `dataset/` directory

To set up the dataset:
```bash
uv run python setup_dataset.py
```

**Note:** Not every TSP instance has a corresponding optimal tour file available in the source dataset. This is normal and expected - the script will only extract tour files that correspond to valid TSP instances.

## Setup Instructions

This project requires Python 3.12 or later.

### Option 1: Using uv (Recommended)

1. **Install uv**: Follow the installation instructions at https://docs.astral.sh/uv/getting-started/installation/

2. **Clone and Set Up**:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   uv sync
   ```

The `uv run` commands will automatically handle the virtual environment for you.

### Option 2: Using standard Python venv

1. **Clone and Set Up**:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e .
   ```

2. **Run commands**:
   ```bash
   python setup_dataset.py
   python main.py dataset/<filename>.tsp
   ```

## Usage

To run the solver:
```
uv run python main.py <path-to-input.tsp>
```

This will output the total distance to stdout and generate `solution.csv` in the current directory.

For development or testing, use `uv run python` to execute scripts in the project environment.

## Project Structure

The project is organized into several key packages and modules:

### Core Modules
- `main.py`: Main solver script and entry point.
- `setup_dataset.py`: Dataset setup script for downloading and filtering TSP instances from TSPLIB95.

### Package Organization

#### `tsp/` - TSP Core Package
- `model.py`: Core data structures (`City`, `TSPInstance`) and distance calculations.
- `io.py`: TSPLIB file parsing utilities for reading `.tsp` files.

#### `algorithm/` - Algorithm Implementations
- `base.py`: Protocol definitions and base classes for iterative TSP solvers.
- `nearest_neighbor.py`: Nearest neighbor constructive algorithm implementation.
- `random_solver.py`: Random permutation solver for baseline comparison.

### Data and Analysis
- `dataset/`: Directory containing TSP instances and optimal tour files (created by setup script).
- `bench_results/`: Directory for storing benchmark results.
- `tsp_analysis.ipynb`: Jupyter notebook for algorithm analysis and visualization.

### Configuration
- `pyproject.toml`: Project configuration and dependencies.
- `uv.lock`: Locked dependencies for reproducibility.
- `README.md`: This file.

