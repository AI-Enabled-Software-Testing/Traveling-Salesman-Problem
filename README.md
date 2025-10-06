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
> python tsp_solver.py aaa.tsp
8934.12
> cat solution.csv
5
4
1
3
2
```

The submission includes the implementation files and a detailed report (as .pdf) describing the solution, approach, and optimizations implemented.

## Dataset Setup

This project includes a dataset setup script that downloads and filters TSP instances from TSPLIB95.

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
   source .venv/bin/activate
   pip install -e .
   ```

2. **Run commands**:
   ```bash
   python setup_dataset.py
   python tsp_solver.py dataset/<filename>.tsp
   ```

## Usage

To run the solver:

```
uv run python tsp_solver.py <problem.tsp>
```

This will output the total distance to stdout and generate `solution.csv` in the current directory.

For development or testing, use `uv run python` to execute scripts in the project environment.

## Generating Figures

To generate performance figures for the TSP algorithms, you can run individual scripts from the `figure_scripts/` directory.

```bash
uv run python -m figure_scripts.box_plot_figures        
uv run python -m figure_scripts.relative_work_figures   
uv run python -m figure_scripts.relative_work_nn_figures 
uv run python -m figure_scripts.time_budget_figures     
uv run python -m figure_scripts.time_budget_nn_figures 
```

To generate all figures at once:

```bash
uv run python generate_figures.py
```

Ensure the dataset is set up (run `uv run python setup_dataset.py` if not already done).

## Hyperparam Tuning

The `tuning/` directory contains scripts for hyperparameter tuning of  GA and SA. These tune parameters over a fixed time budget on the lin105.tsp instance.

```bash
uv run python -m tuning.ga_tuning
uv run python -m tuning.sa_tuning
```

The console output should include output of the best params.

## TSP Analysis Notebook

You can compile the notebook to PDF by running:

```
uv run jupyter nbconvert --to pdf tsp_analysis.ipynb
```

## Project Structure

The project is organized into directories for core functionality, algorithms, data handling, and analysis:

### Core Modules
- Entry point and utilities: `main.py`, `setup_dataset.py`, `generate_figures.py`, `constants.py`, `util.py`.

### Packages
- `tsp/`: Core TSP model and I/O.
- `algorithm/`: Heuristic algorithm implementations (e.g., genetic, simulated annealing, nearest neighbor and random solver).
- `figure_scripts/`: Scripts for generating performance visualizations (e.g., box plots, time budgets, relative work comparisons).
- `tuning/`: Hyperparameter tuning scripts.
- `tests/`: Tests.

### Data and Outputs
- `dataset/`: TSP instances and optimal tours.
- `figures/`: Generated plots.
- `solution.csv`: Solver output file.
