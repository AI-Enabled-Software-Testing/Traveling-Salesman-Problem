import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from constants import INITIAL_TEMP, COOLING_RATE, POPULATION_SIZE, MUTATION_RATE, CROSSOVER_RATE, ELITISM_COUNT, DATASET_FILENAME, PARALLEL_RUNS, NUM_WORKERS, MAX_SECONDS, N_RUNS, CALIBRATION_TIME, MAX_NORMALIZED_STEPS
from util import find_optimal_tour, exponential_cooling
from algorithm.simulated_annealing import SimulatedAnnealing
from algorithm.genetic_algo import GeneticAlgorithmSolver
from algorithm.nearest_neighbor import NearestNeighbor


def load_tsp_instance(tsp_filename=DATASET_FILENAME):
    tsp_path = Path('dataset') / tsp_filename
    instance, optimal_cost = find_optimal_tour(tsp_path)
    instance_data = {'name': instance.name, 'cities': instance.cities}
    return instance, optimal_cost, instance_data


def create_solvers():
    def create_sa_solver():
        instance, _, _ = load_tsp_instance()
        T0 = INITIAL_TEMP
        cool_rate = COOLING_RATE
        schedule = exponential_cooling(cool_rate)
        return SimulatedAnnealing(instance, T0, schedule)
    
    def create_ga_solver():
        instance, _, _ = load_tsp_instance()
        return GeneticAlgorithmSolver(
            instance,
            population_size=POPULATION_SIZE,
            mutation_rate=MUTATION_RATE,
            crossover_rate=CROSSOVER_RATE,
            elitism_count=ELITISM_COUNT
        )
    
    return {
        'SA_random': create_sa_solver,
        'GA_random': create_ga_solver
    }


def create_plot(title, xlabel, ylabel, figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    # Use /caption in LaTeX instead
    # ax.set_title(title)  
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return fig, ax

def get_nn_initial_route(instance):
    """Compute the initial route using Nearest Neighbor heuristic."""
    nn = NearestNeighbor(instance)
    nn.initialize(None)
    n_cities = len(instance.cities)
    for _ in range(n_cities - 1):
        nn.step()
    return nn.get_route()

def compute_nn_baseline(instance):
    """Compute the baseline cost using Nearest Neighbor."""
    nn = NearestNeighbor(instance)
    nn.initialize(None)
    n_cities = len(instance.cities)
    for _ in range(n_cities - 1):
        nn.step()
    return nn.get_cost()

def align_series(x_lists, y_lists, common_x):
    """Align multiple runs' series data via interpolation and compute mean/std."""
    aligned_y = []
    for x, y in zip(x_lists, y_lists):
        if len(x) > 0 and len(y) > 0:
            interp_y = np.interp(common_x, x, y)
            aligned_y.append(interp_y)
    if not aligned_y:
        return np.array([]), np.array([])
    aligned_y = np.array(aligned_y)
    return np.mean(aligned_y, axis=0), np.std(aligned_y, axis=0)

def save_figure(fig, filepath):
    """Save and show the figure."""
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.show()

def add_optimal_line(ax, optimal_cost):
    """Add the optimal cost horizontal line to the axis."""
    ax.axhline(y=optimal_cost, color='green', linestyle=':', label='Optimal', alpha=0.7)

ALGO_COLORS = {
    'SA': 'blue',
    'GA': 'red'
}

ALGO_LINESTYLES = {
    'SA': '-',
    'GA': '--'
}

def run_parallel_trials(trial_func, args_list, parallel=PARALLEL_RUNS, max_workers=NUM_WORKERS, desc="Running trials"):
    """Run trials in parallel or sequentially."""
    if parallel:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(trial_func, arg) for arg in args_list]
            return [f.result() for f in tqdm(as_completed(futures), total=len(args_list), desc=desc)]
    else:
        return [trial_func(arg) for arg in tqdm(args_list, desc=desc)]


def compute_statistics(data, optimal_cost=None):
    """Compute comprehensive statistics for a dataset."""
    if not data or len(data) == 0:
        return {}
    
    data = np.array(data)
    stats = {
        'count': len(data),
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'median': float(np.median(data)),
        'q1': float(np.percentile(data, 25)),
        'q3': float(np.percentile(data, 75)),
        'iqr': float(np.percentile(data, 75) - np.percentile(data, 25))
    }
    
    if optimal_cost and optimal_cost > 0:
        stats['gap_to_optimal_pct'] = float((stats['mean'] / optimal_cost - 1) * 100)
        stats['best_gap_to_optimal_pct'] = float((stats['min'] / optimal_cost - 1) * 100)
        stats['worst_gap_to_optimal_pct'] = float((stats['max'] / optimal_cost - 1) * 100)
    
    return stats


def create_metadata(instance_data, optimal_cost, experiment_type, additional_settings=None):
    """Create standardized metadata for statistics JSON files."""
    base_settings = {
        'max_seconds': MAX_SECONDS,
        'n_runs': N_RUNS,
        'parallel_runs': PARALLEL_RUNS,
        'num_workers': NUM_WORKERS
    }
    
    if additional_settings:
        base_settings.update(additional_settings)
    
    return {
        'generated_at': datetime.now().isoformat(),
        'tsp_instance': instance_data['name'],
        'optimal_cost': optimal_cost,
        'experiment_settings': base_settings
    }


def save_statistics_json(statistics, filename):
    """Save statistics to JSON file with proper formatting."""
    with open(f'figures/{filename}', 'w') as f:
        json.dump(statistics, f, indent=2)
    print(f"Saved figures/{filename}")


def create_box_plot_statistics(instance, optimal_cost, instance_data, costs_by_algo, algorithms):
    """Create statistics structure for box plot experiments."""
    # Compute statistics for each algorithm
    stats_by_algo = {}
    for algo, costs in costs_by_algo.items():
        stats_by_algo[algo] = compute_statistics(costs, optimal_cost)
    
    # Compute NN baseline statistics
    nn_cost = compute_nn_baseline(instance)
    nn_stats = compute_statistics([nn_cost], optimal_cost)
    
    return {
        'metadata': create_metadata(instance_data, optimal_cost, 'box_plot'),
        'algorithms': stats_by_algo,
        'nearest_neighbor_baseline': nn_stats,
        'experiment_config': {
            'max_seconds': MAX_SECONDS,
            'n_runs': N_RUNS,
            'algorithms': algorithms
        }
    }


def create_time_budget_statistics(instance_data, optimal_cost, stats_by_algo, algorithms, use_nn=False):
    """Create statistics structure for time budget experiments."""
    return {
        'metadata': create_metadata(instance_data, optimal_cost, 'time_budget'),
        'algorithms': stats_by_algo,
        'experiment_config': {
            'max_seconds': MAX_SECONDS,
            'n_runs': N_RUNS,
            'algorithms': algorithms,
            'use_nn_initialization': use_nn
        }
    }


def create_relative_work_statistics(instance_data, optimal_cost, stats_by_algo, algorithms, calibration_data, use_nn=False):
    """Create statistics structure for relative work experiments."""
    return {
        'metadata': create_metadata(instance_data, optimal_cost, 'relative_work', {
            'calibration_time': CALIBRATION_TIME,
            'max_normalized_steps': MAX_NORMALIZED_STEPS
        }),
        'algorithms': stats_by_algo,
        'calibration': calibration_data,
        'experiment_config': {
            'max_normalized_steps': MAX_NORMALIZED_STEPS,
            'n_runs': N_RUNS,
            'algorithms': algorithms,
            'use_nn_initialization': use_nn
        }
    }
