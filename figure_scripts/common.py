import matplotlib.pyplot as plt
from pathlib import Path
from constants import INITIAL_TEMP, COOLING_RATE, POPULATION_SIZE, MUTATION_RATE, CROSSOVER_RATE, ELITISM_COUNT, DATASET_FILENAME, PARALLEL_RUNS, NUM_WORKERS
from util import find_optimal_tour, exponential_cooling
from algorithm.simulated_annealing import SimulatedAnnealing
from algorithm.genetic_algo import GeneticAlgorithmSolver
import numpy as np
from tsp.model import TSPInstance
from algorithm.nearest_neighbor import NearestNeighbor
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


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
    ax.set_title(title)
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
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
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
