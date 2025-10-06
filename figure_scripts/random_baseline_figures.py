import numpy as np
import matplotlib.pyplot as plt
import logging
from constants import MAX_SECONDS, N_RUNS, PARALLEL_RUNS
from .common import (
    load_tsp_instance, create_solvers, create_plot, get_nn_initial_route, 
    run_parallel_trials, save_figure, create_box_plot_statistics, save_statistics_json
)
from util import run_algorithm_with_timing
from tsp.model import TSPInstance
from algorithm.random_solver import RandomSolver
from algorithm.nearest_neighbor import NearestNeighbor

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_single_baseline_trial(args):
    """Single trial for random baseline comparison."""
    name, instance_data, use_nn = args
    solvers = create_solvers()
    
    if name == 'Random':
        # Use random solver
        instance = TSPInstance(name=instance_data['name'], cities=instance_data['cities'])
        solver = RandomSolver(instance)
        init_route = None
    else:
        # Use existing solvers
        solver_factory = solvers.get(name.replace('_NN', '_random') if use_nn else name)
        if solver_factory:
            solver = solver_factory()
            init_route = None
            if use_nn:
                instance = TSPInstance(name=instance_data['name'], cities=instance_data['cities'])
                init_route = get_nn_initial_route(instance)
        else:
            raise ValueError(f"Unknown algorithm: {name}")
    
    instance = TSPInstance(name=instance_data['name'], cities=instance_data['cities'])
    _, best_costs, _, _, _ = run_algorithm_with_timing(
        instance, solver, init_route, MAX_SECONDS
    )
    return best_costs[-1] if best_costs else float('inf')

def main():
    _, optimal_cost, instance_data = load_tsp_instance()
    logger.info(f"Generating random baseline comparison for {instance_data['name']} (optimal: {optimal_cost:.2f})")
    
    algorithms = ["Random"]
    
    args_list = []
    for name in algorithms:
        use_nn = name.endswith('_NN')
        for _ in range(N_RUNS):
            args_list.append((name, instance_data, use_nn))
    
    logger.info(f"Running {len(args_list)} trials {'in parallel' if PARALLEL_RUNS else 'sequentially'}")
    
    all_final_costs = run_parallel_trials(run_single_baseline_trial, args_list, desc="Running trials")
    
    # Group by algorithm
    costs_by_algo = {name: [] for name in algorithms}
    for i, cost in enumerate(all_final_costs):
        algo_idx = i // N_RUNS
        algo_name = algorithms[algo_idx]
        costs_by_algo[algo_name].append(cost)
    
    # Compute gaps
    gaps_by_algo = {}
    for algo, costs in costs_by_algo.items():
        gaps = [(c / optimal_cost - 1) * 100 for c in costs if c > 0]
        gaps_by_algo[algo] = gaps
        mean_gap = np.mean(gaps)
        std_gap = np.std(gaps)
        logger.info(f"{algo}: Mean gap {mean_gap:.1f}% ± {std_gap:.1f}%")
    
    # Compute NN baseline with random starts
    instance = TSPInstance(name=instance_data['name'], cities=instance_data['cities'])
    nn_costs = []
    for _ in range(N_RUNS):
        nn = NearestNeighbor(instance, seed=None)  # Different seed each time
        nn.initialize(None)
        n_cities = len(instance.cities)
        for _ in range(n_cities - 1):
            nn.step()
        nn_costs.append(nn.get_cost())
    
    nn_gaps = [(c / optimal_cost - 1) * 100 for c in nn_costs if c > 0]
    mean_nn_gap = np.mean(nn_gaps)
    std_nn_gap = np.std(nn_gaps)
    logger.info(f"Nearest Neighbor: Mean gap {mean_nn_gap:.1f}% ± {std_nn_gap:.1f}%")
    
    # Add NN to the algorithms for plotting
    algorithms_with_nn = algorithms + ["Nearest Neighbor"]
    costs_by_algo["Nearest Neighbor"] = nn_costs
    gaps_by_algo["Nearest Neighbor"] = nn_gaps
    
    # Create and save statistics
    statistics = create_box_plot_statistics(instance, optimal_cost, instance_data, costs_by_algo, algorithms_with_nn)
    save_statistics_json(statistics, 'random_baseline_figures.json')
    
    # Plot random baseline
    fig, ax = create_plot(
        f'Random Baseline Performance Distribution (Final Costs after {MAX_SECONDS}s)', 
        'Algorithms', 
        'Gap to Optimal (%)',
        figsize=(6, 8)  # Make it narrower and taller
    )
    
    # Create box plot with custom colors
    box_colors = ['red', 'orange']  # Random, Nearest Neighbor
    box_plot = ax.boxplot([gaps_by_algo[name] for name in algorithms_with_nn], 
                         tick_labels=algorithms_with_nn, patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(box_plot['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add reference lines
    ax.axhline(y=0, color='green', linestyle=':', label='Optimal', linewidth=2)
    
    # Add mean lines for comparison
    random_gaps = gaps_by_algo['Random']
    random_mean = np.mean(random_gaps)
    ax.axhline(y=random_mean, color='red', linestyle='-', alpha=0.5, label='Random', linewidth=2)
    
    nn_gaps = gaps_by_algo['Nearest Neighbor']
    nn_mean = np.mean(nn_gaps)
    ax.axhline(y=nn_mean, color='orange', linestyle='-', alpha=0.5, label='NN', linewidth=2)
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    save_figure(fig, 'figures/random_baseline_figures.png')
    logger.info("Saved figures/random_baseline_figures.png")

if __name__ == "__main__":
    main()
