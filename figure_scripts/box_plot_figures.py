import numpy as np
import matplotlib.pyplot as plt
import logging
from constants import MAX_SECONDS, N_RUNS, PARALLEL_RUNS, NUM_WORKERS
from .common import (
    load_tsp_instance, create_solvers, create_plot, get_nn_initial_route,
    compute_nn_baseline, run_parallel_trials, save_figure, ALGO_COLORS,
    create_box_plot_statistics, save_statistics_json, compute_statistics
)
from util import run_algorithm_with_timing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from tsp.model import TSPInstance

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_single_box_trial(args):
    name, instance_data, use_nn = args
    solvers = create_solvers()
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
    logger.info(f"Generating box plots for lin105 (optimal: {optimal_cost:.2f})")
    
    algorithms = ["SA_random", "GA_random", "SA_NN", "GA_NN"]
    
    args_list = []
    for name in algorithms:
        use_nn = name.endswith('_NN')
        for _ in range(N_RUNS):
            args_list.append((name, instance_data, use_nn))
    
    logger.info(f"Running {len(args_list)} trials {'in parallel' if PARALLEL_RUNS else 'sequentially'}")
    
    all_final_costs = run_parallel_trials(run_single_box_trial, args_list, desc="Running trials")
    
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
    
    # Compute NN baseline
    instance = TSPInstance(name=instance_data['name'], cities=instance_data['cities'])
    nn_cost = compute_nn_baseline(instance)
    nn_gap = ((nn_cost / optimal_cost - 1) * 100) if optimal_cost else 0
    logger.info("Nearest Neighbor")
    
    # Create and save statistics
    statistics = create_box_plot_statistics(instance, optimal_cost, instance_data, costs_by_algo, algorithms)
    save_statistics_json(statistics, 'box_plot_figures.json')
    
    # Plot box plot
    _, ax = create_plot(f'Algorithm Performance Distribution (Final Costs after {MAX_SECONDS}s)', 'Algorithms', 'Gap to Optimal (%)')
    ax.boxplot([gaps_by_algo[name] for name in algorithms], tick_labels=algorithms)
    ax.axhline(y=0, color='green', linestyle=':', label='Optimal')
    ax.axhline(y=nn_gap, color='orange', linestyle='--', label='NN')
    ax.legend()
    
    save_figure(plt.gcf(), 'figures/box_plot_figures.png')
    logger.info("Saved figures/box_plot_figures.png")

if __name__ == "__main__":
    main()
