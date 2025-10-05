import numpy as np
import matplotlib.pyplot as plt
import logging
from constants import MAX_SECONDS, N_RUNS, PARALLEL_RUNS, NUM_WORKERS
from .common import load_tsp_instance, create_solvers
from util import run_algorithm_with_timing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from tsp.model import TSPInstance
from algorithm.nearest_neighbor import NearestNeighbor

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
            nn = NearestNeighbor(instance)
            nn.initialize(None)
            n_cities = len(instance.cities)
            for _ in range(n_cities - 1):
                nn.step()
            init_route = nn.get_route()
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
    
    if PARALLEL_RUNS:
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(run_single_box_trial, arg) for arg in args_list]
            all_final_costs = [f.result() for f in tqdm(as_completed(futures), total=len(args_list), desc="Running trials")]
    else:
        all_final_costs = []
        for arg in tqdm(args_list, desc="Running trials"):
            final_cost = run_single_box_trial(arg)
            all_final_costs.append(final_cost)
    
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
    nn = NearestNeighbor(instance)
    nn.initialize(None)
    n_cities = len(instance.cities)
    for _ in range(n_cities - 1):
        nn.step()
    nn_cost = nn.get_cost()
    nn_gap = ((nn_cost / optimal_cost - 1) * 100) if optimal_cost else 0
    logger.info("Nearest Neighbor")

    # Plot box plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot([gaps_by_algo[name] for name in algorithms], tick_labels=algorithms)
    ax.set_ylabel('Gap to Optimal (%)')
    ax.set_title(f'Algorithm Performance Distribution (Final Costs after {MAX_SECONDS}s)')
    ax.axhline(y=0, color='green', linestyle=':', label='Optimal')
    ax.axhline(y=nn_gap, color='orange', linestyle='--', label=f'NN Baseline ({nn_gap:.1f}%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/box_plot_figures.png', dpi=300, bbox_inches='tight')
    plt.show()
    logger.info("Saved figures/box_plot_figures.png")

if __name__ == "__main__":
    main()
