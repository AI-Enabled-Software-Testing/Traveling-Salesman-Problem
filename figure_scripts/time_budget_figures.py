import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from tsp.model import TSPInstance

from constants import MAX_SECONDS, N_RUNS, PARALLEL_RUNS, NUM_WORKERS
from util import run_algorithm_with_timing
from .common import (
    load_tsp_instance, create_solvers, create_plot, run_parallel_trials, 
    align_series, save_figure, ALGO_COLORS, ALGO_LINESTYLES, add_optimal_line,
    create_time_budget_statistics, save_statistics_json, compute_statistics
)
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)





def run_single_time_trial(args):
    name, instance_data = args

    solvers = create_solvers()
    solver_factory = solvers.get(name)
    if solver_factory:
        solver = solver_factory()
        init_route = None
    else:
        raise ValueError(f"Unknown algorithm: {name}")
    
    
    instance = TSPInstance(name=instance_data['name'], cities=instance_data['cities'])
    
    iterations, best_costs, current_costs, times, best_route = run_algorithm_with_timing(
        instance, solver, init_route, MAX_SECONDS
    )
    
    return {
        'name': name,
        'iterations': iterations,
        'best_costs': best_costs,
        'current_costs': current_costs,
        'times': times,
        'best_route': best_route
    }


def main():
    _, optimal_cost, instance_data = load_tsp_instance()
    logger.info(f"Generating time-budget figures for {instance_data['name']} (optimal: {optimal_cost:.2f})")
    
    algorithms = ["SA_random", "GA_random"]
    
    args_list = []
    for name in algorithms:
        for _ in range(N_RUNS):
            args_list.append((name, instance_data))
    
    logger.info(f"Running {len(args_list)} trials {'in parallel' if PARALLEL_RUNS else 'sequentially'}")
    
    all_results = run_parallel_trials(run_single_time_trial, args_list, desc="Running trials")
    
    # Group by algorithm
    results_by_algo = {name: [] for name in algorithms}
    for res in all_results:
        results_by_algo[res['name']].append(res)
    
    # Compute statistics for each algorithm
    stats_by_algo = {}
    for algo_name, algo_results in results_by_algo.items():
        if not algo_results:
            continue
        
        final_costs = [r['best_costs'][-1] if r['best_costs'] else float('inf') for r in algo_results]
        iterations = [len(r['iterations']) if r['iterations'] else 0 for r in algo_results]
        convergence_times = [r['times'][-1] if r['times'] else MAX_SECONDS for r in algo_results]
        
        stats_by_algo[algo_name] = {
            'final_costs': compute_statistics(final_costs, optimal_cost),
            'iterations': compute_statistics(iterations),
            'convergence_times': compute_statistics(convergence_times)
        }
    
    # Create and save statistics
    statistics = create_time_budget_statistics(instance_data, optimal_cost, stats_by_algo, algorithms, use_nn=False)
    save_statistics_json(statistics, 'time_budget_figures.json')
    
    fig, ax = create_plot(
        f'TSP Algorithm Comparison: Time Budget ({MAX_SECONDS}s, {N_RUNS} runs each)',
        'Time (seconds)',
        'Best Cost'
    )
    
    for algo_name in algorithms:
        algo_results = results_by_algo[algo_name]
        
        if not algo_results:
            continue
        
        # Find max time reached across runs
        max_time = max(run['times'][-1] for run in algo_results if run['times'])
        num_points = 100
        common_times = np.linspace(0, min(max_time, MAX_SECONDS), num_points)
        
        # Interpolate each run to common grid
        x_lists = [run['times'] for run in algo_results if run['times'] and run['best_costs']]
        y_lists = [run['best_costs'] for run in algo_results if run['times'] and run['best_costs']]
        mean_best, std_best = align_series(x_lists, y_lists, common_times)
        
        if len(mean_best) == 0:
            continue
        
        base_algo = algo_name.split('_')[0]
        color = ALGO_COLORS[base_algo]
        linestyle = ALGO_LINESTYLES[base_algo]
        
        ax.plot(common_times, mean_best, label=f"{algo_name}", 
                color=color, linestyle=linestyle, linewidth=2)
        ax.fill_between(common_times, mean_best - std_best, mean_best + std_best, 
                        alpha=0.2, color=color)
        
        final_mean = mean_best[-1]
        final_std = std_best[-1]
        gap = ((final_mean / optimal_cost - 1) * 100) if optimal_cost else 0
        logger.info(f"{algo_name}: Final mean cost {final_mean:.2f} ± {final_std:.2f} (gap: {gap:.1f}%)")
    
    add_optimal_line(ax, optimal_cost)
    ax.legend()
    
    save_figure(fig, 'figures/time_budget_figures.png')
    logger.info("Saved figures/time_budget_figures.png")


if __name__ == "__main__":
    main()
