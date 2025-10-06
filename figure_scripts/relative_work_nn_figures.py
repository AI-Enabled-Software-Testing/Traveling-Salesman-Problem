import time
import math
import logging
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime
from constants import CALIBRATION_TIME, MAX_NORMALIZED_STEPS, N_RUNS, PARALLEL_RUNS, NUM_WORKERS
from .common import load_tsp_instance, create_solvers, create_plot, get_nn_initial_route, align_series, save_figure, ALGO_COLORS, add_optimal_line, compute_nn_baseline
from concurrent.futures import ProcessPoolExecutor, as_completed
from tsp.model import TSPInstance

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

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


def calibrate_steps_per_second(create_solver_func, initial_route=None, calibration_time=CALIBRATION_TIME):
    solver = create_solver_func()
    solver.initialize(initial_route)
    
    start_time = time.perf_counter()
    step_count = 0
    
    while time.perf_counter() - start_time < calibration_time:
        solver.step()
        step_count += 1
    
    actual_time = time.perf_counter() - start_time
    return step_count / actual_time


def run_single_benchmark(create_solver_func, initial_route, work_factor, max_normalized_steps=MAX_NORMALIZED_STEPS):
    solver = create_solver_func()
    solver.initialize(initial_route)
    
    max_actual_steps = max(1, math.floor(max_normalized_steps / work_factor))
    
    results = []
    for step in range(max_actual_steps):
        solver.step()
        normalized_step = (step + 1) * work_factor
        fitness = solver.get_cost()  # Get current best cost (fitness)
        results.append((normalized_step, fitness))
    
    return results


def worker_benchmark_nn(display_name, orig_key, work_factor):
    """Worker for parallel NN benchmarking - computes initial_route locally."""
    from figure_scripts.relative_work_nn_figures import run_single_benchmark
    from .common import load_tsp_instance, create_solvers, get_nn_initial_route
    _, _, instance_data = load_tsp_instance()
    instance = TSPInstance(name=instance_data['name'], cities=instance_data['cities'])
    initial_route = get_nn_initial_route(instance)  # Compute locally
    solvers = create_solvers()
    create_func = solvers[orig_key]
    result = run_single_benchmark(create_func, initial_route, work_factor)
    return (display_name, result)


def main():
    instance, optimal_cost, instance_data = load_tsp_instance()
    initial_route = get_nn_initial_route(instance)
    logger.info(f"Generating relative work figures for {instance_data['name']} (optimal: {optimal_cost:.2f}) with NN init")
    
    # Calibration phase (single run for speed)
    solvers = create_solvers()
    steps_per_second = {}
    for algo_name, create_func in solvers.items():
        steps_per_second[algo_name] = calibrate_steps_per_second(create_func, initial_route)
    
    logger.info(f"Steps per second: {steps_per_second}")

    # Find the algorithm with the most steps per second (least work per step)
    reference_algo = max(steps_per_second, key=steps_per_second.get)
    reference_steps_per_second = steps_per_second[reference_algo]

    logger.info(f"Reference algorithm (fastest): {reference_algo} with {reference_steps_per_second:.1f} steps/sec")
    
    # Calculate normalization factors (work per step relative to reference algorithm)
    work_per_step = {}
    for algo_name in solvers:
        work_per_step[algo_name] = reference_steps_per_second / steps_per_second[algo_name]
    
    logger.info(f"Work per step: {work_per_step}")
    
    algo_configs = [
        ('SA_NN', 'SA_random'),
        ('GA_NN', 'GA_random')
    ]
    
    args_list = []
    for display_name, orig_key in algo_configs:
        work_factor = work_per_step[orig_key]
        for _ in range(N_RUNS):
            args_list.append((display_name, orig_key, work_factor))
    
    logger.info(f"Running {len(args_list)} trials {'in parallel' if PARALLEL_RUNS else 'sequentially'}")
    
    if PARALLEL_RUNS:
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(worker_benchmark_nn, *arg) for arg in args_list]
            all_results_list = [f.result() for f in tqdm(as_completed(futures), total=len(args_list), desc="Running trials")]
    else:
        all_results_list = []
        # For sequential, use the original initial_route
        for arg in tqdm(args_list, desc="Running trials"):
            display_name, orig_key, work_factor = arg
            create_func = solvers[orig_key]
            result = run_single_benchmark(create_func, initial_route, work_factor)
            all_results_list.append((display_name, result))
    
    all_results = {display: [] for display, _ in algo_configs}
    for display_name, result in all_results_list:
        all_results[display_name].append(result)
    
    # Compute statistics for final costs
    stats_by_algo = {}
    for algo_name, algo_runs in all_results.items():
        if not algo_runs:
            continue
        
        final_costs = []
        for run_data in algo_runs:
            if run_data:
                final_costs.append(run_data[-1][1])  # Last cost in the run
        
        stats_by_algo[algo_name] = compute_statistics(final_costs, optimal_cost)
    
    # Save statistics to JSON
    statistics = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'tsp_instance': instance_data['name'],
            'optimal_cost': optimal_cost,
            'experiment_settings': {
                'calibration_time': CALIBRATION_TIME,
                'max_normalized_steps': MAX_NORMALIZED_STEPS,
                'n_runs': N_RUNS,
                'parallel_runs': PARALLEL_RUNS,
                'num_workers': NUM_WORKERS
            }
        },
        'algorithms': stats_by_algo,
        'calibration': {
            'steps_per_second': steps_per_second,
            'reference_algorithm': reference_algo,
            'work_per_step': work_per_step
        },
        'experiment_config': {
            'max_normalized_steps': MAX_NORMALIZED_STEPS,
            'n_runs': N_RUNS,
            'algorithms': [display for display, _ in algo_configs],
            'use_nn_initialization': True
        }
    }
    
    with open('figures/relative_work_nn_figures.json', 'w') as f:
        json.dump(statistics, f, indent=2)
    logger.info("Saved figures/relative_work_nn_figures.json")
    
    # Plot
    fig, ax = create_plot(
        "Algorithm Performance vs Normalized Work with NN init",
        f"Normalized Steps (Reference: {reference_algo})",
        "Best Cost"
    )
    
    num_points = 100
    common_norm_steps = np.linspace(1, MAX_NORMALIZED_STEPS, num_points)
    
    for algo_name, algo_runs in all_results.items():
        if not algo_runs:
            continue
        
        # Interpolate each run to common normalized steps grid
        x_lists = [[point[0] for point in run_data] for run_data in algo_runs]
        y_lists = [[point[1] for point in run_data] for run_data in algo_runs]
        mean_best, std_best = align_series(x_lists, y_lists, common_norm_steps)
        
        if len(mean_best) == 0:
            continue
        
        base_algo = algo_name.split('_')[0]
        color = ALGO_COLORS[base_algo]
        
        ax.plot(common_norm_steps, mean_best, label=f"{algo_name}", 
                color=color, linewidth=2)
        ax.fill_between(common_norm_steps, mean_best - std_best, mean_best + std_best, 
                        alpha=0.2, color=color)
        
        final_mean = mean_best[-1]
        final_std = std_best[-1]
        logger.info(f"{algo_name}: Final mean cost {final_mean:.2f} ± {final_std:.2f}")
    
    add_optimal_line(ax, optimal_cost)
    
    # Add NN baseline line
    instance = TSPInstance(name=instance_data['name'], cities=instance_data['cities'])
    nn_cost = compute_nn_baseline(instance)
    ax.axhline(y=nn_cost, color='orange', linestyle='--', label='NN', alpha=0.7, linewidth=2)
    
    ax.legend()
    
    save_figure(fig, 'figures/relative_work_nn_figures.png')
    logger.info("Saved figures/relative_work_nn_figures.png")


if __name__ == "__main__":
    main()
