import time
import math
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import numpy as np
from constants import CALIBRATION_TIME, MAX_NORMALIZED_STEPS, N_RUNS, PARALLEL_RUNS, NUM_WORKERS
from .common import load_tsp_instance, create_solvers, create_plot
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)



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


def worker_benchmark(algo_name, work_factor):
    """Worker function for parallel benchmarking - imports locally to avoid pickling."""
    from figure_scripts.relative_work_figures import run_single_benchmark
    from .common import create_solvers
    solvers = create_solvers()
    create_func = solvers[algo_name]
    result = run_single_benchmark(create_func, None, work_factor)
    return (algo_name, result)


def main():
    _, optimal_cost, instance_data = load_tsp_instance()
    logger.info(f"Generating relative work figures for {instance_data['name']} (optimal: {optimal_cost:.2f})")
    
    # Calibration phase (single run for speed)
    solvers = create_solvers()
    steps_per_second = {}
    for algo_name, create_func in solvers.items():
        steps_per_second[algo_name] = calibrate_steps_per_second(create_func)
    
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

    args_list = []
    for algo_name in solvers:
        for _ in range(N_RUNS):
            args_list.append((algo_name, work_per_step[algo_name]))

    logger.info(f"Running {len(args_list)} trials {'in parallel' if PARALLEL_RUNS else 'sequentially'}")

    if PARALLEL_RUNS:
        adjusted_args = [(algo_name, work_per_step[algo_name]) for algo_name in solvers for _ in range(N_RUNS)]
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(worker_benchmark, *arg) for arg in adjusted_args]
            all_results_list = [f.result() for f in tqdm(as_completed(futures), total=len(adjusted_args), desc="Running trials")]
    else:
        all_results_list = []
        for arg in tqdm(args_list, desc="Running trials"):
            algo_name, work_factor = arg
            create_func = solvers[algo_name]
            result = run_single_benchmark(create_func, None, work_factor)
            all_results_list.append((algo_name, result))

    all_results = {name: [] for name in solvers}
    for algo_name, result in all_results_list:
        all_results[algo_name].append(result)

    _, ax = create_plot(
        "Algorithm Performance vs Normalized Work",
        f"Normalized Steps (Reference: {reference_algo})",
        "Best Cost"
    )
    
    colors = {'SA_random': 'blue', 'GA_random': 'red'}
    
    num_points = 100
    common_norm_steps = np.linspace(1, MAX_NORMALIZED_STEPS, num_points)
    
    for algo_name, algo_runs in all_results.items():
        if not algo_runs:
            continue
        
        # Interpolate each run to common normalized steps grid
        aligned_best = []
        for run_data in algo_runs:
            norm_steps = [point[0] for point in run_data]
            costs = [point[1] for point in run_data]
            if len(norm_steps) > 0:
                interp_cost = np.interp(common_norm_steps, norm_steps, costs)
                aligned_best.append(interp_cost)
        
        if not aligned_best:
            continue
        
        aligned_best = np.array(aligned_best)
        
        mean_best = np.mean(aligned_best, axis=0)
        std_best = np.std(aligned_best, axis=0)
        
        ax.plot(common_norm_steps, mean_best, label=f"{algo_name}", 
                color=colors[algo_name], linewidth=2)
        ax.fill_between(common_norm_steps, mean_best - std_best, mean_best + std_best, 
                        alpha=0.2, color=colors[algo_name])
        
        final_mean = mean_best[-1]
        final_std = std_best[-1]
        logger.info(f"{algo_name}: Final mean cost {final_mean:.2f} ± {final_std:.2f}")

    ax.axhline(y=optimal_cost, color='green', linestyle=':', label='Optimal', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('figures/relative_work_figures.png', dpi=300, bbox_inches='tight')
    logger.info("Saved figures/relative_work_figures.png")


if __name__ == "__main__":
    main()
