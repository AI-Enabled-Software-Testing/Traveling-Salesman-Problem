import pandas as pd # for csv output

# Utilities
from pathlib import Path
import time
import matplotlib.pyplot as plt
import random
from constants import *
from util import *
import sys
import logging
from multiprocessing import Pool, cpu_count
import numpy as np
import os

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Based on the analysis from `tsp_analysis.ipynb`,
# we believe the best default algorithm to showcase is:
ALGORITHM = "SimulatedAnnealing_NearestNeighbor"

def verify_args() -> Path:
    # Check if TSP file path is provided as command line argument
    if len(sys.argv) not in (2,3):
        raise ValueError("[ERROR] Usage: python main.py <tsp_file_path> <(OPTIONAL) algorithm name>\nExample: python main.py aaa.tsp SimulatedAnnealing_random")
    
    tsp_file_path = sys.argv[1]
    
    # Validate file exists and has .tsp extension
    tsp_path = Path(tsp_file_path)
    if not tsp_path.exists():
        raise FileNotFoundError(f"[ERROR] File '{tsp_file_path}' not found")
    
    if not tsp_file_path.lower().endswith('.tsp'):
        raise ValueError(f"[ERROR] File must have .tsp extension, got '{tsp_file_path}'")

    return tsp_path

def print_summary(results: dict, optimal_cost: float):
    # Summary Table
    print("\n" + "=" * 85)
    print("SUMMARY")
    print("=" * 85)
    print(f"{'Algorithm':<20} {'Mean Cost':<18} {'vs Optimal':<12} {'Steps/sec':<12} {'CV %':<8}")
    print("-" * 85)

    for name in results.keys():
        mean_cost = results[name]['final_cost_mean']
        std_cost = results[name]['final_cost_std']
        cost_str = f"{mean_cost:.1f} ± {std_cost:.1f}"
        vs_optimal = f"+{((mean_cost / optimal_cost - 1) * 100):.1f}%" if optimal_cost else "N/A"
        
        # Handle different types of stats - some have total_iterations, some don't
        if 'total_iterations' in results[name]:
            steps_per_sec = results[name]['total_iterations']
        elif 'total_time_mean' in results[name]:
            # For iteration-based results, estimate steps per second
            steps_per_sec = 1000 / results[name]['total_time_mean'] if results[name]['total_time_mean'] > 0 else 0
        else:
            steps_per_sec = 0
            
        cv = (std_cost / mean_cost * 100)
        print(f"{name:<20} {cost_str:<18} {vs_optimal:<12} {steps_per_sec:<12.0f} {cv:<8.1f}")

    print(f"{'Optimal':<20} {optimal_cost:<18.2f} {'0.0%':<12} {'-':<12} {'-':<8}")
    print("=" * 85)

def main(algorithm: str = ALGORITHM):
    tsp_file_path = verify_args() # Exception will be raised if invalid

    print(f"[INFO] Processing TSP file: {tsp_file_path}")
    
    # Load TSP instance
    instance, optimal_cost = find_optimal_tour(tsp_file_path)

    # Prepare instance data for serialization
    instance_data = {'name': instance.name, 'cities': instance.cities}
    seed_identity = list(range(len(instance.cities)))

    # Setup
    solver, init_route = setup_algorithm(algorithm, instance)
    args_list = [(algorithm, run_idx, instance_data, init_route, seed_identity) for run_idx in range(N_RUNS)]

    # Time-based benchmark with multiple runs (parallelized)
    print("=" * 70)
    print(f"TIME-BASED BENCHMARK ({MAX_SECONDS}s per algorithm, {N_RUNS} runs, {cpu_count()} CPUs)")
    print("=" * 70)

    # Run in parallel
    with Pool(processes=min(cpu_count(), N_RUNS)) as pool:
        all_runs = pool.map(run_single_trial_by_timing, args_list)

    # Align all runs to common time grid using interpolation
    # Find the run that reached closest to MAX_SECONDS
    max_time_reached = max(run['times'][-1] for run in all_runs)

    # Create uniform time grid from 0 to the maximum time reached
    num_points = 100  # Use 100 points for smooth curves
    common_times = np.linspace(0, min(max_time_reached, MAX_SECONDS), num_points)
    
    # Interpolate each run onto common time grid
    aligned_best = []
    for run in all_runs:
        # Interpolate best costs onto common time grid
        interp_best = np.interp(common_times, run['times'], run['best_costs'])
        aligned_best.append(interp_best)
    
    aligned_best = np.array(aligned_best)
    aligned_times = common_times

    # Calculate statistics
    mean_best = np.mean(aligned_best, axis=0)
    std_best = np.std(aligned_best, axis=0)
    min_best = np.min(aligned_best, axis=0)
    max_best = np.max(aligned_best, axis=0)
    final_costs = [run['best_costs'][-1] for run in all_runs]

    # Run an Algorithm (by time)
    print(f"Running {algorithm} for {MAX_SECONDS} second(s)...")
    time_results: dict = {}
    time_results[algorithm] = {
        'times': aligned_times,
        'mean_best': mean_best,
        'std_best': std_best,
        'min_best': min_best,
        'max_best': max_best,
        'all_runs_best': aligned_best,
        'final_cost_mean': np.mean(final_costs),
        'final_cost_std': np.std(final_costs),
        'final_cost_min': np.min(final_costs),
        'final_cost_max': np.max(final_costs),
        'total_iterations': np.mean([len(run['iterations']) for run in all_runs])
    }
    logger.info(f"Mean: {time_results[algorithm]['final_cost_mean']:.2f} ± {time_results[algorithm]['final_cost_std']:.2f}")

    # Iteration-based benchmark with multiple runs (parallelized)
    print("\n" + "=" * 70)
    print(f"ITERATION-BASED BENCHMARK ({MAX_ITERATIONS} iterations, {N_RUNS} runs)")
    print("=" * 70)

    logger.info(f"Running {algorithm}... BY ITERATION ({N_RUNS} runs in parallel)")

    # Run in parallel
    with Pool(processes=min(cpu_count(), N_RUNS)) as pool:
        results = pool.map(run_single_iteration_trial, args_list)
    
    runs_best = [r['best_costs'] for r in results]
    runs_iters = [r['iters'] for r in results]
    runs_time = [r['times'] for r in results]

    # Align by min length
    min_len = min(len(x) for x in runs_best)
    aligned_best = [run[:min_len] for run in runs_best]
    aligned_iters = runs_iters[0][:min_len]

    mean_best = np.mean(aligned_best, axis=0)
    std_best = np.std(aligned_best, axis=0)
    final_costs = [run[-1] for run in runs_best if run]
    total_times = [t[-1] for t in runs_time if t]

    iteration_stats: dict = {}
    iteration_stats[algorithm] = {
        'iterations': aligned_iters,
        'mean_best': mean_best,
        'std_best': std_best,
        'final_cost_mean': np.mean(final_costs) if final_costs else float('inf'),
        'final_cost_std': np.std(final_costs) if final_costs else 0.0,
        'total_time_mean': np.mean(total_times) if total_times else 0.0,
    }
    logger.info(f"Cost: {iteration_stats[algorithm]['final_cost_mean']:.2f} ± {iteration_stats[algorithm]['final_cost_std']:.2f}")

    # Summary Statistics
    print_summary(
        {
            "time-based results": time_results[algorithm],
            "iteration-based results": iteration_stats[algorithm],
        }, optimal_cost
    )

    # Find and print the optimal route from the best run
    print("\n" + "=" * 40)
    print("OPTIMAL ROUTE FOUND")
    print("=" * 40)
    
    # Find the best route from BOTH time-based and iteration-based results
    best_cost = float('inf')
    best_route = None
    best_source = None
    
    # Check time-based results
    for i, run in enumerate(all_runs):
        final_cost = run['best_costs'][-1] if run['best_costs'] else float('inf')
        if final_cost < best_cost:
            best_cost = final_cost
            best_route = run['best_route']
            best_source = f"time-based run {i+1}"
    
    # Check iteration-based results
    for i, result in enumerate(results):
        final_cost = result['best_costs'][-1] if result['best_costs'] else float('inf')
        if final_cost < best_cost:
            best_cost = final_cost
            best_route = result['best_route']
            best_source = f"iteration-based run {i+1}"
    
    # Output File
    output_csv = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "solution.csv"
    )
    logger.debug(f"cat {os.path.basename(output_csv)}")
    # Print the source and route
    if best_route:
        logger.info(f"Best solution found in: {best_source}")
        logger.info(f"Best cost: {best_cost:.2f}")
        logger.info(f"Distance traveled: {instance.route_cost(best_route):.2f}")
        logger.info("Route:")
        for city in best_route:
            print(city)
        # Print Routes to CSV
        df = pd.DataFrame(best_route)
        df.to_csv(output_csv, index=False, header=False)

    else:
        logger.debug("No route found")


if __name__ == "__main__":
    algorithm = ALGORITHM
    if len(sys.argv) == 3:
        algorithm = sys.argv[2] # Override with custom algorithm
    main(algorithm)