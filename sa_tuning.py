#!/usr/bin/env python3
"""
SA Hyperparameter Tuning using scikit-optimize

This script optimizes SA parameters to minimize route cost over 5 seconds.
"""

import numpy as np
from pathlib import Path
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import time
from numba import jit, prange

from util import find_optimal_tour, exponential_cooling
from algorithm.simulated_annealing import SimulatedAnnealing


@jit(nopython=True)
def jit_calculate_route_cost(route, cities_coords):
    """JIT-compiled route cost calculation."""
    n = len(route)
    total_cost = 0.0
    
    for i in range(n):
        current_city = route[i]
        next_city = route[(i + 1) % n]
        
        # Calculate Euclidean distance
        dx = cities_coords[current_city, 0] - cities_coords[next_city, 0]
        dy = cities_coords[current_city, 1] - cities_coords[next_city, 1]
        total_cost += np.sqrt(dx * dx + dy * dy)
    
    return total_cost


@jit(nopython=True)
def jit_2opt_neighbor(route, i, j):
    """JIT-compiled 2-opt neighbor generation."""
    n = len(route)
    neighbor = np.empty(n, dtype=np.int32)
    
    # Copy route
    for k in range(n):
        neighbor[k] = route[k]
    
    # Reverse segment between i and j
    for k in range(i, j + 1):
        neighbor[k] = route[j - (k - i)]
    
    return neighbor


@jit(nopython=True, parallel=True)
def jit_batch_route_costs(routes, cities_coords):
    """JIT-compiled batch route cost calculation."""
    n_routes, n_cities = routes.shape
    costs = np.zeros(n_routes, dtype=np.float64)
    
    for route_idx in prange(n_routes):
        total_cost = 0.0
        for i in range(n_cities):
            current_city = routes[route_idx, i]
            next_city = routes[route_idx, (i + 1) % n_cities]
            
            # Calculate Euclidean distance
            dx = cities_coords[current_city, 0] - cities_coords[next_city, 0]
            dy = cities_coords[current_city, 1] - cities_coords[next_city, 1]
            total_cost += np.sqrt(dx * dx + dy * dy)
        
        costs[route_idx] = total_cost
    
    return costs


@jit(nopython=True)
def jit_sa_optimized(cities_coords, initial_temp, cooling_rate, max_steps, seed):
    """JIT-compiled SA algorithm with 2-opt moves."""
    n = cities_coords.shape[0]
    
    # Initialize route
    route = np.arange(n, dtype=np.int32)
    # Simple shuffle using linear congruential generator
    rng_state = seed
    for i in range(n):
        j = rng_state % n
        route[i], route[j] = route[j], route[i]
        rng_state = (rng_state * 1664525 + 1013904223) % (2**32)
    
    # Initialize best and current cost
    current_cost = jit_calculate_route_cost(route, cities_coords)
    best_cost = current_cost
    best_route = route.copy()
    temperature = initial_temp
    
    # SA loop
    for step in range(max_steps):
        # Generate random 2-opt move
        rng_state = (rng_state * 1664525 + 1013904223) % (2**32)
        i = int(rng_state % n)
        rng_state = (rng_state * 1664525 + 1013904223) % (2**32)
        j = int(rng_state % n)
        
        if i > j:
            i, j = j, i
        if i == j:
            j = (j + 1) % n
        
        # Create neighbor
        neighbor = jit_2opt_neighbor(route, i, j)
        
        # Calculate neighbor cost only
        neighbor_cost = jit_calculate_route_cost(neighbor, cities_coords)
        
        # Accept or reject
        delta = neighbor_cost - current_cost
        rng_state = (rng_state * 1664525 + 1013904223) % (2**32)
        random_val = (rng_state % 1000) / 1000.0
        
        if delta < 0 or np.exp(-delta / temperature) > random_val:
            # Accept neighbor
            for k in range(n):
                route[k] = neighbor[k]
            current_cost = neighbor_cost
        
        # Update best if improved
        if current_cost < best_cost:
            best_cost = current_cost
            for k in range(n):
                best_route[k] = route[k]
        
        # Cool down
        temperature *= cooling_rate
    
    return best_cost, best_route


def run_single_sa_optimized(params, instance, time_budget_seconds, seed):
    """Fully JIT-compiled SA run with time budget."""
    # Pre-compute cities coordinates for JIT
    cities_coords = np.array([[city.x, city.y] for city in instance.cities], dtype=np.float64)
    
    # Estimate steps based on previous performance (~50k steps/second)
    estimated_steps = int(time_budget_seconds * 50000)
    
    # Run fully JIT-compiled SA
    start_time = time.perf_counter()
    best_cost, best_route = jit_sa_optimized(
        cities_coords, 
        params['initial_temp'], 
        params['cooling_rate'], 
        estimated_steps, 
        seed
    )
    actual_time = time.perf_counter() - start_time
    
    # Calculate actual steps per second
    actual_steps = int(estimated_steps * (time_budget_seconds / actual_time))
    
    return best_cost, actual_steps


def evaluate_sa_params(params, instance, optimal_cost, time_budget_seconds=20.0, n_runs=2):
    """
    Evaluate SA parameters by running multiple times sequentially with time budget.
    
    Args:
        params: Dictionary of SA parameters
        instance: TSP instance
        optimal_cost: Optimal cost for reference
        time_budget_seconds: Time budget per evaluation in seconds
        n_runs: Number of runs to average
    
    Returns:
        Average final cost across runs
    """
    # Validate parameter combinations
    initial_temp = params['initial_temp']
    cooling_rate = params['cooling_rate']
    
    # Ensure reasonable parameter ranges
    if initial_temp <= 0 or initial_temp > 1000:
        return 100000.0  # Large penalty for invalid temperature
    
    if cooling_rate <= 0 or cooling_rate >= 1:
        return 100000.0  # Large penalty for invalid cooling rate
    
    # Run sequentially (no parallel processing)
    costs = []
    steps = []
    
    for run in range(n_runs):
        cost, step_count = run_single_sa_optimized(params, instance, time_budget_seconds, 42 + run)
        costs.append(cost)
        steps.append(step_count)
    
    # Return average cost (lower is better for minimization)
    avg_cost = np.mean(costs)
    avg_steps = np.mean(steps)
    print(f"Params: {params} -> Avg cost: {avg_cost:.2f} (runs: {costs}, avg steps: {avg_steps:.0f})")
    return avg_cost


def main():
    """Main hyperparameter optimization."""
    print("SA Hyperparameter Tuning")
    print("=" * 50)
    
    # Load TSP instance
    instance_path = Path("dataset/lin105.tsp")
    instance, optimal_cost = find_optimal_tour(instance_path)
    print(f"Instance: {instance.name}")
    print(f"Optimal cost: {optimal_cost:.2f}")
    print(f"Cities: {len(instance.cities)}")
    print()
    
    # Define parameter search space
    dimensions = [
        Real(1, 5000, name='initial_temp'),      # Initial temperature
        Real(0.95, 0.9999, name='cooling_rate'),  # Cooling rate
    ]
    
    # Objective function wrapper
    @use_named_args(dimensions=dimensions)
    def objective(**params):
        return evaluate_sa_params(params, instance, optimal_cost, time_budget_seconds=20.0)
    
    print("Starting optimization...")
    print("Each evaluation runs SA for 20 seconds, 2 times sequentially")
    print("Using realistic parameter ranges with validation")
    print()
    
    # Estimate time with a single evaluation
    print("Estimating time with a single evaluation...")
    test_start = time.time()
    test_params = {
        'initial_temp': 100,
        'cooling_rate': 0.995
    }
    evaluate_sa_params(test_params, instance, optimal_cost, time_budget_seconds=20.0)
    test_time = time.time() - test_start
    
    estimated_total_time = test_time * 50  # 50 evaluations
    print(f"Single evaluation took: {test_time:.1f} seconds")
    print(f"Estimated total time: {estimated_total_time:.1f} seconds ({estimated_total_time/60:.1f} minutes)")
    print()
    
    # Ask user if they want to continue
    response = input("Continue with optimization? (y/n): ").lower().strip()
    if response != 'y':
        print("Optimization cancelled.")
        return
    
    # Run Bayesian optimization
    start_time = time.time()
    result = gp_minimize(
        func=objective,
        dimensions=dimensions,
        n_calls=50,  # Number of evaluations
        random_state=42,
        acq_func='EI'  # Expected Improvement
    )
    end_time = time.time()
    
    # Results
    print("\n" + "=" * 50)
    print("OPTIMIZATION RESULTS")
    print("=" * 50)
    print(f"Optimization time: {end_time - start_time:.1f} seconds")
    print(f"Best cost found: {result.fun:.2f}")
    print(f"Optimality gap: {((result.fun / optimal_cost - 1) * 100):.1f}%")
    print()
    
    print("Best parameters:")
    best_params = dict(zip([dim.name for dim in dimensions], result.x))
    for param, value in best_params.items():
        print(f"  {param}: {value:.3f}")
    
    print()
    print("Parameter ranges tested:")
    for dim in dimensions:
        print(f"  {dim.name}: {dim.low} to {dim.high}")
    
    # Test best parameters with more runs
    print("\n" + "=" * 50)
    print("VALIDATION WITH BEST PARAMETERS")
    print("=" * 50)
    
    validation_costs = []
    validation_steps = []
    for run in range(5):  # More runs for validation
        sa = SimulatedAnnealing(
            instance=instance,
            start_temperature=best_params['initial_temp'],
            cooling_schedule=exponential_cooling(best_params['cooling_rate']),
            seed=100 + run
        )
        
        sa.initialize(None)
        
        # Run for 20 seconds
        start_time = time.perf_counter()
        steps = 0
        while time.perf_counter() - start_time < 20.0:
            sa.step()
            steps += 1
        
        validation_costs.append(sa.best_cost)
        validation_steps.append(steps)
        print(f"Run {run+1}: {sa.best_cost:.2f} ({steps} steps)")
    
    print("\nValidation results:")
    print(f"  Mean cost: {np.mean(validation_costs):.2f}")
    print(f"  Std cost: {np.std(validation_costs):.2f}")
    print(f"  Best cost: {np.min(validation_costs):.2f}")
    print(f"  Worst cost: {np.max(validation_costs):.2f}")
    print(f"  Mean optimality gap: {((np.mean(validation_costs) / optimal_cost - 1) * 100):.1f}%")
    print(f"  Mean steps per second: {np.mean(validation_steps) / 20:.0f}")
    print(f"  Steps range: {np.min(validation_steps)} - {np.max(validation_steps)}")


if __name__ == "__main__":
    main()
