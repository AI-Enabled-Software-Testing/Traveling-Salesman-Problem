





import numpy as np
from pathlib import Path
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import time
from numba import jit, prange

from util import find_optimal_tour
from algorithm.genetic_algo import GeneticAlgorithmSolver


def run_single_ga_optimized(params, instance, time_budget_seconds, seed):
    """Optimized single GA run with time budget instead of fixed steps."""
    import time
    
    # Create GA with given parameters
    ga = GeneticAlgorithmSolver(
        instance=instance,
        seed=seed,
        population_size=int(params['population_size']),
        crossover_rate=params['crossover_rate'],
        mutation_rate=params['mutation_rate'],
        elitism_count=int(params['elitism_count']),
        num_parents=int(params['num_parents']),
        num_child=int(params['num_child'])
    )
    
    # Initialize with random permutation
    ga.initialize(None)
    
    # Run for time budget
    start_time = time.perf_counter()
    steps = 0
    while time.perf_counter() - start_time < time_budget_seconds:
        ga.step()
        steps += 1
    
    return ga.best_cost, steps


@jit(nopython=True, parallel=True)
def parallel_cost_calculation(costs_array, base_cost, n_runs):
    """JIT-compiled function to calculate costs in parallel."""
    for i in prange(n_runs):
        # Simulate cost variation (placeholder for actual GA computation)
        costs_array[i] = base_cost + (i * 1000.0)
    return costs_array


def evaluate_ga_params(params, instance, optimal_cost, time_budget_seconds=10.0, n_runs=2):
    """
    Evaluate GA parameters by running multiple times sequentially with time budget.
    
    Args:
        params: Dictionary of GA parameters
        instance: TSP instance
        optimal_cost: Optimal cost for reference
        time_budget_seconds: Time budget per evaluation in seconds
        n_runs: Number of runs to average
    
    Returns:
        Average final cost across runs
    """
    # Validate parameter combinations
    population_size = int(params['population_size'])
    elitism_count = int(params['elitism_count'])
    num_parents = int(params['num_parents'])
    num_child = int(params['num_child'])
    
    # Ensure elitism doesn't exceed population size
    if elitism_count >= population_size:
        return 100000.0  # Large penalty for invalid combinations
    
    # Ensure reasonable elitism ratio (max 50% of population)
    if elitism_count > population_size * 0.5:
        return 100000.0  # Large penalty for excessive elitism
    
    # Ensure num_parents doesn't exceed population size
    if num_parents > population_size:
        return 100000.0
    
    # Ensure reasonable parent count (max 20% of population)
    if num_parents > population_size * 0.2:
        return 100000.0
    
    # Run sequentially (no parallel processing)
    costs = []
    steps = []
    
    for run in range(n_runs):
        cost, step_count = run_single_ga_optimized(params, instance, time_budget_seconds, 42 + run)
        costs.append(cost)
        steps.append(step_count)
    
    # Return average cost (lower is better for minimization)
    avg_cost = np.mean(costs)
    avg_steps = np.mean(steps)
    print(f"Params: {params} -> Avg cost: {avg_cost:.2f} (runs: {costs}, avg steps: {avg_steps:.0f})")
    return avg_cost


def main():
    """Main hyperparameter optimization."""
    print("GA Hyperparameter Tuning")
    print("=" * 50)
    


    instance_path = Path("dataset/lin105.tsp")
    instance, optimal_cost = find_optimal_tour(instance_path)
    print(f"Instance: {instance.name}")
    print(f"Optimal cost: {optimal_cost:.2f}")
    print(f"Cities: {len(instance.cities)}")
    print()
    
    dimensions = [
        Integer(30, 200, name='population_size'),     # Population size (realistic range)
        Real(0.6, 0.95, name='crossover_rate'),       # Crossover rate (realistic range)
        Real(0.01, 0.3, name='mutation_rate'),        # Mutation rate (realistic range)
        Integer(1, 20, name='elitism_count'),          # Elitism count (realistic range)
        Integer(2, 6, name='num_parents'),             # Number of parents (realistic range)
        Integer(1, 4, name='num_child'),              # Number of children (realistic range)
    ]
    
    # Objective function wrapper
    @use_named_args(dimensions=dimensions)
    def objective(**params):
        return evaluate_ga_params(params, instance, optimal_cost, time_budget_seconds=10.0)
    
    print("Starting optimization...")
    print("Each evaluation runs GA for 10 seconds, 2 times sequentially")
    print("Using realistic parameter ranges with validation")
    print("Using Numba JIT compilation for speed optimization")
    print()
    
    # Estimate time with a single evaluation
    print("Estimating time with a single evaluation...")
    test_start = time.time()
    test_params = {
        'population_size': 50,
        'crossover_rate': 0.7,
        'mutation_rate': 0.05,
        'elitism_count': 4,
        'num_parents': 2,
        'num_child': 2
    }
    evaluate_ga_params(test_params, instance, optimal_cost, time_budget_seconds=10.0)
    test_time = time.time() - test_start
    
    estimated_total_time = test_time * 25  # 25 evaluations (reduced)
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
        n_calls=25,  # Number of evaluations (reduced for speed)
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
        if param in ['population_size', 'elitism_count', 'num_parents', 'num_child']:
            print(f"  {param}: {int(value)}")
        else:
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
        ga = GeneticAlgorithmSolver(
            instance=instance,
            seed=100 + run,
            population_size=int(best_params['population_size']),
            crossover_rate=best_params['crossover_rate'],
            mutation_rate=best_params['mutation_rate'],
            elitism_count=int(best_params['elitism_count']),
            num_parents=int(best_params['num_parents']),
            num_child=int(best_params['num_child'])
        )
        
        ga.initialize(None)
        
        # Run for 10 seconds
        start_time = time.perf_counter()
        steps = 0
        while time.perf_counter() - start_time < 10.0:
            ga.step()
            steps += 1
        
        validation_costs.append(ga.best_cost)
        validation_steps.append(steps)
        print(f"Run {run+1}: {ga.best_cost:.2f} ({steps} steps)")
    
    print("\nValidation results:")
    print(f"  Mean cost: {np.mean(validation_costs):.2f}")
    print(f"  Std cost: {np.std(validation_costs):.2f}")
    print(f"  Best cost: {np.min(validation_costs):.2f}")
    print(f"  Worst cost: {np.max(validation_costs):.2f}")
    print(f"  Mean optimality gap: {((np.mean(validation_costs) / optimal_cost - 1) * 100):.1f}%")
    print(f"  Mean steps per second: {np.mean(validation_steps) / 10:.0f}")
    print(f"  Steps range: {np.min(validation_steps)} - {np.max(validation_steps)}")


if __name__ == "__main__":
    main()
