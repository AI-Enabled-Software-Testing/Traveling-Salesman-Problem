import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from util import find_optimal_tour
from algorithm.genetic_algo import GeneticAlgorithmSolver


def run_single_ga(params: dict, instance, time_budget_seconds: float) -> tuple[float, int]:
    """
    Runs a single, randomized GA instance for a fixed time budget.
    """
    ga = GeneticAlgorithmSolver(
        instance=instance,
        population_size=int(params['population_size']),
        crossover_rate=params['crossover_rate'],
        mutation_rate=params['mutation_rate'],
        elitism_count=int(params['elitism_count'])
    )
    ga.initialize(None)
    
    start_time = time.perf_counter()
    steps = 0
    while time.perf_counter() - start_time < time_budget_seconds:
        ga.step()
        steps += 1
    
    return ga.best_cost, steps


def evaluate_ga_params_parallel(params: dict, instance, time_budget_seconds: float, n_runs: int) -> float:
    """
    Evaluates GA parameters by running multiple instances in parallel and returning the average cost.
    """
    population_size = int(params['population_size'])
    elitism_count = int(params['elitism_count'])
    
    if elitism_count >= population_size or elitism_count > population_size * 0.5:
        return 1e9

    costs = []
    with ProcessPoolExecutor() as executor:
        # Submit the same task n_runs times. This is the correct way to call a function
        # with fixed arguments multiple times in parallel.
        futures = [executor.submit(run_single_ga, params, instance, time_budget_seconds) for _ in range(n_runs)]
        
        # Collect results as they complete.
        for future in as_completed(futures):
            cost, _ = future.result()
            costs.append(cost)
    
    avg_cost = np.mean(costs)
    print(f"Params: pop={population_size}, cross={params['crossover_rate']:.2f}, "
          f"mute={params['mutation_rate']:.2f}, elite={elitism_count} -> Avg cost: {avg_cost:.2f}")
    
    return avg_cost


def main():
    ### CONFIGURATION ###
    TIME_BUDGET_PER_EVALUATION_SECONDS = 10.0
    TOTAL_TUNING_TIME_MINUTES = 30.0
    N_RUNS_PER_EVALUATION = 4 
    ### END CONFIGURATION ###

    instance_path = Path("dataset/lin105.tsp")
    instance, optimal_cost = find_optimal_tour(instance_path)
    print(f"Optimizing GA for instance: {instance.name} (Optimal: {optimal_cost:.2f})")
    
    time_per_optimizer_call = TIME_BUDGET_PER_EVALUATION_SECONDS 
    n_calls = max(1, int((TOTAL_TUNING_TIME_MINUTES * 60) / time_per_optimizer_call))
    
    print("\n" + "-"*50)
    print("Tuning Configuration:")
    print(f"  - Time per GA evaluation: {TIME_BUDGET_PER_EVALUATION_SECONDS} seconds")
    print(f"  - Parallel runs per param set: {N_RUNS_PER_EVALUATION}")
    print(f"  - Total tuning goal: ~{TOTAL_TUNING_TIME_MINUTES} minutes")
    print(f"  - Calculated optimizer calls: {n_calls}")
    print("-"*50 + "\n")

    search_space = [
        Integer(30, 250, name='population_size'),
        Real(0.6, 0.95, name='crossover_rate'),
        Real(0.01, 0.4, name='mutation_rate'),
        Integer(1, 25, name='elitism_count'),
    ]
    
    @use_named_args(dimensions=search_space)
    def objective(**params):
        return evaluate_ga_params_parallel(
            params, instance, TIME_BUDGET_PER_EVALUATION_SECONDS, N_RUNS_PER_EVALUATION
        )
    
    start_time = time.time()
    result = gp_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=n_calls,
        acq_func='EI'
    )
    end_time = time.time()
    
    print("\n" + "=" * 50)
    print("OPTIMIZATION COMPLETE")
    print(f"Total time: {(end_time - start_time)/60:.2f} minutes")
    
    best_params = dict(zip([dim.name for dim in search_space], result.x))
    print(f"Best average cost found: {result.fun:.2f}")
    print("Best parameters:")
    print(f"  population_size: {int(best_params['population_size'])}")
    print(f"  crossover_rate: {best_params['crossover_rate']:.3f}")
    print(f"  mutation_rate: {best_params['mutation_rate']:.3f}")
    print(f"  elitism_count: {int(best_params['elitism_count'])}")
    print("=" * 50)

    print("\nValidating best parameters with 5 runs...")
    validation_costs = []
    for i in range(5):
        # The validation runs are also randomized now.
        cost, steps = run_single_ga(
            best_params, instance, time_budget_seconds=TIME_BUDGET_PER_EVALUATION_SECONDS
        )
        validation_costs.append(cost)
        print(f"  Run {i+1}: Cost={cost:.2f} ({steps} steps)")
    
    mean_cost = np.mean(validation_costs)
    gap = ((mean_cost / optimal_cost) - 1) * 100
    print(f"\nValidation Mean Cost: {mean_cost:.2f} (Optimality Gap: {gap:.2f}%)")


if __name__ == "__main__":
    main()