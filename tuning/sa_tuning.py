import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from util import find_optimal_tour, exponential_cooling
from algorithm.simulated_annealing import SimulatedAnnealing


def run_single_sa(params: dict, instance, time_budget_seconds: float) -> tuple[float, int]:
    """
    Runs a single, randomized SA instance for a fixed time budget.
    """
    sa = SimulatedAnnealing(
        instance=instance,
        start_temperature=params['initial_temp'],
        cooling_schedule=exponential_cooling(params['cooling_rate']),
        seed=None  # Ensures each parallel run is unique
    )
    sa.initialize(None)
    
    start_time = time.perf_counter()
    steps = 0
    while time.perf_counter() - start_time < time_budget_seconds:
        sa.step()
        steps += 1
    
    return sa.best_cost, steps


def evaluate_sa_params_parallel(params: dict, instance, time_budget_seconds: float, n_runs: int) -> float:
    """
    Evaluates SA parameters by running multiple instances in parallel and returning the average cost.
    """
    # Basic parameter validation
    if not (0.9 < params['cooling_rate'] < 1.0) or not (params['initial_temp'] > 0):
        return 1e9

    costs = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_single_sa, params, instance, time_budget_seconds) for _ in range(n_runs)]
        
        for future in as_completed(futures):
            cost, _ = future.result()
            costs.append(cost)
    
    avg_cost = np.mean(costs)
    print(f"Params: temp={params['initial_temp']:.2f}, cool={params['cooling_rate']:.4f} -> Avg cost: {avg_cost:.2f}")
    
    return avg_cost


def main():
    ### CONFIGURATION ###
    TIME_BUDGET_PER_EVALUATION_SECONDS = 10.0
    TOTAL_TUNING_TIME_MINUTES = 30.0
    N_RUNS_PER_EVALUATION = 4 
    ### END CONFIGURATION ###

    instance_path = Path("dataset/lin105.tsp")
    instance, optimal_cost = find_optimal_tour(instance_path)
    print(f"Optimizing SA for instance: {instance.name} (Optimal: {optimal_cost:.2f})")
    
    n_calls = max(1, int((TOTAL_TUNING_TIME_MINUTES * 60) / TIME_BUDGET_PER_EVALUATION_SECONDS))
    
    print("\n" + "-"*50)
    print("Tuning Configuration:")
    print(f"  - Time per SA evaluation: {TIME_BUDGET_PER_EVALUATION_SECONDS} seconds")
    print(f"  - Parallel runs per param set: {N_RUNS_PER_EVALUATION}")
    print(f"  - Total tuning goal: ~{TOTAL_TUNING_TIME_MINUTES} minutes")
    print(f"  - Calculated optimizer calls: {n_calls}")
    print("-"*50 + "\n")

    search_space = [
        Real(1.0, 5000.0, name='initial_temp', prior='log-uniform'),
        Real(0.99, 0.9999, name='cooling_rate'),
    ]
    
    @use_named_args(dimensions=search_space)
    def objective(**params):
        return evaluate_sa_params_parallel(
            params, instance, TIME_BUDGET_PER_EVALUATION_SECONDS, N_RUNS_PER_EVALUATION
        )
    
    start_time = time.time()
    result = gp_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=n_calls,
        random_state=42,
        acq_func='EI'
    )
    end_time = time.time()
    
    print("\n" + "=" * 50)
    print("OPTIMIZATION COMPLETE")
    print(f"Total time: {(end_time - start_time)/60:.2f} minutes")
    
    best_params = dict(zip([dim.name for dim in search_space], result.x))
    print(f"Best average cost found: {result.fun:.2f}")
    print("Best parameters:")
    print(f"  initial_temp: {best_params['initial_temp']:.3f}")
    print(f"  cooling_rate: {best_params['cooling_rate']:.5f}")
    print("=" * 50)

    print("\nValidating best parameters with 5 runs...")
    validation_costs = []
    for i in range(5):
        cost, steps = run_single_sa(
            best_params, instance, time_budget_seconds=TIME_BUDGET_PER_EVALUATION_SECONDS
        )
        validation_costs.append(cost)
        print(f"  Run {i+1}: Cost={cost:.2f} ({steps} steps)")
    
    mean_cost = np.mean(validation_costs)
    gap = ((mean_cost / optimal_cost) - 1) * 100
    print(f"\nValidation Mean Cost: {mean_cost:.2f} (Optimality Gap: {gap:.2f}%)")


if __name__ == "__main__":
    main()