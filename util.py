from tsp.io import parse_tsplib_tsp
from pathlib import Path
import time
import logging
from constants import MAX_SECONDS, MAX_ITERATIONS, COOLING_RATE, POPULATION_SIZE, MUTATION_RATE, CROSSOVER_RATE, ELITISM_COUNT

# Import algorithms
from algorithm.nearest_neighbor import NearestNeighbor
from algorithm.simulated_annealing import SimulatedAnnealing
from algorithm.genetic_algo import GeneticAlgorithmSolver
from algorithm.random_solver import RandomSolver

logger = logging.getLogger(__name__)

def setup_algorithm(alg_name: str, instance):
    # Algorithm Specific Setup
    init_route = None
    match alg_name:
        case "SimulatedAnnealing_random":
            INITIAL_TEMP = 100
            solver = SimulatedAnnealing(
                instance, 
                INITIAL_TEMP, 
                exponential_cooling(COOLING_RATE), # Cooling Schedule 
            )
        case "SimulatedAnnealing_NearestNeighbor":
            INITIAL_TEMP = 100
            # Solver
            solver = SimulatedAnnealing(
                instance, 
                INITIAL_TEMP, 
                exponential_cooling(COOLING_RATE), # Cooling Schedule 
            )
        case "GeneticAlgorithm_NearestNeighbor":
            solver = GeneticAlgorithmSolver(
                instance, 
                population_size=POPULATION_SIZE,
                mutation_rate=MUTATION_RATE,
                crossover_rate=CROSSOVER_RATE,
                elitism_count=ELITISM_COUNT,
            )
        case "GeneticAlgorithm":
            solver = GeneticAlgorithmSolver(
                instance, 
                population_size=POPULATION_SIZE,
                mutation_rate=MUTATION_RATE,
                crossover_rate=CROSSOVER_RATE,
                elitism_count=ELITISM_COUNT,
            )
        case "Baseline_Random":
            solver = RandomSolver(instance)
        case _:
            logger.info(" Only the following algorithm names are supported:\nSimulatedAnnealing_random, SimulatedAnnealing_NearestNeighbor, GeneticAlgorithm, Baseline_Random.")
            raise ValueError(f"[ERROR] Unknown algorithm name: {alg_name}")

    # Update seed route for Nearest Neighbor based algorithms
    if alg_name.upper().endswith("NEARESTNEIGHBOR"):
        # Seed routes: identity and Nearest Neighbor
        seed_identity = list(range(len(instance.cities)))

        # Build Nearest Neighbor seed route fully
        _nn_builder = NearestNeighbor(instance)
        _nn_builder.initialize(seed_identity)
        for _ in range(len(instance.cities) - 1):
            _ = _nn_builder.step()
        seed_nn = _nn_builder.get_route()
        init_route = seed_nn
        logger.info(f"[INFO] Using Nearest Neighbor seed route for {alg_name}")

    return solver, init_route

# Dataset should already been downloaded, 
# if not, run `setup_dataset.py` first.
def find_optimal_tour(tsp_path: str | Path):
    # Load instance
    if isinstance(tsp_path, str):
        tsp_path = Path(tsp_path)
    instance = parse_tsplib_tsp(tsp_path)

    # Check for optimal tour file
    opt_tour_path = Path(f"dataset/{instance.name}.opt.tour")
    optimal_cost = None
    if opt_tour_path.exists():
        # Parse optimal tour (simple format: just city indices)
        with open(opt_tour_path, "r") as f:
            lines = f.readlines()
        
        # Find TOUR_SECTION and read tour
        in_tour = False
        tour = []
        for line in lines:
            line = line.strip()
            if line.upper().startswith("TOUR_SECTION"):
                in_tour = True
                continue
            if line.upper().startswith("EOF") or line == "-1":
                break
            if in_tour and line:
                try:
                    # Convert 1-based TSPLIB ids to 0-based indices
                    city_id = int(line) - 1
                    tour.append(city_id)
                except ValueError:
                    continue
        
        if tour:
            optimal_cost = instance.route_cost(tour)
        else:
            logger.warning("Could not parse optimal tour")
    else:
        logger.warning(f"No optimal tour file found for {instance.name}")
    
    return instance, optimal_cost

def run_algorithm_with_timing(instance, solver, initial_route, max_seconds=MAX_SECONDS):
    """Run algorithm for fixed time and collect cost at each step."""
    solver.initialize(initial_route)
    
    iterations = []
    best_costs = []
    current_costs = []
    times = []
    
    start_time = time.perf_counter()
    
    while time.perf_counter() - start_time < max_seconds:
        report = solver.step()
        current_time = time.perf_counter() - start_time
        
        iterations.append(report.iteration)
        best_costs.append(report.best_cost)
        current_costs.append(report.current_cost)
        times.append(current_time)
    
    # Get the final best route
    if hasattr(solver, "get_route"):
        best_route = solver.get_route()
    else: 
        best_route = None
    
    return iterations, best_costs, current_costs, times, best_route

def run_algorithm_with_iterations(instance, solver, initial_route, max_iterations=MAX_ITERATIONS):
    """Run algorithm for fixed number of iterations and collect cost at each step."""
    if hasattr(solver, "initialize"):
        solver.initialize(initial_route)
    else:
        raise ValueError("Solver must have an 'initialize' method")
    
    iterations = []
    best_costs = []
    current_costs = []
    times = []
    
    start_time = time.perf_counter()
    
    for i in range(max_iterations):
        report = solver.step()
        current_time = time.perf_counter() - start_time
        
        iterations.append(report.iteration)
        best_costs.append(report.best_cost)
        current_costs.append(report.current_cost)
        times.append(current_time)
    
    # Get the final best route
    if hasattr(solver, "get_route"):
        best_route = solver.get_route()
    else: 
        best_route = None
    
    return iterations, best_costs, current_costs, times, best_route

# Annealing schedule
def exponential_cooling(alpha: float) -> callable:
    """Return a cooling function with fixed decay rate alpha: T_k+1 = alpha * T_k."""
    def cool(t: float) -> float:
        return t * alpha
    return cool

# Parallel Execution
def run_single_trial_by_timing(args):
    """Run a single algorithm trial - used for parallel processing."""
    algorithm_name, run_idx, instance_data, init_route, seed_identity_data = args
    logger.info(f"Starting run {run_idx} BY TIMING for algorithm {algorithm_name}")

    # Recreate instance from data (needed for multiprocessing)
    from tsp.model import TSPInstance
    instance = TSPInstance(name=instance_data['name'], cities=instance_data['cities'])
    
    # Create solver in worker process to avoid pickling issues
    solver, _ = setup_algorithm(algorithm_name, instance)

    iterations, best_costs, current_costs, times, best_route = run_algorithm_with_timing(
        instance, solver, init_route, MAX_SECONDS
    )
    
    return {
        'iterations': iterations,
        'best_costs': best_costs,
        'current_costs': current_costs,
        'times': times,
        'best_route': best_route
    }

def run_single_iteration_trial(args):
    """Run a single iteration-based trial."""
    algorithm_name, run_idx, instance_data, init_route, seed_identity_data = args
    logger.info(f"Starting run {run_idx} BY ITERATION for algorithm {algorithm_name}")
    
    from tsp.model import TSPInstance
    instance = TSPInstance(name=instance_data['name'], cities=instance_data['cities'])
    
    # Create solver in worker process to avoid pickling issues
    solver, _ = setup_algorithm(algorithm_name, instance)
    
    iters, best_costs, current_costs, times, best_route = run_algorithm_with_iterations(
        instance, solver, init_route, MAX_ITERATIONS
    )
    
    return {'iters': iters, 'best_costs': best_costs, 'times': times, 'best_route': best_route}