from tsp.io import parse_tsplib_tsp
from pathlib import Path
import time
import logging
from constants import MAX_SECONDS, MAX_ITERATIONS

logger = logging.getLogger(__name__)

# Dataset should already been downloaded, 
# if not, run `setup_dataset.py` first.
def find_optimal_tour(tsp_path: str | Path):
    # Load instance
    if isinstance(tsp_path, str):
        tsp_path = Path(tsp_path)
    instance = parse_tsplib_tsp(tsp_path)
    logger.info(f"Loaded {instance.name} with {len(instance.cities)} cities")

    # Check for optimal tour file
    opt_tour_path = Path(f"dataset/{instance.name}.opt.tour")
    optimal_cost = None
    if opt_tour_path.exists():
        logger.info(f"Found optimal tour file: {opt_tour_path}")
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
            logger.info(f"Optimal cost: {optimal_cost:.2f}")
        else:
            logger.warning("Could not parse optimal tour")
    else:
        logger.info("No optimal tour file found")
    
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
    
    return iterations, best_costs, current_costs, times

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
    
    return iterations, best_costs, current_costs, times

# Annealing schedule
def exponential_cooling(alpha: float) -> callable:
    """Return a cooling function with fixed decay rate alpha: T_k+1 = alpha * T_k."""
    def cool(t: float) -> float:
        return t * alpha
    return cool