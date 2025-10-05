# General Settings
MAX_ITERATIONS = 1_000  # Number of iterations for iteration-based benchmark
MAX_SECONDS = 5.0       # Number of seconds for time-based benchmark
N_RUNS = 4              # Number of runs for averaging results in parallel benchmarking

DATASET_FILENAME = 'lin105.tsp'

# Parallelization Settings
PARALLEL_RUNS = True
NUM_WORKERS = 8  # Number of parallel workers for benchmarking trials

# Calibration Constants
CALIBRATION_TIME = 2.0
MAX_NORMALIZED_STEPS = 50_000

# Simulated Annealing (SA) Settings
COOLING_RATE = 0.9999    # Cooling rate for Simulated Annealing (from tuning.json)
INITIAL_TEMP = 4500   # Initial temperature for Simulated Annealing (from tuning.json)

# Genetic Algorithm (GA) Settings
POPULATION_SIZE = 60      # Population size for Genetic Algorithm (from tuning.json)
MUTATION_RATE = 0.3     # Mutation rate for Genetic Algorithm (from tuning.json)
CROSSOVER_RATE = 0.8    # Crossover rate for Genetic Algorithm (from tuning.json)
ELITISM_COUNT = 9         # Number of elite individuals to carry over in Genetic Algorithm (from tuning.json)