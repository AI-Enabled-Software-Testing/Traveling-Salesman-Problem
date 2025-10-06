# General Settings
MAX_ITERATIONS = 1_000  # Number of iterations for relative iteration-based benchmark
MAX_SECONDS = 5.0       # Number of seconds for time-based benchmark
N_RUNS = 5              # Number of runs for averaging results in benchmarking

DATASET_FILENAME = 'lin105.tsp'

# Parallelization Settings
PARALLEL_RUNS = True
NUM_WORKERS = 8  # Number of parallel workers for benchmarking trials

# Calibration Constants
CALIBRATION_TIME = 2.0
MAX_NORMALIZED_STEPS = 100_000

# Simulated Annealing (SA) Settings
COOLING_RATE = 0.99990
INITIAL_TEMP = 167.807

# Genetic Algorithm (GA) Settings
POPULATION_SIZE = 109
MUTATION_RATE = 0.400
CROSSOVER_RATE = 0.600
ELITISM_COUNT = 1