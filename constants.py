# General Settings
MAX_ITERATIONS = 1_000  # Number of iterations for relative iteration-based benchmark
MAX_SECONDS = 5.0       # Number of seconds for time-based benchmark
N_RUNS = 4              # Number of runs for averaging results in benchmarking

DATASET_FILENAME = 'lin105.tsp'

# Parallelization Settings
PARALLEL_RUNS = False
NUM_WORKERS = 8  # Number of parallel workers for benchmarking trials

# Calibration Constants
CALIBRATION_TIME = 2.0
MAX_NORMALIZED_STEPS = 50_000

# Simulated Annealing (SA) Settings
COOLING_RATE = 0.9999 
INITIAL_TEMP = 4500   

# Genetic Algorithm (GA) Settings
POPULATION_SIZE = 60   
MUTATION_RATE = 0.3    
CROSSOVER_RATE = 0.8   
ELITISM_COUNT = 9      