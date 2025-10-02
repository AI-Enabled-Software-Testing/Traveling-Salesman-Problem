# Configuration Constants
MAX_ITERATIONS = 1_000  # Number of iterations for iteration-based benchmark
MAX_SECONDS = 1.0      # Number of seconds for time-based benchmark
RANDOM_SEED = 42       # Random seed for reproducible results
COOLING_RATE = 0.995  # Cooling rate for Simulated Annealing
INITIAL_TEMP = 100    # Initial temperature for Simulated Annealing
POPULATION_SIZE = 50  # Population size for Genetic Algorithm
MUTATION_RATE = 0.05  # Mutation rate for Genetic Algorithm
CROSSOVER_RATE = 0.7   # Crossover rate for Genetic Algorithm
ELITISM_COUNT = 4    # Number of elite individuals to carry over in Genetic Algorithm
NUM_PARENTS = 18       # Number of parents for Genetic Algorithm
NUM_CHILD = 25         # Number of children per crossover for Genetic Algorithm
N_RUNS = 10           # Number of runs for averaging results in parallel benchmarking