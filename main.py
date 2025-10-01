import pandas as pd # for csv output


# Algorithms
from algorithm.nearest_neighbor import NearestNeighbor
from algorithm.simulated_annealing import SimulatedAnnealing
from algorithm.genetic_algo import GeneticAlgorithmSolver
from algorithm.random_solver import RandomSolver

# Utilities
from pathlib import Path
import time
import matplotlib.pyplot as plt
import random
from constants import *
from util import *

def main():
    print("Hello from traveling-salesman-problem!")
    # Setup
    random.seed(RANDOM_SEED)


if __name__ == "__main__":
    main()
