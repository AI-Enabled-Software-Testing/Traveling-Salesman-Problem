import matplotlib.pyplot as plt
from pathlib import Path
from constants import INITIAL_TEMP, COOLING_RATE, POPULATION_SIZE, MUTATION_RATE, CROSSOVER_RATE, ELITISM_COUNT, DATASET_FILENAME
from util import find_optimal_tour, exponential_cooling
from algorithm.simulated_annealing import SimulatedAnnealing
from algorithm.genetic_algo import GeneticAlgorithmSolver


def load_tsp_instance(tsp_filename=DATASET_FILENAME):
    """Load TSP instance from dataset."""
    tsp_path = Path('dataset') / tsp_filename
    instance, optimal_cost = find_optimal_tour(tsp_path)
    instance_data = {'name': instance.name, 'cities': instance.cities}
    return instance, optimal_cost, instance_data


def create_solvers():
    """Create solver factories."""
    def create_sa_solver():
        instance, _, _ = load_tsp_instance()
        T0 = INITIAL_TEMP
        cool_rate = COOLING_RATE
        schedule = exponential_cooling(cool_rate)
        return SimulatedAnnealing(instance, T0, schedule)
    
    def create_ga_solver():
        instance, _, _ = load_tsp_instance()
        return GeneticAlgorithmSolver(
            instance,
            population_size=POPULATION_SIZE,
            mutation_rate=MUTATION_RATE,
            crossover_rate=CROSSOVER_RATE,
            elitism_count=ELITISM_COUNT
        )
    
    return {
        'SA_random': create_sa_solver,
        'GA_random': create_ga_solver
    }


def create_plot(title, xlabel, ylabel, figsize=(10, 6)):
    """Create and configure a plot."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return fig, ax
