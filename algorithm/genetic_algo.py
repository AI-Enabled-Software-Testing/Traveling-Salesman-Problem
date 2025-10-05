from __future__ import annotations

import random
from typing import List
from collections import OrderedDict

from tsp.model import TSPInstance
from .base import IterativeTSPSolver, StepReport


class GeneticAlgorithmSolver(IterativeTSPSolver):
    """Genetic Algorithm for TSP."""

    def __init__(self, instance: TSPInstance, seed: int | float | None = None, population_size: int = 100, crossover_rate: float = 0.7, mutation_rate: float = 0.01, elitism_count: int = 2):
        self.instance = instance
        self.rng = random.Random(seed)
        self.best_route: List[int] = []
        self.best_cost: float = float("inf")
        self.iteration = 0
        self.population_size = population_size
        self.population: List[List[int]] = []
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.fitness_cache = OrderedDict()
        self.max_cache_size = 2 * population_size

    def initialize(self, route: List[int]) -> None:
        n = len(self.instance.cities)
        if route and len(route) == n:
            self.best_route = list(route)
        else:
            self.best_route = list(range(n))
            self.rng.shuffle(self.best_route)
        
        self.population = []
        # Preserve OrderedDict type for LRU behavior
        self.fitness_cache.clear()
        
        # Seed 70% variations of router and 30% random permutations
        if route and len(route) == n:
            self.population.append(list(route))
            target_nn_count = max(1, int(self.population_size * 0.7))
            num_variations = max(0, min(target_nn_count - 1, self.population_size - 1))
            for _ in range(num_variations):
                if len(self.population) >= self.population_size:
                    break

                # 2-opt variation
                variant = list(route)
                num_moves = self.rng.randint(1, 3)
                for _ in range(num_moves):
                    i, j = sorted(self.rng.sample(range(n), 2))
                    variant[i:j+1] = reversed(variant[i:j+1])
                self.population.append(variant)
        
        # Fill remaining ~30% with random permutations
        while len(self.population) < self.population_size:
            individual = list(range(n))
            self.rng.shuffle(individual)
            self.population.append(individual)
        
        self.best_cost = self._get_fitness(self.best_route)
        self.iteration = 0
    
    def _get_fitness(self, route: List[int]) -> float:
        """Get fitness cost with LRU caching."""
        route_tuple = tuple(route)
        if route_tuple not in self.fitness_cache:
            if len(self.fitness_cache) >= self.max_cache_size:
                self.fitness_cache.popitem(last=False)
            self.fitness_cache[route_tuple] = self.instance.route_cost(route)
        else:
            self.fitness_cache.move_to_end(route_tuple)
        return self.fitness_cache[route_tuple]

    def step(self) -> StepReport:
        self.iteration += 1

        # Evaluate fitness for each individual in the population
        population_with_fitness = [(ind, self._get_fitness(ind)) for ind in self.population]
        
        # Update the best-known solution if a better one is found
        improved = False
        fitness_sum = 0.0
        for candidate, fitness in population_with_fitness:
            fitness_sum += fitness
            if fitness < self.best_cost:
                improved = True
                self.best_cost = fitness
                self.best_route = list(candidate)
        
        average_fitness = fitness_sum / len(self.population)

        # Create the next generation
        new_population = []

        # Elitism: carry over the best individuals to the next generation
        if self.elitism_count > 0:
            population_with_fitness.sort(key=lambda x: x[1])
            elite = [ind for ind, _ in population_with_fitness[:self.elitism_count]]
            new_population.extend(elite)

        # Generate the rest of the new population through selection, crossover, and mutation
        while len(new_population) < self.population_size:
            parent1 = self.select_parent()
            parent2 = self.select_parent()
            
            child = self.crossover(parent1, parent2)
            
            if self.rng.random() < self.mutation_rate:
                child = self.mutate(child)
                
            new_population.append(child)
            
        self.population = new_population

        return StepReport(iteration=self.iteration, best_cost=self.best_cost, current_cost=average_fitness, improved=improved)

    def get_route(self) -> List[int]:
        return self.best_route

    def get_cost(self) -> float:
        return self.best_cost
    
    def select_parent(self) -> List[int]:
        """Selects a parent from the population using tournament selection."""
        tournament_size = max(2, self.population_size // 10)
        tournament = self.rng.sample(self.population, tournament_size)
        tournament.sort(key=lambda route: self._get_fitness(route))
        return tournament[0]

    def crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Performs Ordered Crossover (OX) on two parents to create a child."""
        if self.rng.random() > self.crossover_rate:
            return self.rng.choice([parent1, parent2])[:]
        
        n = len(parent1)
        start, end = sorted(self.rng.sample(range(n), 2))
        
        child = [None] * n
        child[start:end] = parent1[start:end]
        
        fill_pos = 0
        for gene in parent2:
            if gene not in child:
                while child[fill_pos] is not None:
                    fill_pos += 1
                child[fill_pos] = gene
        
        return child

    def mutate(self, route: List[int]) -> List[int]:
        """Applies a 2-opt mutation to a route."""
        mutated_route = route[:]
        n = len(mutated_route)
        i, j = sorted(self.rng.sample(range(n), 2))
        mutated_route[i:j+1] = reversed(mutated_route[i:j+1])
        return mutated_route