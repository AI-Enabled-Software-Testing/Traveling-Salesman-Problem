from __future__ import annotations

import random
from typing import List

from tsp.model import TSPInstance
from .base import IterativeTSPSolver, StepReport


class GeneticAlgorithmSolver(IterativeTSPSolver):
    """Genetic Algorithm for TSP."""

    def __init__(self, instance: TSPInstance, seed: int | None = None, population_size: int = 100, crossover_rate: float = 0.7, mutation_rate: float = 0.01, num_parents: int = 2, num_child: int = 2):
        self.instance = instance
        self.rng = random.Random(seed)
        self.best_route: List[int] = []
        self.best_cost: float = float("inf")
        self.iteration = 0
        # Genetic algorithm parameters
        self.population_size = population_size
        self.population: list = [] # List of routes (sublists)
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_parents = num_parents
        self.num_child = num_child

    def initialize(self, route: List[int]) -> None:
        n = len(self.instance.cities)
        if route and len(route) == n:
            self.best_route = list(route)
        else:
            self.best_route = list(range(n))
        
        # Handle the Population List
        for _ in range(self.population_size):
            individual = list(range(n)) # n cities
            self.rng.shuffle(individual)
            self.population.append(individual)
        
        # Initialize Other Constants
        self.best_cost = self.instance.route_cost(self.best_route)
        self.iteration = 0

    def step(self) -> StepReport:
        self.iteration += 1

        # Evaluate Improvement
        improved = False
        current_cost = 0
        for candidate in self.population:
            # Assess Fitness
            current_cost = self.instance.route_cost(candidate)
            if current_cost < self.best_cost:
                improved = True
                self.best_cost = current_cost
                self.best_route = candidate

        # Handle Selection and Crossover
        new_population = []
        # Generate enough offspring to fill the population
        while len(new_population) < self.population_size:
            # Select parents
            selected_parents = []
            for _ in range(self.num_parents):
                parent = self.select_parent()
                selected_parents.append(parent)
            
            # Generate children from these parents
            for _ in range(self.num_child):
                if len(new_population) >= self.population_size:
                    break
                child = self.crossover(selected_parents)
                # Mutation
                if self.rng.random() < self.mutation_rate:
                    child = self.mutate(child)
                new_population.append(child)
            
        self.population = new_population

        return StepReport(iteration=self.iteration, best_cost=self.best_cost, current_cost=current_cost, improved=improved)

    def get_route(self) -> List[int]:
        return self.best_route

    def get_cost(self) -> float:
        return self.best_cost
    
    # Utility Functions for Genetic Algorithm
    def select_parent(self) -> List[int]:
        """Tournament selection."""
        tournament_size = max(2, self.population_size // 10)
        tournament = self.rng.sample(self.population, tournament_size)
        tournament.sort(key=lambda route: self.instance.route_cost(route))
        return tournament[0] # Best in tournament (i.e., lowest cost)

    def crossover(self, parents: List[List[int]]) -> List[int]:
        """Ordered Crossover for TSP from all possible parent pairs (individual city point based)."""
        if len(parents) < 2:
            return parents[0][:] # Return copy of single parent

        # Generate all possible pairs of parents
        from itertools import combinations
        parent_pairs = list(combinations(parents, 2))
        
        # If no pairs possible, return random parent
        if not parent_pairs:
            return self.rng.choice(parents)[:]
        
        # Randomly select one pair for this crossover
        parent1, parent2 = self.rng.choice(parent_pairs)
        n = len(parent1)
        
        # Apply crossover rate
        if self.rng.random() > self.crossover_rate:
            # No crossover, return random parent
            return self.rng.choice([parent1, parent2])[:]
        
        # Choose random crossover points
        start, end = sorted(self.rng.sample(range(n), 2))
        
        # Create child
        child = [-1] * n
        
        # Copy segment from parent1
        child[start:end] = parent1[start:end]
        
        # Fill remaining positions with genes from parent2 in order
        parent2_genes = [gene for gene in parent2 if gene not in child]
        child_index = 0
        
        for gene in parent2_genes:
            # Find next empty position
            while child[child_index] != -1:
                child_index += 1
            child[child_index] = gene
        
        return child

    def mutate(self, route: List[int]) -> List[int]:
        """Swap mutation (route-based)."""
        mutated_route = route[:]  # Create a copy to avoid modifying original
        n = len(mutated_route)
        i, j = self.rng.sample(range(n), 2)
        mutated_route[i], mutated_route[j] = mutated_route[j], mutated_route[i] # Swap
        return mutated_route