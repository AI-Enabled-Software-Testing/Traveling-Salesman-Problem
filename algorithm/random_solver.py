from __future__ import annotations

import random
from typing import List

from tsp.model import TSPInstance
from .base import IterativeTSPSolver, StepReport


class RandomSolver(IterativeTSPSolver):
    """Each iteration samples a new random permutation; keeps track of the best seen."""

    def __init__(self, instance: TSPInstance, seed: int | float | None = None):
        self.instance = instance
        self.rng = random.Random(seed)
        self.best_route: List[int] = []
        self.best_cost: float = float("inf")
        self.iteration = 0

    def initialize(self, route: List[int]) -> None:
        n = len(self.instance.cities)
        if route and len(route) == n:
            self.best_route = list(route)
        else:
            self.best_route = list(range(n))
            self.rng.shuffle(self.best_route)
        self.best_cost = self.instance.route_cost(self.best_route)
        self.iteration = 0

    def step(self) -> StepReport:
        self.iteration += 1
        n = len(self.instance.cities)
        candidate = list(range(n))
        self.rng.shuffle(candidate)
        current_cost = self.instance.route_cost(candidate)
        improved = False
        if current_cost < self.best_cost:
            self.best_cost = current_cost
            self.best_route = candidate
            improved = True
        return StepReport(iteration=self.iteration, best_cost=self.best_cost, current_cost=current_cost, improved=improved)

    def get_route(self) -> List[int]:
        return self.best_route

    def get_cost(self) -> float:
        return self.best_cost


