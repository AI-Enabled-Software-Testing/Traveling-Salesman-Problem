from __future__ import annotations

import random
from typing import List

from tsp.model import TSPInstance
from .base import IterativeTSPSolver, StepReport


class NearestNeighbor(IterativeTSPSolver):
    """Constructive iterative nearest neighbor.

    Each step adds one more nearest unvisited city to the route until complete.
    """

    def __init__(self, instance: TSPInstance, seed: int | float | None = None):
        self.instance = instance
        self.rng = random.Random(seed)
        self.route: List[int] = []
        self.unvisited: List[int] = []
        self.iteration = 0

    def initialize(self, route: list[int] | None = None) -> None:
        n = len(self.instance.cities)
        if route and len(route) > 0:
            self.route = [route[0]]
            self.unvisited = [i for i in range(n) if i not in self.route]
        else:
            start_city = self.rng.randint(0, n - 1)
            self.route = [start_city]
            self.unvisited = [i for i in range(n) if i != start_city]
        self.iteration = 0

    def step(self) -> StepReport:
        self.iteration += 1
        improved = False
        if self.unvisited:
            last = self.route[-1] # FIFO
            # Pick nearest unvisited
            next_city = min(self.unvisited, key=lambda j: self.instance.distance(last, j))
            self.unvisited.remove(next_city)
            self.route.append(next_city)
            improved = True
        current_cost = self.get_cost()
        return StepReport(iteration=self.iteration, best_cost=current_cost, current_cost=current_cost, improved=improved)

    def get_route(self) -> List[int]:
        # If incomplete, return current partial route followed by unvisited
        return self.route + self.unvisited

    def get_cost(self) -> float:
        if len(self.route) < 2:
            return 0.0
        return self.instance.route_cost(self.get_route())


