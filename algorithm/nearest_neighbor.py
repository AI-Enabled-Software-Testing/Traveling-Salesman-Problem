from __future__ import annotations

from typing import List

from tsp.model import TSPInstance
from .base import IterativeTSPSolver, StepReport


class NearestNeighbor(IterativeTSPSolver):
    """Constructive iterative nearest neighbor.

    Each step adds one more nearest unvisited city to the route until complete.
    """

    def __init__(self, instance: TSPInstance):
        self.instance = instance
        self.route: List[int] = []
        self.unvisited: List[int] = []
        self.iteration = 0

    def initialize(self, route: List[int]) -> None:
        # Start from provided first city; if empty, start from city 0
        if route:
            self.route = [route[0]]
        else:
            self.route = [0]
        self.unvisited = [i for i in range(len(self.instance.cities)) if i not in self.route]
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
        return StepReport(iteration=self.iteration, cost=self.get_cost(), improved=improved)

    def get_route(self) -> List[int]:
        # If incomplete, return current partial route followed by unvisited
        return self.route + self.unvisited

    def get_cost(self) -> float:
        if len(self.route) < 2:
            return 0.0
        return self.instance.route_cost(self.get_route())


