from __future__ import annotations

import random
from typing import List

from tsp.model import TSPInstance
from .base import IterativeTSPSolver, StepReport


class NearestNeighbor(IterativeTSPSolver):
    """Nearest neighbor sampling."""

    def __init__(self, instance: TSPInstance, seed: int | float | None = None):
        self.instance = instance
        self.rng = random.Random(seed)
        self.iteration = 0

        self.best_route: List[int] = []
        self.best_cost: float = float("inf")

    def _build_nn_route(self, start_city: int) -> List[int]:
        n = len(self.instance.cities)
        route: List[int] = [start_city]
        unvisited: List[int] = [i for i in range(n) if i != start_city]
        while unvisited:
            last = route[-1]
            next_city = min(unvisited, key=lambda j: self.instance.distance(last, j))
            unvisited.remove(next_city)
            route.append(next_city)
        return route

    def initialize(self, route: List[int] | None = None) -> None:
        n = len(self.instance.cities)
        self.iteration = 0
        if route and len(route) > 0:
            start_city = route[0]
        else:
            start_city = self.rng.randint(0, n - 1)
        candidate = self._build_nn_route(start_city)
        self.best_route = candidate
        self.best_cost = self.instance.route_cost(candidate)

    def step(self) -> StepReport:
        self.iteration += 1
        n = len(self.instance.cities)
        start_city = self.rng.randint(0, n - 1)
        candidate = self._build_nn_route(start_city)
        current_cost = self.instance.route_cost(candidate)

        improved = False
        if current_cost < self.best_cost:
            self.best_cost = current_cost
            self.best_route = candidate
            improved = True

        return StepReport(
            iteration=self.iteration,
            best_cost=self.best_cost,
            current_cost=current_cost,
            improved=improved,
        )

    def get_route(self) -> List[int]:
        return self.best_route

    def get_cost(self) -> float:
        return self.best_cost


