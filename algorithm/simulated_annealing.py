from __future__ import annotations

import math
import random
from typing import List
from typing_extensions import Callable

from tsp.model import TSPInstance
from .base import IterativeTSPSolver, StepReport


class SimulatedAnnealing(IterativeTSPSolver):
    """Iterative simulated annealing."""

    def __init__(
        self,
        instance: TSPInstance,
        start_temperature: float,
        cooling_schedule: Callable[[float], float],
        seed: int | float | None = None,
    ):
        self.instance = instance
        self.start_temperature = start_temperature
        self.cooling_schedule = cooling_schedule
        self.rng = random.Random(seed)
        self.iteration = 0

        self.temperature = start_temperature
        self.route: List[int] = []
        self.best_route: List[int] = []
        self.best_cost = float("inf")
        self.current_cost = float("inf")

    def initialize(self, route: List[int]) -> None:
        n = len(self.instance.cities)
        if route and len(route) == n:
            self.route = list(route)
        else:
            self.route = list(range(n))
            self.rng.shuffle(self.route)
        self.best_route = list(self.route)
        self.current_cost = self.instance.route_cost(self.route)
        self.best_cost = self.current_cost
        self.iteration = 0
        self.temperature = self.start_temperature

    def get_random_neighbour(self, route: List[int]) -> List[int]:
        """Random 2-opt neighbour."""
        n = len(route)
        if n < 2:
            return list(route)
        i, j = sorted(self.rng.sample(range(n), 2))
        neighbour = list(route)
        neighbour[i:j+1] = reversed(neighbour[i:j+1])
        return neighbour

    def step(self) -> StepReport:
        self.iteration += 1
        candidate = self.get_random_neighbour(self.route)
        candidate_cost = self.instance.route_cost(candidate)
        delta = candidate_cost - self.current_cost

        accept = delta < 0
        if not accept and self.temperature > 0:
            accept = self.rng.random() < math.exp(-delta / self.temperature)

        improved = False
        if accept:
            self.route = candidate
            self.current_cost = candidate_cost
            if candidate_cost < self.best_cost:
                self.best_cost = candidate_cost
                self.best_route = list(candidate)
                improved = True

        self.temperature = self.cooling_schedule(self.temperature)

        return StepReport(iteration=self.iteration, best_cost=self.best_cost, current_cost=self.current_cost, improved=improved)

    def get_route(self) -> List[int]:
        return self.best_route

    def get_cost(self) -> float:
        return self.best_cost
