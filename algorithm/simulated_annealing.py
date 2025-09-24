from __future__ import annotations

import math
import random
from typing import List
from typing_extensions import Callable

from tsp.model import TSPInstance
from .base import IterativeTSPSolver, StepReport


class SimulatedAnnealing(IterativeTSPSolver):
    """Iterative simulated annealing.

    At each step, proposes a random tweak (neighbor) to the current route.
    Accepts or rejects based on cost difference and current temperature.
    Temperature is updated according to the cooling schedule.
    """

    def __init__(
        self,
        instance: TSPInstance,
        start_temperature: float,
        cooling_schedule: Callable[[float], float],
        seed: int | None = None,
    ):
        """
        Args:
            instance: TSPInstance to solve.
            start_temperature: Initial temperature.
            cooling_schedule: Function (T) -> new T.
            seed: Optional random seed.
        """
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
        """Return a new route by swapping two random cities."""
        n = len(route)
        if n < 2:
            return list(route)
        i, j = self.rng.sample(range(n), 2)
        neighbour = list(route)
        neighbour[i], neighbour[j] = neighbour[j], neighbour[i]
        return neighbour

    def step(self) -> StepReport:
        self.iteration += 1
        improved = False

        # Propose a tweak (neighbor)
        candidate = self.get_random_neighbour(self.route)
        candidate_cost = self.instance.route_cost(candidate)
        delta = candidate_cost - self.current_cost

        if delta < 0:
            # Always accept better solutions
            accept = True
        elif self.temperature > 0:
            # Accept worse solutions with probability exp(-delta/T)
            accept = self.rng.random() < math.exp(-delta / self.temperature)
        else:
            # When temperature is 0 or negative, only accept better solutions
            accept = False

        if accept:
            self.route = candidate
            self.current_cost = candidate_cost
            if candidate_cost < self.best_cost:
                self.best_cost = candidate_cost
                self.best_route = list(candidate)
                improved = True

        self.temperature = self.cooling_schedule(self.temperature)

        return StepReport(iteration=self.iteration, cost=self.best_cost, improved=improved)

    def get_route(self) -> List[int]:
        return self.best_route

    def get_cost(self) -> float:
        return self.best_cost
