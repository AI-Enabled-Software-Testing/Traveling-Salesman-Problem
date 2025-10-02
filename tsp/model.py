from dataclasses import dataclass
from typing import List
import math


@dataclass
class City:
    id: int
    x: float
    y: float


@dataclass
class TSPInstance:
    name: str
    cities: List[City]

    def distance(self, i: int, j: int) -> float:
        """Calculate Euclidean distance between two cities by their indices."""
        a = self.cities[i]
        b = self.cities[j]
        dx = a.x - b.x
        dy = a.y - b.y
        return math.hypot(dx, dy)

    def route_cost(self, route: List[int]) -> float:
        """Calculate total Euclidean distance of a route."""
        total = 0.0
        for k in range(len(route)):
            i = route[k]
            j = route[(k + 1) % len(route)]
            total += self.distance(i, j)
        return total



