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
    edge_weight_type: str = "EUC_2D" # Only EUC_2D is supported

    def distance(self, i: int = None, j: int = None, alg: str = None) -> float:
        """Calculate distance between ONLY TWO cities by their indices."""
        from .controller import calculate_distance, tour_distance
        match alg:
            case "EUC_2D" | "MAN_2D" | "ATT":
                if i is None or j is None:
                    return tour_distance(self.cities, self)
                return calculate_distance(self.cities[i], self.cities[j], self.edge_weight_type)
            case _:
                if i is None or j is None:
                    # Calculate full distance matrix
                    return tour_distance(self.cities, self)
                a = self.cities[i]
                b = self.cities[j]
                dx = a.x - b.x
                dy = a.y - b.y
                return math.hypot(dx, dy)

    def route_cost(self, route: List[int], norm: str = None) -> float:
        from .controller import calculate_fitness
        match norm:
            case "max":
                return calculate_fitness([self.cities[i] for i in route], self, maximize=True)
            case "min":
                return calculate_fitness([self.cities[i] for i in route], self, maximize=False)
            case _: # Default: total distance
                total = 0.0
                for k in range(len(route)):
                    i = route[k]
                    j = route[(k + 1) % len(route)]
                    total += self.distance(i, j, self.edge_weight_type)
                return total



