from dataclasses import dataclass
from typing import List, Protocol


@dataclass
class StepReport:
    iteration: int
    best_cost: float
    current_cost: float
    improved: bool


class IterativeTSPSolver(Protocol):
    """Protocol for iterative TSP algorithms.

    Contract:
    - Call `initialize(route)` once with the original permutation of cities.
    - Repeatedly call `step()`; it returns a StepReport and whether to continue.
    - `get_route()` returns the current best route
    - `get_cost()` returns the current best cost.
    """

    def initialize(self, route: List[int]) -> None: ...

    def step(self) -> StepReport: ...

    def get_route(self) -> List[int]: ...

    def get_cost(self) -> float: ...


