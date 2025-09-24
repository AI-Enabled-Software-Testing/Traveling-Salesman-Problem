# Fitness Function
import math
from typing import List
from .model import City, TSPInstance

# Note: Traveling Salesman Problem (TSP) has the following common problem types: 
# EUC_2D: Euclidean distance in 2D (your main target).
# MAN_2D: Manhattan distance in 2D.
# ATT: Pseudo-Euclidean (used in some TSPLIB instances like att48).
# ######################################################
# BELOW ONES ARE ALSO COMMON BUT NOT SUPPORTED HERE:
# CEIL_2D: Ceiling of Euclidean distance.
# GEO: Geographical distances on a sphere.
# EXPLICIT: Distances explicitly given in a matrix.

def calculate_distance(city1: City, city2: City, edge_weight_type: str = "EUC_2D") -> float:
    """Calculate distance between two cities based on edge weight type."""
    if edge_weight_type == "EUC_2D":
        return math.sqrt((city1.x - city2.x)**2 + (city1.y - city2.y)**2)
    elif edge_weight_type == "MAN_2D":
        return abs(city1.x - city2.x) + abs(city1.y - city2.y)
    elif edge_weight_type == "ATT":
        # Pseudo-Euclidean distance
        rij = math.sqrt(((city1.x - city2.x)**2 + (city1.y - city2.y)**2) / 10.0)
        return math.ceil(rij) if rij != int(rij) else int(rij)
    else:
        raise NotImplementedError(f"Edge weight type {edge_weight_type} not supported")

def tour_distance(tour: List[int], instance: TSPInstance) -> float:
    """Calculate total distance of a tour."""
    if len(tour) != len(instance.cities):
        raise ValueError("Tour must visit all cities exactly once")
    
    total_distance = 0.0
    for i in range(len(tour)):
        current_city = instance.cities[tour[i]]
        next_city = instance.cities[tour[(i + 1) % len(tour)]]  # Return to start
        total_distance += calculate_distance(current_city, next_city, instance.edge_weight_type)
    
    return total_distance

def fitness_minimize_distance(tour: List[int], instance: TSPInstance) -> float:
    """Fitness function that minimizes tour distance (lower is better)."""
    return tour_distance(tour, instance)

def fitness_maximize_inverse_distance(tour: List[int], instance: TSPInstance) -> float:
    """Fitness function that maximizes 1/distance (higher is better)."""
    distance = tour_distance(tour, instance)
    return 1.0 / (distance + 1e-6)  # Add small epsilon to avoid division by zero

# ###########################
# Normalized Fitness Function
# ###########################

def fitness_normalized(tour: List[int], instance: TSPInstance, reference_distance: float = None) -> float:
    """Normalized fitness function (0-1 scale, higher is better)."""
    distance = tour_distance(tour, instance)
    
    if reference_distance is None:
        # Use a simple heuristic as reference (e.g., nearest neighbor approximation)
        reference_distance = estimate_reference_distance(instance)
    
    # Higher fitness for shorter distances
    return max(0, (reference_distance - distance) / reference_distance)

def estimate_reference_distance(instance: TSPInstance) -> float:
    """Estimate a reference distance using nearest neighbor heuristic."""
    # Simple nearest neighbor starting from city 0
    unvisited = set(range(1, len(instance.cities)))
    current = 0
    total_distance = 0.0
    
    while unvisited:
        nearest = min(
            unvisited, key=lambda city: calculate_distance(
                        instance.cities[current], 
                        instance.cities[city], 
                        instance.edge_weight_type
                    )
            )
        total_distance += calculate_distance(
            instance.cities[current], 
            instance.cities[nearest], 
            instance.edge_weight_type
        )
        current = nearest
        unvisited.remove(nearest)
    
    # Return to start
    total_distance += calculate_distance(
        instance.cities[current], 
        instance.cities[0], 
        instance.edge_weight_type
    )
    return total_distance