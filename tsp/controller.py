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

def tour_distance(tour: List[City], instance: TSPInstance) -> float:
    """Calculate total distance of a tour."""
    if len(tour) != len(instance.cities):
        raise ValueError("Tour must visit all cities exactly once")
    
    total_distance = 0.0
    for i in range(len(tour)):
        current_city = tour[i]
        next_city = tour[(i + 1) % len(tour)]  # Return to start
        total_distance += calculate_distance(current_city, next_city, instance.edge_weight_type)
    return total_distance

def fitness_minimize_distance(tour: List[City], instance: TSPInstance) -> float:
    """Fitness function that minimizes tour distance (lower is better)."""
    return tour_distance(tour, instance)

def fitness_maximize_inverse_distance(tour: List[City], instance: TSPInstance) -> float:
    """Fitness function that maximizes 1/distance (higher is better)."""
    distance = tour_distance(tour, instance)
    if distance == 0:
        distance = 1e-6  # Prevent division by zero
    return 1.0 / distance  # Add small epsilon to avoid division by zero

def calculate_fitness(tour: List[City], instance: TSPInstance, maximize: bool = False) -> float:
    """Calculate fitness of a tour based on whether to maximize or minimize distance."""
    if maximize:
        return fitness_maximize_inverse_distance(tour, instance)
    else:
        return fitness_minimize_distance(tour, instance)

# ###########################
# Normalized Fitness Function
# ###########################

def normalize(fitness: float) -> float:
    """Normalize fitness to [0, 1] range."""
    return 1 / (1 + fitness)