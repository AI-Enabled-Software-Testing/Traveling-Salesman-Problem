from pathlib import Path
from typing import List

from .model import City, TSPInstance


def parse_tsplib_tsp(path: Path) -> TSPInstance:
    """Parse a TSPLIB format .tsp file (EUC_2D only)."""
    name = path.stem
    cities: List[City] = []
    in_coords = False
    dimension = 0  # For validation
    city_count = 0  # Track parsed cities
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.upper().startswith("NAME"):
                try:
                    name = line.split(":", 1)[1].strip()
                except Exception:
                    pass
            # Validation
            if line.upper().startswith("TYPE"):
                if "TSP" not in line.upper():
                    raise NotImplementedError(f"Unsupported TSPLIB type in {path}: {line}")
            if line.upper().startswith("DIMENSION"):
                try:
                    dimension = int(line.split(":", 1)[1].strip())
                except Exception:
                    pass
            if line.upper().startswith("NODE_COORD_SECTION"):
                in_coords = True
                continue
            if line.upper().startswith("EOF"):
                if dimension > 0:
                    assert city_count == dimension, f"Expected {dimension} cities, but parsed {city_count} in {path}"
                break
            if in_coords:
                parts = line.split()
                if len(parts) >= 3:
                    city_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    # TSPLIB ids are 1-based; store as 0-based internally
                    cities.append(City(id=city_id - 1, x=x, y=y))
                    city_count += 1
    if not cities:
        raise ValueError(f"No cities parsed from {path}")
    return TSPInstance(name=name, cities=cities)


