# Test Suite for Traveling Salesman Problem Solver

This directory contains comprehensive unit tests for the TSP solver using pytest.

## Test Structure

### TestAlgorithmSetup
Tests the setup and initialization of different TSP algorithms:
- `SimulatedAnnealing_random`
- `SimulatedAnnealing_NearestNeighbor` 
- `GeneticAlgorithm_NearestNeighbor`
- `Baseline_Random`
- Error handling for invalid algorithm names

### TestArgumentValidation
Tests command line argument validation:
- Valid TSP file arguments
- Missing file error handling
- Wrong file extension error handling
- Incorrect number of arguments

### TestOptimalTourFinding
Tests the optimal tour finding functionality:
- Error handling for non-existent files
- Integration with real TSP files from dataset

### TestPrintSummary
Tests the summary printing functionality:
- Complete data with total_iterations
- Iteration-based data with time calculations
- Proper formatting and percentage calculations

### TestMainFunction
Tests the main function execution:
- Default algorithm execution
- Custom algorithm override
- Error handling and mocking

### TestIntegrationSmokeTests
Integration tests that verify:
- All algorithms can be initialized
- Algorithms can run steps without errors
- Route and cost retrieval works correctly

## Running Tests

### Run all tests:
```bash
python -m pytest tests/ -v
```

### Run specific test class:
```bash
python -m pytest tests/test_main.py::TestAlgorithmSetup -v
```

### Run specific test:
```bash
python -m pytest tests/test_main.py::TestAlgorithmSetup::test_simulated_annealing_random_setup -v
```

## Test Coverage

The tests cover the functionality we manually verified:
- ✅ Algorithm setup with all 4 algorithm types
- ✅ Command line argument parsing (2 and 3 arguments)
- ✅ Error handling for invalid inputs
- ✅ Multiprocessing functionality (through mocking)
- ✅ Route finding and optimization
- ✅ Summary statistics printing
- ✅ Integration smoke tests for basic algorithm operations

## Dependencies

- pytest
- pytest-mock (for mocking functionality)
- All project dependencies (algorithms, utils, TSP model)

## Notes

- Tests use proper TSPInstance and City objects as required by the system
- Main function tests use extensive mocking to avoid multiprocessing complexity
- Integration tests provide smoke testing to ensure basic functionality works
- Real file tests verify compatibility with actual dataset files