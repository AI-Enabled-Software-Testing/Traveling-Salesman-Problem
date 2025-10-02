"""
Unit tests for the Traveling Salesman Problem solver.

Tests cover:
- Algorithm setup and initialization
- Main function execution with different algorithms
- Route finding and optimization
- Error handling
- Multiprocessing functionality
"""

import pytest
import sys
import csv
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from main import main, verify_args, print_summary
from util import setup_algorithm, find_optimal_tour
from tsp.model import TSPInstance, City


class TestAlgorithmSetup:
    """Test algorithm setup functionality."""
    
    @pytest.fixture
    def sample_instance(self):
        """Create a small sample TSP instance for testing."""
        cities = [
            City(id=0, x=0, y=0),
            City(id=1, x=1, y=0), 
            City(id=2, x=1, y=1),
            City(id=3, x=0, y=1)
        ]
        return TSPInstance(name="test", cities=cities)
    
    def test_simulated_annealing_random_setup(self, sample_instance):
        """Test SimulatedAnnealing_random algorithm setup."""
        solver, init_route = setup_algorithm("SimulatedAnnealing_random", sample_instance)
        
        assert solver is not None
        assert hasattr(solver, 'initialize')
        assert hasattr(solver, 'step')
        assert hasattr(solver, 'get_route')
        assert hasattr(solver, 'get_cost')
        assert init_route is None  # Random doesn't use nearest neighbor init
    
    def test_simulated_annealing_nn_setup(self, sample_instance):
        """Test SimulatedAnnealing_NearestNeighbor algorithm setup."""
        solver, init_route = setup_algorithm("SimulatedAnnealing_NearestNeighbor", sample_instance)
        
        assert solver is not None
        assert hasattr(solver, 'initialize')
        assert hasattr(solver, 'step')
        assert hasattr(solver, 'get_route')
        assert hasattr(solver, 'get_cost')
        assert init_route is not None  # NN should provide initial route
        assert len(init_route) == len(sample_instance.cities)
    
    def test_genetic_algorithm_setup(self, sample_instance):
        """Test GeneticAlgorithm_NearestNeighbor algorithm setup."""
        solver, init_route = setup_algorithm("GeneticAlgorithm_NearestNeighbor", sample_instance)
        
        assert solver is not None
        assert hasattr(solver, 'initialize')
        assert hasattr(solver, 'step')
        assert hasattr(solver, 'get_route')
        assert hasattr(solver, 'get_cost')
        assert init_route is not None  # GA with NN should provide initial route
        assert len(init_route) == len(sample_instance.cities)
    
    def test_random_solver_setup(self, sample_instance):
        """Test Baseline_Random algorithm setup."""
        solver, init_route = setup_algorithm("Baseline_Random", sample_instance)
        
        assert solver is not None
        assert hasattr(solver, 'initialize')
        assert hasattr(solver, 'step')
        assert hasattr(solver, 'get_route')
        assert hasattr(solver, 'get_cost')
        assert init_route is None  # Random doesn't use nearest neighbor init
    
    def test_invalid_algorithm_raises_error(self, sample_instance):
        """Test that invalid algorithm names raise ValueError."""
        with pytest.raises(ValueError, match="Unknown algorithm name"):
            setup_algorithm("InvalidAlgorithm", sample_instance)


class TestArgumentValidation:
    """Test command line argument validation."""
    
    def test_verify_args_valid_file(self):
        """Test verify_args with valid TSP file."""
        with patch('sys.argv', ['main.py', 'dataset/berlin52.tsp']):
            with patch('pathlib.Path.exists', return_value=True):
                result = verify_args()
                assert str(result).endswith('berlin52.tsp')
    
    def test_verify_args_missing_file(self):
        """Test verify_args with missing file."""
        with patch('sys.argv', ['main.py', 'nonexistent.tsp']):
            with patch('pathlib.Path.exists', return_value=False):
                with pytest.raises(FileNotFoundError):
                    verify_args()
    
    def test_verify_args_wrong_extension(self):
        """Test verify_args with wrong file extension."""
        with patch('sys.argv', ['main.py', 'file.txt']):
            with patch('pathlib.Path.exists', return_value=True):
                with pytest.raises(ValueError, match="File must have .tsp extension"):
                    verify_args()
    
    def test_verify_args_wrong_number_of_arguments(self):
        """Test verify_args with wrong number of arguments."""
        with patch('sys.argv', ['main.py']):  # Missing file argument
            with pytest.raises(ValueError, match="Usage: python main.py"):
                verify_args()


class TestOptimalTourFinding:
    """Test optimal tour finding functionality."""
    
    def test_find_optimal_tour_with_opt_file(self):
        """Test finding optimal tour when .opt.tour file exists."""
        # This test verifies the general structure - detailed testing would need real files
        # Just test that the function exists and can handle basic cases
        
        # Test with a non-existent file to verify error handling
        with pytest.raises((FileNotFoundError, ValueError)):
            find_optimal_tour("nonexistent.tsp")
    
    def test_find_optimal_tour_with_real_file(self):
        """Test with a real TSP file from the dataset."""
        # Test with berlin52.tsp if it exists
        berlin_file = Path("dataset/berlin52.tsp")
        if berlin_file.exists():
            instance, optimal_cost = find_optimal_tour(berlin_file)
            
            assert instance is not None
            assert instance.name == "berlin52"
            assert len(instance.cities) == 52
            # optimal_cost might be None if no .opt.tour file, which is fine


class TestPrintSummary:
    """Test summary printing functionality."""
    
    def test_print_summary_with_complete_data(self, capsys):
        """Test print_summary with complete data including total_iterations."""
        results = {
            "time-based results": {
                'final_cost_mean': 8000.0,
                'final_cost_std': 100.0,
                'total_iterations': 1000
            }
        }
        optimal_cost = 7500.0
        
        print_summary(results, optimal_cost)
        
        captured = capsys.readouterr()
        assert "SUMMARY" in captured.out
        assert "8000.0 ± 100.0" in captured.out
        assert "+6.7%" in captured.out  # vs optimal percentage
        assert "1000" in captured.out  # steps/sec
    
    def test_print_summary_with_iteration_data(self, capsys):
        """Test print_summary with iteration-based data (no total_iterations)."""
        results = {
            "iteration-based results": {
                'final_cost_mean': 8500.0,
                'final_cost_std': 150.0,
                'total_time_mean': 0.05  # 50ms for 1000 iterations
            }
        }
        optimal_cost = 7500.0
        
        print_summary(results, optimal_cost)
        
        captured = capsys.readouterr()
        assert "SUMMARY" in captured.out
        assert "8500.0 ± 150.0" in captured.out
        assert "+13.3%" in captured.out  # vs optimal percentage


class TestMainFunction:
    """Test the main function execution."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock all external dependencies for main function testing."""
        with patch('main.verify_args') as mock_verify, \
             patch('main.find_optimal_tour') as mock_find_tour, \
             patch('main.setup_algorithm') as mock_setup, \
             patch('multiprocessing.Pool') as mock_pool, \
             patch('main.print_summary') as mock_summary:
            
            # Setup return values
            mock_verify.return_value = Path("test.tsp")
            
            # Mock instance
            mock_instance = MagicMock()
            mock_instance.name = "test"
            mock_instance.cities = [
                City(id=0, x=0, y=0),
                City(id=1, x=1, y=0),
                City(id=2, x=1, y=1),
                City(id=3, x=0, y=1)
            ]
            mock_instance.route_cost.return_value = 4.0  # Mock route cost
            mock_find_tour.return_value = (mock_instance, 100.0)
            
            # Mock solver
            mock_solver = MagicMock()
            mock_setup.return_value = (mock_solver, [0, 1, 2, 3])
            
            # Mock pool results
            mock_pool_instance = MagicMock()
            mock_pool.return_value.__enter__.return_value = mock_pool_instance
            
            # Mock parallel run results
            mock_run_result = {
                'best_costs': [120.0, 110.0, 105.0],
                'best_route': [0, 1, 2, 3],
                'times': [0.1, 0.2, 0.3],
                'iterations': [1, 2, 3]
            }
            mock_pool_instance.map.return_value = [mock_run_result] * 10
            
            yield {
                'verify': mock_verify,
                'find_tour': mock_find_tour,
                'setup': mock_setup,
                'pool': mock_pool,
                'summary': mock_summary
            }
    
    def test_main_with_default_algorithm(self, mock_dependencies, capsys):
        """Test main function with default algorithm."""
        with patch('sys.argv', ['main.py', 'test.tsp']):
            main()
        
        # Verify key functions were called
        mock_dependencies['verify'].assert_called_once()
        mock_dependencies['find_tour'].assert_called_once()
        mock_dependencies['setup'].assert_called()
        mock_dependencies['summary'].assert_called_once()
        
        # Check that route output section is printed to stdout
        captured = capsys.readouterr()
        assert "OPTIMAL ROUTE FOUND" in captured.out
    
    def test_main_with_custom_algorithm(self, mock_dependencies):
        """Test main function with custom algorithm override."""
        with patch('sys.argv', ['main.py', 'test.tsp', 'SimulatedAnnealing_random']):
            main('SimulatedAnnealing_random')
        
        # Verify setup was called with the custom algorithm
        mock_dependencies['setup'].assert_called()
        setup_calls = mock_dependencies['setup'].call_args_list
        assert any('SimulatedAnnealing_random' in str(call) for call in setup_calls)
    
    def test_main_error_handling(self, mock_dependencies):
        """Test main function error handling."""
        # Make verify_args raise an exception
        mock_dependencies['verify'].side_effect = FileNotFoundError("File not found")
        
        with patch('sys.argv', ['main.py', 'nonexistent.tsp']):
            with pytest.raises(FileNotFoundError):
                main()


class TestCSVOutput:
    """Test CSV output functionality."""
    
    def test_csv_file_creation(self, tmp_path):
        """Test that CSV file is created with correct route data."""
        # Mock all dependencies and create a controlled test environment
        with patch('main.verify_args') as mock_verify, \
             patch('main.find_optimal_tour') as mock_find_tour, \
             patch('main.setup_algorithm') as mock_setup, \
             patch('multiprocessing.Pool') as mock_pool, \
             patch('main.print_summary') as mock_summary, \
             patch('os.path.dirname') as mock_dirname, \
             patch('os.path.abspath') as mock_abspath:
            
            # Setup mock returns
            mock_verify.return_value = Path("test.tsp")
            
            # Mock instance
            mock_instance = MagicMock()
            mock_instance.name = "test"
            mock_instance.cities = [
                City(id=0, x=0, y=0),
                City(id=1, x=1, y=0),
                City(id=2, x=1, y=1),
                City(id=3, x=0, y=1)
            ]
            mock_instance.route_cost.return_value = 4.0  # Mock route cost
            mock_find_tour.return_value = (mock_instance, 100.0)
            
            # Mock solver
            mock_solver = MagicMock()
            mock_setup.return_value = (mock_solver, [0, 1, 2, 3])
            
            # Setup CSV output path to our test directory
            test_csv_path = tmp_path / "solution.csv"
            mock_dirname.return_value = str(tmp_path)
            mock_abspath.return_value = str(tmp_path / "main.py")
            
            # Mock pool results with known route
            test_route = [0, 1, 2, 3]
            mock_run_result = {
                'best_costs': [120.0, 110.0, 105.0],
                'best_route': test_route,
                'times': [0.1, 0.2, 0.3],
                'iterations': [1, 2, 3]
            }
            
            mock_pool_instance = MagicMock()
            mock_pool.return_value.__enter__.return_value = mock_pool_instance
            mock_pool_instance.map.return_value = [mock_run_result] * 10
            
            # Run main function
            with patch('sys.argv', ['main.py', 'test.tsp']):
                main()
            
            # Verify CSV file was created
            assert test_csv_path.exists(), "CSV file should be created"
            
            # Verify CSV content
            with open(test_csv_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)
                
                # Check header
                assert 'City' in reader.fieldnames, "CSV should have 'City' column"
                
                # Check content
                assert len(rows) == len(test_route), f"CSV should have {len(test_route)} rows"
                for i, row in enumerate(rows):
                    assert int(row['City']) == test_route[i], f"Row {i} should contain city {test_route[i]}"
    
    def test_csv_content_validation(self, tmp_path):
        """Test that CSV contains valid route data and format."""
        with patch('main.verify_args') as mock_verify, \
             patch('main.find_optimal_tour') as mock_find_tour, \
             patch('main.setup_algorithm') as mock_setup, \
             patch('multiprocessing.Pool') as mock_pool, \
             patch('main.print_summary') as mock_summary, \
             patch('os.path.dirname') as mock_dirname, \
             patch('os.path.abspath') as mock_abspath:
            
            # Setup mocks
            mock_verify.return_value = Path("test.tsp")
            
            mock_instance = MagicMock()
            mock_instance.name = "test"
            mock_instance.cities = [
                City(id=0, x=0, y=0),
                City(id=1, x=1, y=0),
                City(id=2, x=1, y=1),
                City(id=3, x=0, y=1),
                City(id=4, x=0.5, y=0.5)
            ]
            mock_instance.route_cost.return_value = 5.0
            mock_find_tour.return_value = (mock_instance, 100.0)
            
            mock_solver = MagicMock()
            mock_setup.return_value = (mock_solver, [0, 1, 2, 3, 4])
            
            test_csv_path = tmp_path / "solution.csv"
            mock_dirname.return_value = str(tmp_path)
            mock_abspath.return_value = str(tmp_path / "main.py")
            
            # Use a simple consistent mock result
            mock_run_result = {
                'best_costs': [150.0, 120.0, 100.0, 80.0],
                'best_route': [0, 1, 2, 3, 4],
                'times': [0.1, 0.2, 0.3, 0.4],
                'iterations': [1, 2, 3, 4],
                'iters': [1, 2, 3, 4]
            }
            
            mock_pool_instance = MagicMock()
            mock_pool.return_value.__enter__.return_value = mock_pool_instance
            mock_pool_instance.map.return_value = [mock_run_result] * 10
            
            # Run main function
            with patch('sys.argv', ['main.py', 'test.tsp']):
                main()
            
            # Verify CSV file exists and has correct content
            assert test_csv_path.exists(), "CSV file should be created"
            
            # Read and validate CSV content
            with open(test_csv_path, 'r', newline='') as csvfile:
                content = csvfile.read()
                
                # Check that CSV has header
                assert content.startswith('City'), "CSV should start with 'City' header"
                
                # Parse CSV content
                csvfile.seek(0)
                reader = csv.DictReader(csvfile)
                route_data = [int(row['City']) for row in reader]
                
                # Verify the route structure (should have all cities)
                assert len(route_data) == 5, f"Expected 5 cities, got {len(route_data)}"
                assert all(city in [0, 1, 2, 3, 4] for city in route_data), "All cities should be valid city IDs"
                assert len(set(route_data)) == 5, "All cities should be unique (valid TSP route)"


class TestIntegrationSmokeTests:
    """Integration smoke tests using real but minimal data."""
    
    def test_algorithm_integration_smoke(self):
        """Smoke test to ensure algorithms can be initialized and run basic steps."""
        # Create minimal instance
        cities = [
            City(id=0, x=0, y=0),
            City(id=1, x=1, y=0),
            City(id=2, x=1, y=1),
            City(id=3, x=0, y=1)
        ]
        instance = TSPInstance(name="smoke_test", cities=cities)
        
        algorithms = [
            "SimulatedAnnealing_random",
            "SimulatedAnnealing_NearestNeighbor", 
            "GeneticAlgorithm_NearestNeighbor",
            "Baseline_Random"
        ]
        
        for alg_name in algorithms:
            solver, init_route = setup_algorithm(alg_name, instance)
            
            # Initialize solver
            if init_route is not None:
                solver.initialize(init_route)
            else:
                solver.initialize(list(range(len(cities))))
            
            # Run a few steps
            for _ in range(3):
                report = solver.step()
                assert hasattr(report, 'iteration')
                assert hasattr(report, 'best_cost')
                assert hasattr(report, 'current_cost')
                assert hasattr(report, 'improved')
            
            # Verify we can get route and cost
            route = solver.get_route()
            cost = solver.get_cost()
            
            assert isinstance(route, list)
            assert len(route) == len(cities)
            assert isinstance(cost, (int, float))
            assert cost > 0


if __name__ == "__main__":
    pytest.main([__file__])