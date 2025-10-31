import unittest
import numpy as np
import sys
sys.path.append('..')

from core.entities import MissionVehicle, CollaborativeVehicle, AccessPoint, Task
from core.network import VehicularNetwork
from algorithms.stable_matching import StableMatchingAlgorithm
from algorithms.cuckoo_search import DiscreteCuckooSearch

class TestAlgorithms(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        self.network = VehicularNetwork((1000, 1000))
        
        # Add test vehicles
        for i in range(5):
            mv = MissionVehicle(
                id=i,
                position=np.random.rand(2) * 1000,
                velocity=np.random.rand(2) * 30,
                communication_range=250,
                computing_resources=1.0,
                storage_resources=100
            )
            self.network.add_mission_vehicle(mv)
            
        for i in range(3):
            cv = CollaborativeVehicle(
                id=100+i,
                position=np.random.rand(2) * 1000,
                velocity=np.random.rand(2) * 30,
                communication_range=250,
                computing_resources=2.0,
                storage_resources=500,
                idle_computing_resources=1.8,
                idle_storage_resources=400
            )
            self.network.add_collaborative_vehicle(cv)
            
        for i in range(2):
            ap = AccessPoint(
                id=200+i,
                position=np.random.rand(2) * 1000,
                communication_range=500,
                computing_resources=10.0,
                storage_resources=10000,
                max_connections=20
            )
            self.network.add_access_point(ap)
            
    def test_stable_matching(self):
        """Test stable matching algorithm"""
        services = list(range(5))
        sm_algo = StableMatchingAlgorithm(self.network, services)
        matching = sm_algo.run()
        
        self.assertIsInstance(matching, dict)
        # Check that matched CVs exist
        for cv_id in matching.keys():
            self.assertIn(cv_id, [cv.id for cv in self.network.collaborative_vehicles])
            
    def test_cuckoo_search(self):
        """Test discrete cuckoo search algorithm"""
        # Create test tasks
        tasks = []
        for i in range(5):
            task = Task(
                id=i,
                data_size=0.4,
                computation_demand=0.4,
                required_service=i % 3,
                generation_time=0,
                deadline=10,
                source_vehicle=i
            )
            tasks.append(task)
            
        config = {
            'algorithms': {
                'cuckoo': {
                    'nest_size': 10,
                    'max_iterations': 20,
                    'discovery_probability': 0.25,
                    'step_size_factor': 1.0,
                    'levy_beta': 1.5
                },
                'differential_evolution': {
                    'scaling_factor': 0.5,
                    'crossover_probability': 0.9
                }
            }
        }
        
        dcs_algo = DiscreteCuckooSearch(config, self.network, tasks)
        solution, fitness = dcs_algo.run()
        
        self.assertEqual(len(solution), len(tasks))
        self.assertIsInstance(fitness, float)
        # Check that all decisions are valid (0, 1, or 2)
        for decision in solution:
            self.assertIn(decision, [0, 1, 2])

if __name__ == '__main__':
    unittest.main()