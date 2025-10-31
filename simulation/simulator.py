import numpy as np
import yaml
from typing import Dict, List
from core.entities import MissionVehicle, CollaborativeVehicle, AccessPoint, Task
from core.network import VehicularNetwork
from algorithms.stable_matching import StableMatchingAlgorithm
from algorithms.cuckoo_search import DiscreteCuckooSearch

class ECORASimulator:
    """Main simulation engine for ECORA strategy"""
    
    def __init__(self, config_file: str):
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.network = None
        self.tasks = []
        self.current_time = 0
        self.results = {
            'average_delay': [],
            'load_balance': [],
            'cache_hit_rate': [],
            'offload_success_rate': []
        }
        
    def initialize_network(self):
        """Initialize network with vehicles and infrastructure"""
        area_size = tuple(self.config['simulation']['area_size'])
        self.network = VehicularNetwork(area_size)
        
        # Create mission vehicles
        for i in range(self.config['vehicles']['mission_vehicles']):
            position = np.random.rand(2) * area_size
            speed = np.random.uniform(*self.config['vehicles']['speed_range'])
            angle = np.random.uniform(0, 2 * np.pi)
            velocity = np.array([speed * np.cos(angle), speed * np.sin(angle)])
            
            mv = MissionVehicle(
                id=i,
                position=position,
                velocity=velocity,
                communication_range=self.config['vehicles']['communication_range'],
                computing_resources=1.0,  # GHz
                storage_resources=100  # MB
            )
            self.network.add_mission_vehicle(mv)
            
        # Create collaborative vehicles
        for i in range(self.config['vehicles']['collaborative_vehicles']):
            position = np.random.rand(2) * area_size
            speed = np.random.uniform(*self.config['vehicles']['speed_range'])
            angle = np.random.uniform(0, 2 * np.pi)
            velocity = np.array([speed * np.cos(angle), speed * np.sin(angle)])
            
            cv = CollaborativeVehicle(
                id=1000 + i,
                position=position,
                velocity=velocity,
                communication_range=self.config['vehicles']['communication_range'],
                computing_resources=2.0,  # GHz
                storage_resources=500,  # MB
                idle_computing_resources=1.8,
                idle_storage_resources=400
            )
            self.network.add_collaborative_vehicle(cv)
            
        # Create access points
        for i in range(self.config['infrastructure']['rsus']):
            position = np.random.rand(2) * area_size
            
            ap = AccessPoint(
                id=2000 + i,
                position=position,
                communication_range=500,  # meters
                computing_resources=10.0,  # GHz
                storage_resources=10000,  # MB
                max_connections=20
            )
            self.network.add_access_point(ap)
            
    def generate_tasks(self):
        """Generate tasks for current time slot"""
        self.tasks = []
        
        for i, mv in enumerate(self.network.mission_vehicles):
            if np.random.random() < 0.7:  # 70% probability of task generation
                task = Task(
                    id=self.current_time * 1000 + i,
                    data_size=np.random.uniform(*self.config['tasks']['data_size_range']),
                    computation_demand=np.random.uniform(*self.config['tasks']['computation_demand_range']),
                    required_service=np.random.randint(0, self.config['services']['total_services']),
                    generation_time=self.current_time,
                    deadline=self.current_time + 10,
                    source_vehicle=mv.id
                )
                self.tasks.append(task)
                mv.current_task = task
                
    def run_time_slot(self):
        """Execute one time slot of simulation"""
        # Update vehicle positions
        for mv in self.network.mission_vehicles:
            mv.update_position(1.0)
            # Boundary conditions
            mv.position = np.clip(mv.position, 0, self.network.area_size)
            
        for cv in self.network.collaborative_vehicles:
            cv.update_position(1.0)
            cv.position = np.clip(cv.position, 0, self.network.area_size)
            
        # Update network connectivity
        self.network.update_connectivity()
        
        # Generate tasks
        self.generate_tasks()
        
        # Run stable matching for service caching
        services = list(range(self.config['services']['total_services']))
        sm_algorithm = StableMatchingAlgorithm(self.network, services)
        cache_matching = sm_algorithm.run()
        
        # Update cached services based on matching
        for cv_id, ap_id in cache_matching.items():
            cv = next((cv for cv in self.network.collaborative_vehicles if cv.id == cv_id), None)
            if cv:
                # Simplified: randomly cache some services
                num_services = min(3, self.config['services']['total_services'])
                cv.cached_services = np.random.choice(services, num_services, replace=False).tolist()
                
        # Run discrete cuckoo search for task offloading
        if self.tasks:
            dcs_algorithm = DiscreteCuckooSearch(self.config, self.network, self.tasks)
            offloading_solution, avg_delay = dcs_algorithm.run()
            
            # Record results
            self.results['average_delay'].append(avg_delay)
            
            # Calculate other metrics
            self._calculate_metrics(offloading_solution)
            
    # <-- NOTE: The indentation for the following methods is corrected -->
    
    def _calculate_metrics(self, offloading_solution):
        # Handle empty solution
        if len(offloading_solution) == 0:
            return
            
        # Load balance (variance of AP loads)
        ap_loads = {ap.id: 0 for ap in self.network.access_points}
        for decision in offloading_solution:
            if decision == 2:  # Offloaded to AP
                # Simplified: randomly assign to an AP
                if self.network.access_points:
                    ap = np.random.choice(self.network.access_points)
                    ap_loads[ap.id] += 1
                    
        if ap_loads:
            load_variance = np.var(list(ap_loads.values()))
            self.results['load_balance'].append(load_variance)
        else:
            self.results['load_balance'].append(0)
            
        # Cache hit rate (simplified)
        cache_hits = sum(1 for decision in offloading_solution if decision == 1)
        hit_rate = cache_hits / len(offloading_solution) if len(offloading_solution) > 0 else 0
        self.results['cache_hit_rate'].append(hit_rate)
        
        # Offload success rate
        offload_success = sum(1 for decision in offloading_solution if decision > 0)
        success_rate = offload_success / len(offloading_solution) if len(offloading_solution) > 0 else 0
        self.results['offload_success_rate'].append(success_rate)
            
    def run(self):
        """Run complete simulation"""
        print("Initializing ECORA simulation...")
        self.initialize_network()
        
        duration = self.config['simulation']['duration']
        print(f"Running simulation for {duration} time slots...")
        
        for t in range(duration):
            self.current_time = t
            self.run_time_slot()
            
            if (t + 1) % 100 == 0:
                avg_delay = np.mean(self.results['average_delay'][-100:]) if self.results['average_delay'] else 0
                print(f"Time slot {t + 1}/{duration} - Avg delay: {avg_delay:.4f} ms")
                
        print("Simulation completed!")
        return self.results