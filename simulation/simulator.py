import numpy as np
import yaml
from typing import Dict, List
from core.entities import MissionVehicle, CollaborativeVehicle, AccessPoint, Task
from core.network import VehicularNetwork
from algorithms.stable_matching import StableMatchingAlgorithm
from algorithms.cuckoo_search import DiscreteCuckooSearch
from utils.helpers import generate_zipf_popularity

class ECORASimulator:
    """Main simulation engine for ECORA strategy"""
    
    def __init__(self, config_file: str, sim_params: dict = None):
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        if sim_params:
            if 'discovery_probability' in sim_params:
                self.config['algorithms']['cuckoo']['discovery_probability'] = sim_params['discovery_probability']
            if 'step_size_factor' in sim_params:
                self.config['algorithms']['cuckoo']['step_size_factor'] = sim_params['step_size_factor']
            if 'average_speed_kmh' in sim_params:
                self.config['vehicles']['average_speed_kmh'] = sim_params['average_speed_kmh']
            if 'task_generation_prob' in sim_params:
                self.config['tasks']['task_generation_prob'] = sim_params['task_generation_prob']
                
        self.network = None
        self.tasks = []
        self.current_time = 0
        self.dcs_algorithm = None 
        
        self.service_popularities = generate_zipf_popularity(
            self.config['services']['total_services'],
            self.config['services']['zipf_parameter']
        )
        self.service_ids = np.arange(self.config['services']['total_services'])

        
    def initialize_network(self):
        """Initialize network with vehicles and infrastructure"""
        area_size = tuple(self.config['simulation']['area_size'])
        self.network = VehicularNetwork(area_size, self.config['channel'])
        
        speed_kmh = self.config['vehicles']['average_speed_kmh']
        speed_ms = (speed_kmh * 1000) / 3600
        
        road_width = 50.0
        road_center_y = area_size[1] / 2.0
        
        def get_random_road_position():
            x = np.random.rand() * area_size[0]
            y = road_center_y + (np.random.rand() - 0.5) * road_width
            return np.array([x, y])
            
        def get_road_velocity(speed):
            direction = 1.0 if np.random.rand() > 0.5 else -1.0
            return np.array([speed * direction, 0.0])

        for i in range(self.config['vehicles']['mission_vehicles']):
            mv = MissionVehicle(
                id=i, position=get_random_road_position(), velocity=get_road_velocity(speed_ms),
                communication_range=self.config['vehicles']['communication_range'],
                computing_resources=1.0, storage_resources=100
            )
            self.network.add_mission_vehicle(mv)
            
        for i in range(self.config['vehicles']['collaborative_vehicles']):
            cv = CollaborativeVehicle(
                id=1000 + i, position=get_random_road_position(), velocity=get_road_velocity(speed_ms),
                communication_range=self.config['vehicles']['communication_range'],
                computing_resources=2.0, storage_resources=500,
                idle_computing_resources=1.8, idle_storage_resources=400
            )
            self.network.add_collaborative_vehicle(cv)
            
        num_aps = self.config['infrastructure']['rsus']
        for i in range(num_aps):
            x_pos = (area_size[0] / num_aps) * (i + 0.5) 
            y_pos = road_center_y + (road_width / 2.0) 
            position = np.array([x_pos, y_pos])
            ap = AccessPoint(
                id=2000 + i, position=position,
                communication_range=self.config['infrastructure']['communication_range_ap'],
                computing_resources=10.0, storage_resources=10000, max_connections=20
            )
            self.network.add_access_point(ap)

            
    def generate_tasks(self):
        """Generate tasks for current time slot"""
        self.tasks = []
        task_gen_prob = self.config['tasks']['task_generation_prob']
        
        for i, mv in enumerate(self.network.mission_vehicles):
            if np.random.random() < task_gen_prob: 
                
                # --- CRITICAL FIX: Read from 'mbits' and 'gcycles' ---
                data_size = np.random.uniform(*self.config['tasks']['data_size_range_mbits'])
                comp_demand = np.random.uniform(*self.config['tasks']['computation_demand_range_gcycles'])
                
                required_service = np.random.choice(
                    self.service_ids,
                    p=self.service_popularities
                )

                task = Task(
                    id=self.current_time * 1000 + i,
                    data_size=data_size,       # in Mbits
                    computation_demand=comp_demand, # in GCycles
                    required_service=required_service,
                    generation_time=self.current_time,
                    deadline=self.current_time + 10,
                    source_vehicle=mv.id
                )
                self.tasks.append(task)
                mv.current_task = task
        if self.current_time == 0:
             print(f"  > Generated {len(self.tasks)} tasks (Prob: {task_gen_prob})")

                
    def run_time_slot(self):
        """Execute one time slot of simulation"""
        if self.current_time == 0:
            print(f"\n--- Running Time Slot 0 ---")
            
        for mv in self.network.mission_vehicles:
            mv.update_position(1.0)
            mv.position[0] = mv.position[0] % self.config['simulation']['area_size'][0]
        for cv in self.network.collaborative_vehicles:
            cv.update_position(1.0)
            cv.position[0] = cv.position[0] % self.config['simulation']['area_size'][0]
        
        self.network.update_connectivity()
        self.generate_tasks()
        
        if self.current_time == 0:
            print("  > Solving Sub-problem 1: Service Caching (Stable Matching)...")
            
        sm_algorithm = StableMatchingAlgorithm(
            self.network, 
            self.service_popularities,
            self.config['services']['total_services']
        )
        cache_matching = sm_algorithm.run()
        
        for cv_id, ap_id in cache_matching.items():
            cv = next((cv for cv in self.network.collaborative_vehicles if cv.id == cv_id), None)
            if cv:
                num_to_cache = min(3, self.config['services']['total_services'])
                cv.cached_services = np.argsort(self.service_popularities)[-num_to_cache:].tolist()
        
        if self.current_time == 0:
             print(f"  > Caching complete. {len(cache_matching)} CVs matched.")

        history = None
        if self.tasks:
            if self.current_time == 0:
                print("  > Solving Sub-problem 2: Task Offloading (Discrete Cuckoo Search)...")
                print(f"    > Starting {self.config['algorithms']['cuckoo']['max_iterations']} iterations...")

            self.dcs_algorithm = DiscreteCuckooSearch(self.config, self.network, self.tasks)
            offloading_solution, avg_delay_ms, history = self.dcs_algorithm.run()
            
            if self.current_time == 0:
                # We will print in MS, but the result is in SECONDS
                print(f"  > Cuckoo Search complete. Final optimized delay: {avg_delay_ms * 1000:.4f} ms") # Convert S to MS for print
            
            if self.current_time == 0:
                return history
        else:
            if self.current_time == 0:
                print("  > No tasks generated. Skipping Cuckoo Search.")
                return [] 
        
        return None