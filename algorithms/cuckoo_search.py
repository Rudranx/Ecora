import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.special import gamma

class DiscreteCuckooSearch:
    """Discrete Cuckoo Search with Differential Evolution for task offloading"""
    
    def __init__(self, config, network, tasks):
        self.config = config
        self.network = network
        self.tasks = tasks
        
        # Algorithm parameters
        self.nest_size = config['algorithms']['cuckoo']['nest_size']
        self.max_iterations = config['algorithms']['cuckoo']['max_iterations']
        self.pa = config['algorithms']['cuckoo']['discovery_probability']
        self.alpha = config['algorithms']['cuckoo']['step_size_factor']
        self.beta = config['algorithms']['cuckoo']['levy_beta']
        
        # DE parameters
        self.scaling_factor = config['algorithms']['differential_evolution']['scaling_factor']
        self.crossover_prob = config['algorithms']['differential_evolution']['crossover_probability']
        
        # Solution representation
        self.dimension = len(tasks) if tasks else 1  # Fix: use tasks length
        self.nests = None
        self.fitness = None
        self.best_nest = None
        self.best_fitness = float('inf')
        
    def initialize_nests(self):
        """Initialize nest positions randomly"""
        self.nests = np.random.rand(self.nest_size, self.dimension)
        self.fitness = np.full(self.nest_size, float('inf'))
        
        for i in range(self.nest_size):
            self.fitness[i] = self.evaluate_fitness(self.nests[i])
            
        best_idx = np.argmin(self.fitness)
        self.best_nest = self.nests[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]
        
    def levy_flight(self):
        """Generate Levy flight step"""
        # Levy distribution parameters
        sigma_u = (gamma(1 + self.beta) * np.sin(np.pi * self.beta / 2) / 
                  (gamma((1 + self.beta) / 2) * self.beta * 2**((self.beta - 1) / 2)))**(1 / self.beta)
        sigma_v = 1
        
        u = np.random.normal(0, sigma_u, self.dimension)
        v = np.random.normal(0, sigma_v, self.dimension)
        
        step = 0.01 * u / (np.abs(v)**(1 / self.beta) + 1e-10)  # Add small epsilon
        return step
    
    def continuous_to_discrete(self, continuous_solution):
        """Convert continuous solution to discrete offloading decisions"""
        discrete_solution = np.zeros(len(continuous_solution), dtype=int)
        
        for i, value in enumerate(continuous_solution):
            # Map continuous value to discrete offloading decision
            # 0: local execution, 1: offload to CV, 2: offload to AP
            if value < 0.33:
                discrete_solution[i] = 0
            elif value < 0.66:
                discrete_solution[i] = 1
            else:
                discrete_solution[i] = 2
                
        return discrete_solution
    
    def evaluate_fitness(self, solution):
        """Evaluate fitness (average task processing delay)"""
        if not self.tasks:
            return 0.0
            
        discrete_solution = self.continuous_to_discrete(solution)
        total_delay = 0
        
        for i, task in enumerate(self.tasks):
            if i >= len(discrete_solution):
                break
                
            decision = discrete_solution[i]
            
            if decision == 0:
                # Local execution
                delay = self._calculate_local_delay(task)
            elif decision == 1:
                # Offload to collaborative vehicle
                delay = self._calculate_cv_offload_delay(task, i)
            else:
                # Offload to access point
                delay = self._calculate_ap_offload_delay(task, i)
                
            total_delay += delay
            
        return total_delay / len(self.tasks) if self.tasks else 0.0
    
    def _calculate_local_delay(self, task):
        """Calculate local execution delay"""
        # Simplified model
        processing_delay = task.computation_demand / 1.0  # Assuming 1 GHz local processing
        return processing_delay
    
    def _calculate_cv_offload_delay(self, task, task_idx):
        """Calculate delay for offloading to collaborative vehicle"""
        cvs = self.network.collaborative_vehicles
        if not cvs:
            return 10.0  # Return high penalty instead of inf
            
        # Find the source vehicle for this task
        source_vehicle = None
        for mv in self.network.mission_vehicles:
            if mv.current_task and mv.current_task.id == task.id:
                source_vehicle = mv
                break
                
        if not source_vehicle:
            # If no source vehicle found, use the first mission vehicle
            if self.network.mission_vehicles:
                source_vehicle = self.network.mission_vehicles[0]
            else:
                return 10.0
            
        min_delay = 10.0  # Set max delay instead of inf
        
        for cv in cvs:
            # Check if CV has the required service cached
            if task.required_service in cv.cached_services:
                distance = source_vehicle.distance_to(cv)
                if distance <= source_vehicle.communication_range:
                    # Transmission delay
                    channel_quality = max(0.1, self.network._calculate_channel_quality(distance))
                    transmission_delay = task.data_size / (channel_quality * 10)
                    
                    # Processing delay
                    available_resources = max(0.1, cv.computing_resources)
                    processing_delay = task.computation_demand / available_resources
                    
                    total_delay = transmission_delay + processing_delay
                    min_delay = min(min_delay, total_delay)
                    
        return min_delay
    
    def _calculate_ap_offload_delay(self, task, task_idx):
        """Calculate delay for offloading to access point"""
        aps = self.network.access_points
        if not aps:
            return 10.0  # Return high penalty instead of inf
            
        # Find the source vehicle for this task
        source_vehicle = None
        for mv in self.network.mission_vehicles:
            if mv.current_task and mv.current_task.id == task.id:
                source_vehicle = mv
                break
                
        if not source_vehicle:
            # If no source vehicle found, use the first mission vehicle
            if self.network.mission_vehicles:
                source_vehicle = self.network.mission_vehicles[0]
            else:
                return 10.0
            
        min_delay = 10.0  # Set max delay instead of inf
        
        for ap in aps:
            distance = ap.distance_to(source_vehicle)
            if distance <= source_vehicle.communication_range and ap.can_accept_connection():
                # Transmission delay
                channel_quality = max(0.1, self.network._calculate_channel_quality(distance))
                transmission_delay = task.data_size / (channel_quality * 20)
                
                # Processing delay
                available_resources = max(0.1, ap.computing_resources)
                processing_delay = task.computation_demand / available_resources
                
                total_delay = transmission_delay + processing_delay
                min_delay = min(min_delay, total_delay)
                
        return min_delay
    
    def differential_evolution(self, nest_idx):
        """Apply differential evolution operations"""
        if self.nest_size < 3:
            return  # Need at least 3 nests for DE
            
        # Select random indices for mutation
        indices = list(range(self.nest_size))
        indices.remove(nest_idx)
        
        if len(indices) < 2:
            return
            
        p, q = np.random.choice(indices, 2, replace=False)
        
        # Mutation
        mutant = self.nests[nest_idx] + self.scaling_factor * (self.nests[p] - self.nests[q])
        mutant = np.clip(mutant, 0, 1)
        
        # Crossover
        trial = np.copy(self.nests[nest_idx])
        for j in range(self.dimension):
            if np.random.random() < self.crossover_prob:
                trial[j] = mutant[j]
                
        # Selection
        trial_fitness = self.evaluate_fitness(trial)
        if trial_fitness < self.fitness[nest_idx]:
            self.nests[nest_idx] = trial
            self.fitness[nest_idx] = trial_fitness
            
    def run(self):
        """Execute the discrete cuckoo search algorithm"""
        if not self.tasks:
            return np.array([]), 0.0
            
        self.initialize_nests()
        
        for iteration in range(self.max_iterations):
            # Global search using Levy flights
            for i in range(self.nest_size):
                # Generate new solution via Levy flight
                step = self.levy_flight()
                new_nest = self.nests[i] + self.alpha * step * (self.best_nest - self.nests[i])
                new_nest = np.clip(new_nest, 0, 1)
                
                # Evaluate new solution
                new_fitness = self.evaluate_fitness(new_nest)
                
                # Replace if better
                if new_fitness < self.fitness[i]:
                    self.nests[i] = new_nest
                    self.fitness[i] = new_fitness
                    
                    if new_fitness < self.best_fitness:
                        self.best_nest = new_nest.copy()
                        self.best_fitness = new_fitness
                        
            # Local search with differential evolution
            for i in range(self.nest_size):
                if np.random.random() < self.pa:
                    self.differential_evolution(i)
                    
        return self.continuous_to_discrete(self.best_nest), self.best_fitness