import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.special import gamma

class DiscreteCuckooSearch:
    """Discrete Cuckoo Search with Differential Evolution for task offloading"""
    
    def __init__(self, config, network, tasks):
        self.config = config
        self.network = network
        self.tasks = tasks
        
        # ... (Algorithm parameters are fine) ...
        self.nest_size = config['algorithms']['cuckoo']['nest_size']
        self.max_iterations = config['algorithms']['cuckoo']['max_iterations']
        self.pa = config['algorithms']['cuckoo']['discovery_probability']
        self.alpha = config['algorithms']['cuckoo']['step_size_factor']
        self.beta = config['algorithms']['cuckoo']['levy_beta']
        self.scaling_factor = config['algorithms']['differential_evolution']['scaling_factor']
        self.crossover_prob = config['algorithms']['differential_evolution']['crossover_probability']
        
        self.dimension = len(tasks) if tasks else 0 # Fix: use tasks length
        self.nests = None
        self.fitness = None
        self.best_nest = None
        self.best_fitness = float('inf')
        
        # Helper to find source vehicles quickly
        self.source_vehicle_map = {}
        for mv in self.network.mission_vehicles:
            if mv.current_task:
                self.source_vehicle_map[mv.current_task.id] = mv

    def initialize_nests(self):
        # ... (This function is fine) ...
        if self.dimension == 0:
            self.nests = np.array([])
            self.fitness = np.array([])
            self.best_nest = np.array([])
            return
            
        self.nests = np.random.rand(self.nest_size, self.dimension)
        self.fitness = np.full(self.nest_size, float('inf'))
        
        for i in range(self.nest_size):
            self.fitness[i] = self.evaluate_fitness(self.nests[i])
            
        best_idx = np.argmin(self.fitness)
        self.best_nest = self.nests[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]
        
    def levy_flight(self):
        # ... (This function is fine) ...
        sigma_u = (gamma(1 + self.beta) * np.sin(np.pi * self.beta / 2) / 
                  (gamma((1 + self.beta) / 2) * self.beta * 2**((self.beta - 1) / 2)))**(1 / self.beta)
        sigma_v = 1
        u = np.random.normal(0, sigma_u, self.dimension)
        v = np.random.normal(0, sigma_v, self.dimension)
        step = 0.01 * u / (np.abs(v)**(1 / self.beta) + 1e-10)
        return step
    
    def continuous_to_discrete(self, continuous_solution):
        # ... (This function is fine) ...
        discrete_solution = np.zeros(len(continuous_solution), dtype=int)
        for i, value in enumerate(continuous_solution):
            if value < 0.33:
                discrete_solution[i] = 0
            elif value < 0.66:
                discrete_solution[i] = 1
            else:
                discrete_solution[i] = 2
        return discrete_solution
    
    def evaluate_fitness(self, solution):
        """Evaluate fitness (average task processing delay)"""
        if not self.tasks or self.dimension == 0:
            return 0.0
            
        discrete_solution = self.continuous_to_discrete(solution)
        total_delay = 0
        
        for i, task in enumerate(self.tasks):
            if i >= len(discrete_solution):
                break
            decision = discrete_solution[i]
            
            if decision == 0:
                # (Scenario 1) Local execution
                delay = self._calculate_local_delay(task)
            elif decision == 1:
                # (Scenario 2) Offload to collaborative vehicle (tp_m,n)
                delay = self._calculate_cv_offload_delay(task)
            else:
                # (Scenario 3) Offload to access point (tp_m,k or tp_m,k,k')
                delay = self._calculate_ap_offload_delay(task)
                
            total_delay += delay
            
        return total_delay / len(self.tasks) if self.tasks else 0.0
    
    def _get_source_vehicle(self, task):
        """Helper to find the MV that generated the task"""
        return self.source_vehicle_map.get(task.id)

    def _calculate_transmission_delay(self, data_size, distance, bandwidth_mhz=1.0, is_fiber=False):
        """Calculate transmission delay"""
        if is_fiber:
            # Simplified fiber delay (e.g., 1 Gbps)
            return data_size * 8 / (1000 + 1e-6) # data_size in MB
            
        channel_quality = self.network._calculate_channel_quality(distance)
        if channel_quality <= 0:
            return float('inf')
            
        # Simplified capacity (e.g., in Mbps)
        # The paper's QoS model is complex. We simplify.
        capacity = channel_quality * (bandwidth_mhz * 1e6) / 1e6 # in Mbps
        return data_size * 8 / (capacity + 1e-6) # data_size in MB

    def _calculate_processing_delay(self, task, resources):
        """Calculate processing delay (tul)"""
        if resources <= 0:
            return float('inf')
        return task.computation_demand / resources

    def _calculate_local_delay(self, task):
        """(Scenario 1) Local execution delay"""
        source_vehicle = self._get_source_vehicle(task)
        if not source_vehicle:
            return 10.0 # Penalty
        # Assuming 1.0 GHz local processing as in old code
        return self._calculate_processing_delay(task, 1.0) 
    
    def _calculate_cv_offload_delay(self, task):
        """(Scenario 2) Calculate delay for offloading to CV (tp_m,n)"""
        source_vehicle = self._get_source_vehicle(task)
        if not source_vehicle:
            return 10.0 # Penalty
            
        min_delay = float('inf')
        
        for cv in self.network.collaborative_vehicles:
            # Check if CV has the required service cached
            if task.required_service in cv.cached_services:
                distance = source_vehicle.distance_to(cv)
                if distance <= source_vehicle.communication_range:
                    
                    # tup_m,n
                    transmission_delay = self._calculate_transmission_delay(task.data_size, distance)
                    
                    # tul_m,n
                    processing_delay = self._calculate_processing_delay(task, cv.computing_resources)
                    
                    # tp_m,n = tup_m,n + tul_m,n (Eq. 12)
                    total_delay = transmission_delay + processing_delay
                    min_delay = min(min_delay, total_delay)
                    
        return min_delay if min_delay != float('inf') else 10.0 # Return penalty if no CV found

    def _calculate_ap_offload_delay(self, task):
        """(Scenario 3) Calculate delay for offloading to AP (tp_m,k or tp_m,k,k')"""
        source_vehicle = self._get_source_vehicle(task)
        if not source_vehicle:
            return 10.0 # Penalty
            
        if not self.network.access_points:
            return 10.0 # Penalty

        # Find closest (associated) AP
        aps_by_distance = sorted(
            self.network.access_points,
            key=lambda ap: ap.distance_to(source_vehicle)
        )
        
        associated_ap = aps_by_distance[0]
        distance_to_ap = associated_ap.distance_to(source_vehicle)
        
        if distance_to_ap > source_vehicle.communication_range:
            return 10.0 # No AP in range

        # Calculate delay for MV-to-AP transmission (tup_m,k)
        tup_m_k = self._calculate_transmission_delay(task.data_size, distance_to_ap)
        
        # Check if associated AP can handle the task (simplified)
        if associated_ap.can_accept_connection():
            # AP can handle it. Calculate tp_m,k = tup_m,k + tul_m,k (Eq. 13)
            tul_m_k = self._calculate_processing_delay(task, associated_ap.computing_resources)
            return tup_m_k + tul_m_k
        else:
            # AP is busy, must forward to another AP (k')
            # Find the next-closest AP (k')
            if len(aps_by_distance) < 2:
                return 10.0 # No other AP to forward to
                
            forward_ap = aps_by_distance[1]
            
            # ts_k,k' (AP-to-AP fiber delay)
            ts_k_k_prime = self._calculate_transmission_delay(task.data_size, 0, is_fiber=True)
            
            # tul_m,k' (Processing delay at forward AP)
            tul_m_k_prime = self._calculate_processing_delay(task, forward_ap.computing_resources)
            
            # tp_m,k,k' = tup_m,k + ts_k,k' + tul_m,k' (Eq. 14)
            return tup_m_k + ts_k_k_prime + tul_m_k_prime
    
    def differential_evolution(self, nest_idx):
        # ... (This function is fine) ...
        if self.nest_size < 3:
            return
        indices = list(range(self.nest_size))
        indices.remove(nest_idx)
        if len(indices) < 2:
            return
        p, q = np.random.choice(indices, 2, replace=False)
        
        mutant = self.nests[nest_idx] + self.scaling_factor * (self.nests[p] - self.nests[q])
        mutant = np.clip(mutant, 0, 1)
        
        trial = np.copy(self.nests[nest_idx])
        for j in range(self.dimension):
            if np.random.random() < self.crossover_prob:
                trial[j] = mutant[j]
                
        trial_fitness = self.evaluate_fitness(trial)
        if trial_fitness < self.fitness[nest_idx]:
            self.nests[nest_idx] = trial
            self.fitness[nest_idx] = trial_fitness
            
    def run(self):
        """Execute the discrete cuckoo search algorithm"""
        if not self.tasks or self.dimension == 0:
            return np.array([]), 0.0, [] # <-- RETURN EMPTY HISTORY
            
        self.initialize_nests()
        
        fitness_history = [] # <-- ADDED FOR GRAPHING
        
        for iteration in range(self.max_iterations):
            # Global search using Levy flights
            for i in range(self.nest_size):
                step = self.levy_flight()
                new_nest = self.nests[i] + self.alpha * step * (self.best_nest - self.nests[i])
                new_nest = np.clip(new_nest, 0, 1)
                
                new_fitness = self.evaluate_fitness(new_nest)
                
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
            
            fitness_history.append(self.best_fitness) # <-- SAVE BEST FITNESS EACH ITERATION
                    
        return self.continuous_to_discrete(self.best_nest), self.best_fitness, fitness_history # <-- RETURN HISTORY