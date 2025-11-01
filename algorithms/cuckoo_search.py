import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.special import gamma
from collections import defaultdict

class DiscreteCuckooSearch:
    """Discrete Cuckoo Search with Differential Evolution for task offloading"""
    
    def __init__(self, config, network, tasks):
        self.config = config
        self.network = network
        self.tasks = tasks
        
        self.cv_targets = list(network.collaborative_vehicles)
        self.ap_targets = list(network.access_points)
        self.offload_targets = self.cv_targets + self.ap_targets # Y = {CV U AP}
        self.num_targets = len(self.offload_targets)
        
        self.nest_size = config['algorithms']['cuckoo']['nest_size']
        self.max_iterations = config['algorithms']['cuckoo']['max_iterations']
        self.pa = config['algorithms']['cuckoo'].get('discovery_probability', 0.25)
        self.alpha = config['algorithms']['cuckoo'].get('step_size_factor', 1.0)
        self.beta = config['algorithms']['cuckoo']['levy_beta']
        
        self.scaling_factor = config['algorithms']['differential_evolution']['scaling_factor']
        self.crossover_prob = config['algorithms']['differential_evolution']['crossover_probability']
        
        self.dimension = len(tasks) if tasks else 0
        
        self.nests = None
        self.fitness = None
        self.best_nest = None
        self.best_fitness = float('inf')
        
        self.source_vehicle_map = {}
        for mv in self.network.mission_vehicles:
            if mv.current_task:
                self.source_vehicle_map[mv.current_task.id] = mv
                
        self.node_resources = {}
        for target in self.offload_targets:
            self.node_resources[target.id] = target.computing_resources

    def initialize_nests(self):
        if self.dimension == 0:
            self.nests, self.fitness, self.best_nest = [], [], []
            return
            
        self.nests = np.random.rand(self.nest_size, self.dimension)
        self.fitness = np.full(self.nest_size, float('inf'))
        
        for i in range(self.nest_size):
            self.fitness[i] = self.evaluate_fitness(self.nests[i])
            
        if self.fitness.size > 0:
            best_idx = np.nanargmin(self.fitness)
            if not np.isnan(self.fitness[best_idx]):
                self.best_nest = self.nests[best_idx].copy()
                self.best_fitness = self.fitness[best_idx]
            else:
                self.best_nest = self.nests[0].copy()
                self.best_fitness = 100.0 # 100 seconds
        else:
            self.best_fitness = 100.0
        
    def levy_flight(self):
        if self.dimension == 0: return np.array([])
        sigma_u = (gamma(1 + self.beta) * np.sin(np.pi * self.beta / 2) / 
                  (gamma((1 + self.beta) / 2) * self.beta * 2**((self.beta - 1) / 2)))**(1 / self.beta)
        sigma_v = 1
        u = np.random.normal(0, sigma_u, self.dimension)
        v = np.random.normal(0, sigma_v, self.dimension)
        step = 0.01 * u / (np.abs(v)**(1 / self.beta) + 1e-10)
        return step
    
    def continuous_to_discrete(self, continuous_solution):
        """Maps a continuous value [0, 1] to a discrete target node index."""
        if self.num_targets == 0:
            return np.array([], dtype=int)
            
        discrete_solution = np.zeros(len(continuous_solution), dtype=int)
        for i, value in enumerate(continuous_solution):
            target_index = int(value * self.num_targets)
            if target_index >= self.num_targets:
                target_index = self.num_targets - 1
            discrete_solution[i] = target_index
        return discrete_solution
    
    def evaluate_fitness(self, solution):
        """
        Evaluate fitness (average task processing delay in SECONDS)
        """
        if not self.tasks or self.dimension == 0 or self.num_targets == 0:
            return 0.0
            
        discrete_solution = self.continuous_to_discrete(solution)
        total_delay_s = 0
        
        node_load_count = defaultdict(int)
        task_assignments = {} 
        
        best_ap_map = {}
        for task in self.tasks:
            best_ap_map[task.id] = self._find_best_ap_target(task)
        
        for i, task in enumerate(self.tasks):
            source_vehicle = self._get_source_vehicle(task)
            if not source_vehicle:
                task_assignments[task.id] = (None, (100.0, 0)) # 100s penalty
                continue
                
            target_index = discrete_solution[i]
            target_node = self.offload_targets[target_index]
            
            final_node = None
            delay_components = (0, 0) # (transmission_delay, processing_delay_base)
            
            # --- Step 1a: Check if the "guess" is a valid CV ---
            if target_node in self.cv_targets:
                cv_is_valid = False
                if task.required_service in target_node.cached_services:
                    distance = source_vehicle.distance_to(target_node)
                    if distance <= source_vehicle.communication_range: # Check MV's range
                        cv_is_valid = True
                
                if cv_is_valid:
                    final_node = target_node
                    tup = self._calculate_transmission_delay_s(
                        task.data_size, distance, self.network.config['vehicle_tx_power_dbm']
                    )
                    tul_base = task.computation_demand # GCycles
                    delay_components = (tup, tul_base)
            
            # --- Step 1b: If CV is invalid OR AP was chosen, assign to AP ---
            if final_node is None:
                ap_node, trans_delay, proc_base, is_forwarded = best_ap_map[task.id]

                if ap_node is None:
                    task_assignments[task.id] = (None, (100.0, 0)) # Failed
                else:
                    final_node = ap_node
                    delay_components = (trans_delay, proc_base)

            if final_node:
                node_load_count[final_node.id] += 1
                task_assignments[task.id] = (final_node, delay_components)
            elif (None, (100.0, 0)) not in task_assignments:
                task_assignments[task.id] = (None, (100.0, 0)) # Failed


        # --- Step 2: Calculate total delay *based on the calculated load* ---
        for i, task in enumerate(self.tasks):
            final_node, (transmission_delay, processing_base) = task_assignments.get(task.id, (None, (100.0, 0)))
            
            if final_node is None:
                total_delay_s += transmission_delay # 100.0
                continue
            
            node_id = final_node.id
            node_total_resource = self.node_resources.get(node_id, 0.1)
            load = node_load_count[node_id]
            
            if load == 0: load = 1
                
            allocated_resource = node_total_resource / load
            
            processing_delay = self._calculate_processing_delay_s(
                processing_base, allocated_resource
            )
            
            total_delay_s += (transmission_delay + processing_delay)
            
        return total_delay_s / len(self.tasks) if self.tasks else 0.0

    # ---
    # --- Helper functions ---
    # ---

    def _get_source_vehicle(self, task):
        return self.source_vehicle_map.get(task.id)

    def _calculate_transmission_delay_s(self, data_size_mbits: float, distance_m: float, tx_power_dbm: float, is_fiber=False):
        """Calculate transmission delay in SECONDS"""
        if is_fiber:
            delay_sec = data_size_mbits / (1000 + 1e-6) # 1 Gbps
            return delay_sec
            
        capacity_mbps = self.network.get_channel_capacity_mbps(distance_m, tx_power_dbm)
        if capacity_mbps <= 0.1: return 100.0 # 100s penalty
        
        delay_sec = data_size_mbits / capacity_mbps
        return delay_sec

    def _calculate_processing_delay_s(self, task_gcycles: float, allocated_resources_ghz: float):
        """Calculate processing delay in SECONDS"""
        if allocated_resources_ghz <= 0:
            return 100.0
        # GCycles / (GCycles/sec) = seconds
        delay_sec = task_gcycles / allocated_resources_ghz
        return delay_sec

    def _find_associated_ap(self, task):
        """Finds the closest AP in the MV's range for a task."""
        source_vehicle = self._get_source_vehicle(task)
        if not source_vehicle: return None, float('inf')
        
        associated_ap = None
        min_dist = float('inf')
        
        for ap in self.ap_targets:
            distance = ap.distance_to(source_vehicle)
            if distance < min_dist and distance <= source_vehicle.communication_range:
                min_dist = distance
                associated_ap = ap
        
        return associated_ap, min_dist
        
    def _find_best_ap_target(self, task):
        """Finds the best AP target (associated or forwarded) for a task."""
        associated_ap, assoc_dist = self._find_associated_ap(task)
        
        if associated_ap is None:
            return None, 100.0, 0, False # (node, trans_delay, proc_base, is_forwarded)

        tup_m_k = self._calculate_transmission_delay_s(
            task.data_size, assoc_dist, self.network.config['ap_tx_power_dbm']
        )
        tul_base = task.computation_demand # This is now GCycles
        
        # (Eq. 13) Delay for processing at associated AP
        delay_assoc = tup_m_k + self._calculate_processing_delay_s(tul_base, associated_ap.computing_resources)
        
        # (Eq. 14) Delay for processing at *best* forwarded AP
        delay_fwd = float('inf')
        best_fwd_ap = None
        ts_k_k_prime = self._calculate_transmission_delay_s(task.data_size, 0, 0, is_fiber=True)

        for ap in self.ap_targets:
            if ap.id == associated_ap.id:
                continue
            
            fwd_proc_delay = self._calculate_processing_delay_s(tul_base, ap.computing_resources)
            total_fwd_delay = tup_m_k + ts_k_k_prime + fwd_proc_delay
            
            if total_fwd_delay < delay_fwd:
                delay_fwd = total_fwd_delay
                best_fwd_ap = ap
                
        if delay_assoc <= delay_fwd or best_fwd_ap is None:
            return associated_ap, tup_m_k, tul_base, False
        else:
            return best_fwd_ap, (tup_m_k + ts_k_k_prime), tul_base, True


    def differential_evolution(self, nest_idx):
        if self.nest_size < 3 or self.dimension < 1:
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
        if np.isnan(trial_fitness):
            trial_fitness = 100.0

        if trial_fitness < self.fitness[nest_idx]:
            self.nests[nest_idx] = trial
            self.fitness[nest_idx] = trial_fitness
            
    def run(self):
        """Execute the discrete cuckoo search algorithm"""
        if not self.tasks or self.dimension == 0 or self.num_targets == 0:
            return np.array([]), 0.0, []
            
        self.initialize_nests()
        
        fitness_history = []
        
        if self.best_fitness == float('inf'):
             fitness_history.append(10) # High starting value (10s)
        elif np.isnan(self.best_fitness):
             fitness_history.append(10)
        else:
             fitness_history.append(self.best_fitness)
        
        for iteration in range(self.max_iterations):
            if self.best_nest is not None and len(self.best_nest) == self.dimension:
                for i in range(self.nest_size):
                    step = self.levy_flight()
                    if step.size == 0: continue
                    new_nest = self.nests[i] + self.alpha * step * (self.best_nest - self.nests[i])
                    new_nest = np.clip(new_nest, 0, 1)
                    
                    new_fitness = self.evaluate_fitness(new_nest)
                    
                    if new_fitness < self.fitness[i]:
                        self.nests[i] = new_nest
                        self.fitness[i] = new_fitness
                        if new_fitness < self.best_fitness:
                            self.best_nest = new_nest.copy()
                            self.best_fitness = new_fitness
            
            for i in range(self.nest_size):
                if np.random.random() < self.pa:
                    self.differential_evolution(i)
            
            if np.isnan(self.best_fitness) or self.best_fitness == float('inf'):
                fitness_history.append(fitness_history[-1]) 
            else:
                fitness_history.append(self.best_fitness)
                    
        return self.continuous_to_discrete(self.best_nest), self.best_fitness, fitness_history