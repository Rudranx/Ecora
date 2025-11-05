"""
Corrected Implementation of Algorithm 2: Discrete Cuckoo Search Based on Differential Evolution
For task offloading in vehicular networks
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.special import gamma
from collections import defaultdict

class DiscreteCuckooSearch:
    """
    Implements Algorithm 2: Discrete Cuckoo Search Based on Differential Evolution
    Correctly implements Equations 18-24 from the ECORA paper
    """
    
    def __init__(self, config, network, tasks):
        self.config = config
        self.network = network
        self.tasks = tasks
        
        # Get offloading targets
        self.cv_targets = list(network.collaborative_vehicles)
        self.ap_targets = list(network.access_points)
        self.offload_targets = self.cv_targets + self.ap_targets
        self.num_targets = len(self.offload_targets)
        
        # Algorithm parameters from config
        self.nest_size = config['algorithms']['cuckoo']['nest_size']
        self.max_iterations = config['algorithms']['cuckoo']['max_iterations']
        self.pa = config['algorithms']['cuckoo'].get('discovery_probability', 0.25)
        self.alpha3 = config['algorithms']['cuckoo'].get('step_size_factor', 1.0)  # α3 in paper
        self.beta3 = config['algorithms']['cuckoo']['levy_beta']  # β3 in paper
        
        # Differential Evolution parameters
        self.kappa = config['algorithms']['differential_evolution']['scaling_factor']  # κ in paper
        self.CR = config['algorithms']['differential_evolution']['crossover_probability']
        
        # Problem dimension: 3 decisions per task (CV, AP, Forward)
        self.dimension = len(tasks) * 3 if tasks else 0
        
        # Resource idle threshold (ρ^f_th from Equation 5)
        self.idle_threshold = 0.3  # 30% minimum idle resources
        
        # Optimization variables
        self.nests = None
        self.fitness = None
        self.best_nest = None
        self.best_fitness = float('inf')
        
        # Track source vehicles for tasks
        self.source_vehicle_map = {}
        for mv in self.network.mission_vehicles:
            if mv.current_task:
                self.source_vehicle_map[mv.current_task.id] = mv
        
        # Track node resources
        self.node_resources = {}
        for target in self.offload_targets:
            self.node_resources[target.id] = target.computing_resources
    
    def initialize_nests(self):
        """Initialize nest positions in continuous space [0,1]^d"""
        if self.dimension == 0:
            self.nests, self.fitness, self.best_nest = [], [], []
            return
        
        # Random initialization in [0,1]
        self.nests = np.random.rand(self.nest_size, self.dimension)
        self.fitness = np.full(self.nest_size, float('inf'))
        
        # Evaluate initial fitness
        for i in range(self.nest_size):
            self.fitness[i] = self.evaluate_fitness(self.nests[i])
        
        # Find initial best
        if self.fitness.size > 0:
            best_idx = np.nanargmin(self.fitness)
            if not np.isnan(self.fitness[best_idx]):
                self.best_nest = self.nests[best_idx].copy()
                self.best_fitness = self.fitness[best_idx]
            else:
                self.best_nest = self.nests[0].copy()
                self.best_fitness = 100.0  # 100ms penalty
        else:
            self.best_fitness = 100.0
    
    def levy_flight(self, current_position):
        """
        CORRECTED Levy flight implementation (Equations 18-19)
        x^(t+1)_i,m = x^t_i,m + α3 * Rand * Levy(β3)
        where Levy(β3) = 0.01 * u/|v|^(1/β3) * (x^t_j,m - b^t_g,m)
        """
        if self.dimension == 0:
            return current_position
        
        # Calculate sigma_u according to the paper
        sigma_u = (gamma(1 + self.beta3) * np.sin(np.pi * self.beta3 / 2) / 
                  (gamma((1 + self.beta3) / 2) * self.beta3 * 2**((self.beta3 - 1) / 2)))**(1 / self.beta3)
        sigma_v = 1
        
        # Generate u and v
        u = np.random.normal(0, sigma_u, self.dimension)
        v = np.random.normal(0, sigma_v, self.dimension)
        
        # CRITICAL FIX: Include (best_nest - current_position) from Equation 19
        levy_step = 0.01 * (u / (np.abs(v)**(1/self.beta3) + 1e-10)) * (self.best_nest - current_position)
        
        # Apply with random factor and step size (Equation 18)
        rand_factor = np.random.rand()
        new_position = current_position + self.alpha3 * rand_factor * levy_step
        
        # Clip to bounds [0,1]
        return np.clip(new_position, 0, 1)
    
    def continuous_to_discrete(self, continuous_solution):
        """
        Maps continuous solution to discrete offloading decisions
        Implements Equations 20-21 from the paper
        """
        if not self.tasks or self.dimension == 0:
            return {}
        
        discrete_solution = {}
        
        for i, task in enumerate(self.tasks):
            # Get the 3 continuous values for this task
            idx_base = i * 3
            if idx_base + 2 >= len(continuous_solution):
                continue
            
            cv_value = continuous_solution[idx_base]
            ap_value = continuous_solution[idx_base + 1]
            forward_value = continuous_solution[idx_base + 2]
            
            # Apply H() function from Equation 20
            # Then threshold with γ ~ U(0,1) as per Equation 21
            gamma = np.random.rand()
            
            # Binary decisions
            use_cv = 1 if cv_value < gamma else 0
            use_ap = 1 if ap_value < gamma else 0
            use_forward = 1 if forward_value < gamma else 0
            
            # Apply constraint from Equation 4: exactly one should be selected
            # Priority based on paper: CV > Associated AP > Forwarded AP
            source_mv = self._get_source_vehicle(task)
            
            if use_cv == 1 and source_mv:
                # Try to find a suitable CV
                best_cv = self._find_best_cv(task, source_mv)
                if best_cv:
                    discrete_solution[task.id] = ('CV', best_cv)
                    continue
            
            if use_ap == 1 and source_mv:
                # Use associated AP
                best_ap = self._find_associated_ap(task, source_mv)
                if best_ap:
                    discrete_solution[task.id] = ('AP', best_ap)
                    continue
            
            if use_forward == 1 and source_mv:
                # Forward to another AP
                forward_ap = self._find_forward_ap(task, source_mv)
                if forward_ap:
                    discrete_solution[task.id] = ('FORWARD', forward_ap)
                    continue
            
            # Default: use closest AP
            if source_mv:
                default_ap = self._find_associated_ap(task, source_mv)
                discrete_solution[task.id] = ('AP', default_ap)
        
        return discrete_solution
    
    def evaluate_fitness(self, solution):
        """
        Calculate average task processing delay (milliseconds)
        Implements Equations 12-14 from the paper
        """
        if not self.tasks or self.dimension == 0:
            return 0.0
        
        discrete_solution = self.continuous_to_discrete(solution)
        total_delay_ms = 0
        node_load_count = defaultdict(int)
        
        # First pass: count load on each node
        for task in self.tasks:
            decision = discrete_solution.get(task.id, ('AP', None))
            _, target = decision
            if target:
                node_load_count[target.id] += 1
        
        # Second pass: calculate delays
        for task in self.tasks:
            decision = discrete_solution.get(task.id, ('AP', None))
            offload_type, target = decision
            
            if target is None:
                total_delay_ms += 100.0  # Penalty for no valid target
                continue
            
            # Calculate delay based on offloading type
            if offload_type == 'CV':
                delay = self._calculate_cv_delay(task, target, node_load_count[target.id])
            elif offload_type == 'AP':
                delay = self._calculate_ap_delay(task, target, node_load_count[target.id])
            elif offload_type == 'FORWARD':
                delay = self._calculate_forward_delay(task, target, node_load_count[target.id])
            else:
                delay = 100.0
            
            total_delay_ms += delay
        
        return total_delay_ms / len(self.tasks) if self.tasks else 0.0
    
    def _calculate_cv_delay(self, task, cv, load):
        """
        Calculate delay for CV offloading (Equation 12)
        t^p_m,n = t^up_m,n + t^ul_m,n
        """
        source_mv = self._get_source_vehicle(task)
        if not source_mv:
            return 100.0
        
        distance = source_mv.distance_to(cv)
        
        # Transmission delay t^up_m,n (CORRECTED UNITS)
        data_size_mb = task.data_size  # Already in Mb from corrected config
        channel_capacity = self.network.get_channel_capacity_mbps(
            distance, self.network.config['channel']['vehicle_tx_power_dbm']
        )
        if channel_capacity <= 0:
            return 100.0
        t_up = (data_size_mb / channel_capacity) * 1000  # Convert to ms
        
        # Processing delay t^ul_m,n (CORRECTED UNITS)
        computation_ghz = task.computation_demand  # Already in GHz from corrected config
        allocated_resource = cv.computing_resources / max(1, load)
        t_ul = (computation_ghz / allocated_resource) * 1000  # Convert to ms
        
        return t_up + t_ul
    
    def _calculate_ap_delay(self, task, ap, load):
        """
        Calculate delay for AP offloading (Equation 13)
        t^p_m,k = t^up_m,k + t^ul_m,k
        """
        source_mv = self._get_source_vehicle(task)
        if not source_mv:
            return 100.0
        
        distance = source_mv.distance_to(ap)
        
        # Transmission delay t^up_m,k
        data_size_mb = task.data_size  # Mb
        channel_capacity = self.network.get_channel_capacity_mbps(
            distance, self.network.config['channel']['ap_tx_power_dbm']
        )
        if channel_capacity <= 0:
            return 100.0
        t_up = (data_size_mb / channel_capacity) * 1000  # ms
        
        # Processing delay t^ul_m,k
        computation_ghz = task.computation_demand  # GHz
        allocated_resource = ap.computing_resources / max(1, load)
        t_ul = (computation_ghz / allocated_resource) * 1000  # ms
        
        return t_up + t_ul
    
    def _calculate_forward_delay(self, task, forward_ap, load):
        """
        Calculate delay for forwarded AP offloading (Equation 14)
        t^p_m,k,k' = t^up_m,k + t^s_k,k' + t^ul_m,k'
        """
        source_mv = self._get_source_vehicle(task)
        if not source_mv:
            return 100.0
        
        # Find associated AP
        assoc_ap = self._find_associated_ap(task, source_mv)
        if not assoc_ap:
            return 100.0
        
        distance_to_assoc = source_mv.distance_to(assoc_ap)
        
        # MV to associated AP transmission (t^up_m,k)
        data_size_mb = task.data_size  # Mb
        channel_capacity = self.network.get_channel_capacity_mbps(
            distance_to_assoc, self.network.config['channel']['ap_tx_power_dbm']
        )
        if channel_capacity <= 0:
            return 100.0
        t_up_mk = (data_size_mb / channel_capacity) * 1000  # ms
        
        # AP to AP transmission via fiber (t^s_k,k')
        # Assume 1 Gbps fiber connection
        t_s_kk = (data_size_mb / 1000) * 1000  # ms (1 Gbps = 1000 Mbps)
        
        # Processing at forwarded AP (t^ul_m,k')
        computation_ghz = task.computation_demand  # GHz
        allocated_resource = forward_ap.computing_resources / max(1, load)
        t_ul_mk = (computation_ghz / allocated_resource) * 1000  # ms
        
        return t_up_mk + t_s_kk + t_ul_mk
    
    def differential_evolution_update(self, current_nest_idx):
        """
        Implements differential evolution operations (Equations 22-24)
        """
        # Mutation (Equation 22)
        # Select 3 different nests
        indices = list(range(self.nest_size))
        indices.remove(current_nest_idx)
        p, q = np.random.choice(indices, 2, replace=False)
        
        # u^t_i,m = x^t_i,m + κ * (x^t_p,m - x^t_q,m)
        mutant = self.nests[current_nest_idx] + self.kappa * (self.nests[p] - self.nests[q])
        mutant = np.clip(mutant, 0, 1)
        
        # Crossover (Equation 23)
        candidate = np.copy(self.nests[current_nest_idx])
        for j in range(self.dimension):
            alpha4 = np.random.rand()
            beta4 = np.random.randint(0, self.dimension)
            
            if alpha4 < self.CR or j == beta4:
                candidate[j] = mutant[j]
        
        return candidate
    
    def run(self):
        """
        Main optimization loop implementing Algorithm 2
        Returns convergence history for plotting
        """
        self.initialize_nests()
        convergence_history = []
        
        for iteration in range(self.max_iterations):
            # Global search with Levy flight
            for i in range(self.nest_size):
                # Generate new solution via Levy flight
                new_nest = self.levy_flight(self.nests[i])
                
                # Evaluate new solution
                new_fitness = self.evaluate_fitness(new_nest)
                
                # Greedy selection: keep if better
                if new_fitness < self.fitness[i]:
                    self.nests[i] = new_nest
                    self.fitness[i] = new_fitness
                    
                    # Update global best
                    if new_fitness < self.best_fitness:
                        self.best_nest = new_nest.copy()
                        self.best_fitness = new_fitness
            
            # Local search with differential evolution
            for i in range(self.nest_size):
                # With probability (1-Pa), do local search
                if np.random.rand() > self.pa:
                    # Apply differential evolution
                    candidate = self.differential_evolution_update(i)
                    
                    # Evaluate candidate
                    candidate_fitness = self.evaluate_fitness(candidate)
                    
                    # Selection (Equation 24)
                    if candidate_fitness < self.fitness[i]:
                        self.nests[i] = candidate
                        self.fitness[i] = candidate_fitness
                        
                        # Update global best
                        if candidate_fitness < self.best_fitness:
                            self.best_nest = candidate.copy()
                            self.best_fitness = candidate_fitness
            
            # Record convergence
            convergence_history.append(self.best_fitness)
        
        return convergence_history
    
    # Helper methods
    def _get_source_vehicle(self, task):
        """Get the mission vehicle that generated this task"""
        return self.source_vehicle_map.get(task.id)
    
    def _find_best_cv(self, task, source_mv):
        """
        Find best CV for task offloading based on Equation 5 constraints
        """
        best_cv = None
        best_score = -1
        
        for cv in self.cv_targets:
            # Check communication range
            if source_mv.distance_to(cv) > source_mv.communication_range:
                continue
            
            # Check idle resource threshold (ρ^f_y >= ρ^f_th)
            idle_rate = cv.idle_computing_resources / cv.computing_resources
            if idle_rate < self.idle_threshold:
                continue
            
            # Check service cache (S_y,f = 1)
            if task.required_service not in cv.cached_services:
                continue
            
            # Calculate score (higher is better)
            distance = source_mv.distance_to(cv)
            score = idle_rate / (distance + 1)
            
            if score > best_score:
                best_score = score
                best_cv = cv
        
        return best_cv
    
    def _find_associated_ap(self, task, source_mv):
        """Find the closest AP within communication range"""
        best_ap = None
        min_distance = float('inf')
        
        for ap in self.ap_targets:
            distance = source_mv.distance_to(ap)
            if distance <= ap.communication_range and distance < min_distance:
                min_distance = distance
                best_ap = ap
        
        return best_ap
    
    def _find_forward_ap(self, task, source_mv):
        """Find AP with most idle resources for forwarding"""
        best_ap = None
        max_idle = 0
        
        for ap in self.ap_targets:
            # Don't forward to associated AP
            if ap == self._find_associated_ap(task, source_mv):
                continue
            
            idle_rate = ap.idle_computing_resources / ap.computing_resources
            if idle_rate > max_idle:
                max_idle = idle_rate
                best_ap = ap
        
        return best_ap
